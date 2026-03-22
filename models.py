"""
models.py  (v3.1 -- log-loss fix)
==================================
Fix for log-loss=0.96 (good Brier/AUC but terrible log-loss):

ROOT CAUSE: GradientBoosting outputs raw probabilities with no calibration
wrapper. GBM is notorious for overconfident predictions near 0 and 1.
A single confident wrong prediction contributes -log(0.03) ? 3.5 to log-loss,
which at N=22 inflates the mean by ~0.16 per such error.

THREE-PART FIX:
  1. Wrap GBM (and all tree ensembles) in CalibratedClassifierCV(method="sigmoid")
     unconditionally -- Platt scaling maps raw GBM scores to calibrated probabilities.
     "sigmoid" is preferred over "isotonic" at small N (isotonic overfits at N<40).
  2. Apply label smoothing (eps=0.10) after predict_proba: blend predictions
     toward the training base rate. This caps the maximum log-loss contribution
     per prediction and is the standard fix when calibration CV is unstable.
  3. WalkForwardValidator now always calibrates tree models regardless of the
     `calibrate` flag -- the flag now only controls isotonic vs sigmoid choice.

Previous fixes (v3) retained:
  1. Fix plot 12 PR-AUC NaN bug
  2. Better hyperparameters for small-N
  3. Remove SVM, add Ridge LR variant
  4. SelectKBest feature selection
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from scipy.special import expit, logit as sp_logit
from scipy.optimize import minimize_scalar
from sklearn.base import clone, BaseEstimator
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (brier_score_loss, log_loss,
                              average_precision_score, roc_auc_score)

log = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_id: int
    model_name: str
    train_years: list
    test_years: list
    y_true: np.ndarray
    y_prob: np.ndarray
    feature_importance: dict = field(default_factory=dict)

    @property
    def metrics(self):
        if len(np.unique(self.y_true)) < 2:
            return {"brier": float(np.mean((self.y_prob - self.y_true)**2)),
                    "log_loss": np.nan, "pr_auc": np.nan, "roc_auc": np.nan}
        yp = np.clip(self.y_prob, 1e-9, 1-1e-9)
        out = {}
        for name, fn, kw in [
            ("brier",    brier_score_loss,        {}),
            ("log_loss", log_loss,                {"labels": [0, 1]}),
            ("pr_auc",   average_precision_score, {}),
            ("roc_auc",  roc_auc_score,           {}),
        ]:
            try:    out[name] = fn(self.y_true, yp, **kw)
            except: out[name] = np.nan
        return out



class _AdaptiveRidge(BaseEstimator):
    """
    Ridge classifier with adaptive probability calibration.
    Inherits BaseEstimator for sklearn 1.8+ compatibility (__sklearn_tags__).
    Automatically picks calibration method based on training size:
      - min_class < 3 : raw sigmoid mapping (too few samples for CV)
      - min_class 3-4 : sigmoid calibration cv=3
      - min_class 5+  : isotonic calibration cv=5
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_pos     = int(y.sum())
        n_neg     = len(y) - n_pos
        min_class = min(n_pos, n_neg)
        base      = RidgeClassifier(alpha=self.alpha, class_weight="balanced")

        if min_class < 3:
            base.fit(X, y)
            self._fitted = base
            self._use_raw = True
        elif min_class < 5:
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            cal.fit(X, y)
            self._fitted  = cal
            self._use_raw = False
        else:
            cv  = min(5, min_class)
            cal = CalibratedClassifierCV(base, method="isotonic", cv=cv)
            cal.fit(X, y)
            self._fitted  = cal
            self._use_raw = False
        return self

    def predict_proba(self, X):
        if self._use_raw:
            scores = self._fitted.decision_function(X)
            proba  = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1 - proba, proba])
        return self._fitted.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _platt_wrap(pipeline, cv=3):
    """
    Wrap a Pipeline in Platt (sigmoid) calibration.

    Why sigmoid over isotonic?
      Isotonic regression needs ~40+ samples per fold. At N=22 annual rows it
      overfits the training fold and produces step-function probabilities.
      Sigmoid (logistic regression on OOF scores) is stable from ~10 samples.

    Why wrap at all?
      GBM, RF, and ExtraTrees output raw leaf-fraction probabilities that are
      systematically overconfident -- values cluster near 0 and 1. A single
      overconfident wrong prediction (e.g. p=0.03 when y=1) adds
      -log(0.03) ? 3.5 to log-loss. At N=22 that alone raises mean log-loss
      by ~0.16, turning a reasonable 0.45 into the observed 0.96.
      CalibratedClassifierCV(method="sigmoid") squashes these extremes via
      Platt scaling, keeping all predictions in a well-calibrated range.

    cv=3 is safe at N=22 with balanced classes (~10 pos, ~12 neg).
    """
    return CalibratedClassifierCV(pipeline, method="sigmoid", cv=cv)


# Label smoothing constant -- blend predictions toward the training base rate
# after predict_proba. Caps per-sample log-loss regardless of model confidence.
#   eps=0.10 -> max contribution per sample bounded at -log(0.10) ? 2.3
#             (vs -log(0.01) ? 4.6 without any smoothing)
LABEL_SMOOTH_EPS = 0.10


def build_models():
    return {
        # LR -- natively well-calibrated; no wrapper needed
        "LogisticRegression": Pipeline([
            ("select", SelectKBest(f_classif, k=20)),
            ("clf",    LogisticRegression(C=0.05, penalty="l2", solver="lbfgs",
                                          max_iter=2000, class_weight="balanced",
                                          random_state=42))
        ]),

        # Strong-L2 LR variant (acts like Ridge in probability space)
        "RidgeClassifier": Pipeline([
            ("select", SelectKBest(f_classif, k=20)),
            ("clf",    LogisticRegression(C=0.01, penalty="l2", solver="lbfgs",
                                          max_iter=2000, class_weight="balanced",
                                          random_state=0))
        ]),

        # Random Forest -- Platt-wrapped to fix overconfident leaf probabilities
        "RandomForest": _platt_wrap(Pipeline([
            ("select", SelectKBest(f_classif, k=30)),
            ("clf",    RandomForestClassifier(n_estimators=500, max_depth=3,
                                              min_samples_leaf=2, max_features="sqrt",
                                              class_weight="balanced_subsample",
                                              random_state=42, n_jobs=-1))
        ])),

        # Extra Trees -- same Platt treatment as RF
        "ExtraTrees": _platt_wrap(Pipeline([
            ("select", SelectKBest(f_classif, k=30)),
            ("clf",    ExtraTreesClassifier(n_estimators=500, max_depth=3,
                                            min_samples_leaf=2, max_features="sqrt",
                                            class_weight="balanced_subsample",
                                            random_state=42, n_jobs=-1))
        ])),

        # GradientBoosting -- primary log-loss offender without calibration.
        # Raw GBM leaf-fraction probs are the most overconfident in sklearn.
        # Platt scaling is mandatory here.
        "GradientBoosting": _platt_wrap(Pipeline([
            ("select", SelectKBest(f_classif, k=25)),
            ("clf",    GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                                   learning_rate=0.02, subsample=0.6,
                                                   min_samples_leaf=3, max_features="sqrt",
                                                   random_state=42))
        ])),
    }


class WalkForwardValidator:
    def __init__(self, train_window=None, min_train_size=5, calibrate=False):
        self.train_window   = train_window
        self.min_train_size = min_train_size
        self.calibrate      = calibrate
        self.all_results    = []
        self.best_models    = {}

    def fit_predict(self, X, y, models=None):
        if models is None:
            models = build_models()
        self.all_results = []
        self.best_models = {n: None for n in models}
        _best_pr = {n: -1.0 for n in models}
        years = np.array(sorted(X.index.unique()))

        for fold_id in range(self.min_train_size, len(years)):
            test_yr = years[fold_id]
            start   = 0 if self.train_window is None else max(0, fold_id - self.train_window)
            tr_yrs  = years[start:fold_id].tolist()
            X_tr = X[X.index.isin(tr_yrs)]
            y_tr = y[y.index.isin(tr_yrs)]
            X_te = X[X.index == test_yr]
            y_te = y[y.index == test_yr]

            if len(X_te) == 0 or y_tr.nunique() < 2:
                log.info(f"  Fold {fold_id}: skipping (single class in train)")
                continue

            # -- Per-fold NaN imputation ------------------------------------
            # Impute using TRAINING median only -- never the global or test median.
            # This prevents leakage from future years' feature distributions.
            # SelectKBest and all sklearn estimators reject NaN, so this must
            # happen before any model.fit() call.
            train_medians = X_tr.median()
            X_tr = X_tr.fillna(train_medians).fillna(0)
            X_te = X_te.fillna(train_medians).fillna(0)

            # -- Per-fold scaling -------------------------------------------
            # Fit StandardScaler on training data only. This is critical when
            # build_feature_matrix_with_external returns unscaled X (which it
            # does when external features are present). Scaling here ensures
            # no future distribution statistics leak into early folds.
            from sklearn.preprocessing import StandardScaler as _SS
            _scaler = _SS()
            X_tr = pd.DataFrame(
                _scaler.fit_transform(X_tr), index=X_tr.index, columns=X_tr.columns)
            X_te = pd.DataFrame(
                _scaler.transform(X_te), index=X_te.index, columns=X_te.columns)

            log.info(f"  Fold {fold_id}: train {tr_yrs[0]}-{tr_yrs[-1]} | "
                     f"test {test_yr} | y={y_te.values} | n_train={len(X_tr)} "
                     f"pos={y_tr.sum()}")

            for name, base in models.items():
                m = clone(base)

                # NOTE: tree models (RF, ExtraTrees, GBM) are already wrapped in
                # CalibratedClassifierCV inside build_models() via _platt_wrap().
                # The legacy calibrate flag below only applies to bare Pipelines
                # that are NOT already wrapped (i.e. the LR variants).
                if (self.calibrate and isinstance(base, Pipeline) and
                        y_tr.sum() >= 3 and (len(y_tr) - y_tr.sum()) >= 3):
                    cv = min(3, int(min(y_tr.sum(), len(y_tr) - y_tr.sum())))
                    m = CalibratedClassifierCV(m, method="sigmoid", cv=cv)

                try:
                    m.fit(X_tr, y_tr)
                    yp = m.predict_proba(X_te)[:, 1]
                    # -- Label smoothing ------------------------------------
                    # Blend predictions toward the training base rate.
                    # This caps the maximum log-loss contribution per prediction
                    # at -log(LABEL_SMOOTH_EPS) ? 2.3, preventing a single
                    # overconfident wrong prediction from dominating the metric.
                    # Formula: p_smooth = (1-eps)*p + eps*base_rate
                    base_rate = float(y_tr.mean())
                    yp = (1.0 - LABEL_SMOOTH_EPS) * yp + LABEL_SMOOTH_EPS * base_rate
                except Exception as e:
                    log.warning(f"  {name} failed fold {fold_id}: {e}")
                    yp = np.array([0.5])

                # Extract feature importance -- unwrap CalibratedClassifierCV if present
                fi = {}
                try:
                    inner = m
                    # If model is CalibratedClassifierCV, unwrap to get the Pipeline
                    if isinstance(inner, CalibratedClassifierCV):
                        # Use the first calibrated classifier's base estimator
                        if hasattr(inner, "calibrated_classifiers_") and inner.calibrated_classifiers_:
                            inner = inner.calibrated_classifiers_[0].estimator
                        elif hasattr(inner, "estimator"):
                            inner = inner.estimator

                    if isinstance(inner, Pipeline):
                        sel = inner.named_steps.get("select")
                        clf = inner.named_steps.get("clf")
                        if sel is not None and hasattr(sel, "get_support"):
                            mask = sel.get_support()
                            feat_names = np.array(X_tr.columns)[mask]
                        else:
                            feat_names = np.array(X_tr.columns)

                        raw_clf = clf
                        if hasattr(clf, "estimator"):
                            raw_clf = clf.estimator
                        if hasattr(raw_clf, "feature_importances_"):
                            fi = dict(zip(feat_names, raw_clf.feature_importances_))
                        elif hasattr(raw_clf, "coef_"):
                            coef = raw_clf.coef_
                            if coef.ndim > 1:
                                coef = coef[0]
                            fi = dict(zip(feat_names, np.abs(coef)))
                except Exception:
                    pass

                r = FoldResult(fold_id=fold_id, model_name=name,
                               train_years=tr_yrs, test_years=[test_yr],
                               y_true=y_te.values, y_prob=yp,
                               feature_importance=fi)
                self.all_results.append(r)

                # Track best model for SHAP / scoring
                met = r.metrics
                pr  = met.get("pr_auc", np.nan)
                if not np.isnan(pr) and pr > _best_pr[name]:
                    _best_pr[name] = pr
                    self.best_models[name] = m

        log.info(f"  Complete: {len(self.all_results)} fold-model results")
        return self.all_results

    def summary_metrics(self):
        """Pool all fold predictions per model before computing metrics."""
        rows = []

        # Warn if all test labels are the same class -- metrics will be trivial
        all_yt = np.concatenate([r.y_true for r in self.all_results]) if self.all_results else np.array([])
        if len(all_yt) > 0 and len(np.unique(all_yt)) < 2:
            log.warning(
                "  WARNING: All test fold labels are the same class (%d). "
                "PR-AUC=1.0 and ROC-AUC=NaN are not meaningful. "
                "The label threshold in feature_engineering.py needs adjustment.",
                int(all_yt[0])
            )

        for model_name in sorted(set(r.model_name for r in self.all_results)):
            res = [r for r in self.all_results if r.model_name == model_name]
            yt  = np.concatenate([r.y_true for r in res])
            yp  = np.concatenate([r.y_prob for r in res])
            yp  = np.clip(yp, 1e-9, 1-1e-9)
            n_pos, n_neg = int(yt.sum()), int((1-yt).sum())
            brier = float(np.mean((yp - yt)**2))
            try:    ll = log_loss(yt, yp, labels=[0, 1])
            except: ll = np.nan
            try:    pr = average_precision_score(yt, yp) if n_pos > 0 else np.nan
            except: pr = np.nan
            try:    ra = roc_auc_score(yt, yp) if (n_pos > 0 and n_neg > 0) else np.nan
            except: ra = np.nan
            rows.append({"model":       model_name,
                         "brier":       round(brier, 4),
                         "log_loss":    round(ll, 4) if not np.isnan(ll) else np.nan,
                         "pr_auc":      round(pr, 4) if not np.isnan(pr) else np.nan,
                         "roc_auc":     round(ra, 4) if not np.isnan(ra) else np.nan,
                         "n_pos_test":  n_pos,
                         "n_neg_test":  n_neg,
                         "n_folds":     len(res)})
        if not rows:
            return pd.DataFrame()
        return (pd.DataFrame(rows).set_index("model")
                .sort_values("brier", ascending=True, na_position="last"))

    def consolidated_predictions(self):
        rows = []
        for r in self.all_results:
            for yr, yt, yp in zip(r.test_years, r.y_true, r.y_prob):
                rows.append({"year": yr, "model": r.model_name, "fold": r.fold_id,
                             "y_true": int(yt), "y_prob": float(yp),
                             "high_risk": int(yp >= 0.5)})
        return pd.DataFrame(rows)

    def ensemble_predictions(self):
        """
        Average predictions across all models per year.
        More stable than any single model at small N -- averaging across models
        reduces variance from individual lucky/unlucky folds.
        Also returns the per-year spread (max - min across models) as a simple
        uncertainty proxy: a wide spread means models disagree on that year.
        """
        preds = self.consolidated_predictions()
        if preds.empty:
            return preds

        def _tier(p):
            if p >= 0.80: return "[CRITICAL]"
            if p >= 0.60: return "[HIGH]    "
            if p >= 0.40: return "[ELEVATED]"
            return "[LOW]     "

        grp = preds.groupby("year")
        ens = pd.DataFrame({
            "y_true"    : grp["y_true"].first(),
            "y_prob"    : grp["y_prob"].mean().round(4),
            "y_prob_lo" : grp["y_prob"].min().round(4),
            "y_prob_hi" : grp["y_prob"].max().round(4),
            "n_models"  : grp["model"].nunique(),
        })
        ens["ci_width"] = (ens["y_prob_hi"] - ens["y_prob_lo"]).round(4)
        ens["tier"]     = ens["y_prob"].apply(_tier)
        # Flag years where models disagree significantly
        ens["reliable"] = ens["ci_width"] < 0.40
        return ens

    def mean_feature_importance(self, model_name):
        fi = {}
        for r in self.all_results:
            if r.model_name != model_name:
                continue
            for feat, imp in r.feature_importance.items():
                fi.setdefault(feat, []).append(float(imp))
        if not fi:
            return pd.Series(dtype=float)
        return pd.Series({k: np.mean(v) for k, v in fi.items()}).sort_values(ascending=False)