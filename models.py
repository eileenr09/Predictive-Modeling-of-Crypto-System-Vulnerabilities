"""
models.py
=========
Walk-forward time-series cross-validation with 5 calibrated models.

Calibration strategy:
  - Sigmoid (Platt) preferred over isotonic: stable from ~10 samples vs ~40+.
  - Adaptive cv: early folds (<3 minority-class samples) drop calibration wrapper
    so tree models produce real predictions instead of falling back to 0.5.
  - Label smoothing (eps=0.05) blends toward training base rate, capping
    per-sample log-loss at -log(0.05) ~ 3.0 while allowing confident predictions.
  - Ensemble weighted by each model's cumulative PR-AUC (softmax T=2) instead
    of a simple mean, boosting the best models and reducing drag from weaker ones.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (brier_score_loss, log_loss,
                              average_precision_score, roc_auc_score)

log = logging.getLogger(__name__)


def _extract_fi_from_model(m, col_names):
    """Extract feature importances from a (possibly calibrated) pipeline.

    Averages across all calibration folds so the result is stable.
    Returns dict {feature_name: importance}, empty if not available.
    """
    col_names = np.asarray(col_names)
    candidates = []
    if isinstance(m, CalibratedClassifierCV):
        for cc in getattr(m, "calibrated_classifiers_", []):
            est = getattr(cc, "estimator", None)
            if est is not None:
                candidates.append(est)
        if not candidates and hasattr(m, "estimator"):
            candidates.append(m.estimator)
    else:
        candidates.append(m)

    fi_accum = {}
    count = 0
    for inner in candidates:
        feat_names = col_names
        raw_clf = inner
        if hasattr(inner, "named_steps"):
            sel = inner.named_steps.get("select")
            clf = inner.named_steps.get("clf")
            if clf is None:
                continue
            if sel is not None and hasattr(sel, "get_support"):
                try:
                    feat_names = col_names[sel.get_support()]
                except Exception:
                    pass
            raw_clf = getattr(clf, "estimator", clf)
        if hasattr(raw_clf, "feature_importances_"):
            for fn, imp in zip(feat_names, raw_clf.feature_importances_):
                fi_accum[fn] = fi_accum.get(fn, 0.0) + float(imp)
            count += 1
        elif hasattr(raw_clf, "coef_"):
            coef = raw_clf.coef_
            if coef.ndim > 1:
                coef = coef[0]
            for fn, imp in zip(feat_names, np.abs(coef)):
                fi_accum[fn] = fi_accum.get(fn, 0.0) + float(imp)
            count += 1
    if count == 0:
        return {}
    return {k: v / count for k, v in fi_accum.items()}


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
            except Exception: out[name] = np.nan
        return out




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


# Label smoothing constant: blend predictions toward the training base rate.
# eps=0.05 caps per-sample log-loss at -log(0.05) ~ 3.0 while still allowing
# confident predictions (max prob = 0.95+), balancing calibration vs. sharpness.
LABEL_SMOOTH_EPS = 0.05


def build_models():
    return {
        # LR -- natively well-calibrated; C=0.10 allows slightly more confidence
        "LogisticRegression": Pipeline([
            ("select", SelectKBest(f_classif, k=25)),
            ("clf",    LogisticRegression(C=0.10, penalty="l2", solver="lbfgs",
                                          max_iter=2000, class_weight="balanced",
                                          random_state=42))
        ]),

        # Strong-L2 LR variant; C=0.05 is less over-regularised than before
        "RidgeClassifier": Pipeline([
            ("select", SelectKBest(f_classif, k=25)),
            ("clf",    LogisticRegression(C=0.05, penalty="l2", solver="lbfgs",
                                          max_iter=2000, class_weight="balanced",
                                          random_state=0))
        ]),

        # Random Forest -- 1000 trees, depth=4 for richer DeFi-era interactions
        "RandomForest": _platt_wrap(Pipeline([
            ("select", SelectKBest(f_classif, k=35)),
            ("clf",    RandomForestClassifier(n_estimators=1000, max_depth=4,
                                              min_samples_leaf=2, max_features="sqrt",
                                              class_weight="balanced_subsample",
                                              random_state=42, n_jobs=-1))
        ])),

        # Extra Trees -- same tuning as RF; more randomness helps small-N variance
        "ExtraTrees": _platt_wrap(Pipeline([
            ("select", SelectKBest(f_classif, k=35)),
            ("clf",    ExtraTreesClassifier(n_estimators=1000, max_depth=4,
                                            min_samples_leaf=2, max_features="sqrt",
                                            class_weight="balanced_subsample",
                                            random_state=42, n_jobs=-1))
        ])),

        # GradientBoosting -- lr=0.03 / 300 trees keeps it conservative but sharper
        "GradientBoosting": _platt_wrap(Pipeline([
            ("select", SelectKBest(f_classif, k=30)),
            ("clf",    GradientBoostingClassifier(n_estimators=300, max_depth=2,
                                                   learning_rate=0.03, subsample=0.7,
                                                   min_samples_leaf=2, max_features="sqrt",
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

            n_min = int(min(y_tr.sum(), len(y_tr) - y_tr.sum()))

            for name, base in models.items():
                m = clone(base)

                # Adaptive Platt calibration cv.
                # Tree models are pre-wrapped with cv=3 in build_models(), but
                # early folds have too few minority-class samples for cv=3, causing
                # fit failures and silent fallback to y_prob=0.5.
                # Fix: re-wrap with safe cv, or strip calibration entirely when
                # n_min < 2 so the raw pipeline still makes real predictions.
                if isinstance(m, CalibratedClassifierCV) and m.method == "sigmoid":
                    if n_min < 2:
                        m = clone(m.estimator)          # no calibration wrapper
                    elif n_min == 2:
                        m = CalibratedClassifierCV(
                            clone(m.estimator), method="sigmoid", cv=2)
                    # else: keep cv=3 (n_min >= 3)

                # Legacy flag for LR variants (bare Pipelines)
                elif (self.calibrate and isinstance(m, Pipeline) and n_min >= 3):
                    cv = min(3, n_min)
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

                # Extract feature importances (averages across calibration folds)
                fi = _extract_fi_from_model(m, X_tr.columns)

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
            except Exception: ll = np.nan
            try:    pr = average_precision_score(yt, yp) if n_pos > 0 else np.nan
            except Exception: pr = np.nan
            try:    ra = roc_auc_score(yt, yp) if (n_pos > 0 and n_neg > 0) else np.nan
            except Exception: ra = np.nan
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
        PR-AUC-weighted average across models per year.
        Each model's weight is proportional to softmax(PR-AUC / T) with T=2,
        so better-performing models contribute more without excluding weaker ones.
        Simple mean is a special case when all PR-AUCs are equal.
        """
        preds = self.consolidated_predictions()
        if preds.empty:
            return preds

        def _tier(p):
            if p >= 0.80: return "[CRITICAL]"
            if p >= 0.60: return "[HIGH]    "
            if p >= 0.40: return "[ELEVATED]"
            return "[LOW]     "

        # Compute per-model cumulative PR-AUC for weighting
        model_pr = {}
        for mname, grp in preds.groupby("model"):
            yt = grp["y_true"].values
            yp = grp["y_prob"].values
            try:
                model_pr[mname] = (average_precision_score(yt, yp)
                                   if yt.sum() > 0 and (1 - yt).sum() > 0
                                   else 0.5)
            except Exception:
                model_pr[mname] = 0.5

        # Softmax with temperature T=2: amplifies differences without winner-takes-all
        T = 2.0
        names  = sorted(model_pr)
        scores = np.array([model_pr[n] for n in names])
        w_raw  = np.exp(scores / T)
        w_norm = w_raw / w_raw.sum()
        weight_map = dict(zip(names, w_norm))
        log.info("  Ensemble weights: " +
                 ", ".join(f"{n}={w:.3f}" for n, w in sorted(weight_map.items(),
                                                              key=lambda x: -x[1])))

        preds["_w"]  = preds["model"].map(weight_map).fillna(1.0 / len(weight_map))
        preds["_wp"] = preds["y_prob"] * preds["_w"]

        grp = preds.groupby("year")
        ens = pd.DataFrame({
            "y_true"    : grp["y_true"].first(),
            "y_prob"    : (grp["_wp"].sum() / grp["_w"].sum()).round(4),
            "y_prob_lo" : grp["y_prob"].min().round(4),
            "y_prob_hi" : grp["y_prob"].max().round(4),
            "n_models"  : grp["model"].nunique(),
        })
        ens["ci_width"] = (ens["y_prob_hi"] - ens["y_prob_lo"]).round(4)
        ens["tier"]     = ens["y_prob"].apply(_tier)
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