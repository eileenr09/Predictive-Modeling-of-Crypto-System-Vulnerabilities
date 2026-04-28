"""
feature_engineering.py  (v3.2 -- label fix)
============================================
WHAT CHANGED:
  - crypto_binary target now uses a MEDIAN SPLIT on crypto_breach_count
    instead of a hardcoded >=1 (or >=2) threshold.

    WHY: Adding the CE + WA datasets pushes crypto_breach_count to 50-190
    per year from 2015 onward. Every year from ~2012 onward now trivially
    exceeds >=1 or >=2, so all 6 negative years fall in 2004-2009 and are
    consumed entirely by training folds -- test folds see only positives.
    Result: PR-AUC=1.0 and ROC-AUC=NaN, which are meaningless.

    The median split computes the threshold dynamically from the FULL
    yearly series, always giving ~50/50 balance regardless of dataset size.
    It targets "was this a HIGH-VOLUME crypto breach year?" -- a meaningful
    and harder prediction problem than mere presence/absence.

    With 23 years this gives ~11-12 pos / 11-12 neg, including 7 negative
    test folds that make ROC-AUC and PR-AUC genuinely informative.

  - build_feature_matrix_with_external: removed global StandardScaler fit
    (scaler was fit on full X before walk-forward split -- mild leakage).
    Scaler is now fit inside each training fold in models.py instead.
    The scale= parameter is kept for API compatibility but now only applies
    to the base build_feature_matrix path (not the external path).

  - Rolling features: added shift(1) before windowing so year T's rolling
    mean only includes T-3..T-1, not T itself.

  - Variance threshold raised 0.05 -> 0.10, correlation threshold
    lowered 0.97 -> 0.90 for cleaner feature set.

Previous fixes retained:
  - Sector/method fraction features
  - Lag 1/2/3 features on key signals
  - Trend slope and delta features
  - Interaction terms
  - External dataset integration
"""

import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def build_feature_matrix(df, target_mode="crypto_binary", scale=True):
    yr = _aggregate_yearly(df)
    yr = _add_sector_features(df, yr)
    yr = _add_method_features(df, yr)
    yr = _add_crypto_specific_features(df, yr)
    yr = _add_rolling_features(yr)
    yr = _add_trend_features(yr)
    yr = _add_interaction_features(yr)

    # ---- Label ----------------------------------------------------------
    if target_mode == "crypto_binary":
        # Use crypto_FRACTION (crypto breaches / total breaches) rather than
        # raw crypto_breach_count for the median split.
        #
        # WHY FRACTION NOT RAW COUNT:
        # The CE dataset adds ~600 crypto incidents uniformly across 2014-2026,
        # inflating raw counts to 60-300 for every modern year. The median of
        # the raw count series ends up around 55-80, which nearly every
        # post-2014 year exceeds. Result: only the partial 2026 year falls
        # below the threshold, leaving just 1 negative in all test folds.
        # With 1 negative, PR-AUC=1.0 is trivially achieved by any model
        # that ranks that one year last -- it is not a meaningful score.
        #
        # crypto_fraction divides by total breaches, removing the "CE was
        # added" artifact. The fraction varies between 0-12% year-to-year
        # even in modern years (visible in plot 04), giving 7 genuinely
        # different negative years spread across both early and recent folds.
        # This makes PR-AUC and ROC-AUC real discriminative measures.
        median_frac = yr["crypto_fraction"].median()
        y = (yr["crypto_fraction"] > median_frac).astype(int).rename("label")
        log.info(f"  Label: crypto_fraction median split (threshold={median_frac:.4f})")
    elif target_mode == "crypto_count":
        y = yr["crypto_breach_count"].rename("label")
    else:  # high_impact
        threshold = yr["total_records"].quantile(0.75)
        y = (yr["total_records"] >= threshold).astype(int).rename("label")

    log.info(f"  Label distribution: {y.value_counts().sort_index().to_dict()}")

    drop_cols = ["crypto_breach_count", "total_records",
                 "total_breaches", "crypto_fraction"]
    X = yr.drop(columns=[c for c in drop_cols if c in yr.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    # Do NOT fillna here -- leave NaN for per-fold imputation in the validator

    # Variance filter (safe to use full dataset -- this is feature selection not scaling)
    std = X.std()
    X   = X.loc[:, std > 0.10]

    # Correlation filter
    X = _remove_high_correlation(X, threshold=0.90)

    y = y.reindex(X.index).fillna(0).astype(int)

    scaler = None
    if scale and len(X) > 0:
        scaler = StandardScaler()
        X_arr  = scaler.fit_transform(X.fillna(0))
        X_out  = pd.DataFrame(X_arr, index=X.index, columns=X.columns)
    else:
        X_out = X.copy()

    log.info(f"  Feature matrix: {X_out.shape} | Positive rate: {y.mean():.2f} "
             f"| Features: {len(X_out.columns)}")
    return X_out, y, list(X_out.columns), scaler, yr


def _remove_high_correlation(X, threshold=0.90):
    """Drop one of each pair of features with |corr| > threshold."""
    corr    = X.corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        log.info(f"  Dropping {len(to_drop)} highly correlated features")
    return X.drop(columns=to_drop, errors="ignore")


def _aggregate_yearly(df):
    grp = df.groupby("year")
    yr  = pd.DataFrame(index=sorted(df["year"].unique()))
    yr.index.name          = "year"
    yr["total_breaches"]   = grp.size()
    yr["crypto_breach_count"] = grp["is_crypto"].sum().astype(float)
    yr["crypto_fraction"]  = yr["crypto_breach_count"] / yr["total_breaches"].replace(0, 1)
    yr["total_records"]    = grp["records_affected"].sum()
    yr["log_records_total"]= np.log1p(yr["total_records"].fillna(0))
    yr["mean_records"]     = grp["records_affected"].mean().fillna(0)
    yr["log_max_records"]  = np.log1p(grp["records_affected"].max().fillna(0))
    yr["unique_entities"]  = grp["entity"].nunique()
    yr["unique_methods"]   = grp["method_category"].nunique()
    yr["source_diversity"] = grp["source_id"].nunique()
    yr["large_breach_count"] = (
        df[df["records_affected"] > 1_000_000]
        .groupby("year").size().reindex(yr.index, fill_value=0)
    )
    return yr


def _add_sector_features(df, yr):
    pivot = df.groupby(["year", "sector"]).size().unstack(fill_value=0)
    frac  = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)
    frac.columns = [f"sector_{c}" for c in frac.columns]
    return yr.join(frac, how="left").fillna(0)


def _add_method_features(df, yr):
    pivot = df.groupby(["year", "method_category"]).size().unstack(fill_value=0)
    frac  = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0)
    frac.columns = [f"method_{c}" for c in frac.columns]
    return yr.join(frac, how="left").fillna(0)


def _add_crypto_specific_features(df, yr):
    cr = df[df["is_crypto"].astype(bool)]
    if len(cr) == 0:
        return yr
    grp = cr.groupby("year")
    yr["crypto_unique_methods"] = grp["method_category"].nunique().reindex(yr.index, fill_value=0)
    yr["crypto_unique_sectors"] = grp["sector"].nunique().reindex(yr.index, fill_value=0)
    yr["crypto_log_records"]    = np.log1p(
        grp["records_affected"].sum().reindex(yr.index, fill_value=0))
    crypto_total = cr.groupby("year").size().replace(0, 1)
    for method in ["hacking", "phishing", "insider", "malware", "smart_contract_exploit"]:
        sub = cr[cr["method_category"] == method].groupby("year").size()
        yr[f"crypto_pct_{method}"] = (sub / crypto_total).reindex(yr.index, fill_value=0)
    # Smart contract exploits as absolute count (key crypto-specific signal)
    sc_count = cr[cr["method_category"] == "smart_contract_exploit"].groupby("year").size()
    yr["crypto_smart_contract_count"] = sc_count.reindex(yr.index, fill_value=0)
    # Smart contract exploits as fraction of ALL breaches (not just crypto).
    # This captures the DeFi explosion post-2021 relative to the whole landscape.
    sc_all = df[df["method_category"] == "smart_contract_exploit"].groupby("year").size()
    yr["smart_contract_frac_all"] = (
        sc_all / yr["total_breaches"].replace(0, 1)
    ).reindex(yr.index, fill_value=0)
    return yr


def _add_rolling_features(yr):
    """
    Lag and rolling features with shift(1) before windowing.
    This ensures year T's rolling mean only uses T-3..T-1, preventing
    look-ahead leakage where the current year's own value feeds its predictor.
    """
    roll_cols = [
        "total_breaches", "log_records_total",
        "crypto_breach_count", "large_breach_count", "crypto_log_records",
        "crypto_smart_contract_count", "smart_contract_frac_all",
    ]
    for col in roll_cols:
        if col not in yr.columns:
            continue
        for lag in [1, 2, 3]:
            yr[f"{col}_lag{lag}"] = yr[col].shift(lag)
        # Shift FIRST, then roll -- key leakage fix
        base = yr[col].shift(1)
        for w in [2, 3]:
            yr[f"{col}_roll{w}_mean"] = base.rolling(w, min_periods=1).mean()

    # Volatility: shifted base only
    for col in ["log_records_total"]:
        if col in yr.columns:
            base = yr[col].shift(1)
            yr[f"{col}_roll3_std"] = base.rolling(3, min_periods=2).std()

    return yr


def _add_trend_features(yr):
    """Directional signals: is the situation getting better or worse?"""
    for col in ["total_breaches", "log_records_total", "crypto_breach_count"]:
        if col not in yr.columns:
            continue
        chg = yr[col].diff()
        yr[f"{col}_delta"]  = chg
        yr[f"{col}_rising"] = (chg > 0).astype(int)

    yr["cumulative_crypto"] = yr.get(
        "crypto_breach_count", pd.Series(0, index=yr.index)
    ).cumsum()

    def _slope(series: pd.Series, w: int = 3) -> pd.Series:
        """Linear slope over the prior w-1 years, excluding the current year."""
        vals   = series.values.astype(float)
        n      = len(vals)
        slopes = np.zeros(n)
        for i in range(n):
            window = vals[max(0, i - w + 1): i]  # up to w-1 prior values
            if len(window) < 2:
                continue
            x = np.arange(len(window), dtype=float)
            # OLS slope: (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2)
            x_bar = x.mean()
            y_bar = window.mean()
            denom = ((x - x_bar) ** 2).sum()
            slopes[i] = ((x - x_bar) * (window - y_bar)).sum() / denom if denom else 0.0
        return pd.Series(slopes, index=series.index)

    yr["breach_trend_slope"] = _slope(yr["total_breaches"])
    yr["crypto_trend_slope"] = _slope(yr.get(
        "crypto_breach_count", pd.Series(0, index=yr.index)))
    return yr


def _add_interaction_features(yr):
    if ("crypto_breach_count_lag1" in yr.columns and
            "crypto_breach_count_rising" in yr.columns):
        yr["crypto_momentum"] = (yr["crypto_breach_count_lag1"] *
                                 yr["crypto_breach_count_rising"])
    if "method_hacking" in yr.columns and "sector_financial" in yr.columns:
        yr["hacking_x_financial"] = yr["method_hacking"] * yr["sector_financial"]
    if ("crypto_trend_slope" in yr.columns and
            "log_records_total_roll3_std" in yr.columns):
        yr["crypto_slope_x_vol"] = (yr["crypto_trend_slope"] *
                                    yr["log_records_total_roll3_std"])
    return yr


# ---------------------------------------------------------------------------
# External dataset integration
# ---------------------------------------------------------------------------

def build_feature_matrix_with_external(
    df,
    data_dir: str = None,
    target_mode: str = "crypto_binary",
    scale: bool = True,
):
    """
    Extends build_feature_matrix() with external datasets.
    Returns (X_unscaled, y, feat_names, scaler, yr_df).

    NOTE: The StandardScaler is fit on the full X here only when scale=True
    AND no external features are available (legacy path). When external
    features are present the scaler is intentionally omitted -- per-fold
    scaling happens inside WalkForwardValidator using only training data,
    which prevents leakage of future distribution statistics.
    """
    from external_data import load_external_features

    # Build base features -- unscaled so we can join external cols first
    X_base, y, feat_names, _, yr_df = build_feature_matrix(
        df, target_mode=target_mode, scale=False
    )

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    ext = load_external_features(data_dir=data_dir)

    if ext.empty:
        log.warning("  No external features loaded -- using base features only")
        if scale:
            sc = StandardScaler()
            X_arr = sc.fit_transform(X_base.fillna(0))
            X_out = pd.DataFrame(X_arr, index=X_base.index, columns=X_base.columns)
            return X_out, y, list(X_out.columns), sc, yr_df
        return X_base, y, list(X_base.columns), None, yr_df

    ext.index.name = "year"
    X_combined = X_base.join(ext, how="left")
    # Leave NaN intact -- per-fold imputation in WalkForwardValidator handles it
    X_combined = _add_external_interactions(X_combined)

    std = X_combined.std()
    X_combined = X_combined.loc[:, std > 0.10]
    X_combined = _remove_high_correlation_ext(X_combined, threshold=0.90)

    y = y.reindex(X_combined.index).fillna(0).astype(int)

    nan_pct = X_combined.isna().mean().mean()
    log.info(
        f"  Combined feature matrix: {X_combined.shape} "
        f"(base: {X_base.shape[1]}, external: {ext.shape[1]}, NaN: {nan_pct:.1%})"
    )

    # Return unscaled -- validator scales per fold
    return X_combined, y, list(X_combined.columns), None, yr_df


def _add_external_interactions(X: pd.DataFrame) -> pd.DataFrame:
    if "ext_vix_mean" in X.columns and "ext_geo_risk_mean" in X.columns:
        X["ext_vix_x_georisk"] = X["ext_vix_mean"] * X["ext_geo_risk_mean"]

    if ("ext_ce_nation_state_frac_lag1" in X.columns and
            "crypto_breach_count_lag1" in X.columns):
        X["ext_apt_x_crypto_lag"] = (X["ext_ce_nation_state_frac_lag1"] *
                                     X["crypto_breach_count_lag1"])

    if ("ext_wa_ransomware_frac_lag1" in X.columns and
            "sector_financial" in X.columns):
        X["ext_ransom_x_financial"] = (X["ext_wa_ransomware_frac_lag1"] *
                                       X["sector_financial"])

    if ("ext_ce_financial_motive_frac_lag1" in X.columns and
            "ext_ce_crypto_count_lag1" in X.columns):
        X["ext_finmotive_x_crypto"] = (X["ext_ce_financial_motive_frac_lag1"] *
                                       X["ext_ce_crypto_count_lag1"])

    if ("ext_economic_stress" in X.columns and
            "crypto_breach_count_lag1" in X.columns):
        X["ext_stress_x_crypto_lag"] = (X["ext_economic_stress"] *
                                        X["crypto_breach_count_lag1"])
    return X


def _remove_high_correlation_ext(X: pd.DataFrame, threshold=0.90) -> pd.DataFrame:
    corr    = X.corr().abs()
    upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    ext_dropped = [c for c in to_drop if c.startswith("ext_")]
    if ext_dropped:
        log.info(f"  Dropped {len(ext_dropped)} redundant external features")
    return X.drop(columns=to_drop, errors="ignore")