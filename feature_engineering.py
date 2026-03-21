"""
feature_engineering.py  (v3 — optimised)
==========================================
Fixes identified from plot analysis:
  1. Label: use ≥1 crypto breach (presence/absence) not median threshold
     — the median threshold was causing 0% positive rate in 2015-2021 (plot 8 collapse)
  2. Remove rolling std features as primary signals — they dominate importance (plot 9)
     but are unstable; keep only std of the most meaningful cols
  3. Add directional signals: is crypto count increasing YoY?
  4. Better interaction terms aligned with what GBM found useful
  5. Clip extreme rolling values to reduce variance
"""

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

    # ── Label ────────────────────────────────────────────────────────────
    if target_mode == "crypto_binary":
        # Presence/absence of crypto breach — clearest signal
        # Use ≥2 to avoid single-incident noise while keeping meaningful events
        y = (yr["crypto_breach_count"] >= 2).astype(int).rename("label")
    elif target_mode == "crypto_count":
        y = yr["crypto_breach_count"].rename("label")
    else:
        threshold = yr["total_records"].quantile(0.75)
        y = (yr["total_records"] >= threshold).astype(int).rename("label")

    log.info(f"  Label distribution: {y.value_counts().to_dict()}")

    drop_cols = ["crypto_breach_count", "total_records",
                 "total_breaches", "crypto_fraction"]
    X = yr.drop(columns=[c for c in drop_cols if c in yr.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    # Remove near-zero-variance and near-constant features
    std = X.std()
    X = X.loc[:, std > 0.05]

    # Remove features that are highly correlated with each other (keep cleaner set)
    # This reduces multicollinearity that confuses linear models
    X = _remove_high_correlation(X, threshold=0.97)

    y = y.reindex(X.index).fillna(0).astype(int)

    scaler = None
    if scale and len(X) > 0:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    else:
        X_scaled = X.copy()

    log.info(f"  Feature matrix: {X_scaled.shape} | Positive rate: {y.mean():.2f} | Features: {len(X_scaled.columns)}")
    return X_scaled, y, list(X_scaled.columns), scaler, yr


def _remove_high_correlation(X, threshold=0.97):
    """Drop one of each pair of features with |corr| > threshold."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        log.info(f"  Dropping {len(to_drop)} highly correlated features")
    return X.drop(columns=to_drop, errors="ignore")


def _aggregate_yearly(df):
    grp = df.groupby("year")
    yr  = pd.DataFrame(index=sorted(df["year"].unique()))
    yr.index.name = "year"
    yr["total_breaches"]      = grp.size()
    yr["crypto_breach_count"] = grp["is_crypto"].sum().astype(float)
    yr["crypto_fraction"]     = yr["crypto_breach_count"] / yr["total_breaches"].replace(0, 1)
    yr["total_records"]       = grp["records_affected"].sum()
    yr["log_records_total"]   = np.log1p(yr["total_records"].fillna(0))
    yr["mean_records"]        = grp["records_affected"].mean().fillna(0)
    yr["log_max_records"]     = np.log1p(grp["records_affected"].max().fillna(0))
    yr["unique_entities"]     = grp["entity"].nunique()
    yr["unique_methods"]      = grp["method_category"].nunique()
    yr["source_diversity"]    = grp["source_id"].nunique()
    yr["large_breach_count"]  = (
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
    yr["crypto_log_records"]    = np.log1p(grp["records_affected"].sum().reindex(yr.index, fill_value=0))
    # Crypto-specific method fractions
    for method in ["hacking", "phishing", "insider", "malware"]:
        sub   = cr[cr["method_category"] == method].groupby("year").size()
        total = cr.groupby("year").size().replace(0, 1)
        yr[f"crypto_pct_{method}"] = (sub / total).reindex(yr.index, fill_value=0)
    return yr


def _add_rolling_features(yr):
    """
    Lags and rolling MEANS only (no std for primary signals — plot 9 showed
    std features dominated while providing unstable signal).
    Keep std only for the most meaningful 2 columns.
    """
    mean_cols = [
        "total_breaches", "log_records_total", "crypto_fraction",
        "crypto_breach_count", "large_breach_count", "crypto_log_records",
    ]
    for col in mean_cols:
        if col not in yr.columns:
            continue
        for lag in [1, 2, 3]:
            yr[f"{col}_lag{lag}"] = yr[col].shift(lag).fillna(0)
        for w in [2, 3]:
            yr[f"{col}_roll{w}_mean"] = yr[col].rolling(w, min_periods=1).mean()

    # Std only for the 2 most meaningful (volatility of records and crypto fraction)
    for col in ["log_records_total", "crypto_fraction"]:
        if col in yr.columns:
            yr[f"{col}_roll3_std"] = yr[col].rolling(3, min_periods=2).std().fillna(0)

    return yr


def _add_trend_features(yr):
    """Directional signals: is the situation getting better or worse?"""
    for col in ["total_breaches", "crypto_fraction", "log_records_total", "crypto_breach_count"]:
        if col not in yr.columns:
            continue
        chg = yr[col].diff().fillna(0)           # absolute change (more stable than pct)
        yr[f"{col}_delta"]  = chg
        yr[f"{col}_rising"] = (chg > 0).astype(int)  # binary: is it rising this year?

    # Cumulative trend (how far are we above baseline?)
    yr["cumulative_crypto"] = yr.get("crypto_breach_count",
                                      pd.Series(0, index=yr.index)).cumsum()

    # 3-year linear slope
    def _slope(series, w=3):
        slopes = []
        for i in range(len(series)):
            window = series.iloc[max(0, i-w+1): i+1].values
            slopes.append(float(np.polyfit(np.arange(len(window)), window, 1)[0])
                          if len(window) >= 2 else 0.0)
        return pd.Series(slopes, index=series.index)

    yr["breach_trend_slope"] = _slope(yr["total_breaches"])
    yr["crypto_trend_slope"] = _slope(yr.get("crypto_breach_count",
                                              pd.Series(0, index=yr.index)))
    return yr


def _add_interaction_features(yr):
    """Targeted interactions based on what GBM found important (plot 9)."""
    # Crypto momentum: lagged count × is it rising?
    if "crypto_breach_count_lag1" in yr.columns and "crypto_breach_count_rising" in yr.columns:
        yr["crypto_momentum"] = yr["crypto_breach_count_lag1"] * yr["crypto_breach_count_rising"]

    # Hacking in financial sector
    if "method_hacking" in yr.columns and "sector_financial" in yr.columns:
        yr["hacking_x_financial"] = yr["method_hacking"] * yr["sector_financial"]

    # Crypto trend combined with records volatility
    if "crypto_trend_slope" in yr.columns and "log_records_total_roll3_std" in yr.columns:
        yr["crypto_slope_x_volatility"] = (yr["crypto_trend_slope"] *
                                            yr["log_records_total_roll3_std"])
    return yr


# ─────────────────────────────────────────────────────────────────────────────
# External dataset integration
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix_with_external(
    df,
    data_dir: str = None,
    target_mode: str = "crypto_binary",
    scale: bool = True,
):
    """
    Extends build_feature_matrix() with the three new external datasets:
      - Global AI workforce / automation indicators
      - Market trend (VIX, geopolitical risk, sentiment, volatility)
      - World Bank macro (GDP growth, unemployment, inflation, debt)

    Returns same signature as build_feature_matrix().
    """
    from external_data import load_external_features

    # Base feature matrix
    X_base, y, feat_names, scaler, yr_df = build_feature_matrix(
        df, target_mode=target_mode, scale=False  # scale after adding external
    )

    # Load external features
    if data_dir is None:
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    ext = load_external_features(data_dir=data_dir)

    if ext.empty:
        log.warning("  No external features loaded — using base features only")
        if scale:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_scaled = pd.DataFrame(sc.fit_transform(X_base),
                                    index=X_base.index, columns=X_base.columns)
            return X_scaled, y, list(X_scaled.columns), sc, yr_df
        return X_base, y, list(X_base.columns), None, yr_df

    # Align on year index
    ext.index.name = "year"
    X_combined = X_base.join(ext, how="left")
    X_combined  = X_combined.ffill().bfill().fillna(0)

    # Add cross-dataset interaction features
    X_combined = _add_external_interactions(X_combined)

    # Remove near-zero-variance
    std = X_combined.std()
    X_combined = X_combined.loc[:, std > 0.05]

    # Remove high-correlation duplicates
    X_combined = _remove_high_correlation_ext(X_combined, threshold=0.97)

    y = y.reindex(X_combined.index).fillna(0).astype(int)

    log.info(
        f"  Combined feature matrix: {X_combined.shape} "
        f"(base: {X_base.shape[1]}, external: {ext.shape[1]})"
    )

    if scale:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_scaled = pd.DataFrame(
            sc.fit_transform(X_combined),
            index=X_combined.index, columns=X_combined.columns,
        )
        return X_scaled, y, list(X_scaled.columns), sc, yr_df

    return X_combined, y, list(X_combined.columns), None, yr_df


def _add_external_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-dataset interactions that are theoretically motivated:
      - High VIX × high geopolitical risk = compounded threat environment
      - AI investment growth × automation rate = technology disruption index
      - Economic stress × crypto fraction lag = financial pressure on crypto
      - GDP growth decline × hacking method rate = recession-driven attacks
    """
    # VIX × Geo risk (both elevate attacker motivation)
    if "ext_vix_mean" in X.columns and "ext_geo_risk_mean" in X.columns:
        X["ext_vix_x_georisk"] = X["ext_vix_mean"] * X["ext_geo_risk_mean"]

    # AI investment × automation rate (tech disruption drives new attack surfaces)
    if ("ext_ai_investment_mean" in X.columns and
            "ext_automation_rate" in X.columns):
        X["ext_ai_x_automation"] = (X["ext_ai_investment_mean"] *
                                     X["ext_automation_rate"])

    # Economic stress × lagged crypto fraction (financial pressure → crypto attacks)
    if ("ext_economic_stress" in X.columns and
            "crypto_fraction_lag1" in X.columns):
        X["ext_stress_x_crypto_lag"] = (X["ext_economic_stress"] *
                                         X["crypto_fraction_lag1"])

    # GDP decline × hacking rate (recessions historically → more financially motivated hacks)
    if ("ext_gdp_growth" in X.columns and "method_hacking" in X.columns):
        X["ext_gdp_decline_x_hacking"] = (
            (-X["ext_gdp_growth"]).clip(lower=0) * X["method_hacking"]
        )

    # Job displacement × crypto breach count lag (displaced workers → insider threats)
    if ("ext_job_displacement" in X.columns and
            "crypto_breach_count_lag1" in X.columns):
        X["ext_displacement_x_crypto"] = (X["ext_job_displacement"] *
                                           X["crypto_breach_count_lag1"])

    return X


def _remove_high_correlation_ext(X: pd.DataFrame, threshold=0.97) -> pd.DataFrame:
    """Same as base version but logs which external cols are dropped."""
    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    ext_dropped = [c for c in to_drop if c.startswith("ext_")]
    if ext_dropped:
        log.info(f"  Dropped {len(ext_dropped)} redundant external features")
    return X.drop(columns=to_drop, errors="ignore")
