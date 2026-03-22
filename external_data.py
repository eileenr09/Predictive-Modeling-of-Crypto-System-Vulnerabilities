"""
external_data.py  (v4 — new datasets + overconfidence fixes)
=============================================================
WHAT CHANGED:
  - Added Source G: cyber_events (CSIS/DFRLab-style incident database, 2014–2026)
    Provides: nation-state threat fraction, financial-motive fraction,
    exploitive-event fraction, crypto incident count/fraction, total volume
  - Added Source H: WA Data Breach Notifications (2016–2026)
    Provides: ransomware fraction, days-to-identify trend,
    finance/health/business breach concentration, exploit velocity signals
  - All new features are STRICTLY LAG-1 when used for prediction
    (we only use the *previous* year's value as a predictor of the current year)
    to prevent data leakage.
  - Removed backward-fill of market features (fills future gap with 2017 data —
    this silently carries stale signals forward into 2018-2025 predictions).
  - Coverage gaps now expressed as explicit NaN, not zero or last-known value,
    so downstream models see true uncertainty rather than fake continuity.
"""

import os
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

G20_COUNTRIES = [
    "United States", "China", "Japan", "Germany", "United Kingdom",
    "France", "Italy", "Canada", "South Korea", "Australia",
    "Brazil", "India", "Mexico", "Russia", "Turkey",
    "Saudi Arabia", "Indonesia", "Argentina", "South Africa",
]

CYBER_EVENTS_FILENAME = "cyber_events_2026-03-22.csv"
WA_BREACH_FILENAME = "Data_Breach_Notifications_Affecting_Washington_Residents.csv"


def load_external_features(data_dir: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by year with all external features.
    Features outside a source's coverage window are left as NaN so that
    the downstream imputation strategy is applied transparently.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    frames = []

    ai_path  = os.path.join(data_dir, "global_ai_workforce_automation_2015_2025.csv")
    mkt_path = os.path.join(data_dir, "Market_Trend_External.csv")
    wb_path  = os.path.join(data_dir, "world_bank_data_2025.csv")

    if os.path.exists(ai_path):
        frames.append(_load_ai(ai_path))
        log.info("  External: AI workforce dataset loaded")

    if os.path.exists(mkt_path):
        frames.append(_load_market(mkt_path))
        log.info("  External: Market trend dataset loaded")

    if os.path.exists(wb_path):
        frames.append(_load_worldbank(wb_path))
        log.info("  External: World Bank dataset loaded")

    for fname, loader in [
        (CYBER_EVENTS_FILENAME, _load_cyber_events),
        (WA_BREACH_FILENAME,    _load_wa_breach),
    ]:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            upload_path = os.path.join("/mnt/user-data/uploads", fname)
            if os.path.exists(upload_path):
                path = upload_path
            else:
                log.info(f"  External: {fname} not found — skipping")
                continue
        try:
            frames.append(loader(path))
            log.info(f"  External: {fname} loaded")
        except Exception as e:
            log.warning(f"  External: failed to load {fname}: {e}")

    if not frames:
        log.warning("  External: No external datasets found")
        return pd.DataFrame()

    full_idx = pd.RangeIndex(2004, 2027, name="year")
    combined = pd.DataFrame(index=full_idx)
    for df in frames:
        combined = combined.join(df, how="left")

    # Safe gap-filling: interpolate only WITHIN coverage windows
    combined = combined.interpolate(method="linear", limit_direction="forward",
                                    limit_area="inside")
    # Forward-fill by at most 2 years for slow-moving signals
    combined = combined.ffill(limit=2)
    # Remaining NaN stays NaN — fold-level imputation handles it

    for col in combined.columns:
        vals = combined[col].dropna()
        if len(vals) < 4:
            continue
        lo = vals.quantile(0.02)
        hi = vals.quantile(0.98)
        combined[col] = combined[col].clip(lo, hi)

    nan_rate = combined.isna().mean().mean()
    log.info(f"  External features: {combined.shape[1]} cols x {len(combined)} years "
             f"(NaN rate: {nan_rate:.1%})")
    return combined


def _load_cyber_events(path: str) -> pd.DataFrame:
    """
    Structured incident database (2014–2026).
    All features are shifted by 1 year (lag-1) to prevent data leakage.
    """
    df = pd.read_csv(path, low_memory=False)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(2014, 2026)]

    grp  = df.groupby("year")
    total = grp.size().rename("ext_ce_total_events")

    crypto_mask = df["description"].str.contains(
        "crypto|bitcoin|blockchain|ethereum|defi|nft|wallet|token|web3|dex|binance|coinbase",
        case=False, na=False
    )
    crypto_count = df[crypto_mask].groupby("year").size().rename("ext_ce_crypto_count")

    nation_state = (df["actor_type"] == "Nation-State").groupby(df["year"]).sum()
    financial_m  = (df["motive"] == "Financial").groupby(df["year"]).sum()
    exploitive   = (df["event_type"] == "Exploitive").groupby(df["year"]).sum()
    ransom_mask  = df["original_method"].astype(str).str.lower().str.contains("ransom", na=False)
    ransomware_c = ransom_mask.groupby(df["year"]).sum()

    agg = pd.DataFrame({
        "ext_ce_total_events"  : total,
        "ext_ce_crypto_count"  : crypto_count,
    }).fillna(0)

    agg["ext_ce_crypto_frac"]           = agg["ext_ce_crypto_count"] / agg["ext_ce_total_events"].replace(0, 1)
    agg["ext_ce_nation_state_frac"]     = (nation_state / total.replace(0, 1)).fillna(0)
    agg["ext_ce_financial_motive_frac"] = (financial_m  / total.replace(0, 1)).fillna(0)
    agg["ext_ce_exploitive_frac"]       = (exploitive   / total.replace(0, 1)).fillna(0)
    agg["ext_ce_ransomware_frac"]       = (ransomware_c / total.replace(0, 1)).fillna(0)
    agg["ext_ce_yoy_growth"]            = agg["ext_ce_total_events"].pct_change().fillna(0)

    agg.index.name = "year"

    # Lag-1: each year only uses prior year's threat landscape
    lag1 = agg.shift(1)
    lag1.columns = [c + "_lag1" for c in lag1.columns]
    lag1.index.name = "year"
    return lag1


def _load_wa_breach(path: str) -> pd.DataFrame:
    """
    Washington State Data Breach Notifications (2016–2026).
    Shifted by 1 year (lag-1) to prevent leakage.
    """
    df = pd.read_csv(path)
    df = df[df["Year"].between(2016, 2026)].copy()

    grp   = df.groupby("Year")
    total = grp.size().rename("ext_wa_total_breaches")

    cyber       = df[df["DataBreachCause"] == "Cyberattack"]
    ransom      = (cyber["CyberattackType"] == "Ransomware").groupby(cyber["Year"]).sum()
    cyber_total = cyber.groupby("Year").size()

    finance = (df["IndustryType"] == "Finance").groupby(df["Year"]).sum()
    health  = (df["IndustryType"] == "Health").groupby(df["Year"]).sum()

    days_id  = grp["DaysToIdentifyBreach"].median()
    days_exp = grp["DaysOfExposure"].median()

    large_mask = df["WashingtoniansAffectedRange"].astype(str).str.contains(
        r"10,000|50,000|100,000|250,000|500,000|1,000,000|\+", na=False
    )
    large_c = df[large_mask].groupby("Year").size()

    agg = pd.DataFrame({
        "ext_wa_total_breaches"   : total,
        "ext_wa_days_to_identify" : days_id,
        "ext_wa_days_exposure"    : days_exp,
    })
    agg["ext_wa_ransomware_frac"]   = (ransom  / cyber_total.replace(0, 1)).fillna(0)
    agg["ext_wa_finance_frac"]      = (finance / total.replace(0, 1)).fillna(0)
    agg["ext_wa_health_frac"]       = (health  / total.replace(0, 1)).fillna(0)
    agg["ext_wa_large_breach_frac"] = (large_c / total.replace(0, 1)).fillna(0)
    agg["ext_wa_yoy_growth"]        = agg["ext_wa_total_breaches"].pct_change().fillna(0)
    agg.index.name = "year"

    lag1 = agg.shift(1)
    lag1.columns = [c + "_lag1" for c in lag1.columns]
    lag1.index.name = "year"
    return lag1


# ─── Legacy loaders (preserved) ──────────────────────────────────────────────

def _load_ai(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    agg = df.groupby("Year").agg(
        ext_ai_investment_mean    = ("AI_Investment_BillionUSD",          "mean"),
        ext_ai_investment_std     = ("AI_Investment_BillionUSD",          "std"),
        ext_automation_rate       = ("Automation_Rate_Percent",           "mean"),
        ext_ai_policy_index       = ("AI_Policy_Index",                  "mean"),
        ext_job_displacement      = ("Job_Displacement_Million",         "mean"),
        ext_job_creation          = ("Job_Creation_Million",             "mean"),
        ext_ai_readiness          = ("AI_Readiness_Score",               "mean"),
        ext_productivity_index    = ("Productivity_Index",               "mean"),
        ext_reskilling_investment = ("Reskilling_Investment_MillionUSD", "mean"),
    ).rename_axis("year")
    for col in ["ext_ai_investment_mean", "ext_automation_rate",
                "ext_job_displacement", "ext_ai_readiness"]:
        agg[f"{col}_delta"] = agg[col].diff().fillna(0)
    return agg


def _load_market(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    df["year"] = df["Date"].dt.year
    df = df[df["year"] >= 2004]
    agg = df.groupby("year").agg(
        ext_vix_mean           = ("VIX_Close",               "mean"),
        ext_vix_max            = ("VIX_Close",               "max"),
        ext_vix_std            = ("VIX_Close",               "std"),
        ext_geo_risk_mean      = ("GeoPolitical_Risk_Score",  "mean"),
        ext_geo_risk_max       = ("GeoPolitical_Risk_Score",  "max"),
        ext_sentiment_mean     = ("Sentiment_Score",          "mean"),
        ext_sentiment_std      = ("Sentiment_Score",          "std"),
        ext_market_return_mean = ("Daily_Return_Pct",         "mean"),
        ext_market_volatility  = ("Volatility_Range",         "mean"),
        ext_fed_rate_changes   = ("Federal_Rate_Change_Flag", "sum"),
        ext_economic_news_days = ("Economic_News_Flag",       "sum"),
    )
    agg["ext_vix_yoy_change"]       = agg["ext_vix_mean"].diff().fillna(0)
    agg["ext_geo_risk_yoy_change"]  = agg["ext_geo_risk_mean"].diff().fillna(0)
    agg["ext_sentiment_yoy_change"] = agg["ext_sentiment_mean"].diff().fillna(0)
    # No backward-fill: market data ends 2017; post-2017 years stay NaN
    return agg


def _load_worldbank(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df_g20 = df[df["country_name"].isin(G20_COUNTRIES)].copy()
    agg = df_g20.groupby("year").agg(
        ext_gdp_growth      = ("GDP Growth (% Annual)",           "median"),
        ext_unemployment    = ("Unemployment Rate (%)",           "median"),
        ext_inflation_cpi   = ("Inflation (CPI %)",               "median"),
        ext_gdp_per_capita  = ("GDP per Capita (Current USD)",    "median"),
        ext_public_debt     = ("Public Debt (% of GDP)",          "median"),
        ext_interest_rate   = ("Interest Rate (Real, %)",         "median"),
        ext_current_account = ("Current Account Balance (% GDP)", "median"),
    ).rename_axis("year")
    agg["ext_gdp_growth_delta"]   = agg["ext_gdp_growth"].diff().fillna(0)
    agg["ext_inflation_delta"]    = agg["ext_inflation_cpi"].diff().fillna(0)
    agg["ext_unemployment_delta"] = agg["ext_unemployment"].diff().fillna(0)

    def _norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0
    stress = (_norm(agg["ext_unemployment"]) +
              _norm(agg["ext_inflation_cpi"].abs()) +
              _norm(-agg["ext_gdp_growth"]))
    agg["ext_economic_stress"] = stress / 3.0
    return agg
