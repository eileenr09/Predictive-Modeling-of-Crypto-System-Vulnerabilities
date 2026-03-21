"""
external_data.py
================
Loads and harmonises the three new external datasets into
annual signals aligned with the breach feature matrix (2004–2025).

Sources:
  A) global_ai_workforce_automation_2015_2025.csv
     — AI investment, automation rate, policy index, job displacement
       (20 countries × 11 years; aggregated to global yearly means)

  B) Market_Trend_External.csv
     — VIX, GeoPolitical Risk Score, Sentiment Score, market volatility
       (daily; aggregated to yearly stats; 2004–2017 only)

  C) world_bank_data_2025.csv
     — GDP growth, unemployment, inflation, public debt
       (217 countries × 2010–2025; we use US + G20 aggregates)

Integration strategy
--------------------
Each source becomes a set of yearly scalar features. Coverage gaps
(e.g. AI data only from 2015, Market only to 2017) are handled by:
  - Forward-filling recent AI features backward to 2004
    using the earliest available value (conservative baseline)
  - Backward-filling market features forward from 2017
    using the last known value (conservative)
  - World Bank: interpolate missing years linearly
All features are winsorised at 1/99 percentile to clip extreme outliers
before they enter the model.
"""

import os
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# G20 country names as they appear in the World Bank file
G20_COUNTRIES = [
    "United States", "China", "Japan", "Germany", "United Kingdom",
    "France", "Italy", "Canada", "South Korea", "Australia",
    "Brazil", "India", "Mexico", "Russia", "Turkey",
    "Saudi Arabia", "Indonesia", "Argentina", "South Africa",
]


def load_external_features(data_dir: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by year (int) with all external features.
    Missing years are filled conservatively.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    ai_path  = os.path.join(data_dir, "global_ai_workforce_automation_2015_2025.csv")
    mkt_path = os.path.join(data_dir, "Market_Trend_External.csv")
    wb_path  = os.path.join(data_dir, "world_bank_data_2025.csv")

    frames = []
    if os.path.exists(ai_path):
        frames.append(_load_ai(ai_path))
        log.info("  External: AI workforce dataset loaded")
    else:
        log.warning(f"  External: AI file missing: {ai_path}")

    if os.path.exists(mkt_path):
        frames.append(_load_market(mkt_path))
        log.info("  External: Market trend dataset loaded")
    else:
        log.warning(f"  External: Market file missing: {mkt_path}")

    if os.path.exists(wb_path):
        frames.append(_load_worldbank(wb_path))
        log.info("  External: World Bank dataset loaded")
    else:
        log.warning(f"  External: World Bank file missing: {wb_path}")

    if not frames:
        log.warning("  External: No external datasets found — skipping")
        return pd.DataFrame()

    # Align all on a full yearly index 2004–2025
    full_idx = pd.RangeIndex(2004, 2026, name="year")
    combined = pd.DataFrame(index=full_idx)
    for df in frames:
        combined = combined.join(df, how="left")

    # Fill gaps: interpolate → forward-fill → backward-fill
    combined = combined.interpolate(method="linear", limit_direction="both")
    combined = combined.ffill().bfill()

    # Winsorise each column at 1st/99th percentile
    for col in combined.columns:
        lo = combined[col].quantile(0.01)
        hi = combined[col].quantile(0.99)
        combined[col] = combined[col].clip(lo, hi)

    log.info(f"  External features: {combined.shape[1]} columns × {len(combined)} years")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_ai(path: str) -> pd.DataFrame:
    """
    Global yearly means across all 20 countries.
    Coverage: 2015–2025 → backfilled to 2004 using 2015 baseline.
    """
    df = pd.read_csv(path)

    # Global aggregate per year
    agg = df.groupby("Year").agg(
        ext_ai_investment_mean       = ("AI_Investment_BillionUSD",     "mean"),
        ext_ai_investment_std        = ("AI_Investment_BillionUSD",     "std"),
        ext_automation_rate          = ("Automation_Rate_Percent",       "mean"),
        ext_ai_policy_index          = ("AI_Policy_Index",              "mean"),
        ext_job_displacement         = ("Job_Displacement_Million",     "mean"),
        ext_job_creation             = ("Job_Creation_Million",         "mean"),
        ext_ai_readiness             = ("AI_Readiness_Score",           "mean"),
        ext_productivity_index       = ("Productivity_Index",           "mean"),
        ext_reskilling_investment    = ("Reskilling_Investment_MillionUSD", "mean"),
    ).rename_axis("year")

    # YoY deltas (directional signals)
    for col in ["ext_ai_investment_mean", "ext_automation_rate",
                "ext_job_displacement", "ext_ai_readiness"]:
        agg[f"{col}_delta"] = agg[col].diff().fillna(0)

    return agg


def _load_market(path: str) -> pd.DataFrame:
    """
    Daily market data → yearly aggregations.
    Coverage: 2004–2017 (usable range from 30k rows).
    """
    df = pd.read_csv(path, parse_dates=["Date"], low_memory=False)
    df["year"] = df["Date"].dt.year
    df = df[df["year"] >= 2004]

    agg = df.groupby("year").agg(
        ext_vix_mean              = ("VIX_Close",              "mean"),
        ext_vix_max               = ("VIX_Close",              "max"),
        ext_vix_std               = ("VIX_Close",              "std"),
        ext_geo_risk_mean         = ("GeoPolitical_Risk_Score", "mean"),
        ext_geo_risk_max          = ("GeoPolitical_Risk_Score", "max"),
        ext_sentiment_mean        = ("Sentiment_Score",         "mean"),
        ext_sentiment_std         = ("Sentiment_Score",         "std"),
        ext_market_return_mean    = ("Daily_Return_Pct",        "mean"),
        ext_market_volatility     = ("Volatility_Range",        "mean"),
        ext_fed_rate_changes      = ("Federal_Rate_Change_Flag","sum"),
        ext_economic_news_days    = ("Economic_News_Flag",      "sum"),
    )

    # YoY changes for key signals
    agg["ext_vix_yoy_change"]        = agg["ext_vix_mean"].diff().fillna(0)
    agg["ext_geo_risk_yoy_change"]   = agg["ext_geo_risk_mean"].diff().fillna(0)
    agg["ext_sentiment_yoy_change"]  = agg["ext_sentiment_mean"].diff().fillna(0)

    return agg


def _load_worldbank(path: str) -> pd.DataFrame:
    """
    World Bank macro indicators.
    Strategy: compute G20 median (robust to missing) per year.
    Coverage: 2010–2025.
    """
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    # Use G20 countries + global (all countries) aggregate
    df_g20 = df[df["country_name"].isin(G20_COUNTRIES)].copy()

    agg = df_g20.groupby("year").agg(
        ext_gdp_growth        = ("GDP Growth (% Annual)",         "median"),
        ext_unemployment      = ("Unemployment Rate (%)",         "median"),
        ext_inflation_cpi     = ("Inflation (CPI %)",             "median"),
        ext_gdp_per_capita    = ("GDP per Capita (Current USD)",  "median"),
        ext_public_debt       = ("Public Debt (% of GDP)",        "median"),
        ext_interest_rate     = ("Interest Rate (Real, %)",       "median"),
        ext_current_account   = ("Current Account Balance (% GDP)","median"),
    ).rename_axis("year")

    # Derived signals
    agg["ext_gdp_growth_delta"]   = agg["ext_gdp_growth"].diff().fillna(0)
    agg["ext_inflation_delta"]    = agg["ext_inflation_cpi"].diff().fillna(0)
    agg["ext_unemployment_delta"] = agg["ext_unemployment"].diff().fillna(0)

    # Economic stress composite: high unemployment + high inflation + low growth
    # Normalise each component 0-1 then combine
    def _norm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0
    stress = (_norm(agg["ext_unemployment"]) +
              _norm(agg["ext_inflation_cpi"].abs()) +
              _norm(-agg["ext_gdp_growth"]))
    agg["ext_economic_stress"] = stress / 3.0

    return agg
