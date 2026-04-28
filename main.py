"""
main.py
=======
Single-command entry point for the full pipeline.

Usage:
    python main.py                          # uses data/ directory
    python main.py --data_dir /path/to/data
    python main.py --target crypto_binary   # or crypto_count / high_impact
"""

import argparse
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# -- Windows-safe UTF-8 logging -----------------------------------------------
# Windows console defaults to cp1252 which cannot encode emoji or many Unicode
# characters. Force both the file handler and stream handler to UTF-8 so the
# pipeline never crashes on log output regardless of system locale.
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_file_handler   = logging.FileHandler("pipeline.log", encoding="utf-8")
_file_handler.setFormatter(_fmt)

_stream_handler = logging.StreamHandler(stream=sys.stdout)
_stream_handler.setFormatter(_fmt)
# Wrap the stream in a UTF-8 recoder if the terminal doesn't support it
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _stream_handler])
log = logging.getLogger(__name__)

# -- ASCII tier labels (no emoji -- safe on all platforms) ---------------------
def _tier(p: float) -> str:
    if p >= 0.80: return "[CRITICAL]"
    if p >= 0.60: return "[HIGH]    "
    if p >= 0.40: return "[ELEVATED]"
    return "[LOW]     "


def run(data_dir: str = "data", target: str = "crypto_binary",
        out_dir: str = "outputs"):

    os.makedirs(out_dir, exist_ok=True)

    # -- Validate data directory ----------------------------------------------
    abs_data = os.path.abspath(data_dir)
    CORE_FILES = [
        "Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv",
        "Cyber_Security_Breaches.csv",
        "Data_BreachesN_new.csv",
        "Data_Breaches_EN_V2_2004_2017_20180220.csv",
        "df_1.csv",
        "260306_Cyber_Events.pdf",
    ]
    NEW_FILES = [
        "cyber_events_2026-03-22.csv",
        "Data_Breach_Notifications_Affecting_Washington_Residents.csv",
        "defi_hack_labs.csv",
    ]

    if not os.path.isdir(abs_data):
        log.error(f"Data directory not found: {abs_data}")
        raise SystemExit(1)

    missing = [f for f in CORE_FILES if not os.path.exists(os.path.join(abs_data, f))]
    if missing:
        log.warning(f"  {len(missing)} core file(s) missing -- continuing with available data")
        for m in missing:
            log.warning(f"    MISSING: {m}")
    else:
        log.info(f"  All core data files found in: {abs_data}")

    _UPLOAD_DIR = "/mnt/user-data/uploads"
    for f in NEW_FILES:
        found = (os.path.exists(os.path.join(abs_data, f)) or
                 os.path.exists(os.path.join(_UPLOAD_DIR, f)))
        status = "OK" if found else "NOT FOUND"
        log.info(f"  New dataset [{status}]: {f}")

    # -- 1. Ingest ------------------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 1 -- Data Ingestion")
    log.info("=" * 60)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_ingestion import load_all_datasets
    df = load_all_datasets(data_dir=data_dir)
    log.info(f"  Total records : {len(df):,}")
    log.info(f"  Crypto-related: {df['is_crypto'].sum():,}")
    log.info(f"  Year range    : {df['year'].min()} - {df['year'].max()}")
    log.info(f"  Sources       : {df['source_id'].value_counts().to_dict()}")

    # -- 2. Feature engineering -----------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 2 -- Feature Engineering (leakage-fixed)")
    log.info("=" * 60)
    from feature_engineering import build_feature_matrix_with_external
    X, y, feat_names, scaler, yr_df = build_feature_matrix_with_external(
        df, data_dir=data_dir, target_mode=target, scale=True
    )
    log.info(f"  Feature matrix : {X.shape}")
    log.info(f"  Positive rate  : {y.mean():.1%}  ({y.sum()}/{len(y)} years)")
    log.info(f"  Feature count  : {len(feat_names)}")
    nan_rate = X.isna().mean().mean() if hasattr(X, 'isna') else 0.0
    log.info(f"  NaN rate       : {nan_rate:.1%} (handled per-fold)")

    # -- 3. EDA plots ---------------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 3 -- Exploratory Data Analysis")
    log.info("=" * 60)
    try:
        from evaluation import plot_eda
        plot_eda(df, yr_df, out_dir=out_dir)
        log.info("  EDA plots saved.")
    except Exception as e:
        log.warning(f"  EDA plots failed: {e}")

    # -- 4. Walk-forward training ----------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 4 -- Walk-Forward Training (fold-local imputation + calibration)")
    log.info("=" * 60)
    from models import WalkForwardValidator, build_models
    wfv    = WalkForwardValidator(min_train_size=5, calibrate=False)
    models = build_models()
    wfv.fit_predict(X, y, models=models)

    # -- 5. Evaluation ---------------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 5 -- Evaluation (Brier + Log-Loss + PR-AUC)")
    log.info("=" * 60)
    summary = wfv.summary_metrics()
    log.info("\n" + summary.to_string())
    try:
        from evaluation import plot_model_results
        plot_model_results(wfv, out_dir=out_dir)
        log.info("  Model evaluation plots saved.")
    except Exception as e:
        log.warning(f"  Model result plots failed: {e}")

    # -- 6. Ensemble forecast --------------------------------------------------
    log.info("=" * 60)
    log.info("STAGE 6 -- Ensemble Forecast with Uncertainty Bands")
    log.info("=" * 60)

    ens = wfv.ensemble_predictions()

    log.info(f"  {'Year':>4}  {'p(breach)':>9}  {'Model range':>14}  "
             f"{'Spread':>6}  {'Tier':>10}  {'Reliable':>8}  {'Actual':>8}")
    log.info("  " + "-" * 70)

    for yr_val, row in ens.tail(10).iterrows():
        actual   = "BREACH" if row["y_true"] == 1 else "      "
        reliable = "yes" if row["reliable"] else "WIDE  "
        ci_str   = f"[{row['y_prob_lo']:.2f}-{row['y_prob_hi']:.2f}]"
        tier_str = _tier(row["y_prob"])
        log.info(
            f"  {int(yr_val):>4}  {row['y_prob']:>9.3f}  {ci_str:>14}  "
            f"{row['ci_width']:>6.3f}  {tier_str}  {reliable:>8}  {actual}"
        )

    log.info("")
    log.info("  NOTE: WIDE spread means models disagree on that year.")
    log.info("        Treat those years as directional signals only.")

    # -- 7. Save artefacts -----------------------------------------------------
    import pickle

    # Save ensemble CSV alongside the model pickle
    ens_csv = os.path.join(out_dir, "ensemble_forecast.csv")
    ens.to_csv(ens_csv)

    artefact = {
        "summary_metrics"     : summary,
        "ensemble_predictions": ens,
        "all_predictions"     : wfv.consolidated_predictions(),
        "best_models"         : wfv.best_models,
        "feature_names"       : feat_names,
        "scaler"              : scaler,
        "X_last"              : X.iloc[[-1]],
    }
    pkl_path = os.path.join(out_dir, "model_artefacts.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(artefact, f)

    plots = sorted(os.listdir(out_dir))
    log.info(f"  Artefacts saved ({len(plots)} files) -> {out_dir}/")
    log.info("Pipeline complete.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Breach ML Pipeline")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--target",   default="crypto_binary",
                        choices=["crypto_binary", "crypto_count", "high_impact"])
    parser.add_argument("--out_dir",  default="outputs")
    args = parser.parse_args()
    run(data_dir=args.data_dir, target=args.target, out_dir=args.out_dir)