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
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def run(data_dir: str = "data", target: str = "crypto_binary",
        out_dir: str = "outputs"):

    os.makedirs(out_dir, exist_ok=True)

    # ── Validate data directory ──────────────────────────────────────────
    abs_data = os.path.abspath(data_dir)
    REQUIRED_FILES = [
        "Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv",
        "Cyber_Security_Breaches.csv",
        "Data_BreachesN_new.csv",
        "Data_Breaches_EN_V2_2004_2017_20180220.csv",
        "df_1.csv",
        "260306_Cyber_Events.pdf",
    ]
    if not os.path.isdir(abs_data):
        log.error(f"Data directory not found: {abs_data}")
        log.error("Create it and place the 6 data files inside, then re-run.")
        raise SystemExit(1)

    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(abs_data, f))]
    if missing:
        log.warning(f"  {len(missing)} file(s) missing from {abs_data}:")
        for m in missing:
            log.warning(f"    ✗ {m}")
        found = [f for f in REQUIRED_FILES if f not in missing]
        if found:
            log.info(f"  {len(found)} file(s) found — continuing with available data.")
        else:
            log.error("No data files found at all. Cannot proceed.")
            raise SystemExit(1)
    else:
        log.info(f"  All 6 data files found in: {abs_data}")

    # ── 1. Ingest ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 1 — Data Ingestion")
    log.info("=" * 60)
    from data_ingestion import load_all_datasets
    df = load_all_datasets(data_dir=data_dir)
    log.info(f"  Total records : {len(df):,}")
    log.info(f"  Crypto-related: {df['is_crypto'].sum():,}")
    log.info(f"  Year range    : {df['year'].min()} – {df['year'].max()}")
    log.info(f"  Sources       : {df['source_id'].value_counts().to_dict()}")

    # ── 2. Feature engineering ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 2 — Feature Engineering")
    log.info("=" * 60)
    from feature_engineering import build_feature_matrix_with_external
    X, y, feat_names, scaler, yr_df = build_feature_matrix_with_external(
        df, data_dir=data_dir, target_mode=target, scale=True
    )
    log.info(f"  Feature matrix : {X.shape}")
    log.info(f"  Positive rate  : {y.mean():.1%}  ({y.sum()}/{len(y)} years)")
    log.info(f"  Feature count  : {len(feat_names)}")

    # ── 3. EDA plots ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 3 — Exploratory Data Analysis Plots")
    log.info("=" * 60)
    from evaluation import plot_eda
    plot_eda(df, yr_df, out_dir=out_dir)

    # ── 4. Walk-forward training ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 4 — Walk-Forward Model Training")
    log.info("=" * 60)
    from models import WalkForwardValidator, build_models
    wfv = WalkForwardValidator(min_train_size=5, calibrate=False)
    models = build_models()
    wfv.fit_predict(X, y, models=models)

    # ── 5. Evaluation ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 5 — Evaluation")
    log.info("=" * 60)
    summary = wfv.summary_metrics()
    log.info("\n" + summary.to_string())
    from evaluation import plot_model_results
    plot_model_results(wfv, out_dir=out_dir)

    # ── 6. Risk scoring ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STAGE 6 — Risk Scores (Latest Years)")
    log.info("=" * 60)
    preds = wfv.consolidated_predictions()
    best_model = summary["pr_auc"].idxmax()

    tiers = {p: ("CRITICAL" if p >= 0.8 else "HIGH" if p >= 0.6
                  else "ELEVATED" if p >= 0.4 else "LOW")
             for p in [0, 0.1, 0.5, 0.8]}

    def tier(p):
        if p >= 0.80: return "🔴 CRITICAL"
        if p >= 0.60: return "🟠 HIGH"
        if p >= 0.40: return "🟡 ELEVATED"
        return "🟢 LOW"

    best_preds = preds[preds["model"] == best_model].copy()
    best_preds["tier"] = best_preds["y_prob"].apply(tier)
    log.info(f"\n  Best model: {best_model}")
    for _, row in best_preds.tail(8).iterrows():
        flag = "✓ BREACH" if row["y_true"] == 1 else "  "
        log.info(f"  {int(row['year'])}  p={row['y_prob']:.3f}  {row['tier']}  {flag}")

    # ── 7. Save artefacts ────────────────────────────────────────────────
    import pickle
    artefact = {
        "best_model_name": best_model,
        "best_model"     : wfv.best_models.get(best_model),
        "feature_names"  : feat_names,
        "scaler"         : scaler,
        "summary_metrics": summary,
        "X_last"         : X.iloc[[-1]],
    }
    with open(os.path.join(out_dir, "model_artefacts.pkl"), "wb") as f:
        pickle.dump(artefact, f)
    log.info(f"\n  Artefacts saved to {out_dir}/model_artefacts.pkl")

    plots = sorted(os.listdir(out_dir))
    log.info(f"  Outputs ({len(plots)} files): {plots}")
    log.info("Pipeline complete ✓")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Breach ML Pipeline")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--target",   default="crypto_binary",
                        choices=["crypto_binary", "crypto_count", "high_impact"])
    parser.add_argument("--out_dir",  default="outputs")
    args = parser.parse_args()
    run(data_dir=args.data_dir, target=args.target, out_dir=args.out_dir)
