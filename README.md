# Predictive Modeling of Crypto-System Vulnerabilities

**Authors:** Eileen Rashduni & Atharv Koratkar

---

## Project Overview

This project builds a probabilistic machine learning pipeline to predict whether a given year will be a **high crypto-breach-fraction year** — defined as a year in which crypto-related incidents make up an above-median share of all cyber breaches. The model is trained on **19,964 unified breach records** spanning 2004–2026 from eight harmonised data sources, using walk-forward time-series cross-validation so that every prediction is made with only historically available data.

The pipeline can be run end-to-end via `main.py` or explored interactively via the Jupyter notebook `Crypto_Breach_ML_Pipeline.ipynb`.

---

## Repository Structure

```
crypto_ml_project/
├── Crypto_Breach_ML_Pipeline.ipynb   # Interactive notebook (full pipeline + dashboards)
├── main.py                           # CLI entry point (runs all stages)
├── data_ingestion.py                 # Source loading, normalisation, deduplication
├── feature_engineering.py            # Feature matrix construction (37 features)
├── models.py                         # Walk-forward validator + 5 calibrated models
├── evaluation.py                     # Plots 01–12 saved to outputs/
├── generate_notebook.py              # Utility to regenerate the notebook scaffold
├── external_data.py                  # External signal helpers (VIX, WA, CE)
├── data/                             # Raw source files (not tracked in git)
└── outputs/                          # All generated PNGs, CSVs, and model artefacts
    ├── 01_dataset_overview.png
    ├── 02_sector_distribution.png
    ├── 03_method_distribution.png
    ├── 04_crypto_vs_all.png
    ├── 05_records_over_time.png
    ├── 06_model_comparison.png
    ├── 07_pr_curves.png
    ├── 08_probability_timeline.png
    ├── 09_feature_importance.png
    ├── 10_calibration.png
    ├── 11_confusion_matrix_heatmap.png
    ├── 12_brier_logloss_by_fold.png
    ├── executive_summary_dashboard.png
    ├── dashboard_threat_intelligence.png
    ├── dashboard_model_comparison.png
    ├── dashboard_prediction_audit.png
    ├── ensemble_forecast.csv
    └── model_artefacts.pkl
```

---

## Data

### Sources (19,964 records total)

| Source ID | File | Records | Coverage | Description |
|-----------|------|---------|----------|-------------|
| CE | `cyber_events_2026-03-22.csv` | 15,949 | 2014–2026 | Structured incident database — criminal, hacktivist, nation-state actors |
| WA | `Data_Breach_Notifications_Affecting_Washington_Residents.csv` | 1,506 | 2016–2026 | Washington State AG breach notifications |
| HHS | `Cyber_Security_Breaches.csv` | 883 | 2009–2026 | US HHS/OCR healthcare breach portal |
| DHL | `defi_hack_labs.csv` | 685 | 2014–2026 | DeFi protocol exploits and smart contract attacks |
| IIB | `Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv` | 416 | 2004–2022 | Information is Beautiful: high-profile global breaches |
| DBN | `Data_BreachesN_new.csv` | 273 | 2004–2020 | International country-level breach list |
| CSIS | `260306_Cyber_Events.pdf` | 207 | 2006–2026 | CSIS significant cyber incidents (text-mined from PDF) |
| DBEN | `Data_Breaches_EN_V2_2004_2017_20180220.csv` | 37 | 2004–2017 | European breach registry |
| DF1 | `df_1.csv` | 8 | 2004–2020 | Wikipedia-derived breach supplement |
| **Total** | | **19,964** | **2004–2026** | |

De-duplication uses `(entity, year, method_category)` as the composite key. Cross-source overlaps (primarily between IIB, DBN, DBEN, and DF1) are dropped.

### Crypto Coverage

- **1,650 crypto-tagged records** — 8.3% of all ingested incidents
- Crypto fraction ranges from 0% (2004–2009) to ~18% (2024)
- Year range: 2004–2026 (23 annual observations)
- **11 of 23 years** classified as positive (high crypto-breach-fraction): 47.83% base rate

### Sectors

`tech_web`, `financial`, `government`, `healthcare`, `retail`, `academic`, `social_media`, `gaming`, `energy`, `media`, `legal`, `telecom`, `transport`, `other`, `unknown`

### Attack Method Categories

`hacking`, `malware`, `phishing`, `ddos`, `poor_security`, `unauthorized_access`, `insider`, `lost_device`, `smart_contract_exploit`, `supply_chain`, `other`

---

## Quick Start

### Run the full pipeline (CLI)

```bash
pip install -r requirements.txt
python main.py                              # uses data/ directory, outputs to outputs/
python main.py --data_dir /path/to/data
python main.py --target high_impact        # alternative prediction target
python main.py --out_dir my_outputs
```

### Run the interactive notebook

```bash
jupyter notebook Crypto_Breach_ML_Pipeline.ipynb
```

Run all cells top-to-bottom. The notebook mirrors the CLI pipeline and adds visualisation sections 2A–8C plus four dashboards.

### Prediction targets

| Flag | Definition |
|------|------------|
| `crypto_binary` | Was crypto fraction above the series median this year? (default) |
| `high_impact` | Did total records breached exceed the 75th percentile? |
| `crypto_count` | Regression: raw annual crypto breach count |

---

## Pipeline

### Stage 1 — Data Ingestion (`data_ingestion.py`)

All sources are normalised to a canonical schema:
`year · entity · sector · method_category · records_affected · is_crypto · source_id`

- Malformed record counts (`15,000,000`, `1.37e+09`, `3m`, `15.000.000`) are parsed correctly
- CE and WA source files are located automatically — searched in `data/` first, then a fallback upload path
- Crypto flagging uses 30+ keywords: `bitcoin`, `ethereum`, `defi`, `nft`, `wallet`, `exchange`, `blockchain`, `web3`, `dex`, `cex`, `uniswap`, `binance`, `coinbase`, and others

### Stage 2 — Feature Engineering (`feature_engineering.py`)

**37 features** built from yearly aggregates (reduced from ~65 via variance/correlation pruning):

| Category | Features |
|----------|---------|
| Core counts | `log_records_total`, `mean_records`, `unique_entities`, `unique_methods`, `source_diversity`, `large_breach_count` |
| Sector fractions | Annual share per sector (financial, government, healthcare, …) |
| Method fractions | Annual share per attack type (hacking, malware, phishing, …) |
| Crypto signals | `crypto_breach_count`, `crypto_fraction`, `crypto_log_records`, `crypto_momentum` |
| Lag features | 1-, 2-, 3-year lags on key indicators |
| Rolling means | 2- and 3-year windows using `shift(1)` before rolling — prevents look-ahead leakage |
| Trend signals | YoY delta, rising/falling binary (`_rising`), 3-year linear slope |

**Label definition (`crypto_binary`):**
The target is 1 if `crypto_fraction` (crypto breaches / total breaches) exceeds the series median, 0 otherwise. Fraction is used instead of raw count because the CE dataset contributes ~16,000 rows uniformly across 2014–2026, making every modern year a trivial positive on raw count. The fraction-based split produces 11 positives and 12 negatives spread across both early (2004–2013) and recent (2017, 2019) years, making PR-AUC and ROC-AUC genuinely discriminative.

### Stage 3 — Walk-Forward Validation (`models.py`)

Expanding-window time-series cross-validation with `min_train_size=5`:

```
Fold  5: Train 2004–2008  →  Test 2009
Fold  6: Train 2004–2009  →  Test 2010
   ...
Fold 22: Train 2004–2025  →  Test 2026
```

**80 total fold-model combinations** (16 folds × 5 models).

**Per-fold leak prevention applied before every `fit()` call:**
1. NaN imputation using training-fold median only (CE/WA features start in 2014/2016; earlier folds have NaN for those columns)
2. `StandardScaler` fit on training data only, then applied to the single test row

**Models:**

| Model | Hyperparameters | Calibration |
|-------|----------------|-------------|
| LogisticRegression | L2, C=0.10, `class_weight=balanced` | Native (softmax) |
| RidgeClassifier | L2, C=0.05, `class_weight=balanced` | Native (softmax) |
| RandomForest | 1,000 trees, `max_depth=4`, `min_samples_leaf=2`, `max_features=sqrt` | Platt sigmoid, cv=3 |
| ExtraTrees | 1,000 trees, `max_depth=4`, `min_samples_leaf=2`, `max_features=sqrt` | Platt sigmoid, cv=3 |
| GradientBoosting | 300 estimators, `max_depth=2`, `learning_rate=0.03` | Platt sigmoid, cv=3 |

All tree models are wrapped in `SelectKBest(f_classif, k=35)` before the classifier. Platt (sigmoid) calibration is used over isotonic because isotonic overfits at N < 40 samples per calibration fold. Label smoothing (`eps=0.05`) blends predictions toward the training base rate to cap per-sample log-loss contribution.

Feature importances are extracted by averaging across all calibration folds via `_extract_fi_from_model()`.

### Stage 4 — Evaluation (`evaluation.py`)

Saves 12 plots to `outputs/` covering EDA (01–05) and model evaluation (06–12). See the outputs table in the repository structure above.

---

## Results

### Model Performance (walk-forward, all 16 test folds)

| Model | Brier ↓ | Log-Loss ↓ | PR-AUC ↑ | ROC-AUC ↑ | Pos folds | Neg folds |
|-------|---------|-----------|---------|---------|-----------|-----------|
| **RandomForest** | **0.1887** | **0.5738** | 0.8883 | 0.7333 | 10 | 6 |
| ExtraTrees | 0.2100 | 0.6165 | 0.8952 | 0.7500 | 10 | 6 |
| **GradientBoosting** | 0.2113 | 0.7633 | **0.9525** | **0.8833** | 10 | 6 |
| RidgeClassifier | 0.2397 | 0.6588 | 0.7687 | 0.5833 | 10 | 6 |
| LogisticRegression | 0.2673 | 0.7229 | 0.7389 | 0.5333 | 10 | 6 |

**Best model by PR-AUC: GradientBoosting** (0.9525 PR-AUC, 0.8833 ROC-AUC)
**Best model by Brier Score: RandomForest** (0.1887)

Final evaluation on GradientBoosting:
- Brier Score: **0.2113**
- Log-Loss: **0.7633**
- PR-AUC: **0.9525**

### Risk Score Timeline — GradientBoosting

| Year | P(breach) | Tier | Actual Breach |
|------|-----------|------|---------------|
| 2011 | 3.0% | LOW | No |
| 2012 | 2.3% | LOW | No |
| 2013 | 0.8% | LOW | No |
| 2014 | 1.0% | LOW | No |
| 2015 | 0.5% | LOW | **Yes** |
| 2016 | 18.9% | LOW | No |
| 2017 | 35.8% | LOW | **Yes** |
| 2018 | 30.0% | LOW | **Yes** |
| 2019 | 25.9% | LOW | No |
| 2020 | 20.5% | LOW | **Yes** |
| 2021 | 55.8% | ELEVATED | **Yes** |
| 2022 | 69.4% | HIGH | **Yes** |
| 2023 | 77.3% | HIGH | **Yes** |
| 2024 | 59.4% | ELEVATED | **Yes** |
| 2025 | 58.3% | ELEVATED | **Yes** |
| 2026 | 72.8% | HIGH | **Yes** |

### Risk Tiers

| Tier | Probability | Recommended Posture |
|------|-------------|---------------------|
| CRITICAL | ≥ 80% | Incident response readiness |
| HIGH | 60–79% | Active defence posture |
| ELEVATED | 40–59% | Enhanced vigilance |
| LOW | < 40% | Routine monitoring |

---

## Outputs

After a full run, `outputs/` contains **16 PNG images** plus 2 data files:

| File | Generated by | Description |
|------|-------------|-------------|
| `01_dataset_overview.png` | `evaluation.py` | Breach counts by year and source |
| `02_sector_distribution.png` | `evaluation.py` | Sector breakdown across all years |
| `03_method_distribution.png` | `evaluation.py` | Attack method breakdown |
| `04_crypto_vs_all.png` | `evaluation.py` | Crypto vs non-crypto breaches over time |
| `05_records_over_time.png` | `evaluation.py` | Records exposed per year (log scale) |
| `06_model_comparison.png` | `evaluation.py` | Mean metrics bar chart across models |
| `07_pr_curves.png` | `evaluation.py` | Precision-recall curves per model |
| `08_probability_timeline.png` | `evaluation.py` | Predicted risk probability per year |
| `09_feature_importance.png` | `evaluation.py` | Top features (RF + GradientBoosting) |
| `10_calibration.png` | `evaluation.py` | Reliability diagrams per model |
| `11_confusion_matrix_heatmap.png` | `evaluation.py` | Threshold-based confusion matrices |
| `12_brier_logloss_by_fold.png` | `evaluation.py` | Per-fold metric evolution |
| `executive_summary_dashboard.png` | Notebook cell 56 | Dark-theme executive summary (6 panels) |
| `dashboard_threat_intelligence.png` | Notebook cell 57 | Sector/method/trend analysis (5 panels) |
| `dashboard_model_comparison.png` | Notebook cell 58 | All-model comparison + ensemble band (5 panels) |
| `dashboard_prediction_audit.png` | Notebook cell 59 | TP/TN/FP/FN audit + KPI tiles (5 panels) |
| `ensemble_forecast.csv` | `main.py` | Year-by-year ensemble probabilities with CI |
| `model_artefacts.pkl` | `main.py` | Fitted models, scaler, feature names, predictions |

---

## Notebook Structure

The notebook (`Crypto_Breach_ML_Pipeline.ipynb`) mirrors the CLI pipeline with full visualisation:

| Section | Cells | Content |
|---------|-------|---------|
| Setup | 1–3 | Imports, environment check |
| 2: EDA | 8–16 | 9 plots: breach volume, sector/method distributions, crypto trends, cumulative exposure, bubble chart |
| 3: Features | 18–20 | Feature matrix preview, scaled matrix |
| 4: Feature Analysis | 23–27 | Correlation heatmap, label correlation, class-conditional distributions, violin plots, scatter matrix |
| 5: Walk-Forward | 29–32 | Model training, fold visualisation, ensemble consensus |
| 6: Model Evaluation | 34–42 | Summary table, PR/ROC curves, learning curves, calibration, Brier heatmap, probability timeline |
| 7: Feature Importance | 44–49 | RF importance, Pareto, SHAP-style contributions, RF vs GBM comparison, drift, stability |
| 8: Risk Scoring | 51–55 | Risk tiers, confusion matrix, threshold analysis, error spotlight, final printout |
| Dashboards | 56–59 | Executive summary + 3 themed dashboards (all saved to `outputs/`) |

---

## Known Limitations

- **N=23 annual observations** is a fundamental constraint. All metrics carry wide confidence intervals. A PR-AUC of 0.95 at N=23 is encouraging but not statistically robust in the same way as N=1,000+.
- **External features (CE, WA) only cover 2014/2016 onward.** Folds testing 2009–2015 rely entirely on breach-derived features; NaN is imputed per-fold from the training median.
- **2026 is a partial-year observation** (CE data through March 2026 only). Predictions for 2026 should be treated as preliminary.
- **Feature importances unavailable for tree models via `mean_feature_importance()`** due to Platt calibration wrapping. A direct extraction fallback via `_extract_fi_from_model()` from `wfv.best_models` is used in the executive dashboard and notebook 7E/7F cells.
- The `crypto_binary` label (fraction above median) changes semantic meaning if total breach volume shifts drastically. A year with 50% crypto fraction but only 10 total breaches receives the same label as a year with 12% of 10,000 breaches.

---

## Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
jupyter
notebook
pypdf
```

Tested on: `numpy 2.4.3`, `pandas 3.0.1`, `scikit-learn 1.8.0`

No XGBoost, TensorFlow, or GPU required. Python 3.9+. Fully Windows-compatible — logging uses ASCII output and writes to file in UTF-8.

Optional (for CSIS PDF extraction):
```bash
sudo apt-get install poppler-utils   # Linux
brew install poppler                 # macOS
# pip install cryptography>=3.1      # for AES-encrypted PDFs
```

---

## References

- Olushola A & Meenakshi SP (2025). Cybersecurity crimes in cryptocurrency exchanges (2009–2024). *Frontiers in Blockchain* 8:1713637.
- CSIS. *Significant Cyber Incidents Since 2006.* Center for Strategic and International Studies.
- Kaggle / Information is Beautiful. *World's Biggest Data Breaches & Hacks.*
- HHS / OCR. *Breach Portal: Notice to the Secretary of HHS.*
- Washington State Office of the Attorney General. *Data Breach Notifications Affecting Washington Residents.*
- DeFi Hack Labs. *DeFi Protocol Exploit Database.*
