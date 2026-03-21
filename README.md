# Predictive-Modeling-of-Crypto-System-Vulnerabilities

# Predictive Modeling of Crypto-System Vulnerabilities
**Authors:** Eileen Rashduni · Atharv Koratkar

---

## Project Overview

This project builds a probabilistic ML pipeline that predicts the likelihood of a
**crypto-related data breach occurring in a given year**, using six real-world breach
datasets combined with temporal feature engineering and a walk-forward time-series
cross-validation strategy.

It directly implements the methodology described in the project proposal (Section 3–5)
and integrates data from the Olushola & Meenakshi (2025) *Frontiers in Blockchain* paper.

---

```bash
# 1. Install dependencies (all standard — no xgboost needed)
pip install -r requirements.txt

# 2. Run the full pipeline (uses data/ directory)
python main.py

# 3. Open the interactive notebook
jupyter notebook Crypto_Breach_ML_Pipeline.ipynb
```

## Pipeline Stages

### Stage 1 · Data Ingestion (`data_ingestion.py`)
- Loads all 6 sources into a **canonical schema**: `year, entity, sector, method_category, records_affected, is_crypto, source_id`
- Normalises method names (10 categories) and sector names (12 categories)
- Flags crypto-related entities using 30 keywords
- Text-mines the CSIS PDF with `pdftotext`
- Handles messy record formats: `15,000,000` · `1.37e+09` · `3m` · `15.000.000`
- Soft de-duplicates on `(entity, year, method_category)`

### Stage 2 · Feature Engineering (`feature_engineering.py`)
Transforms raw incidents into a **yearly feature matrix (22 × 52)**:
- **Core counts**: total breaches, crypto count, crypto fraction, log-records
- **Sector fractions**: 8 sectors (tech_web, healthcare, financial, government …)
- **Method fractions**: 10 attack types (hacking, phishing, insider, malware …)
- **Lag features**: 1-year and 2-year lags on key indicators
- **Rolling stats**: 2- and 3-year rolling mean and std
- **Interaction terms**: `hacking × financial`, `records × crypto`
- **YoY growth rates**: breach count and records

### Stage 3 · Walk-Forward Validation (`models.py`)
```
Fold  7: Train 2004–2010 → Test 2011
Fold  8: Train 2004–2011 → Test 2012
   ...
Fold 21: Train 2004–2024 → Test 2025
```
**15 folds**, **4 models**:

| Model | Type |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Bagged decision trees |
| Gradient Boosting | Boosted trees (sklearn) |
| SVM (RBF kernel) | Non-linear kernel method |

All predictions pooled across folds for robust metric computation.

### Stage 4 · Evaluation (`evaluation.py`)

**3 primary metrics (Section 5.3 of proposal):**

| Metric | Formula | Best |
|--------|---------|------|
| **Brier Score** | `mean((F_t − O_t)²)` | ↓ lower |
| **Log-Loss** | `−mean[O·log F + (1−O)·log(1−F)]` | ↓ lower |
| **PR-AUC** | Area under Precision-Recall curve | ↑ higher |

**Actual results (GradientBoosting — best model):**
```
PR-AUC      : 0.9276
ROC-AUC     : 0.9107
Brier Score : 0.1503
Log-Loss    : 0.9604
```

### Stage 5 · Risk Scoring
```
🔴 CRITICAL  ≥ 80%   Incident response readiness
🟠 HIGH      60–80%  Active defence posture
🟡 ELEVATED  40–60%  Enhanced vigilance
🟢 LOW       < 40%   Routine monitoring
```

---

## Jupyter Notebook Stages

The notebook (`Crypto_Breach_ML_Pipeline.ipynb`) walks through 8 stages with 20+ cells:

| Stage | Content |
|-------|---------|
| 1 | Data Ingestion — load all 6 sources, preview dataframes |
| 2 | EDA — 4 multi-panel plots: breach counts, sector/method, crypto vs all, records distribution |
| 3 | Feature Engineering — yearly table, scaled feature matrix |
| 4 | Feature Analysis — correlation heatmap, feature-label correlation, class-conditional distributions |
| 5 | Walk-Forward Training — fold log, prediction table |
| 6 | Evaluation — metric table, PR curves, probability timeline, manual Brier/LogLoss verification |
| 7 | Explainability — feature importance bars, Pareto chart, directional SHAP-style impact |
| 8 | Risk Scoring — tier assignment, risk timeline bar chart, final summary |

---

## Key Findings (from actual data)

- **1,825 unique breach incidents** loaded across 22 years (2004–2025)
- **61 crypto-flagged incidents** (3.3% of total)
- **Gradient Boosting** achieves best PR-AUC = **0.928** on held-out years
- Top predictive features: `crypto_fraction_lag1`, `method_hacking`, `sector_financial`, `total_breaches_roll3_mean`
- Attack method taxonomy aligned with Olushola & Meenakshi (2025): wallet/key compromise, system exploits, and phishing dominate both CEX and DEX incidents
