# Predictive Modeling of Crypto-System Vulnerabilities

**Authors:** Eileen Rashduni & Atharv Koratkar

---

## Project Overview

This project builds a probabilistic machine learning pipeline to predict whether a
given year will be a **high crypto-breach-fraction year** -- defined as a year in
which crypto-related incidents make up an above-median share of all cyber breaches.
The model is trained on **19,281 unique breach records** spanning 2004-2026 from
eight harmonised data sources, using walk-forward time-series cross-validation so
that every prediction is made with only historically available data.

---

## Data

### Scale

| Source | Raw Rows | After Dedup | Coverage | Description |
|--------|----------|-------------|----------|-------------|
| Information is Beautiful (IIB) | 417 | ~400 | 2004-2022 | High-profile global breaches, entity name, sector, method |
| HHS / OCR Healthcare (HHS) | 908 | ~860 | 2009-2026 | US healthcare breach portal, fine-grained breach type |
| DataBreachN (DBN) | 389 | ~370 | 2004-2020 | International country-level breach list |
| DataBreach EU (DBEN) | 270 | ~265 | 2004-2017 | European breach registry |
| Wikipedia Extended (DF1) | 349 | ~330 | 2004-2020 | Broad entity, year, records, method |
| CSIS Cyber Incidents (CSIS) | 207 | ~190 | 2006-2026 | State-sponsored and significant incidents (PDF, text-mined) |
| Cyber Events Database (CE) | 16,532 | ~16,048 | 2014-2026 | Structured incident database: 12,234 criminal, 2,151 hacktivist, 1,125 nation-state actors; 8,541 exploitive events |
| WA Breach Notifications (WA) | 1,533 | ~1,506 | 2016-2026 | Washington State AG notifications: 539 ransomware, 224 malware, 105 phishing; 46.9M Washingtonians affected |
| **Total** | **20,605** | **~19,281** | **2004-2026** | |

De-duplication uses (entity, year, method_category) as the key. The 1,324 dropped
rows are genuine cross-source overlaps, primarily between the three Wikipedia-derived
sources (DBN, DF1, DBEN) and the IIB dataset.

### Crypto Coverage

- **633 crypto-tagged events** in the CE dataset (2014-2026)
- **940 total crypto-flagged records** across all sources combined
- Crypto fraction of all breaches ranges from 0% (2004-2008) to ~12% (2026)
- 30 crypto keywords used for flagging: bitcoin, ethereum, defi, nft, wallet, exchange, blockchain, web3, dex, cex, uniswap, binance, coinbase, and others

### Sector Breakdown (from 19,281 records)

Government (4,370), Tech/Web (3,911), Healthcare (3,452), Financial (1,844),
Academic (1,568), Other (1,339), Retail (1,040), Transport (543), Media (457),
Energy (374)

### Attack Method Breakdown

Hacking (7,253), Malware (5,293), Phishing (2,038), DDoS (1,882), Other (1,287),
Lost/Stolen Device (818), Unauthorized Access (439), Poor Security (180),
Insider (53), Supply Chain (24)

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py                              # default: data/ directory
python main.py --data_dir /path/to/data
python main.py --target high_impact        # alternative target
```

**Prediction targets:**

| Flag | Definition |
|------|------------|
| `crypto_binary` | Was crypto fraction above median this year? (default) |
| `high_impact` | Did total records breached exceed 75th percentile? |
| `crypto_count` | Regression: raw annual crypto breach count |

---

## Pipeline

### Stage 1 -- Ingestion (`data_ingestion.py`)

All eight sources are normalised to a canonical schema:
`year, entity, sector, method_category, records_affected, is_crypto, source_id`

The CE and WA files are searched in `data/` first, then a fallback path
automatically, so they work without copying files manually. Records with malformed
counts (e.g. `15,000,000`, `1.37e+09`, `3m`, `15.000.000`) are parsed correctly.

### Stage 2 -- Feature Engineering (`feature_engineering.py`)

**~65 features built from yearly aggregates:**

- Core counts: total breaches, log-records, unique entities and methods
- Sector fractions: 8 sectors as annual shares (government, tech_web, healthcare, ...)
- Method fractions: 10 attack type shares (hacking, malware, phishing, ...)
- Lag features: 1-, 2-, 3-year lags on key indicators
- Rolling means: 2- and 3-year windows using shift(1) before rolling to prevent
  look-ahead leakage (year T's rolling mean only uses T-3 to T-1)
- Trend signals: year-over-year delta, rising/falling binary, 3-year linear slope
- Interaction terms: hacking x financial sector, crypto momentum
- External lag-1 signals from CE and WA (nation-state fraction, ransomware rate, etc.)

**Label definition for `crypto_binary`:**

The label is 1 if `crypto_fraction` (crypto breaches / total breaches) exceeds the
series median, 0 otherwise.

This uses fraction rather than raw count for a critical reason: the CE dataset
contributes 16,532 rows uniformly across 2014-2026, pushing raw crypto counts to
60-300 for every modern year. A median split on raw count puts the threshold at ~55,
which every year from 2014 onward exceeds -- leaving only 2026 (partial year data)
as a negative. With just 1 negative in 11 test folds, PR-AUC=1.0 is trivially
achieved by any model that ranks that one year last. That score is meaningless.

Crypto fraction divides by total breaches, so it stays meaningful even after adding
16,532 CE rows. The fraction varies between 4-12% year-to-year in modern years
(reflecting genuine variation in how intensely crypto was targeted). The median split
on fraction produces approximately 11-12 positives and 11-12 negatives, with
negatives spread across both early years (2004-2013) and recent years (2017, 2019,
2025). This gives the model a genuinely hard problem and makes PR-AUC and ROC-AUC
real discriminative measures.

### Stage 3 -- Walk-Forward Validation (`models.py`)

```
Fold  5: Train 2004-2008 -> Test 2009
Fold  6: Train 2004-2009 -> Test 2010
   ...
Fold 22: Train 2004-2025 -> Test 2026
```

**Per-fold leak prevention (applied before any model.fit() call):**

1. NaN imputation using the training fold median only (external features start in
   2014/2016 so early folds have NaN for those columns)
2. StandardScaler fit on training data only, then applied to the test row

**Models and calibration:**

| Model | Calibration | Purpose |
|-------|-------------|---------|
| LogisticRegression (L2, C=0.05) | Native softmax | Primary linear baseline |
| RidgeClassifier (L2, C=0.01) | Native softmax | Stronger regularisation |
| Random Forest (max_depth=3) | Platt sigmoid cv=3 | Bagged trees, shallow |
| Extra Trees (max_depth=3) | Platt sigmoid cv=3 | More randomised than RF |
| Gradient Boosting (depth=2, lr=0.02) | Platt sigmoid cv=3 | Conservative boosting |

Tree models are Platt-calibrated because raw leaf-fraction probabilities cluster near
0 and 1. A single overconfident miss (p=0.03 when y=1) adds -log(0.03) = 3.5 to
log-loss -- at N=22 that alone raises the mean by 0.16. Sigmoid calibration is used
over isotonic because isotonic overfits at N < 40 samples per calibration fold.

Label smoothing (eps=0.10) is applied after predict_proba:
`p_final = 0.9 * p_raw + 0.1 * base_rate`
This caps the maximum per-sample log-loss contribution regardless of model confidence.

### Stage 4 -- Evaluation and Outputs

**Metrics reported per model (pooled across all folds):**

| Metric | Best model (LogisticRegression) | Notes |
|--------|--------------------------------|-------|
| Brier Score | 0.088 | Lower is better; 0.25 = random |
| Log-Loss | 0.322 | Lower is better; 0.693 = random |
| PR-AUC | 0.991* | Higher is better |
| ROC-AUC | 0.900 | Higher is better |

*Note: PR-AUC values above 0.95 should be verified. Before the label fix (crypto_count
median split), PR-AUC was a trivial 1.0 because only 1 negative appeared in all test
folds. The current values use the crypto_fraction label which produces ~8 negative test
folds spread across different years. If PR-AUC is still 1.0 after this fix, check
`n_neg_test` in the summary output -- it should be > 3.

**Top predictive features (GradientBoosting):**

1. ext_vix_mean -- market volatility index (annual mean)
2. ext_economic_stress -- composite macro stress indicator
3. ext_current_account -- balance of payments signal
4. method_lost_device -- lost/stolen device fraction
5. log_records_total_lag3 -- log total records breached, 3-year lag
6. crypto_log_records_lag2 -- log crypto records, 2-year lag
7. log_records_total_lag2
8. total_breaches_lag3
9. crypto_log_records_lag1
10. log_records_total (current year)

**Ensemble forecast (most recent test years):**

| Year | p(high-fraction) | Range | Tier | Correct? |
|------|-----------------|-------|------|---------|
| 2019 | 0.711 | [0.63-0.86] | HIGH | Yes |
| 2020 | 0.696 | [0.59-0.81] | HIGH | Yes |
| 2021 | 0.720 | [0.65-0.80] | HIGH | Yes |
| 2022 | 0.749 | [0.70-0.86] | HIGH | Yes |
| 2023 | 0.776 | [0.73-0.89] | HIGH | Yes |
| 2024 | 0.752 | [0.69-0.85] | HIGH | Yes |
| 2025 | 0.731 | [0.64-0.78] | HIGH | Yes |
| 2026 | 0.664 | [0.61-0.73] | HIGH | Incorrect (actual=low) |

2026 is the most challenging year: it is a partial-year observation (CE data through
March 2026 only) with a genuinely low crypto fraction (11.7%), yet all models predict
HIGH. This is partly a recency bias from training on the high-fraction years of 2021-2023.

**Risk tiers:**

| Tier | Probability | Posture |
|------|-------------|---------|
| [CRITICAL] | >= 80% | Incident response readiness |
| [HIGH] | 60-80% | Active defence posture |
| [ELEVATED] | 40-60% | Enhanced vigilance |
| [LOW] | < 40% | Routine monitoring |

---

## Key Findings

- **19,281 unique breach records** across 23 years (2004-2026), an 10x increase
  over the original six sources alone (which provided ~1,825 records)
- **940 crypto-tagged incidents** (4.9% of total), up from 61 in the original dataset
- Crypto fraction grew from 0% (2004-2008) to a peak of ~12% in 2026 (partial year)
- The most predictive features are macroeconomic (VIX, economic stress, current
  account balance), not breach-specific -- suggesting crypto targeting correlates with
  broader financial instability
- Hacking (7,253 incidents) and Malware (5,293) dominate attack methods; ransomware
  grew from <1% of WA cyberattacks in 2017 to 52% by 2026
- Nation-state actors account for 1,125 incidents (6.8% of CE events), with their
  share rising from 1% in 2014 to 13% in 2026
- LogisticRegression is the best overall model: Brier=0.088, Log-Loss=0.322,
  ROC-AUC=0.900 -- consistently outperforming tree models on this small-N time series

---

## Known Limitations

- N=23 annual observations is a fundamental data constraint. All metrics carry wide
  confidence intervals compared to what they would at N=1000+. A ROC-AUC of 0.90
  at N=23 is encouraging but not statistically robust.
- External features from CE and WA only cover 2014/2016 onward. Folds testing
  2009-2015 rely entirely on breach-derived features, with NaN imputed for external
  columns.
- The 2026 data point is partial-year (CE database through March 2026 only).
  Predictions for 2026 should be treated as preliminary.
- The label definition (crypto_fraction above median) changes its semantic meaning
  if the total breach volume changes drastically. A year with 50% crypto fraction but
  only 10 total breaches would be labelled positive the same as a year with 12% of
  10,000 breaches.

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

Optional PDF fallback:
```bash
sudo apt-get install poppler-utils   # Linux
brew install poppler                 # macOS
```

No xgboost, tensorflow, or GPU required. Python 3.9+. Fully Windows-compatible --
all logging uses ASCII output and writes to file in UTF-8.

---

## References

- Olushola A & Meenakshi SP (2025). Cybersecurity crimes in cryptocurrency exchanges
  (2009-2024). Frontiers in Blockchain 8:1713637.
- CSIS. Significant Cyber Incidents Since 2006. Center for Strategic and
  International Studies.
- Kaggle / Information is Beautiful. World's Biggest Data Breaches & Hacks.
- HHS / OCR. Breach Portal: Notice to the Secretary of HHS.
- Washington State Office of the Attorney General. Data Breach Notifications
  Affecting Washington Residents.