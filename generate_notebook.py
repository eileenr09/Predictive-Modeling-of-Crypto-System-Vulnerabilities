"""
generate_notebook.py
====================
Generates the full project Jupyter notebook programmatically.
Run:  python generate_notebook.py
Then: jupyter notebook Crypto_Breach_ML_Pipeline.ipynb
"""

import json, os

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def md(source): return {"cell_type":"markdown","metadata":{},"source":source}
def code(source, outputs=None):
    return {"cell_type":"code","execution_count":None,
            "metadata":{"tags":[]},"outputs": outputs or [],"source":source}

# ─────────────────────────────────────────────────────────────────────────────
# Cells
# ─────────────────────────────────────────────────────────────────────────────

cells = []

# ── Title ────────────────────────────────────────────────────────────────────
cells.append(md([
"# 🔐 Predictive Modeling of Crypto-System Vulnerabilities\n",
"## End-to-End Machine Learning Pipeline\n",
"**Authors:** Eileen Rashduni · Atharv Koratkar\n\n",
"This notebook walks through every stage of the pipeline:\n\n",
"| Stage | Description |\n",
"|-------|-------------|\n",
"| **Stage 1** | Data Ingestion — load & harmonise all 6 real datasets |\n",
"| **Stage 2** | EDA — visualise raw breach data |\n",
"| **Stage 3** | Feature Engineering — build model-ready feature matrix |\n",
"| **Stage 4** | Feature Analysis — correlation, distribution, importance |\n",
"| **Stage 5** | Walk-Forward Model Training — 5 models × time-series CV |\n",
"| **Stage 6** | Evaluation — Brier Score, Log-Loss, PR-AUC |\n",
"| **Stage 7** | Explainability — feature importance & SHAP-style analysis |\n",
"| **Stage 8** | Risk Scoring — probability output & alert tier |\n",
]))

# ── Setup ────────────────────────────────────────────────────────────────────
cells.append(md(["## ⚙️ Setup"]))
cells.append(code([
"import sys, os, warnings\n",
"warnings.filterwarnings('ignore')\n\n",
"# Add project root to path\n",
"sys.path.insert(0, os.path.dirname(os.path.abspath('')))\n\n",
"import numpy as np\n",
"import pandas as pd\n",
"import matplotlib\n",
"import matplotlib.pyplot as plt\n",
"import matplotlib.gridspec as gridspec\n",
"from IPython.display import display, HTML\n\n",
"# Inline plotting\n",
"%matplotlib inline\n",
"plt.rcParams.update({\n",
"    'figure.dpi': 120,\n",
"    'axes.spines.top': False,\n",
"    'axes.spines.right': False,\n",
"    'font.size': 10,\n",
"})\n\n",
"PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']\n",
"print('Environment ready ✓')\n",
"print(f'  numpy  {np.__version__}')\n",
"print(f'  pandas {pd.__version__}')\n",
"import sklearn; print(f'  sklearn {sklearn.__version__}')\n",
]))

# ── Stage 1 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 📥 Stage 1 — Data Ingestion\n\n",
"We load and harmonise **6 independent datasets**:\n\n",
"| ID | Source | Records | Description |\n",
"|----|--------|---------|-------------|\n",
"| IIB | Information is Beautiful | ~440 | High-profile global breaches 2004–2022 |\n",
"| HHS | HHS / OCR | ~1 285 | US Healthcare breaches 2009–present |\n",
"| DBN | DataBreachN | ~352 | International, country-level |\n",
"| DBEN | DataBreach EU | ~277 | European registry 2004–2017 |\n",
"| DF1 | Wikipedia Extended | ~352 | Broad entity list |\n",
"| CSIS | CSIS PDF | text-mined | State-sponsored / major incidents 2006–2025 |\n",
]))

cells.append(code([
"from data_ingestion import load_all_datasets, flag_crypto, categorise_method\n\n",
"df = load_all_datasets(data_dir='data')\n\n",
"print(f'\\n✅ Total records loaded: {len(df):,}')\n",
"print(f'   Year range:           {df[\"year\"].min()} – {df[\"year\"].max()}')\n",
"print(f'   Crypto-related:       {df[\"is_crypto\"].sum():,} ({df[\"is_crypto\"].mean()*100:.1f}%)')\n",
"print(f'   Source breakdown:')\n",
"print(df['source_id'].value_counts().to_string())\n",
]))

cells.append(code([
"# Preview the unified DataFrame\n",
"display(HTML('<b>Unified Breach DataFrame (first 10 rows)</b>'))\n",
"display(df.head(10).style.background_gradient(subset=['records_affected'], cmap='Blues'))\n",
]))

cells.append(code([
"# Data quality summary\n",
"print('=== Data Quality ===')\n",
"print(df.dtypes.to_string())\n",
"print('\\nNull counts:')\n",
"print(df.isnull().sum().to_string())\n",
"print(f'\\nMethod categories: {sorted(df[\"method_category\"].unique())}')\n",
"print(f'Sectors:           {sorted(df[\"sector\"].unique())}')\n",
]))

# ── Stage 2 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 📊 Stage 2 — Exploratory Data Analysis\n\n",
"Visualise the raw breach landscape before any feature engineering.\n",
]))

cells.append(code([
"# ── 2A: Breach counts by year, coloured by source ──────────────────────\n",
"fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n",
"fig.suptitle('Dataset Overview', fontsize=13, fontweight='bold')\n\n",
"yr_src = df.groupby(['year','source_id']).size().unstack(fill_value=0)\n",
"yr_src.plot(kind='bar', stacked=True, ax=axes[0],\n",
"            colormap='tab10', legend=True, width=0.85)\n",
"axes[0].set_title('Breaches per Year by Source')\n",
"axes[0].set_xlabel('Year'); axes[0].set_ylabel('Count')\n",
"axes[0].legend(fontsize=7, ncol=2)\n",
"axes[0].tick_params(axis='x', rotation=45, labelsize=7)\n\n",
"src_cnt = df['source_id'].value_counts()\n",
"src_labels = {'IIB':'IIB\\n(Global)','HHS':'HHS\\n(Healthcare)',\n",
"              'DBN':'DBN\\n(Intl)','DBEN':'DBEN\\n(EU)',\n",
"              'DF1':'DF1\\n(Wiki)','CSIS':'CSIS\\n(Gov)'}\n",
"axes[1].bar([src_labels.get(s,s) for s in src_cnt.index],\n",
"            src_cnt.values, color=PALETTE)\n",
"axes[1].set_title('Records per Source'); axes[1].set_ylabel('Count')\n",
"axes[1].grid(axis='y', alpha=0.3)\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── 2B: Sector & Method distributions ─────────────────────────────────\n",
"fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\n",
"sec = df['sector'].value_counts().head(10)\n",
"axes[0].barh(sec.index[::-1], sec.values[::-1], color='#1f77b4', alpha=0.85)\n",
"axes[0].set_title('Top 10 Sectors', fontweight='bold')\n",
"axes[0].set_xlabel('Breach Count'); axes[0].grid(axis='x', alpha=0.3)\n",
"for i, v in enumerate(sec.values[::-1]):\n",
"    axes[0].text(v+2, i, str(v), va='center', fontsize=8)\n\n",
"meth = df['method_category'].value_counts().head(10)\n",
"colours = plt.cm.Set2(np.linspace(0, 1, len(meth)))\n",
"axes[1].barh(meth.index[::-1], meth.values[::-1], color=colours)\n",
"axes[1].set_title('Attack Method Distribution', fontweight='bold')\n",
"axes[1].set_xlabel('Breach Count'); axes[1].grid(axis='x', alpha=0.3)\n",
"for i, v in enumerate(meth.values[::-1]):\n",
"    axes[1].text(v+2, i, str(v), va='center', fontsize=8)\n\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── 2C: Crypto vs Non-Crypto over time ─────────────────────────────────\n",
"yr = df.groupby('year')\n",
"total  = yr.size().rename('total')\n",
"crypto = yr['is_crypto'].sum().rename('crypto')\n",
"comb   = pd.concat([total, crypto], axis=1).fillna(0)\n",
"comb['non_crypto'] = comb['total'] - comb['crypto']\n\n",
"fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))\n",
"fig.suptitle('Crypto vs All Breaches Over Time', fontsize=13, fontweight='bold')\n\n",
"ax1.bar(comb.index, comb['non_crypto'], label='Non-crypto', color='#aec7e8')\n",
"ax1.bar(comb.index, comb['crypto'], bottom=comb['non_crypto'],\n",
"        label='Crypto', color='#d62728')\n",
"ax1.set_title('Stacked Breach Count'); ax1.legend(); ax1.grid(axis='y', alpha=0.3)\n",
"ax1.set_xlabel('Year'); ax1.tick_params(axis='x', rotation=45)\n\n",
"pct = (comb['crypto'] / comb['total'].replace(0,1)) * 100\n",
"ax2.plot(pct.index, pct.values, 'o-', color='#d62728', linewidth=2)\n",
"ax2.fill_between(pct.index, pct.values, alpha=0.15, color='#d62728')\n",
"ax2.set_title('Crypto % of All Breaches'); ax2.set_ylabel('%')\n",
"ax2.set_xlabel('Year'); ax2.grid(alpha=0.3)\n\n",
"# Heatmap: method × year (crypto only)\n",
"crypto_df = df[df['is_crypto']]\n",
"if len(crypto_df) > 0:\n",
"    heat = crypto_df.groupby(['year','method_category']).size().unstack(fill_value=0)\n",
"    im = ax3.imshow(heat.T, aspect='auto', cmap='YlOrRd')\n",
"    ax3.set_yticks(range(len(heat.columns)))\n",
"    ax3.set_yticklabels(heat.columns, fontsize=8)\n",
"    ax3.set_xticks(range(len(heat.index)))\n",
"    ax3.set_xticklabels(heat.index, rotation=45, fontsize=7)\n",
"    ax3.set_title('Crypto Breaches: Year × Method')\n",
"    plt.colorbar(im, ax=ax3, label='Count')\n\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── 2D: Records exposed distribution (log scale) ───────────────────────\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n\n",
"# Histogram of log(records)\n",
"log_rec = np.log1p(df['records_affected'].dropna())\n",
"ax1.hist(log_rec, bins=40, color='#2ca02c', alpha=0.7, edgecolor='white')\n",
"ax1.set_title('Distribution of log(Records Affected)', fontweight='bold')\n",
"ax1.set_xlabel('log(1 + Records)'); ax1.set_ylabel('Frequency')\n",
"ax1.axvline(log_rec.median(), color='red', linestyle='--', label=f'Median={log_rec.median():.1f}')\n",
"ax1.legend(); ax1.grid(alpha=0.3)\n\n",
"# Crypto vs non-crypto record comparison (box plot)\n",
"# Cast to bool first to avoid bitwise NOT issue on mixed-type columns\n",
"is_crypto_mask = df['is_crypto'].astype(bool)\n",
"crypto_rec  = np.log1p(df[is_crypto_mask]['records_affected'].dropna())\n",
"ncrypto_rec = np.log1p(df[~is_crypto_mask]['records_affected'].dropna())\n",
"bp = ax2.boxplot([ncrypto_rec, crypto_rec],\n",
"                 labels=['Non-Crypto', 'Crypto'],\n",
"                 patch_artist=True,\n",
"                 boxprops=dict(facecolor='#aec7e8'),\n",
"                 medianprops=dict(color='red', linewidth=2))\n",
"bp['boxes'][1].set_facecolor('#ffbb78')\n",
"ax2.set_title('Records Exposed: Crypto vs Non-Crypto', fontweight='bold')\n",
"ax2.set_ylabel('log(1 + Records)'); ax2.grid(alpha=0.3)\n\n",
"plt.tight_layout(); plt.show()\n",
"print(f'Crypto median records:     {10**crypto_rec.median():,.0f}')\n",
"print(f'Non-crypto median records: {10**ncrypto_rec.median():,.0f}')\n",
]))

# ── Stage 3 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 🔧 Stage 3 — Feature Engineering\n\n",
"Transform the raw incident table into a **yearly feature matrix** suitable for ML:\n\n",
"- Annual breach counts, crypto fraction, log-records\n",
"- **Sector one-hot fractions** (7 sectors)\n",
"- **Method one-hot fractions** (10 attack types)\n",
"- **Lag features** (1-year, 2-year) for key indicators\n",
"- **Rolling mean/std** windows (2, 3 years)\n",
"- **Interaction terms**: hacking × financial, records × crypto count\n",
"- **Year-over-year growth** rates\n\n",
"Label: `1` if ≥1 crypto breach occurred that year.\n",
]))

cells.append(code([
"from feature_engineering import build_feature_matrix\n\n",
"X, y, feature_names, scaler, yearly_df = build_feature_matrix(\n",
"    df, target_mode='crypto_binary', scale=True\n",
")\n\n",
"print(f'Feature matrix shape: {X.shape}')\n",
"print(f'Years covered:        {X.index.min()} – {X.index.max()}')\n",
"print(f'Positive class rate:  {y.mean():.2%}  ({y.sum()} / {len(y)} years had crypto breaches)')\n",
"print(f'\\nFirst 5 feature names: {feature_names[:5]}')\n",
"print(f'Last 5 feature names:  {feature_names[-5:]}')\n",
]))

cells.append(code([
"# Show the yearly aggregated DataFrame\n",
"display(HTML('<b>Yearly Aggregated Feature Table</b>'))\n",
"display(\n",
"    yearly_df[['total_breaches','crypto_breach_count','crypto_fraction',\n",
"               'log_records_total','unique_entities','unique_methods']]\n",
"    .style\n",
"    .background_gradient(subset=['crypto_breach_count'], cmap='Reds')\n",
"    .background_gradient(subset=['total_breaches'], cmap='Blues')\n",
"    .format({'crypto_fraction': '{:.1%}', 'log_records_total': '{:.2f}'})\n",
")\n",
]))

cells.append(code([
"# Show scaled feature matrix\n",
"display(HTML('<b>Scaled Feature Matrix (first 5 columns, all years)</b>'))\n",
"display(\n",
"    X.iloc[:, :8]\n",
"    .style\n",
"    .background_gradient(cmap='RdBu_r', axis=0)\n",
"    .format('{:.3f}')\n",
")\n",
]))

# ── Stage 4 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 🔍 Stage 4 — Feature Analysis\n\n",
"Examine feature distributions, correlations, and predictive value\nbefore fitting any model.\n",
]))

cells.append(code([
"# ── 4A: Feature correlation heatmap ───────────────────────────────────\n",
"fig, ax = plt.subplots(figsize=(14, 12))\n",
"corr = X.corr()\n",
"im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')\n",
"ax.set_xticks(range(len(corr.columns)))\n",
"ax.set_yticks(range(len(corr.columns)))\n",
"ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)\n",
"ax.set_yticklabels(corr.columns, fontsize=6)\n",
"plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')\n",
"ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── 4B: Feature-label correlation (Pearson r with y) ──────────────────\n",
"corr_y = X.corrwith(y.astype(float)).abs().sort_values(ascending=False)\n",
"top15  = corr_y.head(15)\n\n",
"fig, ax = plt.subplots(figsize=(10, 5))\n",
"bars = ax.barh(top15.index[::-1], top15.values[::-1],\n",
"               color=plt.cm.viridis(np.linspace(0.2, 0.9, 15)))\n",
"ax.set_xlabel('|Pearson r| with Label')\n",
"ax.set_title('Top 15 Features Correlated with Crypto Breach Label',\n",
"             fontweight='bold')\n",
"ax.grid(axis='x', alpha=0.3)\n",
"for bar, val in zip(bars, top15.values[::-1]):\n",
"    ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,\n",
"            f'{val:.3f}', va='center', fontsize=8)\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── 4C: Class-conditional feature distributions ────────────────────────\n",
"top5_feats = corr_y.head(5).index.tolist()\n",
"fig, axes = plt.subplots(1, 5, figsize=(18, 4))\n",
"fig.suptitle('Class-Conditional Feature Distributions (Top 5)', fontweight='bold')\n\n",
"X_with_y = X.copy(); X_with_y['label'] = y\n",
"for ax, feat in zip(axes, top5_feats):\n",
"    for cls, colour, label in [(0,'#1f77b4','No crypto breach'),\n",
"                                (1,'#d62728','Crypto breach')]:\n",
"        vals = X_with_y[X_with_y['label']==cls][feat].dropna()\n",
"        ax.hist(vals, bins=10, alpha=0.6, color=colour, label=label, density=True)\n",
"    ax.set_title(feat[:25], fontsize=8)\n",
"    ax.set_xlabel('Scaled Value'); ax.grid(alpha=0.3)\n",
"    if ax == axes[0]:\n",
"        ax.legend(fontsize=7)\n\n",
"plt.tight_layout(); plt.show()\n",
]))

# ── Stage 5 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 🤖 Stage 5 — Walk-Forward Model Training\n\n",
"We train **5 models** using **walk-forward (expanding-window) time-series CV**:\n\n",
"```\n",
"Fold 0: Train 2004–2008 → Test 2009\n",
"Fold 1: Train 2004–2009 → Test 2010\n",
"...\n",
"Fold N: Train 2004–(last-1) → Test last year\n",
"```\n\n",
"No fold ever uses future data to train. Calibration is applied via isotonic regression.\n",
]))

cells.append(code([
"from models import WalkForwardValidator, build_models\n\n",
"wfv = WalkForwardValidator(\n",
"    train_window=None,   # expanding window\n",
"    min_train_size=5,\n",
"    calibrate=False,\n",
")\n\n",
"# v2 models: LogisticRegression, RandomForest, ExtraTrees, GradientBoosting, SVM\n",
"models = build_models()\n",
"results = wfv.fit_predict(X, y, models=models)\n\n",
"print(f'Folds run: {len(results)}')\n",
"print(f'Models:    {list(models.keys())}')\n",
]))

cells.append(code([
"# Per-fold predictions table\n",
"preds_df = wfv.consolidated_predictions()\n",
"display(HTML('<b>Walk-Forward Predictions (first 20 rows)</b>'))\n",
"display(\n",
"    preds_df.head(20)\n",
"    .style\n",
"    .background_gradient(subset=['y_prob'], cmap='RdYlGn_r')\n",
"    .applymap(lambda v: 'color: red; font-weight: bold' if v == 1 else '',\n",
"              subset=['y_true'])\n",
"    .format({'y_prob': '{:.3f}'})\n",
")\n",
]))

# ── Stage 6 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 📏 Stage 6 — Evaluation Metrics\n\n",
"Three primary metrics from the project proposal (Section 5.3):\n\n",
"| Metric | Formula | Purpose |\n",
"|--------|---------|------|\n",
"| **Brier Score** | `mean((F_t − O_t)²)` | Calibration — lower is better |\n",
"| **Log-Loss** | `−mean[O·log F + (1−O)·log(1−F)]` | Penalises overconfidence |\n",
"| **PR-AUC** | Area under Precision-Recall curve | Best for rare positive class |\n",
]))

cells.append(code([
"# Summary metrics across all models\n",
"summary = wfv.summary_metrics()\n",
"print('=== Walk-Forward Mean Metrics ===')\n",
"display(\n",
"    summary.style\n",
"    .highlight_min(subset=['brier','log_loss'], color='#c6efce')\n",
"    .highlight_max(subset=['pr_auc','roc_auc'], color='#c6efce')\n",
"    .format('{:.4f}')\n",
")\n",
]))

cells.append(code([
"# ── Model comparison bar chart ─────────────────────────────────────────\n",
"metrics = ['pr_auc', 'roc_auc', 'brier', 'log_loss']\n",
"titles  = ['PR-AUC ↑', 'ROC-AUC ↑', 'Brier Score ↓', 'Log-Loss ↓']\n",
"fig, axes = plt.subplots(1, 4, figsize=(18, 5))\n",
"fig.suptitle('Model Comparison (Walk-Forward Mean)', fontsize=13, fontweight='bold')\n\n",
"for ax, metric, title, colour in zip(axes, metrics, titles,\n",
"                                      ['#2ca02c','#1f77b4','#ff7f0e','#d62728']):\n",
"    vals = summary[metric].dropna().sort_values(ascending=(metric in ['brier','log_loss']))\n",
"    bars = ax.bar(vals.index, vals.values, color=colour, alpha=0.85)\n",
"    ax.set_title(title, fontweight='bold')\n",
"    ax.tick_params(axis='x', rotation=30, labelsize=8)\n",
"    ax.grid(axis='y', alpha=0.3)\n",
"    for bar, val in zip(bars, vals.values):\n",
"        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,\n",
"                f'{val:.3f}', ha='center', fontsize=7)\n\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── Precision-Recall curves ────────────────────────────────────────────\n",
"from sklearn.metrics import precision_recall_curve, average_precision_score\n\n",
"fig, ax = plt.subplots(figsize=(8, 6))\n",
"ax.set_title('Precision-Recall Curves (All Folds Combined)', fontweight='bold')\n\n",
"for i, (model_name, grp) in enumerate(preds_df.groupby('model')):\n",
"    yt, yp = grp['y_true'].values, grp['y_prob'].values\n",
"    if yt.sum() == 0: continue\n",
"    prec, rec, _ = precision_recall_curve(yt, yp)\n",
"    auc_val = average_precision_score(yt, yp)\n",
"    ax.plot(rec, prec, linewidth=2, color=PALETTE[i%len(PALETTE)],\n",
"            label=f'{model_name} (AUC={auc_val:.3f})')\n\n",
"ax.set_xlabel('Recall'); ax.set_ylabel('Precision')\n",
"ax.legend(fontsize=8); ax.grid(alpha=0.3)\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── Probability Timeline ───────────────────────────────────────────────\n",
"fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,\n",
"                                gridspec_kw={'height_ratios':[3,1]})\n",
"fig.suptitle('Predicted Breach Probability Over Time', fontsize=13, fontweight='bold')\n\n",
"for i, (mname, grp) in enumerate(preds_df.groupby('model')):\n",
"    yr_prob = grp.groupby('year')['y_prob'].mean()\n",
"    ax1.plot(yr_prob.index, yr_prob.values, 'o-', linewidth=2, markersize=5,\n",
"             color=PALETTE[i%len(PALETTE)], label=mname, alpha=0.85)\n\n",
"ax1.axhline(0.5, color='crimson', linestyle='--', linewidth=1, label='0.5 threshold')\n",
"ax1.set_ylabel('Mean Predicted Probability')\n",
"ax1.legend(fontsize=7, ncol=2); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1)\n\n",
"actual = preds_df.groupby('year')['y_true'].max()\n",
"ax2.vlines(actual[actual==1].index, 0, 1, color='darkred',\n",
"           linewidth=3, alpha=0.8, label='Actual breach year')\n",
"ax2.set_yticks([]); ax2.set_ylabel('Actual\\nBreaches', fontsize=9)\n",
"ax2.set_xlabel('Year'); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)\n\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── Manual metric verification (Section 5.3 formulae) ─────────────────\n",
"from evaluation import brier_score, log_loss_manual\n\n",
"best_model_name = summary['pr_auc'].idxmax()\n",
"best_preds = preds_df[preds_df['model'] == best_model_name]\n",
"yt = best_preds['y_true'].values\n",
"yp = best_preds['y_prob'].values\n\n",
"print(f'Best model: {best_model_name}')\n",
"print(f'  Brier Score (manual):  {brier_score(yt, yp):.4f}')\n",
"print(f'  Log-Loss (manual):     {log_loss_manual(yt, yp):.4f}')\n",
"print(f'  PR-AUC:                {average_precision_score(yt, yp):.4f}')\n",
]))

# ── Stage 7 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 🧩 Stage 7 — Feature Explainability\n\n",
"Which societal/breach variables drive the model's predictions?\n\n",
"We use the **built-in feature importances** from Random Forest and\nGradient Boosting (tree-based SHAP equivalent).\n",
]))

cells.append(code([
"# ── Feature importance: RF ─────────────────────────────────────────────\n",
"fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
"fig.suptitle('Mean Feature Importance Across Folds', fontsize=13, fontweight='bold')\n\n",
"for ax, mname in zip(axes, ['RandomForest', 'GradientBoosting']):\n",
"    fi = wfv.mean_feature_importance(mname)\n",
"    if fi.empty: ax.set_visible(False); continue\n",
"    top = fi.head(20)\n",
"    colours = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(top)))\n",
"    ax.barh(top.index[::-1], top.values[::-1], color=colours[::-1])\n",
"    ax.set_title(f'{mname}: Top-20 Features', fontweight='bold')\n",
"    ax.set_xlabel('Mean Importance')\n",
"    ax.grid(axis='x', alpha=0.3)\n\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── Cumulative importance (Pareto) ─────────────────────────────────────\n",
"fi_rf = wfv.mean_feature_importance('RandomForest')\n",
"if not fi_rf.empty:\n",
"    cum = fi_rf.cumsum() / fi_rf.sum() * 100\n",
"    fig, ax = plt.subplots(figsize=(11, 4))\n",
"    ax.bar(range(len(fi_rf)), fi_rf.values / fi_rf.sum() * 100,\n",
"           color='#1f77b4', alpha=0.7, label='Individual importance')\n",
"    ax2_twin = ax.twinx()\n",
"    ax2_twin.plot(range(len(cum)), cum.values, 'r-o', markersize=3, linewidth=2,\n",
"                  label='Cumulative')\n",
"    ax2_twin.axhline(80, color='grey', linestyle='--', linewidth=0.8)\n",
"    ax.set_xticks(range(len(fi_rf)))\n",
"    ax.set_xticklabels(fi_rf.index, rotation=90, fontsize=7)\n",
"    ax.set_ylabel('Individual Importance (%)')\n",
"    ax2_twin.set_ylabel('Cumulative Importance (%)')\n",
"    ax.set_title('Random Forest — Pareto Feature Importance', fontweight='bold')\n",
"    ax.grid(axis='y', alpha=0.3)\n",
"    n80 = (cum < 80).sum() + 1\n",
"    print(f'Features needed to explain 80% of variance: {n80}')\n",
"    plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── SHAP-style signed contribution (permutation approximation) ─────────\n",
"# We compute mean(pred | feat=high) - mean(pred | feat=low) for top features\n",
"best_fitted = wfv.best_models.get('RandomForest') or wfv.best_models.get('GradientBoosting')\n",
"if best_fitted is not None:\n",
"    fi_vals = wfv.mean_feature_importance('RandomForest')\n",
"    top_feats = fi_vals.head(8).index.tolist()\n",
"\n",
"    signed_impacts = {}\n",
"    for feat in top_feats:\n",
"        if feat not in X.columns: continue\n",
"        X_hi = X.copy(); X_hi[feat] = X[feat].quantile(0.75)\n",
"        X_lo = X.copy(); X_lo[feat] = X[feat].quantile(0.25)\n",
"        try:\n",
"            p_hi = best_fitted.predict_proba(X_hi)[:, 1].mean()\n",
"            p_lo = best_fitted.predict_proba(X_lo)[:, 1].mean()\n",
"            signed_impacts[feat] = p_hi - p_lo\n",
"        except: pass\n",
"\n",
"    si = pd.Series(signed_impacts).sort_values()\n",
"    colours = ['#d62728' if v > 0 else '#1f77b4' for v in si.values]\n",
"    fig, ax = plt.subplots(figsize=(9, 5))\n",
"    ax.barh(si.index, si.values, color=colours)\n",
"    ax.axvline(0, color='black', linewidth=0.8)\n",
"    ax.set_xlabel('Δ Predicted Probability (75th - 25th percentile)')\n",
"    ax.set_title('Directional Feature Impact (SHAP-Style Approximation)',\n",
"                 fontweight='bold')\n",
"    ax.grid(axis='x', alpha=0.3)\n",
"    plt.tight_layout(); plt.show()\n",
"    print('\\nRed = feature increase RAISES breach probability')\n",
"    print('Blue = feature increase LOWERS breach probability')\n",
]))

# ── Stage 8 ──────────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 🚨 Stage 8 — Risk Scoring & Alerts\n\n",
"Generate a **probability-based risk tier** for any given year's features.\n\n",
"| Tier | Probability | Action |\n",
"|------|-------------|--------|\n",
"| 🟢 LOW | < 40% | Routine monitoring |\n",
"| 🟡 ELEVATED | 40–60% | Enhanced vigilance |\n",
"| 🟠 HIGH | 60–80% | Active defence posture |\n",
"| 🔴 CRITICAL | ≥ 80% | Incident response readiness |\n",
]))

cells.append(code([
"def risk_tier(prob):\n",
"    if prob >= 0.80: return '🔴 CRITICAL'\n",
"    if prob >= 0.60: return '🟠 HIGH'\n",
"    if prob >= 0.40: return '🟡 ELEVATED'\n",
"    return '🟢 LOW'\n\n",
"# Score every test year for the best model\n",
"best_model_name = summary['pr_auc'].idxmax()\n",
"scored = preds_df[preds_df['model'] == best_model_name].copy()\n",
"scored['tier'] = scored['y_prob'].apply(risk_tier)\n\n",
"display(HTML(f'<b>Risk Scores — {best_model_name}</b>'))\n",
"display(\n",
"    scored[['year','y_prob','tier','y_true']]\n",
"    .rename(columns={'y_prob':'probability','y_true':'actual_breach'})\n",
"    .set_index('year')\n",
"    .style\n",
"    .background_gradient(subset=['probability'], cmap='RdYlGn_r')\n",
"    .applymap(lambda v: 'color: red; font-weight: bold' if v == 1 else '',\n",
"              subset=['actual_breach'])\n",
"    .format({'probability': '{:.1%}'})\n",
")\n",
]))

cells.append(code([
"# ── Risk tier distribution visualisation ──────────────────────────────\n",
"tier_order = ['🟢 LOW', '🟡 ELEVATED', '🟠 HIGH', '🔴 CRITICAL']\n",
"tier_cnt   = scored['tier'].value_counts().reindex(tier_order, fill_value=0)\n",
"tier_col   = ['#2ca02c', '#f4d03f', '#ff7f0e', '#d62728']\n\n",
"fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))\n",
"fig.suptitle(f'Risk Score Summary — {best_model_name}', fontsize=13, fontweight='bold')\n\n",
"ax1.bar(tier_order, tier_cnt.values, color=tier_col, alpha=0.9, edgecolor='white')\n",
"ax1.set_ylabel('Years in Tier'); ax1.set_title('Risk Tier Distribution')\n",
"ax1.grid(axis='y', alpha=0.3)\n",
"for i, (v, label) in enumerate(zip(tier_cnt.values, tier_order)):\n",
"    ax1.text(i, v+0.1, str(v), ha='center', fontweight='bold')\n\n",
"yr_prob_best = scored.set_index('year')['y_prob'].sort_index()\n",
"colours_line = [tier_col[['🟢 LOW','🟡 ELEVATED','🟠 HIGH','🔴 CRITICAL'].index(\n",
"                risk_tier(p))] for p in yr_prob_best.values]\n",
"ax2.bar(yr_prob_best.index, yr_prob_best.values, color=colours_line, alpha=0.85)\n",
"ax2.axhline(0.5, color='black', linestyle='--', linewidth=1)\n",
"ax2.set_xlabel('Year'); ax2.set_ylabel('Predicted Probability')\n",
"ax2.set_title('Yearly Risk Score Timeline')\n",
"ax2.grid(axis='y', alpha=0.3)\n\n",
"plt.tight_layout(); plt.show()\n",
]))

cells.append(code([
"# ── Final summary printout ─────────────────────────────────────────────\n",
"from sklearn.metrics import brier_score_loss, log_loss, average_precision_score\n\n",
"yt = scored['y_true'].values\n",
"yp = scored['y_prob'].values\n\n",
"print('=' * 55)\n",
"print(f'  FINAL EVALUATION — {best_model_name}')\n",
"print('=' * 55)\n",
"print(f'  Brier Score : {brier_score_loss(yt, yp):.4f}')\n",
"print(f'  Log Loss    : {log_loss(yt, yp):.4f}')\n",
"print(f'  PR-AUC      : {average_precision_score(yt, yp):.4f}')\n",
"print('='*55)\n",
"print(f'\\n  Records processed from ALL sources: {len(df):,}')\n",
"print(f'  Crypto-related incidents flagged:   {df[\"is_crypto\"].sum():,}')\n",
"print(f'  Feature dimensions:                 {X.shape[1]}')\n",
"print(f'  Walk-forward folds:                 {len(wfv.all_results) // (len(models)+1)}')\n",
]))

# ── Requirements ─────────────────────────────────────────────────────────────
cells.append(md([
"---\n",
"## 📦 Requirements\n\n",
"```\n",
"numpy>=1.24\n",
"pandas>=2.0\n",
"scikit-learn>=1.3\n",
"matplotlib>=3.7\n",
"jupyter\n",
"```\n\n",
"To run: `jupyter notebook Crypto_Breach_ML_Pipeline.ipynb`\n",
]))

# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────────────────────────────────────

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = "Crypto_Breach_ML_Pipeline.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook written to: {out_path}")
