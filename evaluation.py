"""
evaluation.py
=============
All evaluation metrics, plots, and report generation.

Metrics (Section 5.3 of the proposal):
  - Brier Score    - calibration of probabilistic output
  - Log Loss       - penalises overconfident wrong predictions
  - PR-AUC         - best for rare / imbalanced events
  - ROC-AUC        - overall discrimination ability

Plots generated (saved to outputs/):
  01_dataset_overview.png          - breach counts by year and source
  02_sector_distribution.png       - sector pie / bar across all years
  03_method_distribution.png       - attack method breakdown
  04_crypto_vs_all.png             - crypto vs non-crypto breaches over time
  05_records_over_time.png         - records exposed per year (log scale)
  06_model_comparison.png          - mean metrics across models
  07_pr_curves.png                 - precision-recall curves per model
  08_probability_timeline.png      - predicted risk probability per year
  09_feature_importance.png        - top features (RF + GBM)
  10_calibration.png               - reliability diagram per model
  11_confusion_matrix_heatmap.png  - threshold-based confusion matrices
  12_brier_logloss_by_fold.png     - per-fold metric evolution
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

log = logging.getLogger(__name__)

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# -----------------------------------------------------------------------------
# 1-5: EDA plots (use the raw unified DataFrame)
# -----------------------------------------------------------------------------

def plot_eda(df: pd.DataFrame, yearly_df: pd.DataFrame, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    _plot_01_overview(df, out_dir)
    _plot_02_sector(df, out_dir)
    _plot_03_method(df, out_dir)
    _plot_04_crypto_vs_all(df, out_dir)
    _plot_05_records(yearly_df, out_dir)
    log.info("  EDA plots saved.")


def _plot_01_overview(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Overview: Breach Counts", fontsize=13, fontweight="bold")

    # Left: by year
    yr_cnt = df.groupby("year").size()
    axes[0].bar(yr_cnt.index, yr_cnt.values, color="#1f77b4", alpha=0.85)
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Number of Breaches")
    axes[0].set_title("Breaches per Year (All Sources)")
    axes[0].grid(axis="y", alpha=0.3)

    # Right: by source
    src_cnt = df["source_id"].value_counts()
    src_labels = {"IIB": "Information\nIs Beautiful", "HHS": "HHS / OCR\nHealthcare",
                  "DBN": "DataBreach\nN", "DBEN": "DataBreach\nEU",
                  "DF1": "Wikipedia\nExtended", "CSIS": "CSIS\nCyber Events",
                  "CE": "Cyber Events\nDatabase", "WA": "WA Breach\nNotifications",
                  "DHL": "DeFiHack\nLabs"}
    labels = [src_labels.get(s, s) for s in src_cnt.index]
    axes[1].barh(labels, src_cnt.values, color=PALETTE[:len(src_cnt)])
    axes[1].set_xlabel("Records Loaded")
    axes[1].set_title("Records per Source Dataset")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_dataset_overview.png"), dpi=150)
    plt.close()


def _plot_02_sector(df, out_dir):
    top10 = df["sector"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top10.index, top10.values, color=PALETTE * 2)
    ax.set_title("Breaches by Sector (Top 10)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count"); ax.set_xlabel("Sector")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, top10.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_sector_distribution.png"), dpi=150)
    plt.close()


def _plot_03_method(df, out_dir):
    top_m = df["method_category"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top_m.index[::-1], top_m.values[::-1],
                   color=PALETTE[:len(top_m)])
    ax.set_title("Attack Method Distribution (Top 10)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, top_m.values[::-1]):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "03_method_distribution.png"), dpi=150)
    plt.close()


def _plot_04_crypto_vs_all(df, out_dir):
    yr = df.groupby("year")
    total  = yr.size().rename("total")
    crypto = yr["is_crypto"].sum().rename("crypto")
    comb   = pd.concat([total, crypto], axis=1).fillna(0)
    comb["non_crypto"] = comb["total"] - comb["crypto"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Crypto vs All Breaches Over Time", fontsize=13, fontweight="bold")

    ax1.bar(comb.index, comb["non_crypto"], label="Non-crypto", color="#aec7e8")
    ax1.bar(comb.index, comb["crypto"], bottom=comb["non_crypto"],
            label="Crypto-related", color="#d62728")
    ax1.set_ylabel("Breach Count"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    ax2.plot(comb.index, (comb["crypto"] / comb["total"].replace(0, 1)) * 100,
             color="#d62728", linewidth=2, marker="o", markersize=5)
    ax2.set_ylabel("Crypto % of Total"); ax2.set_xlabel("Year")
    ax2.axhline(y=5, color="grey", linestyle="--", linewidth=0.8, label="5% line")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "04_crypto_vs_all.png"), dpi=150)
    plt.close()


def _plot_05_records(yr_df, out_dir):
    if "log_records_total" not in yr_df.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(yr_df.index, yr_df["log_records_total"], alpha=0.35, color="#2ca02c")
    ax.plot(yr_df.index, yr_df["log_records_total"], color="#2ca02c", linewidth=2)
    ax.set_title("log(Total Records Exposed) per Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("log(1 + Records)"); ax.set_xlabel("Year")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "05_records_over_time.png"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# 6-12: Model evaluation plots
# -----------------------------------------------------------------------------

def plot_model_results(wfv, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    summary  = wfv.summary_metrics()
    preds_df = wfv.consolidated_predictions()

    if summary.empty or preds_df.empty:
        log.warning("  No results to plot.")
        return

    _plot_06_model_comparison(summary, out_dir)
    _plot_07_pr_curves(wfv, preds_df, out_dir)
    _plot_08_probability_timeline(preds_df, out_dir)
    _plot_09_feature_importance(wfv, out_dir)
    _plot_10_calibration(preds_df, out_dir)
    _plot_11_confusion(preds_df, out_dir)
    _plot_12_fold_metrics(wfv, out_dir)
    log.info("  Model evaluation plots saved.")


def _plot_06_model_comparison(summary, out_dir):
    metrics = ["pr_auc", "roc_auc", "brier", "log_loss"]
    titles  = ["PR-AUC  (higher better)", "ROC-AUC  (higher better)", "Brier Score  (lower better)", "Log-Loss  (lower better)"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Model Comparison (Walk-Forward Mean)", fontsize=13, fontweight="bold")

    for ax, metric, title, colour in zip(axes, metrics, titles, PALETTE):
        vals = summary[metric].dropna()
        bars = ax.bar(vals.index, vals.values, color=colour, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_model_comparison.png"), dpi=150)
    plt.close()


def _plot_07_pr_curves(wfv, preds_df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Precision-Recall Curves (All Folds Combined)", fontsize=12, fontweight="bold")

    for i, (model_name, grp) in enumerate(preds_df.groupby("model")):
        y_true = grp["y_true"].values
        y_prob = grp["y_prob"].values
        if y_true.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auc = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, linewidth=2, color=PALETTE[i % len(PALETTE)],
                label=f"{model_name} (AUC={auc:.3f})")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "07_pr_curves.png"), dpi=150)
    plt.close()


def _plot_08_probability_timeline(preds_df, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Predicted Breach Probability Over Time", fontsize=13, fontweight="bold")

    for i, (model_name, grp) in enumerate(preds_df.groupby("model")):
        yr_prob = grp.groupby("year")["y_prob"].mean()
        axes[0].plot(yr_prob.index, yr_prob.values,
                     marker="o", linewidth=1.5, markersize=4,
                     color=PALETTE[i % len(PALETTE)], label=model_name, alpha=0.85)

    axes[0].axhline(0.5, color="crimson", linestyle="--", linewidth=1,
                    label="0.5 threshold")
    axes[0].set_ylabel("Mean Predicted Probability")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Actual breach indicator
    actual = preds_df.groupby("year")["y_true"].max()
    axes[1].vlines(actual[actual == 1].index, 0, 1,
                   color="darkred", linewidth=2, alpha=0.7, label="Breach year")
    axes[1].set_yticks([]); axes[1].set_ylabel("Actual\nBreaches", fontsize=9)
    axes[1].set_xlabel("Year"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "08_probability_timeline.png"), dpi=150)
    plt.close()


def _plot_09_feature_importance(wfv, out_dir):
    models_with_fi = ["RandomForest", "GradientBoosting"]
    fig, axes = plt.subplots(1, len(models_with_fi), figsize=(14, 7))
    if len(models_with_fi) == 1:
        axes = [axes]

    for ax, mname in zip(axes, models_with_fi):
        fi = wfv.mean_feature_importance(mname)
        if fi.empty:
            ax.set_visible(False)
            continue
        top = fi.head(20)
        colours = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(top)))
        ax.barh(top.index[::-1], top.values[::-1], color=colours[::-1])
        ax.set_title(f"{mname}: Top-20 Features", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Importance"); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "09_feature_importance.png"), dpi=150)
    plt.close()


def _plot_10_calibration(preds_df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for i, (mname, grp) in enumerate(preds_df.groupby("model")):
        y_true = grp["y_true"].values
        y_prob = grp["y_prob"].values
        if y_true.sum() < 2:
            continue
        n_bins = min(5, int(y_true.sum()))
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
            ax.plot(mean_pred, frac_pos, "s-",
                    color=PALETTE[i % len(PALETTE)], linewidth=2,
                    label=mname)
        except Exception:
            pass

    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration / Reliability Diagram", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "10_calibration.png"), dpi=150)
    plt.close()


def _plot_11_confusion(preds_df, out_dir):
    models = preds_df["model"].unique()
    ncols  = min(3, len(models))
    nrows  = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("Confusion Matrices (threshold=0.5)", fontsize=12, fontweight="bold")

    for ax, mname in zip(axes, models):
        grp    = preds_df[preds_df["model"] == mname]
        y_true = grp["y_true"].values
        y_pred = (grp["y_prob"].values >= 0.5).astype(int)
        cm_vals = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm_vals, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["Actual 0", "Actual 1"])
        ax.set_title(mname, fontsize=9)
        for (r, c), val in np.ndenumerate(cm_vals):
            ax.text(c, r, str(val), ha="center", va="center",
                    color="white" if val > cm_vals.max() / 2 else "black",
                    fontsize=12)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "11_confusion_matrix_heatmap.png"), dpi=150)
    plt.close()


def _plot_12_fold_metrics(wfv, out_dir):
    """
    Plot 12 fix: PR-AUC is NaN for single-sample folds (can't compute with 1 point).
    Solution: plot cumulative rolling PR-AUC (computed over expanding window of predictions)
    and per-fold Brier Score (which works even for single samples).
    """
    rows = []
    for r in wfv.all_results:
        m = r.metrics
        rows.append({"model": r.model_name, "fold": r.fold_id,
                     "test_year": r.test_years[0] if r.test_years else None,
                     **m})
    df = pd.DataFrame(rows).dropna(subset=["brier"])
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-Fold Metric Evolution", fontsize=12, fontweight="bold")

    # Left: Brier score per fold (works for single samples)
    for i, (mname, grp) in enumerate(df.groupby("model")):
        grp_s = grp.sort_values("test_year")
        axes[0].plot(grp_s["test_year"], grp_s["brier"],
                     marker="o", linewidth=1.5, markersize=4,
                     color=PALETTE[i % len(PALETTE)], label=mname, alpha=0.85)
    axes[0].set_title("Brier Score per Fold (lower = better)", fontweight="bold")
    axes[0].set_ylabel("Brier Score"); axes[0].set_xlabel("Test Year")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)

    # Right: Cumulative (rolling) PR-AUC -- pools all predictions up to each year
    # This fixes the NaN problem: single-sample folds contribute to the running pool
    from sklearn.metrics import average_precision_score
    preds_df = wfv.consolidated_predictions()
    if not preds_df.empty:
        for i, (mname, grp) in enumerate(preds_df.groupby("model")):
            grp_s = grp.sort_values("year")
            cumulative_pr = []
            years_plot    = []
            for yr in grp_s["year"].unique():
                pool = grp_s[grp_s["year"] <= yr]
                if pool["y_true"].sum() > 0 and (1 - pool["y_true"]).sum() > 0:
                    try:
                        pr = average_precision_score(pool["y_true"], pool["y_prob"])
                        cumulative_pr.append(pr)
                        years_plot.append(yr)
                    except Exception:
                        pass
            if years_plot:
                axes[1].plot(years_plot, cumulative_pr, marker="s", linewidth=1.5,
                             markersize=4, color=PALETTE[i % len(PALETTE)],
                             label=mname, alpha=0.85)

    axes[1].set_title("Cumulative PR-AUC Over Time (higher = better)", fontweight="bold")
    axes[1].set_ylabel("Cumulative PR-AUC"); axes[1].set_xlabel("Test Year")
    axes[1].set_ylim(0, 1.05)
    # Explicitly constrain x-axis to actual test year range to prevent
    # matplotlib auto-scaling to a 200-year range when few points exist
    all_test_years = sorted(preds_df["year"].unique()) if not preds_df.empty else []
    if len(all_test_years) >= 2:
        axes[1].set_xlim(all_test_years[0] - 1, all_test_years[-1] + 1)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "12_brier_logloss_by_fold.png"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Metric helpers (standalone)
# -----------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """BS = mean((F_t - O_t)^2)"""
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_manual(y_true: np.ndarray, y_prob: np.ndarray, eps=1e-15) -> float:
    """LogLoss = -mean(O*log(F) + (1-O)*log(1-F))"""
    p = np.clip(y_prob, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))