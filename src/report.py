"""
report.py — Generate a comprehensive PDF summary of the HPC Backdoor Detector pipeline.

Sections:
  1. Title Page
  2. Project Overview
  3. EDA Summary (table + figures)
  4. Preprocessing & Feature Engineering
  5. Benchmark Results (table + heatmap)
  6. Threshold Calibration (trade-off curve)
  7. Final Model Summary & CLI Usage
  8. Limitations & Future Work

Usage:
    python src/report.py [--output reports/hpc_backdoor_report.pdf]
"""

import sys
import argparse
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from src.config import (
    RANDOM_SEED, PROJECT_ROOT,
    DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR,
)

# Unified path dict used throughout this module
PATHS = {
    "data":    DATA_DIR,
    "reports": REPORTS_DIR,
    "figures": FIGURES_DIR,
    "models":  MODELS_DIR,
}

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page_setup(fig, title: str = ""):
    """Add a consistent header + footer to every page."""
    fig.text(0.5, 0.97, "HPC Backdoor Detector — Pipeline Report",
             ha="center", va="top", fontsize=9, color="#555555",
             style="italic")
    if title:
        fig.text(0.5, 0.94, title, ha="center", va="top",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01, "CONFIDENTIAL — for evaluation purposes only",
             ha="center", va="bottom", fontsize=7, color="#888888")


def _df_to_table(ax, df: pd.DataFrame, col_widths=None, fontsize=8):
    """Render a pandas DataFrame as a matplotlib table inside `ax`."""
    ax.axis("off")
    df = df.copy().reset_index(drop=True)
    # Truncate long values
    df = df.apply(lambda col: col.map(lambda v: str(v)[:40] if isinstance(v, str) else v))
    col_labels = list(df.columns)
    cell_text = df.values.tolist()
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    if col_widths:
        for (row, col), cell in tbl.get_celld().items():
            if col < len(col_widths):
                cell.set_width(col_widths[col])
    # Style header row
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#3c78d8")
        tbl[0, col].set_text_props(color="white", fontweight="bold")
    # Alternating row colours
    for row in range(1, len(cell_text) + 1):
        fc = "#f3f3f3" if row % 2 == 0 else "white"
        for col in range(len(col_labels)):
            tbl[row, col].set_facecolor(fc)
    tbl.scale(1, 1.3)
    return tbl


def _embed_image(ax, path: Path, caption: str = ""):
    """Display an image file inside a matplotlib axes."""
    if not path.exists():
        ax.text(0.5, 0.5, f"[Figure not found: {path.name}]",
                ha="center", va="center", fontsize=9, color="red",
                transform=ax.transAxes)
        ax.axis("off")
        return
    img = plt.imread(str(path))
    ax.imshow(img, aspect="auto")
    ax.axis("off")
    if caption:
        ax.set_title(caption, fontsize=9, pad=4)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _page_title(pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#1a1a2e")

    fig.text(0.5, 0.65, "HPC Backdoor Detector",
             ha="center", va="center", fontsize=36, fontweight="bold",
             color="white")
    fig.text(0.5, 0.56,
             "Unsupervised Anomaly Detection on Hardware Performance Counter Traces",
             ha="center", va="center", fontsize=16, color="#aaaacc")
    fig.text(0.5, 0.46,
             "Pipeline Report  —  ISEA ML Challenge",
             ha="center", va="center", fontsize=13, color="#888888",
             style="italic")

    # Decorative rule
    ax = fig.add_axes([0.1, 0.42, 0.8, 0.002])
    ax.axhline(0, color="#3c78d8", linewidth=2)
    ax.axis("off")

    specs = [
        ("Training data", "800 clean HPC traces (trace.csv)"),
        ("Features", "cache-references, cycles, LLC-loads"),
        ("Models trained", "7 (IF, OCSVM, LOF, GMM, Mahalanobis, AE, Ensemble)"),
        ("Final model", "LOF (k=20, RobustScaler)"),
        ("Threshold (P95)", "−0.2009   →   FPR ≈ 5 %"),
        ("Random seed", "42"),
    ]
    for i, (k, v) in enumerate(specs):
        y = 0.36 - i * 0.05
        fig.text(0.28, y, f"• {k}:", ha="left", fontsize=11,
                 color="#aaaacc", fontweight="bold")
        fig.text(0.52, y, v, ha="left", fontsize=11, color="white")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_overview(pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    _page_setup(fig, "Project Overview")
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.80])
    ax.axis("off")

    text = textwrap.dedent("""\
    TASK
    ────
    Detect hardware backdoors injected into neural-network inference workloads using only
    Hardware Performance Counter (HPC) telemetry.  Training data is exclusively CLEAN traces
    (one-class / unsupervised learning).  The hidden test set contains clean + backdoor rows
    and is scored on Accuracy, TPR, FPR, F1, and AUROC.

    PIPELINE
    ────────
      1. EDA            (src/eda.py)           — statistics, visualisations, notes
      2. Preprocessing  (src/preprocess.py)    — feature engineering, scaling
      3. Validation     (src/validation.py)    — 5-fold CV, synthetic anomalies, evaluate()
      4. Models         (src/models.py)        — 7 anomaly detectors, common interface
      5. Benchmark      (src/benchmark.py)     — grid search, metric table, heatmap
      6. Threshold      (src/threshold.py)     — OOF calibration, trade-off plot, config.json
      7. Train Final    (src/train_final.py)   — fit on all 800 rows → models/detector.joblib
      8. Predict        (src/predict.py)       — CLI + Python API for the hidden test set

    DESIGN CHOICES
    ──────────────
    • One-class training  — anomaly scores are derived purely from the clean-data manifold.
    • OOF threshold calibration — threshold set at the P95 of out-of-fold clean scores to
      avoid optimism; empirical FPR on training data = 5 %.
    • Synthetic anomaly evaluation — five proxy anomaly types (mean-shift, variance
      inflation, correlation-breaking, ratio perturbation, uniform bounding-box) provide a
      rough upper-bound on detection capability; real backdoor shifts may be subtler.
    • Robust scaling (RobustScaler) chosen after EDA showed heavy tails and bimodality.
    • LOF (k=20) selected as the best individual detector (AUROC 0.9241 ± 0.0031,
      worst-case 0.7577); Ensemble is recommended for robustness in production.
    """)

    ax.text(0.0, 1.0, text, ha="left", va="top", fontsize=9,
            fontfamily="monospace", transform=ax.transAxes,
            wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_eda_summary(pdf: PdfPages):
    summary_path = PATHS["reports"] / "eda_summary.csv"
    fig = plt.figure(figsize=(11, 8.5))
    _page_setup(fig, "EDA — Summary Statistics")

    if summary_path.exists():
        df = pd.read_csv(summary_path, index_col=0).T.reset_index()
        df.rename(columns={"index": "Statistic"}, inplace=True)
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce").apply(
                lambda v: f"{v:.4g}" if isinstance(v, float) and not pd.isna(v) else (str(v) if not pd.isna(v) else "")
            )
        ax = fig.add_axes([0.05, 0.25, 0.90, 0.60])
        _df_to_table(ax, df, fontsize=9)
    else:
        ax = fig.add_axes([0.05, 0.1, 0.90, 0.80])
        ax.text(0.5, 0.5, "eda_summary.csv not found", ha="center",
                va="center", fontsize=12, color="red", transform=ax.transAxes)
        ax.axis("off")

    notes_path = PATHS["reports"] / "eda_notes.md"
    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8")[:900].replace("#", "")
        fig.text(0.07, 0.22, notes, va="top", fontsize=7.5,
                 fontfamily="monospace", wrap=True)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_eda_figures(pdf: PdfPages):
    figures_dir = PATHS["figures"]
    targets = [
        ("hist_kde.png",           "Histograms + KDE"),
        ("boxplots.png",           "Boxplots"),
        ("pairplot.png",           "Pairwise Scatter Plot"),
        ("correlation_heatmap.png","Correlation Heatmap"),
        ("row_index_drift.png",    "Row-index Drift Check"),
        ("qq_plots.png",           "Q-Q Plots"),
    ]

    # 2 figures per page
    for i in range(0, len(targets), 2):
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        _page_setup(fig, "EDA — Visualisations")
        for j, ax in enumerate(axes):
            if i + j < len(targets):
                fname, caption = targets[i + j]
                _embed_image(ax, figures_dir / fname, caption)
            else:
                ax.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _page_preprocessing(pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    _page_setup(fig, "Preprocessing & Feature Engineering")
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.80])
    ax.axis("off")

    text = textwrap.dedent("""\
    SOURCE:  src/preprocess.py  |  src/config.py

    Raw features (3):
      • cache-references   — number of cache reference events
      • cycles             — total CPU cycles
      • LLC-loads          — Last-Level Cache load events

    Engineered ratio features (3):
      • cache_ref_per_cycle  = cache-references / cycles
      • llc_load_per_cycle   = LLC-loads / cycles
      • llc_per_cache_ref    = LLC-loads / cache-references
      Zero-division is guarded with eps = 1e-12.

    Log-ratio features (3):
      • log_cache_ref_per_cycle  = log(cache_ref_per_cycle + eps)
      • log_llc_load_per_cycle   = log(llc_load_per_cycle + eps)
      • log_llc_per_cache_ref    = log(llc_per_cache_ref + eps)

    Final best config (from benchmark):
      • use_ratios = False  (raw features only — LOF performed best without ratios)
      • scaler     = RobustScaler (robust to heavy tails and bimodal clusters)
      • log1p      = False
      • seed       = 42

    API:
      build_features(df, use_ratios=True) -> DataFrame (deterministic, row or batch)
      get_preprocessing_pipeline(config, scaler_type, use_ratios, use_log1p) -> Pipeline
    """)

    ax.text(0.0, 1.0, text, ha="left", va="top", fontsize=10,
            fontfamily="monospace", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_benchmark(pdf: PdfPages):
    results_path = PATHS["reports"] / "benchmark_results.csv"
    fig = plt.figure(figsize=(11, 8.5))
    _page_setup(fig, "Benchmark — 5-Fold CV Results (mean AUROC ± std)")

    if results_path.exists():
        df = pd.read_csv(results_path)
        # Keep only key columns for readability
        keep = ["Model", "Overall_AUROC", "Clean_FPR", "Worst_Case_AUROC (Robustness)",
                "mean_shift_1std", "var_inflation_1.5x", "correlation_breaking",
                "ratio_perturb_10pct", "uniform_box_20pct"]
        keep = [c for c in keep if c in df.columns]
        df_disp = df[keep].copy()
        ax = fig.add_axes([0.03, 0.15, 0.94, 0.72])
        _df_to_table(ax, df_disp, fontsize=7.5)
        fig.text(0.07, 0.12,
                 "★ LOF achieves the highest worst-case AUROC (most robust); Ensemble recommended for production.",
                 fontsize=9, color="#1a56db")
    else:
        ax = fig.add_axes([0.05, 0.1, 0.90, 0.80])
        ax.text(0.5, 0.5, "benchmark_results.csv not found", ha="center",
                va="center", fontsize=12, color="red", transform=ax.transAxes)
        ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_benchmark_figures(pdf: PdfPages):
    figures_dir = PATHS["figures"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    _page_setup(fig, "Benchmark — Visual Summary")
    _embed_image(axes[0], figures_dir / "benchmark_heatmap.png", "AUROC Heatmap by Model × Anomaly Type")
    _embed_image(axes[1], figures_dir / "top3_roc_curves.png",   "ROC Curves — Top 3 Models")
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_threshold(pdf: PdfPages):
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    _page_setup(fig, "Threshold Calibration")

    _embed_image(axes[0], PATHS["figures"] / "threshold_tradeoff.png",
                 "FPR vs TPR at P90/P95/P97.5/P99 (OOF scores)")

    # Right: config summary
    axes[1].axis("off")
    cfg_path = PATHS["models"] / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cal = cfg.get("calibration", {})
        lines = [
            "THRESHOLD CONFIG (models/config.json)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Model          : {cfg.get('model_name', 'LOF')}",
            f"Percentile     : P{cal.get('percentile', 95):.0f}",
            f"Target FPR     : {cal.get('target_fpr', 0.05)*100:.0f} %",
            f"Threshold      : {cal.get('calibrated_threshold', 'N/A'):.6f}",
            f"Training FPR   : {cal.get('empirical_training_fpr_percent', 5):.1f} %",
            f"Training N     : {cal.get('training_samples', 800)}",
            "",
            "RECOMMENDATION",
            "━━━━━━━━━━━━━━",
            "P95 chosen: balances low FPR (5%) with",
            "adequate sensitivity to synthetic anomalies.",
            "Adjust to P90 if the deployment environment",
            "can tolerate 10% clean-data false alarms.",
        ]
        axes[1].text(0.05, 0.92, "\n".join(lines), va="top", ha="left",
                     fontsize=9, fontfamily="monospace",
                     transform=axes[1].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_final_model(pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    _page_setup(fig, "Final Model & CLI Usage")
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.80])
    ax.axis("off")

    cfg_path = PATHS["models"] / "config.json"
    dist = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        dist = cfg.get("training_score_distribution", {})

    dist_lines = "\n".join(
        f"      {k:<24}: {v:.4f}" if isinstance(v, float) else f"      {k:<24}: {v}"
        for k, v in dist.items()
    )

    text = textwrap.dedent(f"""\
    ARTIFACT:  models/detector.joblib
    ───────────────────────────────────
    Format : joblib dict bundle  {{ pipeline, model, threshold, feature_names, version, metadata }}
    Model  : LocalOutlierFactor (n_neighbors=20, metric='minkowski', novelty=True)
    Scaler : RobustScaler (centering=True, scaling=True)
    Features (raw only, 3 columns):
      cache-references, cycles, LLC-loads
    Threshold  : −0.200866  (P95 of training scores, FPR=5 %)

    TRAINING-SET ANOMALY-SCORE DISTRIBUTION
    ────────────────────────────────────────
{dist_lines}

    CLI USAGE
    ─────────
    # Predict on a test CSV (must have cache-references, cycles, LLC-loads columns)
    python src/predict.py --input path/to/test.csv --output predictions.csv

    # Python API
    from src.predict import predict
    scores, labels = predict(df)   # df: pandas DataFrame with the 3 HPC columns

    OUTPUT FORMAT (predictions.csv)
    ────────────────────────────────
    anomaly_score   : continuous score (higher = more anomalous)
    prediction      : 0 = Clean Neural Network Inference
                      1 = Backdoor Anomaly Detected

    INPUT ROBUSTNESS
    ────────────────
    • Extra columns (e.g. label) are silently ignored.
    • Column order does not matter — reordered automatically.
    • NaN / Inf values are imputed with clean-baseline medians.
    """)

    ax.text(0.0, 1.0, text, ha="left", va="top", fontsize=9,
            fontfamily="monospace", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_limitations(pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    _page_setup(fig, "Limitations & Future Work")
    ax = fig.add_axes([0.07, 0.08, 0.86, 0.80])
    ax.axis("off")

    text = textwrap.dedent("""\
    LIMITATIONS
    ───────────
    1. Synthetic anomaly proxy
       The evaluation suite (5 anomaly types × 11 scenarios) measures detection of
       stereotyped perturbations.  Real hardware backdoors may cause subtler, correlated
       shifts that look more like in-distribution data.  AUROC figures are therefore
       optimistic upper bounds.

    2. Small training set (N = 800)
       LOF and GMM are sensitive to the local neighbourhood structure.  With more data,
       re-tune n_neighbors and n_components.

    3. No temporal labels
       The row-index drift check showed no detectable ordering effect in the training set.
       If the test set captures multiple time windows, consider a sliding-window approach.

    4. Single inference context
       All 800 clean rows come from one model / one hardware context.  Generalisation to
       different NN architectures or CPUs is unknown without additional clean traces.

    5. Autoencoder underperforms (AUROC 0.877)
       With only 800 training samples the 2-D bottleneck may underfit.  Consider VAE or
       larger hidden dims with regularisation once more data is available.

    FUTURE WORK
    ───────────
    • Collect clean traces across multiple NN models and hardware configurations.
    • Replace synthetic anomalies with real backdoor traces (even a few) to calibrate
      the proxy quality.
    • Explore COPOD, ECOD, or DeepSVDD for further AUROC gains.
    • Use Conformal Anomaly Detection to provide statistically valid p-values per prediction.
    • Retrain periodically as the HW/SW baseline evolves (concept drift).
    • Ensemble LOF + GMM + OCSVM with Bayesian model averaging for production robustness.
    """)

    ax.text(0.0, 1.0, text, ha="left", va="top", fontsize=10,
            fontfamily="monospace", transform=ax.transAxes)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_report(output_path: Path | None = None) -> Path:
    """Generate the full pipeline PDF report and return its path."""
    if output_path is None:
        output_path = PATHS["reports"] / "hpc_backdoor_report.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating report -> {output_path}")

    with PdfPages(str(output_path)) as pdf:
        _page_title(pdf)
        _page_overview(pdf)
        _page_eda_summary(pdf)
        _page_eda_figures(pdf)
        _page_preprocessing(pdf)
        _page_benchmark(pdf)
        _page_benchmark_figures(pdf)
        _page_threshold(pdf)
        _page_final_model(pdf)
        _page_limitations(pdf)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = "HPC Backdoor Detector — Pipeline Report"
        d["Author"] = "hpc-backdoor-detector automated pipeline"
        d["Subject"] = "Unsupervised Anomaly Detection on Hardware Performance Counters"
        d["Keywords"] = "HPC, backdoor, anomaly detection, LOF, one-class learning"

    print(f"[OK] Report saved: {output_path}  ({output_path.stat().st_size // 1024} KB)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF pipeline report.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PDF path (default: reports/hpc_backdoor_report.pdf)")
    args = parser.parse_args()
    generate_report(args.output)
