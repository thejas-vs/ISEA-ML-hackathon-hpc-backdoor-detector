"""
EDA Module for HPC Backdoor Detector.
Loads data/trace.csv and generates comprehensive summary statistics and diagnostic figures.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure matplotlib uses a writable temp cache directory and non-interactive backend
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_hpc_eda")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def get_project_paths():
    """Return dictionary of essential project paths."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "trace.csv"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base_dir,
        "data": data_path,
        "reports": reports_dir,
        "figures": figures_dir,
    }


def load_data(data_path: Path) -> pd.DataFrame:
    """Load trace.csv and perform initial validation."""
    if not data_path.exists():
        raise FileNotFoundError(f"Trace file not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from: {data_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def compute_summary_statistics(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Compute standard and distribution statistics (mean, std, skew, kurtosis, IQR)."""
    desc = df.describe().T
    desc["skewness"] = df.skew()
    desc["kurtosis"] = df.kurtosis()  # Excess kurtosis (Fisher: normal = 0)
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc["median"] = df.median()
    
    # Reorganize column order for readability
    ordered_cols = [
        "count", "mean", "std", "min", "25%", "median", "75%", "max",
        "IQR", "skewness", "kurtosis"
    ]
    summary_df = desc[ordered_cols]
    
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(summary_df.to_string())
    
    summary_df.to_csv(out_path)
    print(f"\nSaved summary statistics to: {out_path}")
    return summary_df


def compute_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute count and percentage of outliers per feature using IQR and Z-score rules."""
    outlier_records = []
    
    for col in df.columns:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        
        iqr_outliers = series[(series < iqr_lower) | (series > iqr_upper)]
        iqr_count = len(iqr_outliers)
        iqr_pct = (iqr_count / len(series)) * 100
        
        # Z-score rule: |z| > 3
        mean = series.mean()
        std = series.std(ddof=0)
        if std > 0:
            z_scores = np.abs((series - mean) / std)
            z_outliers = series[z_scores > 3.0]
            z_count = len(z_outliers)
        else:
            z_count = 0
        z_pct = (z_count / len(series)) * 100
        
        outlier_records.append({
            "Feature": col,
            "IQR_Lower_Bound": iqr_lower,
            "IQR_Upper_Bound": iqr_upper,
            "IQR_Outlier_Count": iqr_count,
            "IQR_Outlier_Pct (%)": round(iqr_pct, 2),
            "ZScore_Outlier_Count (|z|>3)": z_count,
            "ZScore_Outlier_Pct (%)": round(z_pct, 2),
        })
        
    outlier_df = pd.DataFrame(outlier_records)
    print("\n" + "=" * 60)
    print("OUTLIER COUNTS (IQR & Z-SCORE > 3)")
    print("=" * 60)
    print(outlier_df.to_string(index=False))
    return outlier_df


def plot_histograms_kde(df: pd.DataFrame, out_path: Path):
    """Plot Histogram + KDE for each feature."""
    cols = list(df.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.5))
    
    for ax, col in zip(axes, cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4", bins=30, stat="density", alpha=0.6)
        ax.set_title(f"Distribution: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Histograms+KDE to: {out_path}")


def plot_boxplots(df: pd.DataFrame, out_path: Path):
    """Plot Boxplots for each feature."""
    cols = list(df.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(15, 5))
    
    for ax, col in zip(axes, cols):
        sns.boxplot(y=df[col], ax=ax, color="#72b7b2", width=0.4, fliersize=4)
        ax.set_title(f"Boxplot: {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel(col, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Boxplots to: {out_path}")


def plot_pairwise_scatter(df: pd.DataFrame, out_path: Path):
    """Plot Pairwise scatter plot using seaborn pairplot."""
    sns.set_theme(style="ticks")
    g = sns.pairplot(df, corner=False, diag_kind="kde",
                     plot_kws={"alpha": 0.6, "color": "#2b5c8f", "s": 25},
                     diag_kws={"fill": True, "color": "#2b5c8f"})
    g.fig.subplots_adjust(top=0.93)
    g.fig.suptitle("Pairwise Feature Scatter & Density (Pairplot)", fontsize=14, fontweight="bold")
    g.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Pairplot to: {out_path}")


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path):
    """Plot Pearson and Spearman correlation heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Pearson
    pearson_corr = df.corr(method="pearson")
    sns.heatmap(pearson_corr, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1,
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1)
    ax1.set_title("Pearson Linear Correlation", fontsize=12, fontweight="bold")
    
    # Spearman (rank correlation, robust to monotonic non-linear relations & outliers)
    spearman_corr = df.corr(method="spearman")
    sns.heatmap(spearman_corr, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1,
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax2)
    ax2.set_title("Spearman Rank Correlation", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Correlation Heatmap to: {out_path}")


def plot_row_index_drift(df: pd.DataFrame, out_path: Path):
    """Plot feature values across row indices to evaluate time-ordering / drift."""
    cols = list(df.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 8), sharex=True)
    
    for ax, col in zip(axes, cols):
        ax.plot(df.index, df[col], color="#e15759", linewidth=1.0, alpha=0.8, label="Observation")
        # Moving average trendline (window=25)
        rolling_mean = df[col].rolling(window=25, min_periods=5).mean()
        ax.plot(df.index, rolling_mean, color="#4e79a7", linewidth=2.0, label="Rolling Mean (window=25)")
        ax.set_ylabel(col, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
        
    axes[-1].set_xlabel("Row Index (Time-order / Sample Sequence)", fontsize=11, fontweight="bold")
    fig.suptitle("Sequential Trace Values vs. Row Index (Drift & Stability Check)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Row-Index Line Plots to: {out_path}")


def plot_qq_plots(df: pd.DataFrame, out_path: Path):
    """Plot Q-Q plots against standard normal distribution for normality check."""
    cols = list(df.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.8))
    
    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        stats.probplot(data, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor("#4e79a7")
        ax.get_lines()[0].set_markeredgecolor("#4e79a7")
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color("#d62728")
        ax.get_lines()[1].set_linewidth(1.8)
        
        # Calculate Shapiro-Wilk or D'Agostino normality test p-value
        stat_val, p_val = stats.shapiro(data)
        ax.set_title(f"Q-Q: {col}\n(Shapiro-Wilk p={p_val:.2e})", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Q-Q plots to: {out_path}")


def main():
    paths = get_project_paths()
    print(f"Project root: {paths['base']}")
    
    df = load_data(paths["data"])
    
    # 1. Summary statistics
    summary_csv = paths["reports"] / "eda_summary.csv"
    compute_summary_statistics(df, summary_csv)
    
    # 2. Outliers
    compute_outliers(df)
    
    # 3. Visualizations
    plot_histograms_kde(df, paths["figures"] / "hist_kde_features.png")
    plot_boxplots(df, paths["figures"] / "boxplots_features.png")
    plot_pairwise_scatter(df, paths["figures"] / "pairwise_pairplot.png")
    plot_correlation_heatmap(df, paths["figures"] / "correlation_heatmap.png")
    plot_row_index_drift(df, paths["figures"] / "row_index_drift.png")
    plot_qq_plots(df, paths["figures"] / "qq_plots_normality.png")
    
    print("\nEDA processing complete! All figures and summary statistics successfully saved.")


if __name__ == "__main__":
    main()
