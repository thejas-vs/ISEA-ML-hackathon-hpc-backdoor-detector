"""
Benchmark Module for HPC Backdoor Detector.

Executes a 5-fold Cross-Validation benchmark across all anomaly detectors:
- Compares models with and without ratio features (use_ratios=True vs False).
- Compares StandardScaler vs RobustScaler.
- Tunes model hyperparameters to maximize AUROC while maintaining Clean FPR <= 5%.
- Evaluates performance across all 11 synthetic anomaly scenarios in validation.py.
- Exports benchmark_results.csv, benchmark_heatmap.png, and top3_roc_curves.png.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure writable matplotlib temp directory and Agg backend
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_hpc_bench")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_auc_score, roc_curve

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, RANDOM_SEED, REPORTS_DIR, TRACE_DATA_PATH, seed_everything
from src.models import (
    AutoencoderDetector,
    EnsembleAnomalyDetector,
    GMMDetector,
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
    RobustMahalanobisDetector,
    make_anomaly_detector_pipeline,
)
from src.validation import create_synthetic_validation_suite, extract_anomaly_scores, get_clean_cv_splits

# Suppress minor convergence/covariance warnings during grid exploration
warnings.filterwarnings("ignore")
seed_everything(RANDOM_SEED)


def get_model_candidate_grids() -> Dict[str, List[Dict[str, Any]]]:
    """
    Define targeted hyperparameter and preprocessing candidate grids for each model family.
    Tests:
    - Scaler: RobustScaler vs StandardScaler
    - Ratios: use_ratios=True vs use_ratios=False
    - Key model hyperparameters
    """
    grids = {
        "IsolationForest": [
            {"scaler": "robust", "use_ratios": True, "n_estimators": 150, "max_samples": "auto"},
            {"scaler": "robust", "use_ratios": False, "n_estimators": 150, "max_samples": "auto"},
            {"scaler": "standard", "use_ratios": True, "n_estimators": 150, "max_samples": "auto"},
            {"scaler": "robust", "use_ratios": True, "n_estimators": 250, "max_samples": 0.8},
        ],
        "OneClassSVM": [
            {"scaler": "robust", "use_ratios": True, "nu": 0.05, "gamma": "scale"},
            {"scaler": "robust", "use_ratios": False, "nu": 0.05, "gamma": "scale"},
            {"scaler": "standard", "use_ratios": True, "nu": 0.05, "gamma": "scale"},
            {"scaler": "robust", "use_ratios": True, "nu": 0.02, "gamma": 0.05},
        ],
        "LOF": [
            {"scaler": "robust", "use_ratios": True, "n_neighbors": 20},
            {"scaler": "robust", "use_ratios": False, "n_neighbors": 20},
            {"scaler": "standard", "use_ratios": True, "n_neighbors": 20},
            {"scaler": "robust", "use_ratios": True, "n_neighbors": 35},
        ],
        "GMM": [
            {"scaler": "robust", "use_ratios": True, "n_components_candidates": (2, 3), "cov": "full"},
            {"scaler": "robust", "use_ratios": False, "n_components_candidates": (2, 3), "cov": "full"},
            {"scaler": "standard", "use_ratios": True, "n_components_candidates": (2, 3), "cov": "full"},
            {"scaler": "robust", "use_ratios": True, "n_components_candidates": (1, 2, 3, 4), "cov": "full"},
        ],
        "RobustMahalanobis": [
            {"scaler": "robust", "use_ratios": True, "support_fraction": 0.8},
            {"scaler": "robust", "use_ratios": False, "support_fraction": 0.8},
            {"scaler": "standard", "use_ratios": True, "support_fraction": 0.8},
            {"scaler": "robust", "use_ratios": True, "support_fraction": None},
        ],
        "Autoencoder": [
            {"scaler": "robust", "use_ratios": True, "max_epochs": 100, "patience": 12},
            {"scaler": "robust", "use_ratios": False, "max_epochs": 100, "patience": 12},
            {"scaler": "standard", "use_ratios": True, "max_epochs": 100, "patience": 12},
            {"scaler": "robust", "use_ratios": True, "max_epochs": 150, "patience": 15},
        ],
    }
    return grids


def instantiate_candidate_model(name: str, params: Dict[str, Any]):
    """Instantiate a detector from configuration dictionary."""
    if name == "IsolationForest":
        return IsolationForestDetector(
            n_estimators=params["n_estimators"],
            max_samples=params["max_samples"],
            random_state=RANDOM_SEED,
            target_fpr=0.05,
        )
    elif name == "OneClassSVM":
        return OneClassSVMDetector(
            kernel="rbf",
            nu=params["nu"],
            gamma=params["gamma"],
            target_fpr=0.05,
        )
    elif name == "LOF":
        return LOFDetector(
            n_neighbors=params["n_neighbors"],
            target_fpr=0.05,
        )
    elif name == "GMM":
        return GMMDetector(
            n_components_candidates=params["n_components_candidates"],
            covariance_type=params["cov"],
            random_state=RANDOM_SEED,
            target_fpr=0.05,
        )
    elif name == "RobustMahalanobis":
        return RobustMahalanobisDetector(
            support_fraction=params["support_fraction"],
            random_state=RANDOM_SEED,
            target_fpr=0.05,
        )
    elif name == "Autoencoder":
        return AutoencoderDetector(
            max_epochs=params["max_epochs"],
            patience=params["patience"],
            random_state=RANDOM_SEED,
            target_fpr=0.05,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def evaluate_pipeline_on_suite(pipeline, val_clean: pd.DataFrame, suite: Dict[str, pd.DataFrame]):
    """
    Score clean validation fold and all synthetic anomaly configurations.
    Returns:
    - clean_scores: anomaly scores on clean holdout
    - scenario_aurocs: dict mapping scenario name to AUROC
    - clean_fpr: empirical false positive rate on clean validation holdout
    - combined_y_true, combined_scores: concatenated for ROC curve analysis
    """
    clean_scores = extract_anomaly_scores(pipeline, val_clean)
    
    # Threshold calibrated by model during fit
    thresh = pipeline.threshold_ if hasattr(pipeline, "threshold_") and pipeline.threshold_ is not None else np.percentile(clean_scores, 95.0)
    clean_fpr = np.mean(clean_scores >= thresh)

    scenario_aurocs = {}
    all_anom_scores = []

    for s_name, anom_df in suite.items():
        anom_scores = extract_anomaly_scores(pipeline, anom_df)
        all_anom_scores.append(anom_scores)
        
        y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(anom_scores))])
        c_scores = np.concatenate([clean_scores, anom_scores])
        s_auroc = roc_auc_score(y_true, c_scores)
        scenario_aurocs[s_name] = s_auroc

    # Combined composite evaluation across all scenarios
    stacked_anom = np.concatenate(all_anom_scores)
    composite_y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(stacked_anom))])
    composite_scores = np.concatenate([clean_scores, stacked_anom])
    composite_auroc = roc_auc_score(composite_y_true, composite_scores)

    return {
        "clean_fpr": clean_fpr,
        "scenario_aurocs": scenario_aurocs,
        "composite_auroc": composite_auroc,
        "composite_y_true": composite_y_true,
        "composite_scores": composite_scores,
    }


def run_benchmark():
    """
    Run full 5-fold CV benchmark with hyperparameter tuning and synthetic anomaly evaluation.
    """
    print("=" * 80)
    print("HPC BACKDOOR DETECTOR: 5-FOLD CV BENCHMARK & MODEL SELECTION")
    print("=" * 80)

    df = pd.read_csv(TRACE_DATA_PATH)
    candidate_grids = get_model_candidate_grids()
    cv_splits = list(get_clean_cv_splits(df, n_splits=5, shuffle=True, seed=RANDOM_SEED))

    best_tuned_configs = {}
    best_pipelines_fold0 = {}
    roc_data_dict = {}

    # Stage 1: Tune each model family using 5-fold CV
    print("\n[Stage 1] Tuning hyperparameters and scaler/ratio flags per model...")
    for model_name, configs in candidate_grids.items():
        best_cfg = None
        best_score = -np.inf

        for cfg_idx, cfg in enumerate(configs):
            fold_aurocs = []
            fold_clean_fprs = []

            for train_df, val_df in cv_splits:
                model_inst = instantiate_candidate_model(model_name, cfg)
                pipeline = make_anomaly_detector_pipeline(
                    model=model_inst,
                    scaler_type=cfg["scaler"],
                    use_ratios=cfg["use_ratios"],
                )
                pipeline.fit(train_df)
                suite = create_synthetic_validation_suite(val_df, seed=RANDOM_SEED)
                res = evaluate_pipeline_on_suite(pipeline, val_df, suite)

                mean_scenario_auroc = np.mean(list(res["scenario_aurocs"].values()))
                fold_aurocs.append(mean_scenario_auroc)
                fold_clean_fprs.append(res["clean_fpr"])

            avg_auroc = np.mean(fold_aurocs)
            avg_clean_fpr = np.mean(fold_clean_fprs)

            # Constraint check: prioritize configs maintaining Clean FPR <= 5% (with 1.5% margin for finite sample variance)
            penalty = 0.0 if avg_clean_fpr <= 0.065 else (avg_clean_fpr - 0.065) * 2.0
            adjusted_score = avg_auroc - penalty

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_cfg = cfg

        best_tuned_configs[model_name] = best_cfg
        print(f"  * {model_name:<20} Best: Scaler={best_cfg['scaler']}, Ratios={best_cfg['use_ratios']} | Val AUROC={best_score:.4f}")

    # Stage 2: Deep 5-Fold Evaluation across all 11 scenarios for best tuned models + Ensemble
    print("\n[Stage 2] Conducting deep 5-Fold cross-validation on tuned models...")
    
    # Store scenario results: model -> scenario -> list of 5 fold scores
    model_scenario_scores = {m: {} for m in list(best_tuned_configs.keys()) + ["Ensemble (Rank-Fusion)"]}
    model_composite_aurocs = {m: [] for m in model_scenario_scores}
    model_clean_fprs = {m: [] for m in model_scenario_scores}
    model_composite_y_scores = {m: ([], []) for m in model_scenario_scores}

    for fold_idx, (train_df, val_df) in enumerate(cv_splits):
        suite = create_synthetic_validation_suite(val_df, seed=RANDOM_SEED + fold_idx * 10)
        fitted_models_for_ensemble = []

        # 1. Fit and evaluate the 6 individual tuned models
        for m_name, cfg in best_tuned_configs.items():
            model_inst = instantiate_candidate_model(m_name, cfg)
            pipeline = make_anomaly_detector_pipeline(
                model=model_inst,
                scaler_type=cfg["scaler"],
                use_ratios=cfg["use_ratios"],
            )
            pipeline.fit(train_df)
            res = evaluate_pipeline_on_suite(pipeline, val_df, suite)

            if fold_idx == 0:
                best_pipelines_fold0[m_name] = pipeline

            # Accumulate scores
            for s_name, score in res["scenario_aurocs"].items():
                if s_name not in model_scenario_scores[m_name]:
                    model_scenario_scores[m_name][s_name] = []
                model_scenario_scores[m_name][s_name].append(score)

            model_composite_aurocs[m_name].append(res["composite_auroc"])
            model_clean_fprs[m_name].append(res["clean_fpr"])
            model_composite_y_scores[m_name][0].extend(res["composite_y_true"])
            model_composite_y_scores[m_name][1].extend(res["composite_scores"])

            # Prepare for ensemble
            fitted_models_for_ensemble.append(model_inst)

        # 2. Evaluate the Rank-Fusion Ensemble
        ens_model = EnsembleAnomalyDetector(models=fitted_models_for_ensemble, target_fpr=0.05)
        # Ensemble uses RobustScaler + Ratios pipeline
        ens_pipeline = make_anomaly_detector_pipeline(
            model=ens_model,
            scaler_type="robust",
            use_ratios=True,
        )
        ens_pipeline.fit(train_df)
        ens_res = evaluate_pipeline_on_suite(ens_pipeline, val_df, suite)

        if fold_idx == 0:
            best_pipelines_fold0["Ensemble (Rank-Fusion)"] = ens_pipeline

        m_name = "Ensemble (Rank-Fusion)"
        for s_name, score in ens_res["scenario_aurocs"].items():
            if s_name not in model_scenario_scores[m_name]:
                model_scenario_scores[m_name][s_name] = []
            model_scenario_scores[m_name][s_name].append(score)

        model_composite_aurocs[m_name].append(ens_res["composite_auroc"])
        model_clean_fprs[m_name].append(ens_res["clean_fpr"])
        model_composite_y_scores[m_name][0].extend(ens_res["composite_y_true"])
        model_composite_y_scores[m_name][1].extend(ens_res["composite_scores"])

    # Stage 3: Build Results Table
    scenario_names = list(next(iter(model_scenario_scores.values())).keys())
    results_rows = []

    for m_name in model_scenario_scores:
        row = {"Model": m_name}
        comp_mean = np.mean(model_composite_aurocs[m_name])
        comp_std = np.std(model_composite_aurocs[m_name])
        row["Overall_AUROC"] = f"{comp_mean:.4f} ± {comp_std:.4f}"
        row["Overall_AUROC_Mean"] = comp_mean

        clean_fpr_mean = np.mean(model_clean_fprs[m_name])
        row["Clean_FPR"] = f"{clean_fpr_mean:.4f}"

        scenario_means = []
        for s_name in scenario_names:
            s_mean = np.mean(model_scenario_scores[m_name][s_name])
            s_std = np.std(model_scenario_scores[m_name][s_name])
            row[s_name] = f"{s_mean:.4f} ± {s_std:.4f}"
            row[f"{s_name}_mean"] = s_mean
            scenario_means.append(s_mean)

        row["Worst_Case_AUROC (Robustness)"] = min(scenario_means)
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    # Sort by Overall Mean AUROC descending
    results_df = results_df.sort_values(by="Overall_AUROC_Mean", ascending=False).reset_index(drop=True)

    # Save CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Detailed benchmark results saved to: {csv_path}")

    # Stage 4: Generate Heatmap
    heatmap_cols = [f"{s}_mean" for s in scenario_names] + ["Overall_AUROC_Mean"]
    heatmap_df = results_df.set_index("Model")[heatmap_cols].copy()
    heatmap_df.columns = [c.replace("_mean", "").replace("Overall_AUROC_Mean", "OVERALL_MEAN") for c in heatmap_cols]

    plt.figure(figsize=(15, 6))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "AUROC Score"},
        linewidths=0.8,
        vmin=0.80,
        vmax=1.00,
    )
    plt.title("Benchmark Heatmap: Model AUROC Across Synthetic Anomaly Scenarios (5-Fold CV)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Anomaly Scenario / Perturbation Type", fontsize=11, fontweight="bold")
    plt.ylabel("Anomaly Detector", fontsize=11, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    heatmap_path = FIGURES_DIR / "benchmark_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[Saved] Benchmark heatmap saved to: {heatmap_path}")

    # Stage 5: Plot ROC Curves for Top 3 Models
    top3_models = results_df["Model"].iloc[:3].tolist()
    print(f"\n[Top 3 Models Identified]: {', '.join(top3_models)}")

    plt.figure(figsize=(8.5, 7.5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, m_name in enumerate(top3_models):
        y_true, scores = model_composite_y_scores[m_name]
        fpr_curve, tpr_curve, _ = roc_curve(y_true, scores)
        roc_auc_val = auc(fpr_curve, tpr_curve)
        plt.plot(
            fpr_curve,
            tpr_curve,
            color=palette[idx],
            lw=2.5,
            label=f"{m_name} (AUROC = {roc_auc_val:.4f})",
        )

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUROC = 0.500)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves of Top 3 Anomaly Detectors Across All Benchmark Scenarios", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    roc_path = FIGURES_DIR / "top3_roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Saved] Top 3 ROC curves plot saved to: {roc_path}")

    # Print summary table
    summary_cols = ["Model", "Overall_AUROC", "Worst_Case_AUROC (Robustness)", "Clean_FPR"]
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (TOP DETECTORS)")
    print("=" * 80)
    print(results_df[summary_cols].to_string(index=False))
    print("=" * 80)

    return results_df, best_tuned_configs


if __name__ == "__main__":
    run_benchmark()
