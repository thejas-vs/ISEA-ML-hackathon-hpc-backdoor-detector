"""
Validation & Synthetic Anomaly Evaluation Module for HPC Backdoor Detector.

IMPORTANT METHODOLOGICAL NOTE:
Synthetic anomalies are a controlled PROXY for model evaluation and hyperparameter
selection. In reality, backdoor triggers in neural network inference can be subtle
and non-linear. The synthetic distributions implemented here span a spectrum from
mild microarchitectural perturbations to structural correlation collapses.
SYNTHETIC ANOMALIES ARE NEVER USED IN TRAINING; they are exclusively generated for
out-of-sample validation to estimate TPR, FPR, F1, and AUROC.
"""

from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

from src.config import RANDOM_SEED, RAW_FEATURE_COLS, seed_everything

# Enforce reproducibility
seed_everything(RANDOM_SEED)


# =====================================================================
# 1. Cross-Validation Splitting
# =====================================================================

def get_clean_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = RANDOM_SEED,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Split clean HPC dataset into 5-fold cross-validation folds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean HPC traces.
    n_splits : int, default=5
        Number of CV folds.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    seed : int, default=42
        Random seed for shuffling.
        
    Yields
    ------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_val_clean) for each fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df_train = df.iloc[train_idx].copy().reset_index(drop=True)
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)
        yield df_train, df_val


# =====================================================================
# 2. Synthetic Anomaly Generators (For Validation Only)
# =====================================================================

def generate_mean_shift_anomalies(
    df: pd.DataFrame,
    std_multiplier: float = 2.0,
    direction: str = "both",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Mean shift anomaly: shifts features by +/- 1, 2, or 3 standard deviations.
    Simulates hardware performance counter offsets from injected backdoor loops.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean baseline dataset.
    std_multiplier : float, default=2.0
        Number of standard deviations to shift (e.g., 1.0, 2.0, 3.0).
    direction : {"both", "positive", "negative"}
        Direction of the shift.
    seed : int, default=42
        Random seed.
    """
    rng = np.random.default_rng(seed)
    anom_df = df[RAW_FEATURE_COLS].copy()
    
    stds = df[RAW_FEATURE_COLS].std(axis=0)
    
    for col in RAW_FEATURE_COLS:
        if direction == "positive":
            signs = np.ones(len(df))
        elif direction == "negative":
            signs = -np.ones(len(df))
        else:
            signs = rng.choice([-1.0, 1.0], size=len(df))
            
        shift = signs * std_multiplier * stds[col]
        anom_df[col] = np.maximum(anom_df[col] + shift, 1.0)
        
    return anom_df


def generate_variance_inflation_anomalies(
    df: pd.DataFrame,
    scale: float = 2.0,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Variance inflation anomaly: scales deviation from feature means by 1.5x, 2x, or 3x.
    Simulates jitter or erratic execution behavior caused by asynchronous backdoor activations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean baseline dataset.
    scale : float, default=2.0
        Expansion factor for feature variance (e.g. 1.5, 2.0, 3.0).
    seed : int, default=42
        Random seed.
    """
    anom_df = df[RAW_FEATURE_COLS].copy()
    means = df[RAW_FEATURE_COLS].mean(axis=0)
    
    for col in RAW_FEATURE_COLS:
        diff = anom_df[col] - means[col]
        anom_df[col] = np.maximum(means[col] + scale * diff, 1.0)
        
    return anom_df


def generate_correlation_breaking_anomalies(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Correlation-breaking anomaly: independently shuffles features across rows.
    Preserves marginal distributions while breaking the strong physical coupling
    (e.g. r=0.920 between cache-references and LLC-loads).
    Simulates backdoors that decorrelate cache memory access from CPU cycles.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean baseline dataset.
    seed : int, default=42
        Random seed.
    """
    rng = np.random.default_rng(seed)
    anom_df = pd.DataFrame(index=df.index)
    
    for col in RAW_FEATURE_COLS:
        vals = df[col].to_numpy().copy()
        rng.shuffle(vals)
        anom_df[col] = vals
        
    return anom_df


def generate_ratio_perturbation_anomalies(
    df: pd.DataFrame,
    delta: float = 0.10,
    direction: str = "both",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Ratio perturbation anomaly: perturbs LLC-loads relative to cache-references by delta%.
    Simulates targeted memory access anomalies (e.g. extra cache misses or cache bypass).
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean baseline dataset.
    delta : float, default=0.10
        Fractional perturbation magnitude (e.g. 0.05 for 5%, 0.10 for 10%, 0.20 for 20%).
    direction : {"both", "positive", "negative"}
        Whether to increase, decrease, or randomly perturb the ratio.
    seed : int, default=42
        Random seed.
    """
    rng = np.random.default_rng(seed)
    anom_df = df[RAW_FEATURE_COLS].copy()
    
    if direction == "positive":
        factors = 1.0 + delta
    elif direction == "negative":
        factors = np.maximum(1.0 - delta, 0.01)
    else:
        signs = rng.choice([-1.0, 1.0], size=len(df))
        factors = np.maximum(1.0 + signs * delta, 0.01)
        
    # Perturb LLC-loads while maintaining cache-references and cycles
    anom_df["LLC-loads"] = np.maximum(anom_df["LLC-loads"] * factors, 1.0)
    return anom_df


def generate_uniform_bounding_box_anomalies(
    df: pd.DataFrame,
    expansion: float = 0.20,
    n_samples: Optional[int] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Uniform noise anomalies sampled within a bounding box expanded by 20% around the data.
    Simulates unstructured or unconstrained out-of-distribution executions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clean baseline dataset.
    expansion : float, default=0.20
        Fractional expansion beyond the min-max bounding box (20%).
    n_samples : int, optional
        Number of samples to draw. Defaults to len(df).
    seed : int, default=42
        Random seed.
    """
    rng = np.random.default_rng(seed)
    n = n_samples if n_samples is not None else len(df)
    anom_df = pd.DataFrame(index=range(n))
    
    for col in RAW_FEATURE_COLS:
        c_min = df[col].min()
        c_max = df[col].max()
        c_range = c_max - c_min
        
        low = np.maximum(c_min - expansion * c_range, 1.0)
        high = c_max + expansion * c_range
        anom_df[col] = rng.uniform(low=low, high=high, size=n)
        
    return anom_df


def create_synthetic_validation_suite(
    df_clean_holdout: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    Create a comprehensive suite of synthetic validation anomalies across all types and severities.
    
    Returns a dictionary mapping descriptive anomaly names to synthetic DataFrames.
    """
    suite = {
        # a) Mean shifts
        "mean_shift_1std": generate_mean_shift_anomalies(df_clean_holdout, std_multiplier=1.0, seed=seed),
        "mean_shift_2std": generate_mean_shift_anomalies(df_clean_holdout, std_multiplier=2.0, seed=seed + 1),
        "mean_shift_3std": generate_mean_shift_anomalies(df_clean_holdout, std_multiplier=3.0, seed=seed + 2),
        
        # b) Variance inflation
        "var_inflation_1.5x": generate_variance_inflation_anomalies(df_clean_holdout, scale=1.5, seed=seed + 3),
        "var_inflation_2.0x": generate_variance_inflation_anomalies(df_clean_holdout, scale=2.0, seed=seed + 4),
        "var_inflation_3.0x": generate_variance_inflation_anomalies(df_clean_holdout, scale=3.0, seed=seed + 5),
        
        # c) Correlation breaking (shuffled features)
        "correlation_breaking": generate_correlation_breaking_anomalies(df_clean_holdout, seed=seed + 6),
        
        # d) Ratio perturbation (LLC / cache)
        "ratio_perturb_5pct": generate_ratio_perturbation_anomalies(df_clean_holdout, delta=0.05, seed=seed + 7),
        "ratio_perturb_10pct": generate_ratio_perturbation_anomalies(df_clean_holdout, delta=0.10, seed=seed + 8),
        "ratio_perturb_20pct": generate_ratio_perturbation_anomalies(df_clean_holdout, delta=0.20, seed=seed + 9),
        
        # e) Uniform bounding box noise
        "uniform_box_20pct": generate_uniform_bounding_box_anomalies(df_clean_holdout, expansion=0.20, seed=seed + 10),
    }
    return suite


# =====================================================================
# 3. Model Evaluation Function
# =====================================================================

def extract_anomaly_scores(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """
    Extract continuous anomaly scores from any supported anomaly detector model.
    HIGHER score indicates higher probability / degree of anomaly.
    """
    # If wrapped in a Pipeline, inspect the final estimator
    last_est = model.steps[-1][1] if hasattr(model, "steps") else model
    
    # 1. Custom detector methods or BaseAnomalyDetector
    if hasattr(model, "predict_anomaly_score"):
        return np.asarray(model.predict_anomaly_score(X), dtype=np.float64)
    elif hasattr(model, "reconstruction_error"):
        return np.asarray(model.reconstruction_error(X), dtype=np.float64)
    
    # Check if estimator implements our HPC anomaly detector interface (higher = more anomalous)
    if getattr(last_est, "_is_hpc_detector", False) or hasattr(last_est, "set_threshold_from_fpr"):
        if hasattr(model, "score_samples"):
            return np.asarray(model.score_samples(X), dtype=np.float64)
        elif hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(X), dtype=np.float64)

    # 2. Raw scikit-learn models (where score_samples is higher for inliers)
    if hasattr(model, "score_samples"):
        scores = model.score_samples(X)
        return -np.asarray(scores, dtype=np.float64)
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X)
        return -np.asarray(dec, dtype=np.float64)
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.asarray(probs[:, 1], dtype=np.float64)
        return np.asarray(probs, dtype=np.float64).ravel()
    else:
        raise AttributeError(
            f"Model {type(model)} does not provide an anomaly scoring interface."
        )


def evaluate(
    model: Any,
    X_clean_holdout: Union[pd.DataFrame, np.ndarray],
    X_synth_anomaly: Union[pd.DataFrame, np.ndarray],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Evaluate an anomaly detector on clean holdout traces vs synthetic anomaly traces.
    
    Ground Truth Convention:
        0 = Clean (Negative)
        1 = Anomaly (Positive)
        
    Parameters
    ----------
    model : Any
        Fitted anomaly detection model.
    X_clean_holdout : DataFrame or ndarray
        Clean validation samples.
    X_synth_anomaly : DataFrame or ndarray
        Synthetic anomaly validation samples.
    threshold : float, optional
        Decision threshold applied to anomaly scores. If None, the threshold is
        inferred from the model's native `predict(X)` or set to the 95th percentile
        of clean holdout scores (corresponding to an intended 5% FPR).
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metrics: Accuracy, TPR, FPR, F1, AUROC, Precision, and Threshold.
    """
    n_clean = len(X_clean_holdout)
    n_anom = len(X_synth_anomaly)
    
    # Ground truth labels: 0 for clean, 1 for anomaly
    y_true = np.concatenate([np.zeros(n_clean, dtype=int), np.ones(n_anom, dtype=int)])
    
    # Combine inputs for evaluation
    if isinstance(X_clean_holdout, pd.DataFrame) and isinstance(X_synth_anomaly, pd.DataFrame):
        X_combined = pd.concat([X_clean_holdout, X_synth_anomaly], axis=0, ignore_index=True)
    else:
        X_combined = np.vstack([np.asarray(X_clean_holdout), np.asarray(X_synth_anomaly)])
        
    # 1. Compute Continuous Anomaly Scores
    scores = extract_anomaly_scores(model, X_combined)
    clean_scores = scores[:n_clean]
    anom_scores = scores[n_clean:]
    
    # 2. Compute Threshold-Independent AUROC
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception:
        auroc = 0.5
        
    # 3. Determine Predictions & Operating Threshold
    chosen_threshold = threshold
    last_est = model.steps[-1][1] if hasattr(model, "steps") else model
    
    if chosen_threshold is not None:
        y_pred = (scores >= chosen_threshold).astype(int)
    else:
        # Check if the model has an explicitly calibrated threshold
        if hasattr(last_est, "threshold_") and last_est.threshold_ is not None:
            chosen_threshold = float(last_est.threshold_)
            y_pred = (scores >= chosen_threshold).astype(int)
        elif hasattr(model, "predict"):
            preds = model.predict(X_combined)
            # Sklearn convention: 1 for inlier/clean, -1 for outlier/anomaly
            if -1 in preds:
                y_pred = (preds == -1).astype(int)
                chosen_threshold = float(np.median(clean_scores))
            else:
                y_pred = np.asarray(preds, dtype=int)
                chosen_threshold = 0.5
        else:
            # Default threshold: 95th percentile of clean scores (5% false alarm design)
            chosen_threshold = float(np.percentile(clean_scores, 95.0))
            y_pred = (scores >= chosen_threshold).astype(int)
            
    # 4. Confusion Matrix Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    accuracy = float(accuracy_score(y_true, y_pred))
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0  # False Alarm Rate
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        "Accuracy": round(accuracy, 4),
        "TPR": round(tpr, 4),
        "FPR": round(fpr, 4),
        "F1": round(f1, 4),
        "AUROC": round(auroc, 4),
        "Precision": round(precision, 4),
        "Threshold": round(chosen_threshold, 6),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
