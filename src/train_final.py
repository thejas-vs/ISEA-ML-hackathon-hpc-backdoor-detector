"""
Final Training & Model Serialization Module for HPC Backdoor Detector.

Fits the finalized preprocessing pipeline and top anomaly detector (LOF with RobustScaler)
on ALL 800 clean HPC trace rows.
Serializes a self-contained, production-ready artifact to models/detector.joblib
and records final hyperparameters, seeds, and calibration metadata to models/config.json.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure writable matplotlib temp directory if needed
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_hpc_final")

import joblib
import numpy as np
import pandas as pd
import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RANDOM_SEED, RAW_FEATURE_COLS, TRACE_DATA_PATH, seed_everything
from src.models import (
    AnomalyDetectorPipeline,
    AutoencoderDetector,
    LOFDetector,
    make_anomaly_detector_pipeline,
)

# Enforce strict reproducibility
seed_everything(RANDOM_SEED)


# =====================================================================
# Production Predictor Wrapper
# =====================================================================

class HPCBackdoorDetector:
    """
    Self-contained deployment wrapper around the fitted pipeline and threshold.
    Provides predict(X), score_samples(X), and decision_function(X).
    """

    def __init__(
        self,
        pipeline: AnomalyDetectorPipeline,
        model: Any,
        threshold: float,
        feature_names: List[str],
        version: str = "1.0.0",
        model_name: str = "LOF",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.pipeline = pipeline
        self.model = model
        self.threshold = float(threshold)
        self.feature_names = list(feature_names)
        self.version = str(version)
        self.model_name = str(model_name)
        self.metadata = metadata or {}
        self.is_fitted = True

    def score_samples(self, X: Union[pd.DataFrame, pd.Series, np.ndarray, dict]) -> np.ndarray:
        """
        Compute continuous anomaly scores.
        HIGHER score indicates MORE ANOMALOUS / higher backdoor likelihood.
        """
        X_df = self._validate_and_format_input(X)
        return self.pipeline.score_samples(X_df)

    def predict(self, X: Union[pd.DataFrame, pd.Series, np.ndarray, dict]) -> np.ndarray:
        """
        Predict binary labels:
        0 = Clean Inference
        1 = Backdoor Anomaly
        """
        scores = self.score_samples(X)
        return (scores >= self.threshold).astype(int)

    def decision_function(self, X: Union[pd.DataFrame, pd.Series, np.ndarray, dict]) -> np.ndarray:
        """
        Signed distance from the decision threshold: (score - threshold).
        Positive values indicate flagged backdoor anomalies.
        """
        return self.score_samples(X) - self.threshold

    def _validate_and_format_input(self, X: Any) -> pd.DataFrame:
        """Validate input format and column presence."""
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            df = X.to_frame().T
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                df = pd.DataFrame([X], columns=self.feature_names)
            else:
                df = pd.DataFrame(X, columns=self.feature_names)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for detector: {type(X)}")

        # Ensure required features exist
        for col in self.feature_names:
            if col not in df.columns:
                raise KeyError(
                    f"Missing required HPC counter column '{col}'. Required: {self.feature_names}"
                )
        return df[self.feature_names]


def load_detector(artifact_path: Optional[Union[str, Path]] = None) -> HPCBackdoorDetector:
    """
    Load the finalized serialized detector artifact from models/detector.joblib.
    
    Parameters
    ----------
    artifact_path : str or Path, optional
        Path to detector.joblib. Defaults to models/detector.joblib.
        
    Returns
    -------
    HPCBackdoorDetector
        Loaded self-contained detector instance ready for inference.
    """
    path = Path(artifact_path) if artifact_path is not None else MODELS_DIR / "detector.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Detector artifact not found at {path}")

    loaded = joblib.load(path)
    
    if isinstance(loaded, dict):
        pipeline = loaded["pipeline"]
        model = loaded.get("model", pipeline.steps[-1][1])
        threshold = loaded["threshold"]
        feature_names = loaded.get("feature_names", RAW_FEATURE_COLS)
        version = loaded.get("version", "1.0.0")
        metadata = loaded.get("metadata", {})
        
        # If the underlying model uses PyTorch weights, check for separate weight file
        torch_weights_path = path.parent / "autoencoder_weights.pt"
        if torch_weights_path.exists():
            if isinstance(model, AutoencoderDetector) and model.net_ is not None:
                model.net_.load_state_dict(torch.load(torch_weights_path))
                model.net_.eval()

        model_name = loaded.get("model_name", metadata.get("model_name", "LOF"))
        return HPCBackdoorDetector(
            pipeline=pipeline,
            model=model,
            threshold=threshold,
            feature_names=feature_names,
            version=version,
            model_name=model_name,
            metadata=metadata,
        )
    elif hasattr(loaded, "predict") and hasattr(loaded, "score_samples"):
        return loaded
    else:
        raise ValueError(f"Unrecognized artifact structure in {path}: {type(loaded)}")


# =====================================================================
# Main Training & Serialization Workflow
# =====================================================================

def train_and_serialize_final_model():
    """
    Train final model on all 800 clean traces and serialize artifacts.
    """
    print("=" * 80)
    print("TRAINING FINAL HPC BACKDOOR DETECTOR (ALL 800 CLEAN SAMPLES)")
    print("=" * 80)

    # 1. Load the complete clean training dataset
    df_clean = pd.read_csv(TRACE_DATA_PATH)
    n_samples, n_features = df_clean.shape
    print(f"Loaded training dataset: {n_samples} rows, {n_features} columns from {TRACE_DATA_PATH}")

    # 2. Configure Best Pipeline from Benchmark:
    # Model: Local Outlier Factor (novelty=True, k=20, metric='minkowski')
    # Preprocessing: RobustScaler on raw features (use_ratios=False, use_log1p=False)
    # Target FPR: 0.05 (5.0%)
    model_name = "LOF"
    detector = LOFDetector(
        n_neighbors=20,
        metric="minkowski",
        p=2,
        target_fpr=0.05,
    )

    pipeline = make_anomaly_detector_pipeline(
        model=detector,
        scaler_type="robust",
        use_ratios=False,
        use_log1p=False,
    )

    # 3. Fit pipeline on ALL 800 clean samples
    print("\nFitting preprocessing pipeline and LOF detector on all 800 clean traces...")
    pipeline.fit(df_clean)

    # 4. Compute Anomaly Score Distribution on Full Training Data
    train_scores = pipeline.score_samples(df_clean)
    
    # Finalized threshold calibrated from 95.0th percentile of clean training distribution
    # (Matches Out-Of-Fold calibrated threshold: -0.1695)
    calibrated_threshold = float(np.percentile(train_scores, 95.0))
    detector.threshold_ = calibrated_threshold

    # Calculate empirical FPR on full training data
    train_preds = (train_scores >= calibrated_threshold).astype(int)
    empirical_train_fpr = float(np.mean(train_preds))
    num_flagged = int(np.sum(train_preds))

    # 5. Anomaly Score Distribution Summary
    score_percentiles = {
        "min": float(np.min(train_scores)),
        "p25": float(np.percentile(train_scores, 25.0)),
        "median (p50)": float(np.median(train_scores)),
        "mean": float(np.mean(train_scores)),
        "std": float(np.std(train_scores)),
        "p75": float(np.percentile(train_scores, 75.0)),
        "p90": float(np.percentile(train_scores, 90.0)),
        "p95 (threshold)": calibrated_threshold,
        "p99": float(np.percentile(train_scores, 99.0)),
        "max": float(np.max(train_scores)),
    }

    print("\n" + "=" * 60)
    print("TRAINING-SET ANOMALY SCORE DISTRIBUTION SUMMARY:")
    print("=" * 60)
    for k, v in score_percentiles.items():
        print(f"  {k:<20} : {v:+.6f}")
    print("=" * 60)
    print(f"  Calibrated Threshold (95.0%) : {calibrated_threshold:+.6f}")
    print(f"  Flagged Samples on Train     : {num_flagged} / {n_samples}")
    print(f"  Empirical Training FPR       : {empirical_train_fpr * 100.0:.2f}% (Target: 5.00%)")
    print("=" * 60)

    # 6. Bundle and Save models/detector.joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib_path = MODELS_DIR / "detector.joblib"

    metadata = {
        "training_date": datetime.now().isoformat(),
        "training_samples": n_samples,
        "score_distribution": score_percentiles,
        "empirical_train_fpr": empirical_train_fpr,
        "random_seed": RANDOM_SEED,
        "model_name": model_name,
    }

    artifact_bundle = {
        "pipeline": pipeline,
        "model": detector,
        "threshold": calibrated_threshold,
        "feature_names": RAW_FEATURE_COLS,
        "version": "1.0.0",
        "metadata": metadata,
    }

    joblib.dump(artifact_bundle, joblib_path)
    print(f"\n[Saved] Self-contained detector artifact saved to: {joblib_path}")

    # If PyTorch Autoencoder is ever bundled, save separate torch weights file
    if isinstance(detector, AutoencoderDetector) and detector.net_ is not None:
        torch_path = MODELS_DIR / "autoencoder_weights.pt"
        torch.save(detector.net_.state_dict(), torch_path)
        print(f"[Saved] PyTorch autoencoder weights saved to: {torch_path}")

    # 7. Update models/config.json with final deployment specifications
    config_json_path = MODELS_DIR / "config.json"
    final_config = {
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "model_name": model_name,
        "model_type": "LocalOutlierFactor",
        "hyperparameters": {
            "n_neighbors": 20,
            "metric": "minkowski",
            "p": 2,
            "novelty": True,
        },
        "preprocessing": {
            "scaler": "RobustScaler",
            "with_centering": True,
            "with_scaling": True,
            "use_ratios": False,
            "use_log1p": False,
            "raw_features": RAW_FEATURE_COLS,
        },
        "calibration": {
            "percentile": 95.0,
            "target_fpr": 0.05,
            "calibrated_threshold": calibrated_threshold,
            "training_samples": n_samples,
            "empirical_training_fpr": empirical_train_fpr,
            "empirical_training_fpr_percent": round(empirical_train_fpr * 100.0, 2),
        },
        "training_score_distribution": score_percentiles,
        "artifact_path": str(joblib_path),
        "input_specification": {
            "columns": RAW_FEATURE_COLS,
            "types": ["float64", "float64", "float64"],
            "expected_shape": "(N, 3)",
        },
        "output_specification": {
            "labels": {"0": "Clean Neural Network Inference", "1": "Backdoor Anomaly Detected"},
            "continuous_score_semantics": "Higher score indicates higher anomaly likelihood",
        },
    }

    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)
    print(f"[Saved] Final configuration saved to: {config_json_path}")

    # 8. Sanity check: Reload artifact and run inference on test sample
    print("\n[Sanity Check] Verifying serialized artifact inference...")
    loaded_detector = load_detector(joblib_path)
    sample_clean = df_clean.iloc[[0]]
    sample_score = float(loaded_detector.score_samples(sample_clean)[0])
    sample_pred = int(loaded_detector.predict(sample_clean)[0])
    print(f"  Single Row Sample Score : {sample_score:+.6f}")
    print(f"  Prediction (0=Clean)    : {sample_pred}")
    assert sample_pred == 0, "Sanity check failed: clean sample flagged as anomaly!"
    print("  Sanity check PASSED successfully!\n")


if __name__ == "__main__":
    train_and_serialize_final_model()
