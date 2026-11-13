"""
Unit tests for models serialization and pickling.
Verifies that all 7 anomaly detectors can be saved and loaded with joblib
and produce identical predictions and anomaly scores.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import RANDOM_SEED, TRACE_DATA_PATH
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


def test_models_picklable():
    """Verify all 7 models can be serialized and deserialized with joblib."""
    df = pd.read_csv(TRACE_DATA_PATH).iloc[:200]
    
    models = [
        ("IsolationForest", IsolationForestDetector(n_estimators=30, random_state=RANDOM_SEED)),
        ("OneClassSVM", OneClassSVMDetector(nu=0.05)),
        ("LOF", LOFDetector(n_neighbors=15)),
        ("GMM", GMMDetector(n_components_candidates=(2,), random_state=RANDOM_SEED)),
        ("RobustMahalanobis", RobustMahalanobisDetector(random_state=RANDOM_SEED)),
        ("Autoencoder", AutoencoderDetector(max_epochs=20, random_state=RANDOM_SEED)),
        ("Ensemble", EnsembleAnomalyDetector(
            models=[
                IsolationForestDetector(n_estimators=20, random_state=RANDOM_SEED),
                LOFDetector(n_neighbors=10),
            ]
        )),
    ]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        for name, model in models:
            pipe = make_anomaly_detector_pipeline(model=model, scaler_type="robust")
            pipe.fit(df)
            
            orig_scores = pipe.score_samples(df.iloc[:20])
            orig_preds = pipe.predict(df.iloc[:20])
            
            # Save model with joblib
            save_file = tmp_path / f"{name}.joblib"
            joblib.dump(pipe, save_file)
            assert save_file.exists(), f"Failed to save {name} to {save_file}"
            
            # Reload model with joblib
            loaded_pipe = joblib.load(save_file)
            loaded_scores = loaded_pipe.score_samples(df.iloc[:20])
            loaded_preds = loaded_pipe.predict(df.iloc[:20])
            
            # Verify numerical equivalence
            np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(orig_preds, loaded_preds)
            print(f"  [PASS] {name:<22} picklable and restored identically via joblib")


def test_autoencoder_torch_save():
    """Verify AutoencoderDetector supports direct torch.save / torch.load."""
    df = pd.read_csv(TRACE_DATA_PATH).iloc[:100]
    ae = AutoencoderDetector(max_epochs=15, random_state=RANDOM_SEED)
    ae.fit(df)
    
    orig_scores = ae.score_samples(df.iloc[:10])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "autoencoder.pt"
        torch.save(ae.net_.state_dict(), save_file)
        assert save_file.exists()
        
        # Reload state dict into new model
        restored_ae = AutoencoderDetector(random_state=RANDOM_SEED)
        restored_ae.input_dim_ = 3
        restored_ae.net_ = ae.net_.__class__(3)
        restored_ae.net_.load_state_dict(torch.load(save_file))
        restored_ae.net_.eval()
        
        restored_scores = restored_ae.score_samples(df.iloc[:10])
        np.testing.assert_allclose(orig_scores, restored_scores, rtol=1e-5, atol=1e-5)
        print("  [PASS] AutoencoderDetector state_dict saved & loaded cleanly via torch.save")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING MODEL SERIALIZATION & PICKLABILITY TESTS")
    print("=" * 60)
    test_models_picklable()
    test_autoencoder_torch_save()
    print("=" * 60)
    print("ALL MODEL PICKLE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
