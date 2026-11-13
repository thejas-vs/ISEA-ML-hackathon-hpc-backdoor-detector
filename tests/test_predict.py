"""
Unit & Integration Tests for src/predict.py CLI and Python API.

Verifies:
(a) Clean traces from data/trace.csv have flagged rate near expected ~5% FPR.
(b) Synthetic anomalies get flagged at a high detection rate (>75%).
(c) CLI subprocess execution produces correct CSV format (anomaly_score, prediction).
(d) Handles scrambled column order, extra columns, and NaN/inf values gracefully.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, RAW_FEATURE_COLS, TRACE_DATA_PATH
from src.predict import predict, sanitize_input_data
from src.validation import generate_mean_shift_anomalies, generate_ratio_perturbation_anomalies


def test_predict_on_clean_traces():
    """Verify clean dataset flagged rate is near expected 5% FPR."""
    df_clean = pd.read_csv(TRACE_DATA_PATH)
    scores, labels = predict(df_clean)
    
    assert len(scores) == 800
    assert len(labels) == 800
    assert set(np.unique(labels)).issubset({0, 1})
    assert np.isfinite(scores).all(), "Scores contain non-finite values!"
    
    flagged_rate = np.mean(labels == 1)
    print(f"  Clean traces flagged rate: {flagged_rate * 100.0:.2f}% (Expected ~5.0%)")
    
    # Must be near 5% FPR (between 3% and 7%, exact 5.0% on train)
    assert 0.03 <= flagged_rate <= 0.07, f"Flagged rate {flagged_rate} unexpected for clean data!"
    print("  [PASS] test_predict_on_clean_traces")


def test_predict_on_synthetic_anomalies():
    """Verify synthetic anomalies get flagged at high TPR (>75%)."""
    df_clean = pd.read_csv(TRACE_DATA_PATH)
    
    # 1. 2.5 std mean shift anomalies
    anom_shift = generate_mean_shift_anomalies(df_clean, std_multiplier=2.5, seed=RANDOM_SEED)
    _, labels_shift = predict(anom_shift)
    flagged_shift = np.mean(labels_shift == 1)
    print(f"  Mean shift (2.5std) flagged rate: {flagged_shift * 100.0:.2f}%")
    assert flagged_shift > 0.85, f"Mean shift detection rate too low: {flagged_shift}"
    
    # 2. 15% ratio perturbation anomalies
    anom_ratio = generate_ratio_perturbation_anomalies(df_clean, delta=0.15, seed=RANDOM_SEED)
    _, labels_ratio = predict(anom_ratio)
    flagged_ratio = np.mean(labels_ratio == 1)
    print(f"  Ratio perturbation (15%) flagged rate: {flagged_ratio * 100.0:.2f}%")
    assert flagged_ratio > 0.90, f"Ratio perturbation detection rate too low: {flagged_ratio}"
    
    print("  [PASS] test_predict_on_synthetic_anomalies")


def test_input_sanitization_and_robustness():
    """Verify handling of out-of-order columns, extra columns, and NaN/inf values."""
    # Test with scrambled column order + extra 'label' and 'id' columns
    scrambled_df = pd.DataFrame({
        "id": [1, 2, 3],
        "LLC-loads": [6.65e6, 6.70e6, np.nan],  # includes NaN
        "extra_info": ["x", "y", "z"],
        "cache-references": [1.83e8, np.inf, 1.85e8],  # includes Inf
        "cycles": [4.77e9, 4.78e9, 4.79e9],
        "label": [0, 0, 1],  # extra ground truth column
    })
    
    sanitized = sanitize_input_data(scrambled_df)
    assert list(sanitized.columns) == RAW_FEATURE_COLS
    assert len(sanitized) == 3
    assert np.isfinite(sanitized.to_numpy()).all(), "Sanitized data still contains non-finite values!"
    
    scores, labels = predict(scrambled_df)
    assert len(scores) == 3
    assert len(labels) == 3
    assert np.isfinite(scores).all()
    print("  [PASS] test_input_sanitization_and_robustness")


def test_cli_execution():
    """Verify predict.py runs as a CLI and produces expected CSV output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_csv = tmp_path / "test_input.csv"
        output_csv = tmp_path / "predictions.csv"
        
        # Write small test CSV with 10 rows
        df_clean = pd.read_csv(TRACE_DATA_PATH).iloc[:10]
        df_clean.to_csv(input_csv, index=False)
        
        # Execute CLI
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "predict.py"),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"CLI failed with error:\n{result.stderr}"
        assert output_csv.exists(), "Predictions CSV was not created!"
        
        pred_df = pd.read_csv(output_csv)
        assert list(pred_df.columns) == ["anomaly_score", "prediction"]
        assert len(pred_df) == 10
        assert np.isfinite(pred_df["anomaly_score"]).all()
        assert set(pred_df["prediction"].unique()).issubset({0, 1})
        print("  [PASS] test_cli_execution (CLI subprocess output format verified)")


def run_all_tests():
    """Run full test suite for prediction module."""
    print("=" * 60)
    print("RUNNING INFERENCE & CLI TEST SUITE")
    print("=" * 60)
    test_predict_on_clean_traces()
    test_predict_on_synthetic_anomalies()
    test_input_sanitization_and_robustness()
    test_cli_execution()
    print("=" * 60)
    print("ALL PREDICTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
