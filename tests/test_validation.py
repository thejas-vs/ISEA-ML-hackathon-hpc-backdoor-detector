"""
Unit tests for the validation and synthetic anomaly evaluation module.
Verifies CV splitting, synthetic anomaly generators, and model evaluation metrics.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.config import TRACE_DATA_PATH, RAW_FEATURE_COLS, RANDOM_SEED
from src.preprocess import HPCFeatureEngineer
from src.validation import (
    get_clean_cv_splits,
    generate_mean_shift_anomalies,
    generate_variance_inflation_anomalies,
    generate_correlation_breaking_anomalies,
    generate_ratio_perturbation_anomalies,
    generate_uniform_bounding_box_anomalies,
    create_synthetic_validation_suite,
    evaluate,
)


def load_test_data() -> pd.DataFrame:
    """Load trace.csv for testing."""
    df = pd.read_csv(TRACE_DATA_PATH)
    return df


def test_cv_splits():
    """Verify 5-fold CV generates mutually exclusive, complete partitions."""
    df = load_test_data()
    splits = list(get_clean_cv_splits(df, n_splits=5, shuffle=True, seed=RANDOM_SEED))
    assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"
    
    val_indices_count = 0
    for fold, (train_df, val_df) in enumerate(splits):
        assert len(train_df) == 640, f"Fold {fold} train size {len(train_df)} != 640"
        assert len(val_df) == 160, f"Fold {fold} val size {len(val_df)} != 160"
        assert list(train_df.columns) == RAW_FEATURE_COLS
        val_indices_count += len(val_df)
        
    assert val_indices_count == 800, "Validation folds do not cover all 800 samples"
    print("  [PASS] test_cv_splits (5-fold, 640 train / 160 val)")


def test_synthetic_generators_finite():
    """Verify all synthetic anomaly generators produce valid, finite DataFrames."""
    df = load_test_data()
    val_clean = df.iloc[:160].copy()
    
    # 1. Mean shift
    for mult in [1.0, 2.0, 3.0]:
        anom = generate_mean_shift_anomalies(val_clean, std_multiplier=mult)
        assert anom.shape == (160, 3)
        assert np.isfinite(anom.to_numpy()).all()
        assert not anom.isna().any().any()
        
    # 2. Variance inflation
    for scale in [1.5, 2.0, 3.0]:
        anom = generate_variance_inflation_anomalies(val_clean, scale=scale)
        assert anom.shape == (160, 3)
        assert np.isfinite(anom.to_numpy()).all()
        
    # 3. Correlation breaking
    anom_corr = generate_correlation_breaking_anomalies(val_clean)
    assert anom_corr.shape == (160, 3)
    assert np.isfinite(anom_corr.to_numpy()).all()
    # Confirm that correlation between cache-references and LLC-loads dropped from ~0.92 to near 0
    orig_corr = val_clean["cache-references"].corr(val_clean["LLC-loads"])
    shuff_corr = anom_corr["cache-references"].corr(anom_corr["LLC-loads"])
    assert orig_corr > 0.85, f"Original correlation unexpectedly low: {orig_corr}"
    assert abs(shuff_corr) < 0.35, f"Shuffled correlation did not drop: {shuff_corr}"
    
    # 4. Ratio perturbation
    for delta in [0.05, 0.10, 0.20]:
        anom = generate_ratio_perturbation_anomalies(val_clean, delta=delta)
        assert anom.shape == (160, 3)
        assert np.isfinite(anom.to_numpy()).all()
        
    # 5. Uniform bounding box
    anom_box = generate_uniform_bounding_box_anomalies(val_clean, expansion=0.20)
    assert anom_box.shape == (160, 3)
    assert np.isfinite(anom_box.to_numpy()).all()
    
    # 6. Full suite
    suite = create_synthetic_validation_suite(val_clean)
    assert len(suite) == 11, f"Expected 11 anomaly scenarios, got {len(suite)}"
    for name, s_df in suite.items():
        assert s_df.shape == (160, 3), f"Scenario {name} shape mismatch: {s_df.shape}"
        assert np.isfinite(s_df.to_numpy()).all(), f"Scenario {name} has non-finite values"
        
    print("  [PASS] test_synthetic_generators_finite (all 11 scenarios verified)")


def test_evaluate_function():
    """Verify evaluate function computes Accuracy, TPR, FPR, F1, AUROC on an anomaly detector."""
    df = load_test_data()
    train_clean = df.iloc[:640].copy()
    val_clean = df.iloc[640:].copy()
    
    # Create synthetic anomalies for validation
    val_anom = generate_mean_shift_anomalies(val_clean, std_multiplier=2.5)
    
    # Fit a simple Isolation Forest pipeline
    pipeline = Pipeline([
        ("fe", HPCFeatureEngineer(use_ratios=True)),
        ("scaler", RobustScaler()),
        ("model", IsolationForest(random_state=RANDOM_SEED, contamination=0.05)),
    ])
    pipeline.fit(train_clean)
    
    metrics = evaluate(pipeline, val_clean, val_anom)
    
    # Validate return keys
    required_keys = ["Accuracy", "TPR", "FPR", "F1", "AUROC", "Precision", "Threshold"]
    for k in required_keys:
        assert k in metrics, f"Missing metric key '{k}'"
        
    # Validate ranges
    assert 0.0 <= metrics["Accuracy"] <= 1.0, f"Invalid Accuracy: {metrics['Accuracy']}"
    assert 0.0 <= metrics["TPR"] <= 1.0, f"Invalid TPR: {metrics['TPR']}"
    assert 0.0 <= metrics["FPR"] <= 1.0, f"Invalid FPR: {metrics['FPR']}"
    assert 0.0 <= metrics["F1"] <= 1.0, f"Invalid F1: {metrics['F1']}"
    assert 0.5 <= metrics["AUROC"] <= 1.0, f"AUROC unexpectedly low: {metrics['AUROC']}"
    
    print(f"  [PASS] test_evaluate_function (AUROC={metrics['AUROC']:.4f}, TPR={metrics['TPR']:.4f}, FPR={metrics['FPR']:.4f}, F1={metrics['F1']:.4f})")


def run_all_tests():
    """Run full validation module test suite."""
    print("=" * 60)
    print("RUNNING VALIDATION MODULE TESTS")
    print("=" * 60)
    test_cv_splits()
    test_synthetic_generators_finite()
    test_evaluate_function()
    print("=" * 60)
    print("ALL VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
