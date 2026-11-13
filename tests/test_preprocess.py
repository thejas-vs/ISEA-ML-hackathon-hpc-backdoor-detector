"""
Unit tests for the preprocessing and feature engineering pipeline.
Verifies determinism, single-row vs full-dataframe processing,
reproducibility, and ensures no NaN/inf values are produced.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import TRACE_DATA_PATH, RAW_FEATURE_COLS, RANDOM_SEED, seed_everything
from src.preprocess import build_features, get_preprocessing_pipeline, HPCFeatureEngineer


def load_test_data() -> pd.DataFrame:
    """Load trace.csv for testing."""
    assert TRACE_DATA_PATH.exists(), f"Trace file missing at {TRACE_DATA_PATH}"
    df = pd.read_csv(TRACE_DATA_PATH)
    assert len(df) == 800, f"Expected 800 rows, found {len(df)}"
    assert list(df.columns) == RAW_FEATURE_COLS, f"Columns mismatch: {df.columns}"
    return df


def test_full_dataset_no_nan_or_inf():
    """Verify build_features on full 800-row dataset produces no NaN or Inf."""
    seed_everything(RANDOM_SEED)
    df = load_test_data()
    
    # 1. With ratios & log-ratios
    feat_df = build_features(df, use_ratios=True, use_log_ratios=True, use_log1p=False)
    assert feat_df.shape == (800, 9), f"Expected shape (800, 9), got {feat_df.shape}"
    assert not feat_df.isna().any().any(), "Found NaN in engineered features!"
    assert not np.isneginf(feat_df.to_numpy()).any(), "Found -Inf in engineered features!"
    assert not np.isposinf(feat_df.to_numpy()).any(), "Found +Inf in engineered features!"
    
    # Check expected column names
    expected_cols = [
        "cache-references", "cycles", "LLC-loads",
        "ratio_cache_per_cycle", "ratio_llc_per_cycle", "ratio_llc_per_cache",
        "log_ratio_cache_per_cycle", "log_ratio_llc_per_cycle", "log_ratio_llc_per_cache",
    ]
    assert list(feat_df.columns) == expected_cols, f"Columns mismatch: {list(feat_df.columns)}"
    print("  [PASS] test_full_dataset_no_nan_or_inf (all 9 features finite)")


def test_single_row_compatibility():
    """Verify build_features works on single row DataFrame, Series, and dict."""
    df = load_test_data()
    
    # Single-row DataFrame
    row_df = df.iloc[[0]]
    out_df = build_features(row_df, use_ratios=True)
    assert out_df.shape == (1, 9), f"Expected shape (1, 9), got {out_df.shape}"
    assert not out_df.isna().any().any(), "Single-row DataFrame produced NaN!"
    assert np.isfinite(out_df.to_numpy()).all(), "Single-row DataFrame produced non-finite values!"
    
    # Pandas Series (1 row)
    row_series = df.iloc[0]
    out_series = build_features(row_series, use_ratios=True)
    assert out_series.shape == (1, 9), f"Expected shape (1, 9), got {out_series.shape}"
    assert np.isfinite(out_series.to_numpy()).all(), "Series produced non-finite values!"
    
    # Dictionary
    row_dict = df.iloc[0].to_dict()
    out_dict = build_features(row_dict, use_ratios=True)
    assert out_dict.shape == (1, 9), f"Expected shape (1, 9), got {out_dict.shape}"
    assert np.isfinite(out_dict.to_numpy()).all(), "Dictionary produced non-finite values!"
    
    # Consistency check: values must match across input formats
    pd.testing.assert_frame_equal(out_df, out_series, check_like=True)
    pd.testing.assert_frame_equal(out_df, out_dict, check_like=True)
    print("  [PASS] test_single_row_compatibility (DataFrame, Series, dict)")


def test_determinism():
    """Verify build_features produces bitwise identical results on multiple runs."""
    df = load_test_data()
    run1 = build_features(df, use_ratios=True, use_log_ratios=True, use_log1p=True)
    run2 = build_features(df, use_ratios=True, use_log_ratios=True, use_log1p=True)
    pd.testing.assert_frame_equal(run1, run2)
    print("  [PASS] test_determinism (deterministic bitwise identical outputs)")


def test_log1p_and_no_ratios_options():
    """Verify feature flags use_ratios=False and use_log1p=True."""
    df = load_test_data()
    
    # use_ratios=False should return exactly the 3 raw columns
    no_ratios = build_features(df, use_ratios=False)
    assert no_ratios.shape == (800, 3)
    assert list(no_ratios.columns) == RAW_FEATURE_COLS
    assert np.isfinite(no_ratios.to_numpy()).all()
    
    # use_log1p=True
    with_log = build_features(df, use_ratios=True, use_log1p=True)
    assert with_log.shape == (800, 9)
    assert np.isfinite(with_log.to_numpy()).all()
    # Confirm log1p of raw counters is in reasonable range (~15-23)
    assert (with_log["cycles"] < 25.0).all()
    print("  [PASS] test_log1p_and_no_ratios_options")


def test_sklearn_pipelines():
    """Verify sklearn-compatible pipeline with RobustScaler and StandardScaler."""
    df = load_test_data()
    
    for scaler_name in ["robust", "standard"]:
        for use_log in [False, True]:
            pipe = get_preprocessing_pipeline(
                scaler_type=scaler_name,
                use_ratios=True,
                use_log1p=use_log,
            )
            # Fit and transform
            X_transformed = pipe.fit_transform(df)
            assert X_transformed.shape == (800, 9), f"Unexpected transformed shape: {X_transformed.shape}"
            assert np.isfinite(X_transformed).all(), f"Pipeline with {scaler_name} produced non-finite values!"
            
            # Single row transform
            single_transformed = pipe.transform(df.iloc[[0]])
            assert single_transformed.shape == (1, 9)
            assert np.isfinite(single_transformed).all()
            np.testing.assert_allclose(single_transformed[0], X_transformed[0], rtol=1e-7, atol=1e-7)
            
    print("  [PASS] test_sklearn_pipelines (RobustScaler, StandardScaler, log1p)")


def test_zero_division_guard():
    """Verify division by zero / negative edge cases are safely handled."""
    edge_df = pd.DataFrame([
        {"cache-references": 0.0, "cycles": 0.0, "LLC-loads": 0.0},
        {"cache-references": 100.0, "cycles": 0.0, "LLC-loads": 5.0},
    ])
    edge_out = build_features(edge_df, use_ratios=True, use_log_ratios=True)
    assert edge_out.shape == (2, 9)
    assert np.isfinite(edge_out.to_numpy()).all(), "Edge cases produced NaN/Inf!"
    print("  [PASS] test_zero_division_guard (handles zeros/eps safely)")


def run_all_tests():
    """Run full test suite."""
    print("=" * 60)
    print("RUNNING PREPROCESSING & FEATURE ENGINEERING TESTS")
    print("=" * 60)
    test_full_dataset_no_nan_or_inf()
    test_single_row_compatibility()
    test_determinism()
    test_log1p_and_no_ratios_options()
    test_sklearn_pipelines()
    test_zero_division_guard()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
