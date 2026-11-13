"""
Inference CLI & Python API for HPC Backdoor Detector.

Usage as CLI:
    python src/predict.py --input path/to/test.csv --output predictions.csv

Python API:
    from src.predict import predict
    scores, labels = predict(df)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RAW_FEATURE_COLS
from src.train_final import HPCBackdoorDetector, load_detector

# Baseline clean trace medians for robust NaN/Inf imputation fallback
CLEAN_BASELINE_MEDIANS = {
    "cache-references": 183919582.0,
    "cycles": 4772836184.0,
    "LLC-loads": 6656922.0,
}


def sanitize_input_data(df: Union[pd.DataFrame, pd.Series, np.ndarray, dict]) -> pd.DataFrame:
    """
    Validate, reorder, and sanitize input HPC traces for inference.
    
    Guarantees:
    - Verifies required columns: 'cache-references', 'cycles', 'LLC-loads'.
    - Reorders columns to canonical order.
    - Gracefully ignores any extra columns (e.g. 'label', 'id', 'timestamp').
    - Handles NaN and +/- Inf by imputing with clean baseline medians.
    """
    # 1. Normalize input to DataFrame
    if isinstance(df, dict):
        df_proc = pd.DataFrame([df])
    elif isinstance(df, pd.Series):
        df_proc = df.to_frame().T
    elif isinstance(df, np.ndarray):
        if df.ndim == 1:
            df_proc = pd.DataFrame([df], columns=RAW_FEATURE_COLS)
        else:
            df_proc = pd.DataFrame(df, columns=RAW_FEATURE_COLS)
    elif isinstance(df, pd.DataFrame):
        df_proc = df.copy()
    else:
        raise TypeError(f"Unsupported input type for prediction: {type(df)}")

    # 2. Validate presence of required HPC columns
    missing_cols = [col for col in RAW_FEATURE_COLS if col not in df_proc.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing required HPC counter columns: {missing_cols}. "
            f"Expected columns: {RAW_FEATURE_COLS}. "
            f"Columns found: {list(df_proc.columns)}"
        )

    # 3. Extract and reorder only canonical features (ignoring extra columns like 'label')
    sanitized = df_proc[RAW_FEATURE_COLS].copy()

    # 4. Handle non-finite values (NaN, +Inf, -Inf) gracefully
    for col in RAW_FEATURE_COLS:
        col_series = sanitized[col].astype(np.float64)
        has_non_finite = not np.isfinite(col_series).all()
        if has_non_finite:
            median_val = CLEAN_BASELINE_MEDIANS[col]
            # Replace Inf with NaN then impute with clean baseline median
            sanitized[col] = col_series.replace([np.inf, -np.inf], np.nan).fillna(median_val)
        else:
            sanitized[col] = col_series

    return sanitized


def predict(
    df: Union[pd.DataFrame, pd.Series, np.ndarray, dict],
    model_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference using the trained HPC backdoor detector.
    
    Parameters
    ----------
    df : DataFrame, Series, ndarray, or dict
        Input data containing the required HPC counter columns.
    model_path : str or Path, optional
        Path to detector.joblib. Defaults to models/detector.joblib.
        
    Returns
    -------
    scores : np.ndarray
        1D float array of continuous anomaly scores (higher = more anomalous).
    labels : np.ndarray
        1D int array of binary predictions:
        0 = Clean Inference
        1 = Backdoor Anomaly
    """
    detector = load_detector(model_path)
    clean_df = sanitize_input_data(df)
    
    scores = detector.score_samples(clean_df)
    labels = detector.predict(clean_df)
    
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=int)


def run_cli():
    """CLI entrypoint for batch predictions."""
    parser = argparse.ArgumentParser(
        description="Run HPC Backdoor Anomaly Detection on inference traces."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to input test CSV file containing HPC traces.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Path to output predictions CSV file.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        type=str,
        help="Optional custom path to detector.joblib artifact.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading input traces from: {input_path}")
    df_input = pd.read_csv(input_path)
    print(f"Loaded {len(df_input)} rows.")

    scores, labels = predict(df_input, model_path=args.model)

    output_df = pd.DataFrame({
        "anomaly_score": scores,
        "prediction": labels,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(output_df)} predictions to: {output_path}")

    # Print summary statistics of predictions
    num_flagged = int(np.sum(labels == 1))
    num_clean = int(np.sum(labels == 0))
    flagged_pct = (num_flagged / len(labels)) * 100.0

    print("\nPrediction Summary:")
    print(f"  Total Inferences Scored : {len(labels)}")
    print(f"  Flagged as Backdoor (1) : {num_flagged} ({flagged_pct:.2f}%)")
    print(f"  Classified as Clean (0) : {num_clean} ({100.0 - flagged_pct:.2f}%)")
    print(f"  Score Range             : [{scores.min():.4f}, {scores.max():.4f}]")


if __name__ == "__main__":
    run_cli()
