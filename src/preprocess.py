"""
Preprocessing and Feature Engineering pipeline for HPC Backdoor Detector.
Provides a deterministic, scikit-learn compatible pipeline with ratio features,
log transformations, and configurable scaling (RobustScaler vs StandardScaler).
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

from src.config import (
    DEFAULT_PREPROCESS_CONFIG,
    RANDOM_SEED,
    RAW_FEATURE_COLS,
    PreprocessingConfig,
    seed_everything,
)

# Enforce deterministic behavior on import
seed_everything(RANDOM_SEED)


def build_features(
    df: Union[pd.DataFrame, pd.Series, np.ndarray, dict],
    use_ratios: bool = True,
    use_log_ratios: bool = True,
    use_log1p: bool = False,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Construct engineered HPC features deterministically.
    
    Accepts a full DataFrame, a single row DataFrame, a pandas Series, a dictionary,
    or a 2D numpy array with the canonical 3 columns:
    ['cache-references', 'cycles', 'LLC-loads'].
    
    Parameters
    ----------
    df : DataFrame, Series, ndarray, or dict
        Input data containing the three raw HPC counter metrics.
    use_ratios : bool, default=True
        Whether to compute cross-counter ratios.
    use_log_ratios : bool, default=True
        Whether to compute log-transformed ratio features (if use_ratios is True).
    use_log1p : bool, default=False
        Whether to apply np.log1p transformation to the raw counter values.
    eps : float, default=1e-12
        Epsilon constant added to denominators to prevent division by zero.
        
    Returns
    -------
    pd.DataFrame
        Deterministic DataFrame with engineered features without NaNs or Infs.
    """
    # 1. Normalize input to a pandas DataFrame
    if isinstance(df, pd.Series):
        df_proc = df.to_frame().T
    elif isinstance(df, dict):
        df_proc = pd.DataFrame([df])
    elif isinstance(df, np.ndarray):
        if df.ndim == 1:
            df_proc = pd.DataFrame([df], columns=RAW_FEATURE_COLS)
        else:
            df_proc = pd.DataFrame(df, columns=RAW_FEATURE_COLS)
    elif isinstance(df, pd.DataFrame):
        df_proc = df.copy()
    else:
        raise TypeError(f"Unsupported input type for build_features: {type(df)}")
        
    # 2. Validate presence of raw features
    for col in RAW_FEATURE_COLS:
        if col not in df_proc.columns:
            raise KeyError(
                f"Missing required raw feature '{col}'. Expected columns: {RAW_FEATURE_COLS}"
            )
            
    # Keep only float-cast raw columns in deterministic order
    out_df = pd.DataFrame(index=df_proc.index)
    
    cache_ref = df_proc["cache-references"].astype(np.float64)
    cycles = df_proc["cycles"].astype(np.float64)
    llc_loads = df_proc["LLC-loads"].astype(np.float64)
    
    if use_log1p:
        out_df["cache-references"] = np.log1p(np.maximum(cache_ref, 0.0))
        out_df["cycles"] = np.log1p(np.maximum(cycles, 0.0))
        out_df["LLC-loads"] = np.log1p(np.maximum(llc_loads, 0.0))
    else:
        out_df["cache-references"] = cache_ref
        out_df["cycles"] = cycles
        out_df["LLC-loads"] = llc_loads
        
    # 3. Engineered Relationship Ratios
    # Backdoors induce microarchitectural anomalies (e.g. cache stalls or trigger activations)
    # that distort ratios between memory accesses and execution cycles.
    if use_ratios:
        # Prevent division by zero
        safe_cycles = np.where(cycles > 0, cycles, eps)
        safe_cache_ref = np.where(cache_ref > 0, cache_ref, eps)
        
        ratio_cache_cycle = cache_ref / safe_cycles
        ratio_llc_cycle = llc_loads / safe_cycles
        ratio_llc_cache = llc_loads / safe_cache_ref
        
        out_df["ratio_cache_per_cycle"] = ratio_cache_cycle
        out_df["ratio_llc_per_cycle"] = ratio_llc_cycle
        out_df["ratio_llc_per_cache"] = ratio_llc_cache
        
        if use_log_ratios:
            # Safe natural logarithm of ratios
            out_df["log_ratio_cache_per_cycle"] = np.log(np.maximum(ratio_cache_cycle, eps))
            out_df["log_ratio_llc_per_cycle"] = np.log(np.maximum(ratio_llc_cycle, eps))
            out_df["log_ratio_llc_per_cache"] = np.log(np.maximum(ratio_llc_cache, eps))
            
    # 4. Final sanity check: replace any residual non-finite values if present
    out_df = out_df.replace([np.inf, -np.inf], np.nan)
    if out_df.isna().any().any():
        out_df = out_df.fillna(0.0)
        
    return out_df


class HPCFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for HPC feature engineering.
    """
    
    def __init__(
        self,
        use_ratios: bool = True,
        use_log_ratios: bool = True,
        use_log1p: bool = False,
        eps: float = 1e-12,
    ):
        self.use_ratios = use_ratios
        self.use_log_ratios = use_log_ratios
        self.use_log1p = use_log1p
        self.eps = eps
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit transformer and determine output feature names."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            self.feature_names_in_ = list(RAW_FEATURE_COLS)
            
        dummy_out = build_features(
            X.iloc[:2] if isinstance(X, pd.DataFrame) else X[:2],
            use_ratios=self.use_ratios,
            use_log_ratios=self.use_log_ratios,
            use_log1p=self.use_log1p,
            eps=self.eps,
        )
        self.feature_names_out_ = list(dummy_out.columns)
        return self
        
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform input features using build_features."""
        out_df = build_features(
            X,
            use_ratios=self.use_ratios,
            use_log_ratios=self.use_log_ratios,
            use_log1p=self.use_log1p,
            eps=self.eps,
        )
        return out_df.to_numpy(dtype=np.float64)
        
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return names of output features."""
        return list(self.feature_names_out_)


def get_preprocessing_pipeline(
    config: Optional[PreprocessingConfig] = None,
    scaler_type: Optional[str] = None,
    use_ratios: Optional[bool] = None,
    use_log1p: Optional[bool] = None,
    use_log_ratios: Optional[bool] = None,
) -> Pipeline:
    """
    Build an end-to-end, reproducible scikit-learn Pipeline.
    
    Parameters
    ----------
    config : PreprocessingConfig, optional
        Pre-configured settings dataclass.
    scaler_type : {"robust", "standard"}, optional
        Override for scaler selection.
    use_ratios : bool, optional
        Override for ratio feature generation.
    use_log1p : bool, optional
        Override for log1p transformation of raw features.
    use_log_ratios : bool, optional
        Override for log-ratios.
        
    Returns
    -------
    Pipeline
        Configured scikit-learn pipeline ready for fit / transform.
    """
    cfg = config or DEFAULT_PREPROCESS_CONFIG
    
    # Allow explicit overrides
    _scaler_type = scaler_type if scaler_type is not None else cfg.scaler_type
    _use_ratios = use_ratios if use_ratios is not None else cfg.use_ratios
    _use_log1p = use_log1p if use_log1p is not None else cfg.use_log1p
    _use_log_ratios = use_log_ratios if use_log_ratios is not None else cfg.use_log_ratios
    
    # Feature engineering step
    fe_step = HPCFeatureEngineer(
        use_ratios=_use_ratios,
        use_log_ratios=_use_log_ratios,
        use_log1p=_use_log1p,
        eps=cfg.epsilon,
    )
    
    # Scaler step: switchable between RobustScaler and StandardScaler
    if _scaler_type.lower() == "robust":
        scaler_step = RobustScaler(with_centering=True, with_scaling=True, unit_variance=False)
    elif _scaler_type.lower() == "standard":
        scaler_step = StandardScaler(with_mean=True, with_std=True)
    else:
        raise ValueError(
            f"Unsupported scaler_type '{_scaler_type}'. Supported: 'robust', 'standard'"
        )
        
    pipeline = Pipeline(
        steps=[
            ("feature_engineer", fe_step),
            ("scaler", scaler_step),
        ]
    )
    return pipeline
