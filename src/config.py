"""
Configuration module for HPC Backdoor Detector.
Defines seeds, directory paths, feature flags, and global reproducibility helpers.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import numpy as np

# Global reproducibility seed
RANDOM_SEED: int = 42

# Project root and directory paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
TRACE_DATA_PATH: Path = DATA_DIR / "trace.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Canonical raw input feature columns expected from HPC trace
RAW_FEATURE_COLS: List[str] = [
    "cache-references",
    "cycles",
    "LLC-loads",
]


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Set random seed across Python, NumPy, and PyTorch for strict reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # PyTorch might not be loaded or WDAC may restrict native torch_python.dll in sandbox
        pass


@dataclass
class PreprocessingConfig:
    """Configuration dataclass for feature engineering and scaling."""
    use_ratios: bool = True
    use_log_ratios: bool = True
    use_log1p: bool = False
    scaler_type: Literal["robust", "standard"] = "robust"
    epsilon: float = 1e-12
    random_seed: int = RANDOM_SEED
    raw_features: List[str] = field(default_factory=lambda: list(RAW_FEATURE_COLS))


# Default global instance
DEFAULT_PREPROCESS_CONFIG = PreprocessingConfig()
