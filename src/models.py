"""
Anomaly Detection Models for HPC Backdoor Detector.

Common Interface:
- fit(X, y=None, **kwargs): Train on clean HPC traces only.
- score_samples(X) -> np.ndarray: Higher score indicates MORE ANOMALOUS.
- predict(X) -> np.ndarray: 0 = Clean, 1 = Backdoor Anomaly.
- set_threshold_from_fpr(X_clean, target_fpr): Calibrate decision boundary.

Implemented Models:
1. IsolationForestDetector (Isolation Forest)
2. OneClassSVMDetector (One-Class SVM with RBF kernel)
3. LOFDetector (Local Outlier Factor with novelty=True)
4. GMMDetector (Gaussian Mixture Model with BIC component selection)
5. RobustMahalanobisDetector (Minimum Covariance Determinant / Robust Mahalanobis)
6. AutoencoderDetector (PyTorch Neural Autoencoder with early stopping)
7. EnsembleAnomalyDetector (Rank-normalized score fusion across detectors)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
from pathlib import Path

# Add project root to sys.path so 'src' imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from src.config import RANDOM_SEED, seed_everything
from src.preprocess import get_preprocessing_pipeline

# Ensure global determinism
seed_everything(RANDOM_SEED)


# =====================================================================
# Base Anomaly Detector Class
# =====================================================================

class BaseAnomalyDetector(BaseEstimator, ClassifierMixin):
    """
    Abstract base class establishing the common anomaly detector interface.
    
    Attributes
    ----------
    target_fpr : float, default=0.05
        Target false alarm rate on clean training data used to calibrate threshold.
    threshold_ : float, optional
        Operating score threshold above which samples are flagged as anomalous (1).
    """
    _is_hpc_detector: bool = True

    def __init__(self, target_fpr: float = 0.05):
        self.target_fpr = target_fpr
        self.threshold_: Optional[float] = None
        self.clean_scores_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None, **kwargs):
        raise NotImplementedError

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute continuous anomaly scores.
        HIGHER score indicates MORE ANOMALOUS.
        """
        raise NotImplementedError

    def set_threshold_from_fpr(
        self, X_clean: Union[pd.DataFrame, np.ndarray], target_fpr: Optional[float] = None
    ):
        """
        Set operating threshold based on target FPR on clean reference data.
        
        threshold = (1 - target_fpr) quantile of clean anomaly scores.
        """
        if target_fpr is not None:
            self.target_fpr = target_fpr
        scores = self.score_samples(X_clean)
        self.clean_scores_ = np.sort(scores.copy())
        pct = 100.0 * (1.0 - self.target_fpr)
        self.threshold_ = float(np.percentile(scores, pct))
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict binary labels:
        0 = Clean
        1 = Backdoor Anomaly
        """
        scores = self.score_samples(X)
        if self.threshold_ is None:
            raise ValueError(
                "Model threshold is uncalibrated. Call fit() or set_threshold_from_fpr() first."
            )
        return (scores >= self.threshold_).astype(int)

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Returns distance from threshold (higher = more anomalous).
        Positive values indicate predicted anomalies.
        """
        scores = self.score_samples(X)
        if self.threshold_ is None:
            return scores
        return scores - self.threshold_


# =====================================================================
# 1. Isolation Forest Detector
# =====================================================================

class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest anomaly detector.
    Recursively partitions feature space using random orthogonal hyperplanes.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_samples: Union[str, int, float] = "auto",
        contamination: str = "auto",
        random_state: int = RANDOM_SEED,
        target_fpr: float = 0.05,
    ):
        super().__init__(target_fpr=target_fpr)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.model_: Optional[IsolationForest] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_arr)
        self.set_threshold_from_fpr(X_arr, self.target_fpr)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # In sklearn, score_samples returns opposite of anomaly score (lower = anomalous).
        # We negate it so HIGHER = MORE ANOMALOUS.
        X_arr = np.asarray(X, dtype=np.float64)
        return -self.model_.score_samples(X_arr)


# =====================================================================
# 2. One-Class SVM Detector
# =====================================================================

class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class Support Vector Machine with Radial Basis Function (RBF) kernel.
    Fits tight non-linear support enclosing the clean distribution.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: Union[str, float] = "scale",
        target_fpr: float = 0.05,
    ):
        super().__init__(target_fpr=target_fpr)
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model_: Optional[OneClassSVM] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.model_ = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
        )
        self.model_.fit(X_arr)
        self.set_threshold_from_fpr(X_arr, self.target_fpr)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # decision_function is positive for inliers and negative for outliers.
        # Negate so HIGHER = MORE ANOMALOUS.
        X_arr = np.asarray(X, dtype=np.float64)
        return -self.model_.decision_function(X_arr)


# =====================================================================
# 3. Local Outlier Factor (LOF) Detector
# =====================================================================

class LOFDetector(BaseAnomalyDetector):
    """
    Local Outlier Factor (novelty=True) detector.
    Measures local density of points relative to their k-nearest neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        p: int = 2,
        target_fpr: float = 0.05,
    ):
        super().__init__(target_fpr=target_fpr)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.model_: Optional[LocalOutlierFactor] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            p=self.p,
            novelty=True,
            n_jobs=-1,
        )
        self.model_.fit(X_arr)
        self.set_threshold_from_fpr(X_arr, self.target_fpr)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # decision_function gives shifted opposite of local outlier factor (higher = inlier).
        # Negate so HIGHER = MORE ANOMALOUS.
        X_arr = np.asarray(X, dtype=np.float64)
        return -self.model_.decision_function(X_arr)


# =====================================================================
# 4. Gaussian Mixture Model (GMM) Detector
# =====================================================================

class GMMDetector(BaseAnomalyDetector):
    """
    Gaussian Mixture Model detector.
    Selects optimal number of components using Bayesian Information Criterion (BIC).
    Score = Negative Log-Likelihood under the mixture density.
    """

    def __init__(
        self,
        n_components_candidates: Tuple[int, ...] = (1, 2, 3, 4, 5),
        covariance_type: str = "full",
        random_state: int = RANDOM_SEED,
        target_fpr: float = 0.05,
    ):
        super().__init__(target_fpr=target_fpr)
        self.n_components_candidates = n_components_candidates
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model_: Optional[GaussianMixture] = None
        self.best_n_components_: Optional[int] = None
        self.bic_scores_: Dict[int, float] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        best_bic = np.inf
        best_gmm = None
        best_k = self.n_components_candidates[0]

        for k in self.n_components_candidates:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=3,
                max_iter=200,
            )
            gmm.fit(X_arr)
            bic_val = gmm.bic(X_arr)
            self.bic_scores_[k] = float(bic_val)
            if bic_val < best_bic:
                best_bic = bic_val
                best_gmm = gmm
                best_k = k

        self.model_ = best_gmm
        self.best_n_components_ = best_k
        self.set_threshold_from_fpr(X_arr, self.target_fpr)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # score_samples returns per-sample log-likelihood.
        # Negative log-likelihood (NLL) => HIGHER = MORE ANOMALOUS.
        X_arr = np.asarray(X, dtype=np.float64)
        return -self.model_.score_samples(X_arr)


# =====================================================================
# 5. Robust Mahalanobis Distance Detector
# =====================================================================

class RobustMahalanobisDetector(BaseAnomalyDetector):
    """
    Robust Mahalanobis Distance detector using Minimum Covariance Determinant (MCD).
    Robust to outliers and heavy tails when estimating location and covariance.
    """

    def __init__(
        self,
        support_fraction: Optional[float] = None,
        random_state: int = RANDOM_SEED,
        target_fpr: float = 0.05,
    ):
        super().__init__(target_fpr=target_fpr)
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.model_: Optional[MinCovDet] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = MinCovDet(
                support_fraction=self.support_fraction,
                random_state=self.random_state,
            )
            self.model_.fit(X_arr)
        self.set_threshold_from_fpr(X_arr, self.target_fpr)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # Compute Mahalanobis distance sqrt((x - mu)^T Sigma^-1 (x - mu)).
        # Higher distance = HIGHER = MORE ANOMALOUS.
        X_arr = np.asarray(X, dtype=np.float64)
        sq_dist = self.model_.mahalanobis(X_arr)
        return np.sqrt(np.maximum(sq_dist, 0.0))


# =====================================================================
# 6. PyTorch Neural Autoencoder Detector
# =====================================================================

class _AutoencoderNet(nn.Module):
    """
    PyTorch MLP Autoencoder architecture:
    Input -> 8 -> 2 (bottleneck) -> 8 -> Input
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderDetector(BaseAnomalyDetector):
    """
    Neural Autoencoder anomaly detector trained solely on clean traces.
    Reconstruction error (MSE) serves as anomaly score.
    Includes early stopping on held-out clean fold. Picklable via state_dict.
    """

    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        max_epochs: int = 200,
        batch_size: int = 32,
        patience: int = 15,
        val_split: float = 0.15,
        random_state: int = RANDOM_SEED,
        target_fpr: float = 0.05,
    ):
        super().__init__(target_fpr=target_fpr)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self.input_dim_: Optional[int] = None
        self.net_: Optional[_AutoencoderNet] = None
        self.best_loss_: float = np.inf
        self.epochs_trained_: int = 0

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ):
        seed_everything(self.random_state)
        X_arr = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X_arr.shape[1]

        # Partition clean validation set for early stopping if not explicitly provided
        if X_val is not None:
            X_v = np.asarray(X_val, dtype=np.float32)
            X_t = X_arr
        else:
            n_total = len(X_arr)
            n_val = max(int(n_total * self.val_split), 1)
            rng = np.random.default_rng(self.random_state)
            perm = rng.permutation(n_total)
            train_idx, val_idx = perm[n_val:], perm[:n_val]
            X_t, X_v = X_arr[train_idx], X_arr[val_idx]

        self.net_ = _AutoencoderNet(self.input_dim_)
        optimizer = optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        train_tensor = torch.from_numpy(X_t)
        val_tensor = torch.from_numpy(X_v)

        dataset = torch.utils.data.TensorDataset(train_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_state = None
        best_val_loss = float("inf")
        patience_counter = 0

        self.net_.train()
        for epoch in range(1, self.max_epochs + 1):
            for (batch_x,) in loader:
                optimizer.zero_grad()
                pred_x = self.net_(batch_x)
                loss = criterion(pred_x, batch_x)
                loss.backward()
                optimizer.step()

            # Validation evaluation for early stopping
            self.net_.eval()
            with torch.no_grad():
                val_pred = self.net_(val_tensor)
                val_loss = criterion(val_pred, val_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.epochs_trained_ = epoch
                    break
            self.net_.train()
            self.epochs_trained_ = epoch

        # Restore best model state
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.best_loss_ = float(best_val_loss)

        # Calibrate threshold on clean training data
        self.set_threshold_from_fpr(X_arr, self.target_fpr)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        # Score = per-sample mean squared reconstruction error.
        # HIGHER error = MORE ANOMALOUS.
        X_arr = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_arr)
            pred_x = self.net_(tensor_x)
            mse_per_sample = torch.mean((pred_x - tensor_x) ** 2, dim=1).cpu().numpy()
        return mse_per_sample.astype(np.float64)

    # Joblib / pickle support for PyTorch Module
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.net_ is not None:
            state["net_state_dict"] = self.net_.state_dict()
            state["net_"] = None
        return state

    def __setstate__(self, state):
        net_state_dict = state.pop("net_state_dict", None)
        self.__dict__.update(state)
        if net_state_dict is not None and self.input_dim_ is not None:
            self.net_ = _AutoencoderNet(self.input_dim_)
            self.net_.load_state_dict(net_state_dict)
            self.net_.eval()


# =====================================================================
# 7. Ensemble Anomaly Detector (Rank-Normalized Score Fusion)
# =====================================================================

class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """
    Rank-Normalized Score Averaging Ensemble.
    Combines diverse anomaly detectors by converting heterogeneous score scales
    into uniform quantile ranks relative to clean reference scores and averaging them.
    """

    def __init__(
        self,
        models: Optional[List[BaseAnomalyDetector]] = None,
        weights: Optional[List[float]] = None,
        target_fpr: float = 0.05,
    ):
        super().__init__(target_fpr=target_fpr)
        if models is None:
            # Default diverse committee of 6 base anomaly detectors
            self.models = [
                IsolationForestDetector(random_state=RANDOM_SEED),
                OneClassSVMDetector(nu=0.05, gamma="scale"),
                LOFDetector(n_neighbors=20),
                GMMDetector(n_components_candidates=(2, 3), random_state=RANDOM_SEED),
                RobustMahalanobisDetector(random_state=RANDOM_SEED),
                AutoencoderDetector(max_epochs=120, random_state=RANDOM_SEED),
            ]
        else:
            self.models = models

        self.weights = weights
        self.ref_scores_: List[np.ndarray] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None, **kwargs):
        X_arr = np.asarray(X, dtype=np.float64)
        self.ref_scores_ = []

        for model in self.models:
            model.fit(X_arr, **kwargs)
            m_scores = model.score_samples(X_arr)
            self.ref_scores_.append(np.sort(m_scores))

        self.set_threshold_from_fpr(X_arr, self.target_fpr)
        return self

    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Rank-normalize each model's scores against reference clean distributions
        and compute weighted average fusion score in [0.0, 1.0].
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        n_models = len(self.models)

        norm_weights = (
            np.ones(n_models) / n_models
            if self.weights is None
            else np.array(self.weights, dtype=np.float64) / np.sum(self.weights)
        )

        rank_matrix = np.zeros((n_samples, n_models), dtype=np.float64)

        for i, model in enumerate(self.models):
            raw_scores = model.score_samples(X_arr)
            ref_scores = self.ref_scores_[i]
            n_ref = len(ref_scores)
            if n_ref > 0:
                # Empirical CDF rank: fraction of clean reference points with score <= raw_score
                ranks = np.searchsorted(ref_scores, raw_scores, side="right") / float(n_ref)
            else:
                ranks = stats.rankdata(raw_scores) / float(len(raw_scores))
            rank_matrix[:, i] = ranks

        ensemble_scores = np.dot(rank_matrix, norm_weights)
        return ensemble_scores


# =====================================================================
# Factory Helper: Build End-to-End Pipeline
# =====================================================================

class AnomalyDetectorPipeline(Pipeline):
    """
    Scikit-learn Pipeline extension providing transparent delegation
    of score_samples, predict, and decision_function to the underlying AnomalyDetector.
    """
    
    @property
    def threshold_(self):
        return getattr(self.steps[-1][1], "threshold_", None)
        
    def score_samples(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        Xt = X
        for _, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][1].score_samples(Xt)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        Xt = X
        for _, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][1].predict(Xt)

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        Xt = X
        for _, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][1].decision_function(Xt)


def make_anomaly_detector_pipeline(
    model: BaseAnomalyDetector,
    scaler_type: str = "robust",
    use_ratios: bool = True,
    use_log1p: bool = False,
) -> AnomalyDetectorPipeline:
    """
    Wrap an anomaly detector model in an AnomalyDetectorPipeline with preprocessing.
    """
    prep_pipe = get_preprocessing_pipeline(
        scaler_type=scaler_type,
        use_ratios=use_ratios,
        use_log1p=use_log1p,
    )
    full_pipeline = AnomalyDetectorPipeline(
        steps=[
            ("prep", prep_pipe),
            ("detector", model),
        ]
    )
    return full_pipeline


# =====================================================================
# Smoke-Test Runner
# =====================================================================

def smoke_test_all_models():
    """
    Smoke-test all 7 models end-to-end on trace.csv through the preprocessing pipeline.
    """
    from src.config import TRACE_DATA_PATH
    from src.validation import evaluate, generate_mean_shift_anomalies

    print("=" * 70)
    print("RUNNING SMOKE TEST ON ALL 7 ANOMALY DETECTOR MODELS")
    print("=" * 70)

    df = pd.read_csv(TRACE_DATA_PATH)
    train_clean = df.iloc[:640].copy()
    val_clean = df.iloc[640:].copy()
    val_anom = generate_mean_shift_anomalies(val_clean, std_multiplier=2.5, seed=RANDOM_SEED)

    models_to_test = [
        ("1. Isolation Forest", IsolationForestDetector(n_estimators=100, random_state=RANDOM_SEED)),
        ("2. One-Class SVM", OneClassSVMDetector(nu=0.05, gamma="scale")),
        ("3. Local Outlier Factor", LOFDetector(n_neighbors=20)),
        ("4. Gaussian Mixture Model", GMMDetector(n_components_candidates=(2, 3), random_state=RANDOM_SEED)),
        ("5. Robust Mahalanobis", RobustMahalanobisDetector(random_state=RANDOM_SEED)),
        ("6. PyTorch Autoencoder", AutoencoderDetector(max_epochs=100, patience=10, random_state=RANDOM_SEED)),
        ("7. Rank-Averaged Ensemble", EnsembleAnomalyDetector()),
    ]

    results = []
    for name, model in models_to_test:
        pipeline = make_anomaly_detector_pipeline(
            model=model,
            scaler_type="robust",
            use_ratios=True,
            use_log1p=False,
        )
        pipeline.fit(train_clean)
        metrics = evaluate(pipeline, val_clean, val_anom)
        results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "TPR": metrics["TPR"],
            "FPR": metrics["FPR"],
            "F1": metrics["F1"],
            "AUROC": metrics["AUROC"],
            "Threshold": metrics["Threshold"],
        })
        print(f"  [PASS] {name:<26} | AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1']:.4f} | TPR: {metrics['TPR']:.4f} | FPR: {metrics['FPR']:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY RESULTS TABLE:")
    print("=" * 70)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("=" * 70)
    print("SMOKE TEST COMPLETE - ALL 7 MODELS OPERATIONAL END-TO-END!")
    print("=" * 70)
    return res_df


if __name__ == "__main__":
    smoke_test_all_models()
