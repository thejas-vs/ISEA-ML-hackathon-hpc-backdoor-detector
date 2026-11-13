# Exploratory Data Analysis (EDA) Notes & Interpretations

**Dataset:** `data/trace.csv` (800 rows $\times$ 3 columns)  
**Features:** `cache-references`, `cycles`, `LLC-loads`  
**Task:** Unsupervised Anomaly Detection for Backdoor Detection in Neural Network Inferences  
**Artifacts Generated:** `reports/eda_summary.csv` and diagnostic plots in `reports/figures/`

---

## 1. Summary Statistics & Key Numerical Observations

| Feature | Mean | Median | Std | Min | Max | IQR | Skewness | Kurtosis (Fisher) | Outliers (IQR) | Outliers ($|z|>3$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **cache-references** | $1.824 \times 10^8$ | $1.839 \times 10^8$ | $4.999 \times 10^6$ | $1.679 \times 10^8$ | $1.916 \times 10^8$ | $7.532 \times 10^6$ | **-0.671** | **-0.669** | 0 (0.00%) | 0 (0.00%) |
| **cycles** | $4.772 \times 10^9$ | $4.773 \times 10^9$ | $2.647 \times 10^7$ | $4.699 \times 10^9$ | $4.996 \times 10^9$ | $3.217 \times 10^7$ | **+0.898** | **+6.808** | 7 (0.88%) | 3 (0.38%) |
| **LLC-loads** | $6.593 \times 10^6$ | $6.657 \times 10^6$ | $2.475 \times 10^5$ | $5.916 \times 10^6$ | $7.108 \times 10^6$ | $3.739 \times 10^5$ | **-0.539** | **-0.663** | 0 (0.00%) | 0 (0.00%) |

---

## 2. Core Diagnostic Findings

### A. Correlation Analysis
- **Extremely High Physical Coupling ($r = 0.920$, Spearman $\rho = 0.896$):**  
  `cache-references` and `LLC-loads` exhibit a very strong linear and monotonic correlation. This aligns with microarchitectural reality: increases in total cache memory access requests directly drive requests that reach the Last Level Cache (LLC).
- **Moderate Correlation with Execution Cycles ($r \approx 0.53 - 0.54$):**  
  `cycles` has moderate positive linear correlation with both `cache-references` ($0.529$) and `LLC-loads` ($0.540$). While cache misses stall the pipeline and consume additional cycles, execution cycles also depend on pipeline hazards, branch mispredictions, and arithmetic intensity.
- **Physical Invariant Constraint:**  
  Clean inferences tightly inhabit a low-dimensional manifold in 3D feature space (almost a 1D/2D curve). An adversarial backdoor trigger that alters control flow, injects hidden computation, or stalls memory hierarchy will likely violate this strict correlation, making off-manifold residual projection an exceptionally strong anomaly signal.

### B. Distribution Shape: Skewness, Kurtosis & Normality
- **Non-Gaussianity Across All Features:**  
  The Shapiro-Wilk normality test emphatically rejects the Gaussian hypothesis for all 3 features ($p < 10^{-14}$):
  - `cache-references`: $W = 0.922$, $p = 7.34 \times 10^{-20}$
  - `cycles`: $W = 0.955$, $p = 6.91 \times 10^{-15}$
  - `LLC-loads`: $W = 0.950$, $p = 7.63 \times 10^{-16}$
- **Heavy Tails & Spikes in `cycles`:**  
  `cycles` is right-skewed ($+0.898$) and distinctly **leptokurtic** (excess kurtosis $+6.808$). The maximum value reaches $4.996 \times 10^9$ cycles, with 7 IQR outliers ($0.88\%$) and 3 severe statistical outliers beyond $3\sigma$ ($0.38\%$). These spikes represent transient operating system interrupts, CPU frequency throttling, or thread context switches during clean inferences.
- **Platykurtic & Left-Skewed Cache Behavior:**  
  `cache-references` and `LLC-loads` have negative skew ($-0.671$ and $-0.539$) and platykurtic negative excess kurtosis ($-0.669$ and $-0.663$), reflecting a flatter, capped upper distribution without extreme outliers (0 outliers detected).

### C. Multimodality
- Kernel Density Estimation (KDE) reveals **bimodality** in both `cache-references` and `LLC-loads`:
  - **Secondary / Lower Mode:** `cache-references` $\approx 1.75 \times 10^8$, `LLC-loads` $\approx 6.24 \times 10^6$
  - **Primary / Main Mode:** `cache-references` $\approx 1.86 \times 10^8$, `LLC-loads` $\approx 6.73 \times 10^6$
- This bimodality corresponds to two natural operational regimes or input paths within clean inferences (e.g., varying input dimensions, caching states, or dynamic control flow branches in the clean neural network).

### D. Stationarity & Drift Analysis
- **Zero Temporal / Sequence Drift:**  
  - Feature correlation with row index is practically zero:
    - `cache-references`: $r = +0.0288$
    - `cycles`: $r = +0.0496$
    - `LLC-loads`: $r = +0.0308$
  - Mean shift between the first 400 rows and the last 400 rows is $< 0.35\%$ across all metrics ($+0.28\%$ for cache-references, $+0.06\%$ for cycles, $+0.33\%$ for LLC-loads).
  - The row-index rolling mean (window = 25) remains completely flat.
  - **Conclusion:** The 800 samples can be treated as exchangeable, independently and identically distributed (i.i.d.) clean inferences without needing sequence modeling (e.g. LSTM/ARIMA) or drift compensation.

---

## 3. Implications for Feature Scaling

| Scaler Method | Assessment & Recommendation |
| :--- | :--- |
| **StandardScaler (Z-Score)** | **Suboptimal for distance metrics.** The high kurtosis ($+6.81$) and extreme positive outliers in `cycles` distort the sample mean and inflate the sample variance, compressing the dense clean cluster and degrading distance-based anomaly thresholds. |
| **RobustScaler (Median / IQR)** | **Recommended as baseline.** Uses the median and Interquartile Range (IQR), making the scaling resilient to the positive spikes in `cycles` while preserving the exact bimodal morphology and relative spread of the clean data. |
| **Log-Transformation / PowerTransformer** | **Selective Use.** Taking $\log(\text{cycles})$ or applying a Yeo-Johnson power transform specifically to `cycles` compresses its heavy right tail into near-symmetry. However, applying a naive log-transform to `cache-references` or `LLC-loads` is unhelpful because they are already left-skewed with bounded upper values. |
| **QuantileTransformer (Uniform / Normal)** | **Viable alternative.** Non-parametrically maps features to standard normal or uniform distributions, dampening outliers and linearizing complex densities, but can distort subtle inter-feature physical correlations ($r=0.920$). |

---

## 4. Implications for Anomaly Detection Modeling

1. **Avoid Single-Gaussian Unimodal Models:**  
   Simple unimodal Gaussian estimators (e.g., EllipticEnvelope / Mahalanobis distance with a single mean vector) will struggle because:
   - They cannot accommodate the bimodal distribution of `cache-references` and `LLC-loads`.
   - The valley between the two modes would either be penalized as false positives or the covariance ellipsoid would be artificially dilated.

2. **Top Recommended Unsupervised Detector Candidates:**
   - **Gaussian Mixture Models (GMM) ($k=2$ to $k=3$):**  
     Natively captures the distinct modes observed in the cache features and outputs calibrated negative log-likelihood scores.
   - **Isolation Forest (iForest):**  
     Tree-based recursive partitioning does not make parametric distribution assumptions, easily encapsulates multiple clusters, and remains robust to the heavy tail in `cycles`.
   - **Autoencoder (Reconstruction Error / PyTorch):**  
     Because clean data lies on a tight 1D/2D manifold ($r=0.920$), an autoencoder (e.g., bottleneck dimension = 1 or 2) trained on clean traces will learn to accurately reconstruct normal clean points. Backdoor points that violate the correlated structure will yield high reconstruction errors ($MSE$ or Huber loss).
   - **Local Outlier Factor (LOF) / $k$-NN Distance:**  
     Non-parametric local density estimators adapt seamlessly to clusters with differing local densities and detect isolated points in low-density valleys.
   - **PCA / Robust Mahalanobis Decomposition:**  
     Reconstruction error from the top principal component (which captures $>96\%$ of variance along the cache/LLC axis) will immediately isolate backdoor triggers that cause disproportionate cache misses or cycle stalls.
