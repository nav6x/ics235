# Data Cleaning, Missing Imputation, Scaling & Preprocessing

[← Back to Course README](../README.md)

- [1. Identifying & Categorizing Missing Values](#1-identifying-categorizing-missing-values)
  - [1.1 Missing Completely at Random (MCAR)](#11-missing-completely-at-random-mcar)
  - [1.2 Missing at Random (MAR)](#12-missing-at-random-mar)
  - [1.3 Missing Not at Random (MNAR)](#13-missing-not-at-random-mnar)
- [2. Missing Value Imputation Strategies](#2-missing-value-imputation-strategies)
  - [2.1 Listwise & Pairwise Deletion](#21-listwise-pairwise-deletion)
  - [2.2 Mean, Median & Mode Imputation](#22-mean-median-mode-imputation)
  - [2.3 K-Nearest Neighbors (KNN) Imputation](#23-k-nearest-neighbors-knn-imputation)
- [3. Outlier Detection & Treatment Methods](#3-outlier-detection-treatment-methods)
  - [3.1 Parametric Z-Score Method](#31-parametric-z-score-method)
  - [3.2 Non-Parametric Interquartile Range (IQR) Rule](#32-non-parametric-interquartile-range-iqr-rule)
- [4. Data Transformation & Feature Scaling](#4-data-transformation-feature-scaling)
  - [4.1 Min-Max Normalization](#41-min-max-normalization)
  - [4.2 Z-Score Standardization](#42-z-score-standardization)
  - [4.3 Robust Scaling](#43-robust-scaling)
- [5. Categorical Encoding & Deduplication Engine](#5-categorical-encoding-deduplication-engine)
- [6. Python Mechanics: End-to-End Cleaning Pipeline](#6-python-mechanics-end-to-end-cleaning-pipeline)
- [7. 5 Solved Practice & Analytical Numerical Questions](#7-5-solved-practice-analytical-numerical-questions)
  - [Question 1: Listwise Deletion vs Mean Imputation Variance](#question-1-listwise-deletion-vs-mean-imputation-variance)
  - [Question 2: Parametric Z-Score Outlier Flagging](#question-2-parametric-z-score-outlier-flagging)
  - [Question 3: Non-Parametric IQR Outer Fences](#question-3-non-parametric-iqr-outer-fences)
  - [Question 4: Min-Max Normalization Calculation](#question-4-min-max-normalization-calculation)
  - [Question 5: Z-Score Standardization Calculation](#question-5-z-score-standardization-calculation)
> **Topic**: Data Cleaning, Missing Imputation, Scaling & Preprocessing: 1. Identifying & Categorizing Missing Values, 2. Missing Value Imputation Strategies, 3. Outlier Detection & Treatment Methods, 4. Data Transformation & Feature Scaling

---

## 1. Identifying & Categorizing Missing Values

Let dataset matrix be $X = (X_{\text{obs}}, X_{\text{mis}})$ and indicator matrix $M_{ij} = 1$ if $X_{ij}$ is missing, 0 otherwise.

### 1.1 Missing Completely at Random (MCAR)
Missingness mechanism is independent of observed and unobserved data:

$$P(M \mid X_{\text{obs}}, X_{\text{mis}}) = P(M)$$

### 1.2 Missing at Random (MAR)
Missingness depends on observed features $X_{\text{obs}}$, but not unobserved values $X_{\text{mis}}$:

$$P(M \mid X_{\text{obs}}, X_{\text{mis}}) = P(M \mid X_{\text{obs}})$$

### 1.3 Missing Not at Random (MNAR)
Missingness depends directly on missing values $X_{\text{mis}}$:

$$P(M \mid X_{\text{obs}}, X_{\text{mis}}) \neq P(M \mid X_{\text{obs}})$$

---

## 2. Missing Value Imputation Strategies

### 2.1 Listwise & Pairwise Deletion
* **Listwise Deletion**: Removes entire record if any feature is missing. Unbiased only under MCAR; reduces sample size $n$.
* **Pairwise Deletion**: Uses available feature pairs for specific covariance calculations.

### 2.2 Mean, Median & Mode Imputation
Replaces missing value $x_{ij}$ with feature summary statistic:

$$x_{ij}^{\text{imp}} = \mu_j \quad (\text{Mean}), \quad \tilde{x}_j \quad (\text{Median})$$

### 2.3 K-Nearest Neighbors (KNN) Imputation
Identifies $K$ nearest neighbors using Gower's or Euclidean distance over observed features:

$$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\frac{1}{|O_{ij}|} \sum_{k \in O_{ij}} (x_{ik} - x_{jk})^2}$$

$$x_{im}^{\text{imp}} = \frac{\sum_{j \in \mathcal{N}_K(\mathbf{x}_i)} w_{ij} x_{jm}}{\sum_{j \in \mathcal{N}_K(\mathbf{x}_i)} w_{ij}}$$

---

## 3. Outlier Detection & Treatment Methods

### 3.1 Parametric Z-Score Method
For Gaussian feature $X \sim \mathcal{N}(\mu, \sigma^2)$, point $x_i$ is an outlier if:

$$|Z_i| = \left| \frac{x_i - \mu}{\sigma} \right| > 3.0$$

### 3.2 Non-Parametric Interquartile Range (IQR) Rule
Defines inner fence bounds $[Q_1 - 1.5 \cdot \text{IQR}, Q_3 + 1.5 \cdot \text{IQR}]$ where $\text{IQR} = Q_3 - Q_1$:

$$\text{Outlier}(x_i) = \mathbb{I}(x_i < Q_1 - 1.5 \cdot \text{IQR} \quad \lor \quad x_i > Q_3 + 1.5 \cdot \text{IQR})$$

---

## 4. Data Transformation & Feature Scaling

### 4.1 Min-Max Normalization
Rescales feature values into specified range $[a, b]$ (typically $[0, 1]$):

$$x_{\text{scaled}} = a + \frac{(x - x_{\min})(b - a)}{x_{\max} - x_{\min}}$$

### 4.2 Z-Score Standardization
Transforms feature distribution to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

### 4.3 Robust Scaling
Uses median and IQR to scale features containing extreme outliers:

$$x_{\text{robust}} = \frac{x - \text{Median}}{\text{IQR}}$$

---

## 5. Categorical Encoding & Deduplication Engine

One-Hot Encoding converts categorical variable with $K$ states into $K$ binary features $\mathbf{e}_k \in \{0, 1\}$. To prevent dummy variable multicollinearity in linear models, drop one category ($K-1$ encoding).

---

## 6. Python Mechanics: End-to-End Cleaning Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import RobustScaler

# Generate Raw Dirty Dataset
np.random.seed(42)
n = 100

data = {
    'age': np.random.choice([25, 30, 45, np.nan, 35, 120, 28], size=n),
    'income': np.random.choice([50000, 65000, 90000, np.nan, 120000, 1500000], size=n),
    'education': np.random.choice(['BS', 'MS', 'PhD', np.nan], size=n)
}

df_dirty = pd.DataFrame(data)

# Pipeline Preprocessing
num_features = ['age', 'income']
cat_features = ['education']

num_pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=3)),
    ('scaler', RobustScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

X_clean = preprocessor.fit_transform(df_dirty)
print(f"Clean Transformed Data Shape: {X_clean.shape}")
```

---

## 7. 5 Solved Practice & Analytical Numerical Questions

### Question 1: Listwise Deletion vs Mean Imputation Variance
**Problem**: Feature values are `[10, 20, np.nan, 30, 40]`. Compute sample variance before deletion (on non-NaNs) and after mean imputation.
```python
# Solution
import numpy as np
import pandas as pd

s_raw = pd.Series([10, 20, np.nan, 30, 40])
s_clean = s_raw.dropna()
var_clean = s_clean.var(ddof=1) # Variance on complete cases

mean_val = s_clean.mean() # 25.0
s_imp = s_raw.fillna(mean_val)
var_imp = s_imp.var(ddof=1)

print(f"Question 1 -> Variance Complete Cases: {var_clean:.2f}, Variance Mean Imputed: {var_imp:.2f}")
# Complete case var = 166.67, Mean imputed var = 125.00 (Mean imputation under-estimates variance!)
```

### Question 2: Parametric Z-Score Outlier Flagging
**Problem**: Given sample `[12, 14, 15, 13, 12, 14, 95]`, compute sample mean $\mu$, sample std $\sigma$, and flag values with $|Z| > 2.0$.
```python
# Solution
data = np.array([12, 14, 15, 13, 12, 14, 95])
mu = np.mean(data)
sigma = np.std(data, ddof=1)

z_scores = np.abs((data - mu) / sigma)
outliers = data[z_scores > 2.0]
print(f"Question 2 -> Mean: {mu:.2f}, Std: {sigma:.2f}, Outliers (|Z|>2): {outliers}")
```

### Question 3: Non-Parametric IQR Outer Fences
**Problem**: Compute $Q_1$, $Q_3$, IQR, and outer bounds for dataset `[5, 18, 20, 22, 25, 28, 30, 85]`.
```python
# Solution
s_iqr = pd.Series([5, 18, 20, 22, 25, 28, 30, 85])
q1 = s_iqr.quantile(0.25)
q3 = s_iqr.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers_iqr = s_iqr[(s_iqr < lower_bound) | (s_iqr > upper_bound)].values
print(f"Question 3 -> Q1: {q1:.1f}, Q3: {q3:.1f}, IQR: {iqr:.1f}, Outliers: {outliers_iqr}")
```

### Question 4: Min-Max Normalization Calculation
**Problem**: Scale $x = 45$ into range $[0, 1]$ given $x_{\min} = 10, x_{\max} = 60$.
```python
# Solution
x, x_min, x_max = 45, 10, 60
x_norm = (x - x_min) / (x_max - x_min)
print(f"Question 4 -> Min-Max Scaled Value: {x_norm:.4f}") # 35/50 = 0.7000
```

### Question 5: Z-Score Standardization Calculation
**Problem**: Standardize value $x = 140$ given sample mean $\mu = 100$ and sample standard deviation $\sigma = 20$.
```python
# Solution
x, mu, sigma = 140, 100, 20
z_std = (x - mu) / sigma
print(f"Question 5 -> Standardized Value Z: {z_std:.2f}") # (140 - 100)/20 = 2.00
```
