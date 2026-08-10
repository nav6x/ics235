# Correlation Analysis & Bivariate Dependency Metrics

[← Back to Course README](../README.md)

- [1. Mathematical Formulation of Pearson Correlation](#1-mathematical-formulation-of-pearson-correlation)
- [2. Spearman Rank Correlation Coefficient](#2-spearman-rank-correlation-coefficient)
- [3. Kendall's Tau Rank Correlation](#3-kendalls-tau-rank-correlation)
- [4. Mutual Information & Non-Linear Dependency](#4-mutual-information-non-linear-dependency)
- [5. Multicollinearity Detection & Matrix Filtering](#5-multicollinearity-detection-matrix-filtering)
- [6. Python Mechanics: Correlation Matrix Suite](#6-python-mechanics-correlation-matrix-suite)
- [7. 5 Solved Analytical & Numerical Practice Questions](#7-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Manual Pearson Correlation Derivation](#question-1-manual-pearson-correlation-derivation)
  - [Question 2: Spearman Rank Formula Computation](#question-2-spearman-rank-formula-computation)
  - [Question 3: Kendall's Tau Concordance Calculation](#question-3-kendalls-tau-concordance-calculation)
  - [Question 4: P-Value Test for Pearson Significance](#question-4-p-value-test-for-pearson-significance)
  - [Question 5: Automated Correlation Matrix Filter](#question-5-automated-correlation-matrix-filter)
> **Topic**: Correlation Analysis & Bivariate Dependency Metrics: 1. Mathematical Formulation of Pearson Correlation, 2. Spearman Rank Correlation Coefficient, 3. Kendall's Tau Rank Correlation, 4. Mutual Information & Non-Linear Dependency

---

## 1. Mathematical Formulation of Pearson Correlation

Pearson product-moment correlation coefficient $r_{XY}$ measures strength and direction of linear relationship between continuous random variables $X$ and $Y$:

$$r_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

Where $r_{XY} \in [-1, +1]$.

---

## 2. Spearman Rank Correlation Coefficient

Evaluates monotonic relationships by computing Pearson correlation over sample ranks $R(X)$ and $R(Y)$:

$$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}$$

Where $d_i = R(x_i) - R(y_i)$ is difference between ranks of sample $i$.

---

## 3. Kendall's Tau Rank Correlation

Measures ordinal association based on concordant and discordant pairs:

$$\tau = \frac{C - D}{\frac{1}{2} n(n - 1)}$$

Where $C$ is number of concordant pairs ($x_i > x_j$ and $y_i > y_j$) and $D$ is number of discordant pairs ($x_i > x_j$ and $y_i < y_j$).

---

## 4. Mutual Information & Non-Linear Dependency

Quantifies mutual dependence between variables using information theory:

$$I(X; Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log\left( \frac{p(x, y)}{p(x) p(y)} \right) = H(X) + H(Y) - H(X, Y)$$

Where $I(X; Y) = 0$ if and only if $X$ and $Y$ are strictly independent.

---

## 5. Multicollinearity Detection & Matrix Filtering

Extremely high correlations ($|r| > 0.85$) indicate redundant features. Automated correlation matrix filtering prunes collinear columns.

---

## 6. Python Mechanics: Correlation Matrix Suite

```python
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression

# Generate Non-Linear Quadratic Dataset
np.random.seed(42)
X = np.linspace(-3, 3, 200)
Y_quad = X ** 2 + np.random.normal(0, 0.5, 200)

r_pear, p_pear = pearsonr(X, Y_quad)
rho_spear, _ = spearmanr(X, Y_quad)
tau_kend, _ = kendalltau(X, Y_quad)
mi_score = mutual_info_regression(X.reshape(-1, 1), Y_quad, random_state=42)[0]

print(f"Pearson r:      {r_pear:.4f} (Fails on symmetric non-linear)")
print(f"Spearman rho:   {rho_spear:.4f}")
print(f"Kendall tau:    {tau_kend:.4f}")
print(f"Mutual Info:    {mi_score:.4f} (Detects non-linear interaction!)")
```

---

## 7. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Manual Pearson Correlation Derivation
**Problem**: Calculate Pearson correlation $r_{XY}$ for points $(1, 2), (2, 5), (3, 8)$.
```python
# Solution
import numpy as np

x = np.array([1, 2, 3])
y = np.array([2, 5, 8])

cov_xy = np.mean((x - np.mean(x)) * (y - np.mean(y)))
std_x = np.std(x)
std_y = np.std(y)

r_calc = cov_xy / (std_x * std_y)
print(f"Question 1 -> Pearson r: {r_calc:.4f}") # 1.0000
```

### Question 2: Spearman Rank Formula Computation
**Problem**: Ranks for 4 samples are $R(X) = [1, 2, 3, 4]$ and $R(Y) = [2, 1, 4, 3]$. Compute Spearman $\rho$.
```python
# Solution
rx = np.array([1, 2, 3, 4])
ry = np.array([2, 1, 4, 3])
d = rx - ry # [-1, 1, -1, 1]
d_sq_sum = np.sum(d**2) # 4

n = 4
rho_val = 1.0 - (6.0 * d_sq_sum) / (n * (n**2 - 1))
print(f"Question 2 -> Spearman rho: {rho_val:.4f}") # 1 - 24/60 = 0.6000
```

### Question 3: Kendall's Tau Concordance Calculation
**Problem**: Given pairs $(1, 10), (2, 20), (3, 15)$, count concordant ($C$) and discordant ($D$) pairs and compute $\tau$.
```python
# Solution
# Pairs: P1(1,10), P2(2,20), P3(3,15)
# (P1, P2): x1<x2, y1<y2 -> Concordant
# (P1, P3): x1<x3, y1<y3 -> Concordant
# (P2, P3): x2<x3, y2>y3 -> Discordant
C, D = 2, 1
n = 3
tau_val = (C - D) / (0.5 * n * (n - 1))
print(f"Question 3 -> Kendall tau: {tau_val:.4f}") # (2 - 1)/3 = 0.3333
```

### Question 4: P-Value Test for Pearson Significance
**Problem**: Test if $r = 0.40$ over $n = 50$ samples is statistically significant at $\alpha = 0.05$.
```python
# Solution
from scipy.stats import pearsonr

np.random.seed(42)
x_sim = np.random.normal(0, 1, 50)
y_sim = 0.4 * x_sim + np.random.normal(0, 0.9, 50)

r_stat, p_val = pearsonr(x_sim, y_sim)
print(f"Question 4 -> r: {r_stat:.4f}, p-value: {p_val:.4e} -> Significant: {p_val < 0.05}")
```

### Question 5: Automated Correlation Matrix Filter
**Problem**: Drop feature pairs with $|r| > 0.85$ from DataFrame.
```python
# Solution
df_corr = pd.DataFrame({
    'f1': [1, 2, 3, 4, 5],
    'f2': [1.05, 2.01, 3.02, 4.01, 5.04], # Highly collinear with f1
    'f3': [9, 3, 1, 5, 2]
})

cm = df_corr.corr().abs()
upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.85)]
print(f"Question 5 -> Dropped Collinear Columns: {drop_cols}") # ['f2']
```
