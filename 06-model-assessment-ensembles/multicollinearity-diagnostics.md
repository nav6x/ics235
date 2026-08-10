# Multicollinearity Diagnostics & Variance Inflation Factor

[← Back to Course README](../README.md)

- [1. Multicollinearity Fundamentals](#1-multicollinearity-fundamentals)
- [2. Variance Inflation Factor (VIF)](#2-variance-inflation-factor-vif)
- [3. Condition Index & Eigenvalue Diagnostics](#3-condition-index-eigenvalue-diagnostics)
- [4. Remediation Strategies](#4-remediation-strategies)
- [5. Python Implementation](#5-python-implementation)
- [6. Industry Case Study & Solved Exercises](#6-industry-case-study-solved-exercises)
  - [Question 1: VIF Calculation from R^2](#question-1-vif-calculation-from-r2)
  - [Question 2: Condition Index Calculation](#question-2-condition-index-calculation)
  - [Question 3: Standard Error Inflation Factor](#question-3-standard-error-inflation-factor)
  - [Question 4: Iterative VIF Pruner Function](#question-4-iterative-vif-pruner-function)
  - [Question 5: Ridge Regularization Weight Stabilization](#question-5-ridge-regularization-weight-stabilization)
> **Topic**: Multicollinearity Diagnostics & Variance Inflation Factor: 1. Multicollinearity Fundamentals, 2. Variance Inflation Factor (VIF), 3. Condition Index & Eigenvalue Diagnostics, 4. Remediation Strategies

---

## 1. Multicollinearity Fundamentals

Multicollinearity occurs when predictor variables in regression are highly linearly correlated, inflating weight variance $\text{Var}(\hat{\beta}_j) = \sigma^2 (X^T X)^{-1}_{jj}$.

---

## 2. Variance Inflation Factor (VIF)

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is $R^2$ score from regressing feature $X_j$ against all other features.
* $\text{VIF}_j > 5.0$: High collinearity.

---

## 3. Condition Index & Eigenvalue Diagnostics

Condition index $\kappa = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}$ evaluates matrix condition number. $\kappa > 30$ indicates severe collinearity.

---

## 4. Remediation Strategies

1. Feature elimination via iterative VIF pruning.
2. Principal Component Analysis (PCA).
3. Ridge L2 regularization.

---

## 5. Python Implementation

```python
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create dataset with collinearity
np.random.seed(42)
x1 = np.random.rand(100)
x2 = x1 * 0.98 + np.random.rand(100) * 0.02
x3 = np.random.rand(100)

df_vif = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
vifs = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
for c, v in zip(df_vif.columns, vifs):
    print(f"Feature {c} VIF: {v:.2f}")
```

---

## 6. Industry Case Study & Solved Exercises

### Question 1: VIF Calculation from R^2
**Problem**: Auxiliary regression $R_j^2 = 0.95$. Compute VIF.
```python
# Solution
r2_j = 0.95
vif_val = 1.0 / (1.0 - r2_j)
print(f"Question 1 -> VIF: {vif_val:.2f}") # 20.00
```

### Question 2: Condition Index Calculation
**Problem**: Eigenvalues are $\lambda_{\max} = 90.0, \lambda_{\min} = 0.1$. Compute Condition Index.
```python
# Solution
import math
cond_idx = math.sqrt(90.0 / 0.1)
print(f"Question 2 -> Condition Index: {cond_idx:.2f}") # sqrt(900) = 30.00
```

### Question 3: Standard Error Inflation Factor
**Problem**: For VIF = 25.0, calculate standard error inflation factor $\sqrt{\text{VIF}}$.
```python
# Solution
print(f"Question 3 -> SE Inflation Factor: {math.sqrt(25.0):.1f}x") # 5.0x
```

### Question 4: Iterative VIF Pruner Function
**Problem**: Prune features with VIF > 5.0 iteratively.
```python
# Solution
def prune(df_in):
    curr = df_in.copy()
    while True:
        v_list = [variance_inflation_factor(curr.values, i) for i in range(curr.shape[1])]
        if max(v_list) > 5.0:
            curr = curr.drop(columns=[curr.columns[np.argmax(v_list)]])
        else: break
    return curr

print(f"Question 4 -> Remaining Columns: {list(prune(df_vif).columns)}")
```

### Question 5: Ridge Regularization Weight Stabilization
**Problem**: Compare weight magnitudes of LinearRegression vs Ridge on collinear inputs.
```python
# Solution
from sklearn.linear_model import LinearRegression, Ridge
X_c = np.column_stack([x1, x2])
y_c = 2*x1 + 3*x2 + np.random.normal(0, 0.1, 100)

print(f"Question 5 -> OLS Coefs: {LinearRegression().fit(X_c, y_c).coef_}")
print(f"Ridge Coefs: {Ridge(alpha=5.0).fit(X_c, y_c).coef_}")
```
