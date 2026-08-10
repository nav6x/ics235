# Bagging & Random Forests Ensemble Architecture

[← Back to Course README](../README.md)

- [1. Ensemble Variance Reduction Principles](#1-ensemble-variance-reduction-principles)
- [2. Bootstrap Aggregating (Bagging)](#2-bootstrap-aggregating-bagging)
- [3. Out-of-Bag (OOB) Error Estimation](#3-out-of-bag-oob-error-estimation)
- [4. Random Forests & Feature Subspace Sampling](#4-random-forests-feature-subspace-sampling)
- [5. Scikit-Learn Python Implementation](#5-scikit-learn-python-implementation)
- [6. Industry Case Study & Solved Exercises](#6-industry-case-study-solved-exercises)
  - [Question 1: Bootstrap Inclusion Probability](#question-1-bootstrap-inclusion-probability)
  - [Question 2: Ensemble Variance Reduction](#question-2-ensemble-variance-reduction)
  - [Question 3: Classification Subspace Size](#question-3-classification-subspace-size)
  - [Question 4: Extract Feature Importances](#question-4-extract-feature-importances)
  - [Question 5: Tree vs Forest Generalization](#question-5-tree-vs-forest-generalization)
> **Topic**: Bagging & Random Forests Ensemble Architecture: 1. Ensemble Variance Reduction Principles, 2. Bootstrap Aggregating (Bagging), 3. Out-of-Bag (OOB) Error Estimation, 4. Random Forests & Feature Subspace Sampling

---

## 1. Ensemble Variance Reduction Principles

For $B$ estimators with individual variance $\sigma^2$ and pairwise correlation $\rho$:

$$\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

---

## 2. Bootstrap Aggregating (Bagging)

Generates $B$ bootstrap samples $\mathcal{D}_b$ by sampling $n$ items uniformly with replacement.

---

## 3. Out-of-Bag (OOB) Error Estimation

Probability of omitting sample $i$ in bootstrap sample of size $n$:

$$\lim_{n \to \infty} \left( 1 - \frac{1}{n} \right)^n = e^{-1} \approx 0.3679$$

OOB samples (~36.8%) provide unbiased validation error without cross-validation.

---

## 4. Random Forests & Feature Subspace Sampling

Randomly samples feature subset of size $m$ at each split to decorrelate trees ($\rho \to 0$):
* Classification: $m = \lfloor \sqrt{d} \rfloor$
* Regression: $m = \lfloor d / 3 \rfloor$

---

## 5. Scikit-Learn Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, random_state=42).fit(X, y)
print(f"Random Forest OOB Score: {rf.oob_score_:.4f}")
```

---

## 6. Industry Case Study & Solved Exercises

### Question 1: Bootstrap Inclusion Probability
**Problem**: Calculate probability that a specific item is included in bootstrap sample size $n \to \infty$.
```python
# Solution
import math
p_in = 1.0 - (1.0 / math.e)
print(f"Question 1 -> Bootstrap Inclusion Prob: {p_in:.4f}") # ~0.6321
```

### Question 2: Ensemble Variance Reduction
**Problem**: Compute ensemble variance for $B=100$, $\sigma^2=16$, and tree correlation $\rho=0.25$.
```python
# Solution
B, sig2, rho = 100, 16.0, 0.25
v_ens = rho * sig2 + ((1 - rho) / B) * sig2
print(f"Question 2 -> Ensemble Variance: {v_ens:.4f}") # 4.0 + 0.12 = 4.1200
```

### Question 3: Classification Subspace Size
**Problem**: Compute classification subspace size $m = \sqrt{d}$ for $d = 64$ features.
```python
# Solution
d = 64
print(f"Question 3 -> Feature Subspace Size m: {math.floor(math.sqrt(d))}") # 8
```

### Question 4: Extract Feature Importances
**Problem**: Print top 3 feature importances from Random Forest model.
```python
# Solution
import numpy as np
imp = rf.feature_importances_
print(f"Question 4 -> Top 3 Feature Indices: {np.argsort(imp)[::-1][:3]}")
```

### Question 5: Tree vs Forest Generalization
**Problem**: Compare single DecisionTree vs RandomForest R^2 score.
```python
# Solution
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X_r, y_r = make_regression(n_samples=200, n_features=10, noise=5.0, random_state=42)
dt_s = DecisionTreeRegressor(random_state=42).fit(X_r, y_r).score(X_r, y_r)
rf_s = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_r, y_r).score(X_r, y_r)
print(f"Question 5 -> Single Tree Score: {dt_s:.4f}, Forest Score: {rf_s:.4f}")
```
