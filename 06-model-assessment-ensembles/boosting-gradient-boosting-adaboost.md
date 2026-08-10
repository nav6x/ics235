# Boosting Algorithms & Heterogeneous Stacking Ensembles

[← Back to Course README](../README.md)

- [1. Sequential Boosting Principles](#1-sequential-boosting-principles)
- [2. Adaptive Boosting (AdaBoost)](#2-adaptive-boosting-adaboost)
- [3. Gradient Boosting Decision Trees (GBDT)](#3-gradient-boosting-decision-trees-gbdt)
- [4. Heterogeneous Ensembles: Voting & Stacking](#4-heterogeneous-ensembles-voting-stacking)
- [5. Scikit-Learn Python Implementation](#5-scikit-learn-python-implementation)
- [6. Industry Case Study & Solved Exercises](#6-industry-case-study-solved-exercises)
  - [Question 1: AdaBoost Alpha Weight](#question-1-adaboost-alpha-weight)
  - [Question 2: GBDT Residual Calculation](#question-2-gbdt-residual-calculation)
  - [Question 3: Soft Voting Probability](#question-3-soft-voting-probability)
  - [Question 4: AdaBoost Sample Weight Update](#question-4-adaboost-sample-weight-update)
  - [Question 5: GBDT Learning Rate Comparison](#question-5-gbdt-learning-rate-comparison)
> **Topic**: Boosting Algorithms & Heterogeneous Stacking Ensembles: 1. Sequential Boosting Principles, 2. Adaptive Boosting (AdaBoost), 3. Gradient Boosting Decision Trees (GBDT), 4. Heterogeneous Ensembles: Voting & Stacking

---

## 1. Sequential Boosting Principles

Fits estimators sequentially to focus on previously misclassified or high-residual samples.

---

## 2. Adaptive Boosting (AdaBoost)

Weak estimator weight $\alpha_m$:

$$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)$$

Sample weight update:

$$w_i^{(m+1)} = w_i^{(m)} \exp\left( -\alpha_m y_i h_m(\mathbf{x}_i) \right)$$

---

## 3. Gradient Boosting Decision Trees (GBDT)

Fits base estimators $h_m(\mathbf{x})$ to negative gradient pseudo-residuals:

$$r_{im} = - \left[ \frac{\partial L(y_i, f(\mathbf{x}_i))}{\partial f(\mathbf{x}_i)} \right]_{f=f_{m-1}}$$

---

## 4. Heterogeneous Ensembles: Voting & Stacking

* **Voting Classifier**: Combines predictions from diverse models via hard majority or soft probability averaging.
* **Stacking**: Trains meta-estimator on out-of-fold predictions of base estimators.

---

## 5. Scikit-Learn Python Implementation

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
estimators = [('dt', DecisionTreeClassifier(max_depth=2)), ('knn', KNeighborsClassifier(n_neighbors=3))]
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()).fit(X, y)
print(f"Stacking Ensemble Accuracy: {stack.score(X, y):.4f}")
```

---

## 6. Industry Case Study & Solved Exercises

### Question 1: AdaBoost Alpha Weight
**Problem**: Compute $\alpha_m$ for weak estimator error $\epsilon_m = 0.20$.
```python
# Solution
import math
eps = 0.20
alpha = 0.5 * math.log((1 - eps) / eps)
print(f"Question 1 -> AdaBoost Estimator Weight Alpha: {alpha:.4f}") # 0.5 * ln(4) = 0.6931
```

### Question 2: GBDT Residual Calculation
**Problem**: Target $y = 20.0$, previous prediction $f_{m-1}(x) = 16.0$. Compute MSE residual.
```python
# Solution
print(f"Question 2 -> GBDT Pseudo-Residual: {20.0 - 16.0:.2f}") # 4.00
```

### Question 3: Soft Voting Probability
**Problem**: Model probabilities are `[0.70, 0.80, 0.90]`. Compute soft voting prediction.
```python
# Solution
probs = [0.70, 0.80, 0.90]
avg_p = sum(probs) / len(probs)
print(f"Question 3 -> Soft Voting Prob: {avg_p:.4f} -> Class: {1 if avg_p >= 0.5 else 0}")
```

### Question 4: AdaBoost Sample Weight Update
**Problem**: Initial weight $w_i = 0.10, \alpha_1 = 0.6931, y_i = 1, h(x_i) = -1$ (Misclassified). Compute updated unnormalized weight $w_i'$.
```python
# Solution
w_old, alpha_1, y_i, h_i = 0.10, 0.6931, 1, -1
w_new = w_old * math.exp(-alpha_1 * y_i * h_i) # 0.10 * exp(0.6931) = 0.20
print(f"Question 4 -> Updated Sample Weight: {w_new:.4f}") # 0.2000
```

### Question 5: GBDT Learning Rate Comparison
**Problem**: Compare GBDT accuracy with learning rate 0.1 vs 1.0.
```python
# Solution
gb1 = GradientBoostingClassifier(learning_rate=0.1, random_state=42).fit(X, y)
gb2 = GradientBoostingClassifier(learning_rate=1.0, random_state=42).fit(X, y)
print(f"Question 5 -> LR 0.1 Acc: {gb1.score(X, y):.4f}, LR 1.0 Acc: {gb2.score(X, y):.4f}")
```
