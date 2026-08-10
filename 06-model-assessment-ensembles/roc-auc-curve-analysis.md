# ROC Curve & AUC Score Analysis

[← Back to Course README](../README.md)

- [1. Receiver Operating Characteristic (ROC) Curve](#1-receiver-operating-characteristic-roc-curve)
- [2. Area Under the Curve (ROC-AUC)](#2-area-under-the-curve-roc-auc)
- [3. Decision Threshold Optimization](#3-decision-threshold-optimization)
- [4. Scikit-Learn Python Implementation](#4-scikit-learn-python-implementation)
- [5. Industry Case Study & Solved Exercises](#5-industry-case-study-solved-exercises)
  - [Question 1: ROC-AUC Calculation](#question-1-roc-auc-calculation)
  - [Question 2: Youden's J Threshold](#question-2-youdens-j-threshold)
  - [Question 3: Random Predictor ROC-AUC](#question-3-random-predictor-roc-auc)
  - [Question 4: PR-AUC for Imbalanced Class](#question-4-pr-auc-for-imbalanced-class)
  - [Question 5: Multi-Class ROC-AUC (OVR)](#question-5-multi-class-roc-auc-ovr)
> **Topic**: ROC Curve & AUC Score Analysis: 1. Receiver Operating Characteristic (ROC) Curve, 2. Area Under the Curve (ROC-AUC), 3. Decision Threshold Optimization, 4. Scikit-Learn Python Implementation

---

## 1. Receiver Operating Characteristic (ROC) Curve

Plots Sensitivity (True Positive Rate) against 1 - Specificity (False Positive Rate) across decision thresholds $\tau \in [0, 1]$:

$$\text{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}, \quad \text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$$

---

## 2. Area Under the Curve (ROC-AUC)

ROC-AUC quantifies probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample:

$$\text{AUC} = P(\hat{p}(\mathbf{x}_+) > \hat{p}(\mathbf{x}_-))$$

---

## 3. Decision Threshold Optimization

Youden's J Statistic identifies optimal decision threshold $\tau^*$:

$$J(\tau) = \text{TPR}(\tau) - \text{FPR}(\tau)$$

---

## 4. Scikit-Learn Python Implementation

```python
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

y_true = np.array([1, 1, 0, 0, 1, 0])
y_scores = np.array([0.95, 0.85, 0.60, 0.40, 0.30, 0.10])

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc_val = roc_auc_score(y_true, y_scores)
print(f"ROC-AUC Performance Score: {auc_val:.4f}")
```

---

## 5. Industry Case Study & Solved Exercises

### Question 1: ROC-AUC Calculation
**Problem**: Compute ROC-AUC for ground truth `[1, 1, 0, 0]` and scores `[0.9, 0.7, 0.3, 0.1]`.
```python
# Solution
print(f"Question 1 -> ROC-AUC Score: {roc_auc_score([1, 1, 0, 0], [0.9, 0.7, 0.3, 0.1]):.4f}") # 1.0000
```

### Question 2: Youden's J Threshold
**Problem**: Select threshold maximizing $J = \text{TPR} - \text{FPR}$ for `tpr = [0, 0.9, 1.0]` and `fpr = [0, 0.1, 0.5]`.
```python
# Solution
tpr = np.array([0, 0.9, 1.0])
fpr = np.array([0, 0.1, 0.5])
thresholds = np.array([1.0, 0.5, 0.0])

j_scores = tpr - fpr # [0, 0.8, 0.5]
opt_t = thresholds[np.argmax(j_scores)]
print(f"Question 2 -> Optimal Threshold: {opt_t:.2f}") # 0.50
```

### Question 3: Random Predictor ROC-AUC
**Problem**: Calculate expected ROC-AUC for random guessing classifier.
```python
# Solution
print("Question 3 -> Random Guessing ROC-AUC: 0.5000")
```

### Question 4: PR-AUC for Imbalanced Class
**Problem**: Compute Average Precision (PR-AUC) using `average_precision_score`.
```python
# Solution
from sklearn.metrics import average_precision_score
print(f"Question 4 -> PR-AUC Score: {average_precision_score([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3]):.4f}") # 1.0000
```

### Question 5: Multi-Class ROC-AUC (OVR)
**Problem**: Compute One-vs-Rest ROC-AUC for 3 classes.
```python
# Solution
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
probs = LogisticRegression(max_iter=200).fit(X, y).predict_proba(X)
print(f"Question 5 -> Multiclass OVR ROC-AUC: {roc_auc_score(y, probs, multi_class='ovr'):.4f}")
```
