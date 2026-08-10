# Confusion Matrix, Model Assessment & Regression Metrics

[← Back to Course README](../README.md)

- [1. Assessment Paradigms: Batch vs Rank-Ordered](#1-assessment-paradigms-batch-vs-rank-ordered)
- [2. Confusion Matrix & Classification Metrics](#2-confusion-matrix-classification-metrics)
- [3. Assessing Regression Models](#3-assessing-regression-models)
- [4. Scikit-Learn Python Implementation](#4-scikit-learn-python-implementation)
- [5. Industry Case Study & Solved Exercises](#5-industry-case-study-solved-exercises)
  - [Question 1: Precision, Recall & F1 Calculation](#question-1-precision-recall-f1-calculation)
  - [Question 2: R^2 Score Calculation](#question-2-r2-score-calculation)
  - [Question 3: MAE vs RMSE Calculation](#question-3-mae-vs-rmse-calculation)
  - [Question 4: Top Decile Lift Calculation](#question-4-top-decile-lift-calculation)
  - [Question 5: Multiclass Confusion Matrix](#question-5-multiclass-confusion-matrix)
> **Topic**: Confusion Matrix, Model Assessment & Regression Metrics: 1. Assessment Paradigms: Batch vs Rank-Ordered, 2. Confusion Matrix & Classification Metrics, 3. Assessing Regression Models, 4. Scikit-Learn Python Implementation

---

## 1. Assessment Paradigms: Batch vs Rank-Ordered

* **Batch Assessment**: Evaluates static predictions using fixed threshold metrics (Accuracy, F1-Score).
* **Rank-Ordered Assessment**: Ranks instances by predicted probability to evaluate Cumulative Gain and Lift Charts for targeted campaigns.

---

## 2. Confusion Matrix & Classification Metrics

| Actual \ Predicted | Predicted Positive (1) | Predicted Negative (0) |
| :--- | :--- | :--- |
| **Actual Positive (1)** | True Positive (TP) | False Negative (FN) |
| **Actual Negative (0)** | False Positive (FP) | True Negative (TN) |

* **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
* **Precision**: $\frac{TP}{TP + FP}$
* **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$
* **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

---

## 3. Assessing Regression Models

* **MSE**: $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$
* **RMSE**: $\sqrt{\text{MSE}}$
* **MAE**: $\frac{1}{n} \sum |y_i - \hat{y}_i|$
* **Coefficient of Determination ($R^2$)**:

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

---

## 4. Scikit-Learn Python Implementation

```python
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_squared_error
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])

print("Classification Report:")
print(classification_report(y_true, y_pred))
```

---

## 5. Industry Case Study & Solved Exercises

### Question 1: Precision, Recall & F1 Calculation
**Problem**: TP=90, FP=10, FN=20, TN=880. Compute Precision, Recall, and F1-Score.
```python
# Solution
tp, fp, fn, tn = 90, 10, 20, 880
prec = tp / (tp + fp) # 0.90
rec = tp / (tp + fn) # 90/110 = 0.8182
f1 = 2 * (prec * rec) / (prec + rec)

print(f"Question 1 -> Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
```

### Question 2: R^2 Score Calculation
**Problem**: Calculate $R^2$ score for actual values `[10, 20, 30]` and predictions `[11, 19, 31]`.
```python
# Solution
from sklearn.metrics import r2_score

y_t = np.array([10, 20, 30])
y_p = np.array([11, 19, 31])
print(f"Question 2 -> R^2 Score: {r2_score(y_t, y_p):.4f}") # ~0.9700
```

### Question 3: MAE vs RMSE Calculation
**Problem**: Errors are `[0, 2, 8]`. Compute MAE and RMSE.
```python
# Solution
errs = np.array([0, 2, 8])
mae = np.mean(np.abs(errs))
rmse = np.sqrt(np.mean(errs**2))
print(f"Question 3 -> MAE: {mae:.2f}, RMSE: {rmse:.2f}") # MAE=3.33, RMSE=4.76
```

### Question 4: Top Decile Lift Calculation
**Problem**: Overall churn rate is 10%. Top decile model selection yields 50% churn rate. Compute Lift score.
```python
# Solution
base_rate = 0.10
model_rate = 0.50
lift = model_rate / base_rate
print(f"Question 4 -> Top Decile Lift Score: {lift:.2f}x") # 5.0x
```

### Question 5: Multiclass Confusion Matrix
**Problem**: Generate multiclass confusion matrix using `confusion_matrix`.
```python
# Solution
cm = confusion_matrix([0, 1, 2, 0, 1, 2], [0, 1, 1, 0, 2, 2])
print("Question 5 -> Multiclass Confusion Matrix:")
print(cm)
```
