# Support Vector Machines (SVM), Margin Theory & Kernel Methods

[← Back to Course README](../README.md)

- [1. Maximum Margin Hyperplane & Geometric Margin](#1-maximum-margin-hyperplane-geometric-margin)
- [2. Hard-Margin SVM Optimization Derivation](#2-hard-margin-svm-optimization-derivation)
  - [2.1 Primal Optimization Formulation](#21-primal-optimization-formulation)
  - [2.2 Lagrangian & Dual Formulation via KKT Conditions](#22-lagrangian-dual-formulation-via-kkt-conditions)
- [3. Soft-Margin SVM & Slack Variables](#3-soft-margin-svm-slack-variables)
- [4. The Kernel Trick & Mercer's Theorem](#4-the-kernel-trick-mercers-theorem)
- [5. Scikit-Learn Python Implementation](#5-scikit-learn-python-implementation)
- [6. End-to-End Industry Case Study: Medical Image Diagnosis Engine](#6-end-to-end-industry-case-study-medical-image-diagnosis-engine)
- [7. 5 Solved Analytical & Numerical Practice Questions](#7-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Geometric Margin Width Calculation](#question-1-geometric-margin-width-calculation)
  - [Question 2: Weight Vector Calculation from Dual Multipliers](#question-2-weight-vector-calculation-from-dual-multipliers)
  - [Question 3: RBF Kernel Calculation](#question-3-rbf-kernel-calculation)
  - [Question 4: Linear Hyperplane Decision Value](#question-4-linear-hyperplane-decision-value)
  - [Question 5: Polynomial Kernel Inner Product](#question-5-polynomial-kernel-inner-product)
> **Topic**: Support Vector Machines (SVM), Margin Theory & Kernel Methods: 1. Maximum Margin Hyperplane & Geometric Margin, 2. Hard-Margin SVM Optimization Derivation, 3. Soft-Margin SVM & Slack Variables, 4. The Kernel Trick & Mercer's Theorem

---

## 1. Maximum Margin Hyperplane & Geometric Margin

Support Vector Machines construct optimal decision hyperplanes $\mathbf{w}^T \mathbf{x} + b = 0$ maximizing functional and geometric margin $\gamma$:

$$\text{Geometric Margin } \gamma = \frac{2}{\|\mathbf{w}\|_2}$$

---

## 2. Hard-Margin SVM Optimization Derivation

### 2.1 Primal Optimization Formulation
Minimizes squared weight norm subject to strict zero classification error constraints:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad \forall i=1, \dots, n$$

### 2.2 Lagrangian & Dual Formulation via KKT Conditions
Formulating Lagrangian function with multipliers $\alpha_i \ge 0$:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} \|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i \left[ y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1 \right]$$

Setting stationarity partial derivatives to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = \mathbf{0} \implies \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0$$

Substituting back yields **Dual Optimization Problem**:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j (\mathbf{x}_i^T \mathbf{x}_j) \quad \text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \ge 0$$

---

## 3. Soft-Margin SVM & Slack Variables

Allows controlled margin violations using slack variables $\xi_i \ge 0$ penalized by regularization parameter $C$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

Dual box constraint bounds multipliers: $0 \le \alpha_i \le C$.

---

## 4. The Kernel Trick & Mercer's Theorem

Replaces inner products $\mathbf{x}_i^T \mathbf{x}_j$ with non-linear kernel functions $K(\mathbf{x}_i, \mathbf{x}_j) = \Phi(\mathbf{x}_i)^T \Phi(\mathbf{x}_j)$:

* **Polynomial Kernel**: $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + c)^d$
* **Radial Basis Function (RBF / Gaussian Kernel)**:

$$K(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \|\mathbf{x} - \mathbf{x}'\|^2 \right)$$

---

## 5. Scikit-Learn Python Implementation

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_m, y_m = make_moons(n_samples=300, noise=0.2, random_state=42)
X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_m, y_m, test_size=0.3, random_state=42)

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_tr_m, y_tr_m)
print(f"RBF Kernel SVM Test Accuracy: {svm_rbf.score(X_te_m, y_te_m):.4f}")
print(f"Total Support Vectors Count: {len(svm_rbf.support_)}")
```

---

## 6. End-to-End Industry Case Study: Medical Image Diagnosis Engine

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10.0, probability=True))
svm_pipeline.fit(X_cancer, y_cancer)

print(f"Medical Diagnosis SVM Score: {svm_pipeline.score(X_cancer, y_cancer):.4f}")
```

---

## 7. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Geometric Margin Width Calculation
**Problem**: Given weight vector norm $\|\mathbf{w}\|_2 = 5.0$, compute geometric margin width $\frac{2}{\|\mathbf{w}\|_2}$.
```python
# Solution
w_norm = 5.0
margin_width = 2.0 / w_norm
print(f"Question 1 -> Geometric Margin Width: {margin_width:.4f}") # 0.4000
```

### Question 2: Weight Vector Calculation from Dual Multipliers
**Problem**: Support vectors are $\mathbf{x}_1 = [1, 1]$ ($y_1=1, \alpha_1=0.5$) and $\mathbf{x}_2 = [2, 0]$ ($y_2=-1, \alpha_2=0.5$). Compute primal weight vector $\mathbf{w} = \sum \alpha_i y_i \mathbf{x}_i$.
```python
# Solution
x1 = np.array([1, 1])
x2 = np.array([2, 0])
a1, y1 = 0.5, 1
a2, y2 = 0.5, -1

w_primal = a1 * y1 * x1 + a2 * y2 * x2 # [0.5, 0.5] - [1.0, 0.0] = [-0.5, 0.5]
print(f"Question 2 -> Primal Weight Vector w: {w_primal}")
```

### Question 3: RBF Kernel Calculation
**Problem**: Compute RBF kernel similarity for $\mathbf{x}_1 = [0, 0]$ and $\mathbf{x}_2 = [2, 2]$ with $\gamma = 0.25$.
```python
# Solution
import math

x1 = np.array([0, 0])
x2 = np.array([2, 2])
gamma = 0.25

sq_dist = np.sum((x1 - x2)**2) # 8.0
k_rbf = math.exp(-gamma * sq_dist) # exp(-2.0) = 0.1353
print(f"Question 3 -> RBF Kernel Similarity: {k_rbf:.4f}")
```

### Question 4: Linear Hyperplane Decision Value
**Problem**: For hyperplane $z = \mathbf{w}^T \mathbf{x} + b$ with $\mathbf{w} = [1, 2], b = -3$, evaluate prediction for $\mathbf{x} = [2, 1]$.
```python
# Solution
w = np.array([1, 2])
b = -3
x = np.array([2, 1])

z = np.dot(w, x) + b # 2 + 2 - 3 = 1 -> Class 1
print(f"Question 4 -> Functional Value z: {z:.2f} -> Class: {1 if z >= 0 else -1}")
```

### Question 5: Polynomial Kernel Inner Product
**Problem**: Compute 2nd degree polynomial kernel $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + 1)^2$ for $\mathbf{x} = [1, 2]$ and $\mathbf{x}' = [3, 4]$.
```python
# Solution
x = np.array([1, 2])
x_prime = np.array([3, 4])

dot_prod = np.dot(x, x_prime) # 3 + 8 = 11
k_poly = (dot_prod + 1) ** 2 # 12^2 = 144
print(f"Question 5 -> Polynomial Kernel Result: {k_poly:.2f}")
```
