# Linear Regression & Logistic Regression

[← Back to Course README](../README.md)

- [1. Ordinary Least Squares (OLS) Linear Regression](#1-ordinary-least-squares-ols-linear-regression)
  - [1.1 Mathematical Model & Matrix Formulation](#11-mathematical-model-matrix-formulation)
  - [1.2 Step-by-Step Matrix Calculus Derivation of Normal Equations](#12-step-by-step-matrix-calculus-derivation-of-normal-equations)
  - [1.3 Gauss-Markov Theorem & Statistical Assumptions](#13-gauss-markov-theorem-statistical-assumptions)
- [2. Logistic Regression for Binary Classification](#2-logistic-regression-for-binary-classification)
  - [2.1 The Sigmoid (Logistic) Function & Log-Odds Logit](#21-the-sigmoid-logistic-function-log-odds-logit)
  - [2.2 Maximum Likelihood Estimation & Binary Cross-Entropy Loss](#22-maximum-likelihood-estimation-binary-cross-entropy-loss)
  - [2.3 Gradient Descent Optimization Derivation](#23-gradient-descent-optimization-derivation)
- [3. Regularized Regression (Ridge & LASSO)](#3-regularized-regression-ridge-lasso)
- [4. From-Scratch NumPy & Scikit-Learn Production Implementations](#4-from-scratch-numpy-scikit-learn-production-implementations)
- [5. End-to-End Industry Case Study: Brain Weight vs Body Weight Case Study](#5-end-to-end-industry-case-study-brain-weight-vs-body-weight-case-study)
- [6. 5 Solved Analytical & Numerical Practice Questions](#6-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Closed-Form OLS Normal Equations Computation](#question-1-closed-form-ols-normal-equations-computation)
  - [Question 2: Sigmoid Probability & Log-Odds Transformation](#question-2-sigmoid-probability-log-odds-transformation)
  - [Question 3: Binary Cross-Entropy Loss Calculation](#question-3-binary-cross-entropy-loss-calculation)
  - [Question 4: Ridge Closed-Form Weight Shrinkage](#question-4-ridge-closed-form-weight-shrinkage)
  - [Question 5: Logistic Regression Decision Boundary Location](#question-5-logistic-regression-decision-boundary-location)
> **Topic**: Linear Regression & Logistic Regression: 1. Ordinary Least Squares (OLS) Linear Regression, 2. Logistic Regression for Binary Classification, 3. Regularized Regression (Ridge & LASSO), 4. From-Scratch NumPy & Scikit-Learn Production Implementations

---

## 1. Ordinary Least Squares (OLS) Linear Regression

### 1.1 Mathematical Model & Matrix Formulation
Linear regression models continuous target $y \in \mathbb{R}$ as a linear combination of input features $\mathbf{x} \in \mathbb{R}^d$:

$$y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i = w_0 + \sum_{j=1}^d w_j x_{ij} + \epsilon_i$$

In matrix notation for $n$ samples and $d+1$ parameters (including bias $w_0$):

$$\mathbf{y} = X \mathbf{w} + \boldsymbol{\epsilon}$$

Where $\mathbf{y} \in \mathbb{R}^n$, design matrix $X \in \mathbb{R}^{n \times (d+1)}$, weight vector $\mathbf{w} \in \mathbb{R}^{d+1}$, and residual vector $\boldsymbol{\epsilon} \in \mathbb{R}^n$.

### 1.2 Step-by-Step Matrix Calculus Derivation of Normal Equations
We seek weight vector $\hat{\mathbf{w}}$ minimizing Residual Sum of Squares (RSS) loss function $L(\mathbf{w})$:

$$L(\mathbf{w}) = \|\mathbf{y} - X \mathbf{w}\|^2 = (\mathbf{y} - X \mathbf{w})^T (\mathbf{y} - X \mathbf{w})$$

Expanding matrix transpose product:

$$L(\mathbf{w}) = (\mathbf{y}^T - \mathbf{w}^T X^T)(\mathbf{y} - X \mathbf{w}) = \mathbf{y}^T \mathbf{y} - \mathbf{y}^T X \mathbf{w} - \mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}$$

Since $\mathbf{y}^T X \mathbf{w}$ is a scalar, $(\mathbf{y}^T X \mathbf{w})^T = \mathbf{w}^T X^T \mathbf{y}$. Combining scalar terms:

$$L(\mathbf{w}) = \mathbf{y}^T \mathbf{y} - 2 \mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}$$

Taking gradient with respect to vector $\mathbf{w}$ (using matrix gradient identities $\nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{a}) = \mathbf{a}$ and $\nabla_{\mathbf{w}} (\mathbf{w}^T A \mathbf{w}) = 2 A \mathbf{w}$ for symmetric $A = X^T X$):

$$\nabla_{\mathbf{w}} L(\mathbf{w}) = -2 X^T \mathbf{y} + 2 X^T X \mathbf{w}$$

Setting gradient to zero vector $\mathbf{0}$:

$$-2 X^T \mathbf{y} + 2 X^T X \hat{\mathbf{w}} = \mathbf{0} \implies X^T X \hat{\mathbf{w}} = X^T \mathbf{y}$$

Assuming $X^T X$ is non-singular (invertible), the closed-form **OLS Normal Equation Solution** is:

$$\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$$

### 1.3 Gauss-Markov Theorem & Statistical Assumptions
Under Gauss-Markov assumptions ($\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$, homoscedasticity $\text{Var}(\boldsymbol{\epsilon}) = \sigma^2 I$, uncorrelated residuals $\text{Cov}(\epsilon_i, \epsilon_j) = 0$), OLS estimator $\hat{\mathbf{w}}$ is the **BLUE** (Best Linear Unbiased Estimator), possessing minimum variance among all linear unbiased estimators.

---

## 2. Logistic Regression for Binary Classification

### 2.1 The Sigmoid (Logistic) Function & Log-Odds Logit
Logistic regression models conditional class probability $P(Y=1 \mid \mathbf{x})$ via Sigmoid activation function $\sigma(z)$:

$$\hat{p} = P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}$$

The **Log-Odds (Logit)** transformation maps probability $\hat{p} \in (0, 1)$ into linear feature space:

$$\text{logit}(\hat{p}) = \ln \left( \frac{\hat{p}}{1 - \hat{p}} \right) = \mathbf{w}^T \mathbf{x} = w_0 + \sum_{j=1}^d w_j x_j$$

### 2.2 Maximum Likelihood Estimation & Binary Cross-Entropy Loss
For Bernoulli distributed target $y_i \in \{0, 1\}$, conditional probability density is:

$$P(y_i \mid \mathbf{x}_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

Likelihood over $n$ independent samples:

$$L(\mathbf{w}) = \prod_{i=1}^n \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

Log-likelihood $\ell(\mathbf{w}) = \ln L(\mathbf{w})$:

$$\ell(\mathbf{w}) = \sum_{i=1}^n \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

Minimizing Negative Log-Likelihood (NLL) yields **Binary Cross-Entropy (BCE) Loss**:

$$J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

### 2.3 Gradient Descent Optimization Derivation
Using chain rule, derivative of loss with respect to weight $w_j$:

$$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n (\hat{p}_i - y_i) x_{ij} \implies \nabla_{\mathbf{w}} J(\mathbf{w}) = \frac{1}{n} X^T (\hat{\mathbf{p}} - \mathbf{y})$$

Gradient update rule with learning rate $\alpha$:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \frac{1}{n} X^T (\hat{\mathbf{p}} - \mathbf{y})$$

---

## 3. Regularized Regression (Ridge & LASSO)

To prevent overfitting, objective function incorporates structural complexity penalties:
* **Ridge Regression ($L_2$ Penalty)**:

$$J_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X \mathbf{w}\|^2 + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 \implies \hat{\mathbf{w}}_{\text{Ridge}} = (X^T X + n \lambda I)^{-1} X^T \mathbf{y}$$

* **LASSO Regression ($L_1$ Penalty)**:

$$J_{\text{LASSO}}(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X \mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_1$$

---

## 4. From-Scratch NumPy & Scikit-Learn Production Implementations

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, accuracy_score

# From-Scratch OLS Normal Equation Solver
class FromScratchOLSRegression:
    def fit(self, X, y):
        # Add bias column of ones
        X_b = np.column_stack([np.ones(len(X)), X])
        # Closed form: w = (X^T X)^-1 X^T y
        self.w_ = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self
        
    def predict(self, X):
        X_b = np.column_stack([np.ones(len(X)), X])
        return X_b @ self.w_

# Execute Comparison
np.random.seed(42)
X_sim = np.random.uniform(1, 10, (100, 2))
y_sim = 3.5 * X_sim[:, 0] - 2.0 * X_sim[:, 1] + 5.0 + np.random.normal(0, 0.5, 100)

custom_ols = FromScratchOLSRegression().fit(X_sim, y_sim)
sklearn_ols = LinearRegression().fit(X_sim, y_sim)

print(f"Custom OLS Weights:  {custom_ols.w_}")
print(f"Sklearn OLS Intercept: {sklearn_ols.intercept_:.4f}, Coefs: {sklearn_ols.coef_}")
```

---

## 5. End-to-End Industry Case Study: Brain Weight vs Body Weight Case Study

```python
# Brain and Body Weight Logarithmic Regression Case Study
# Modeling relationship between Body Weight (kg) and Brain Weight (g) across mammal species
body_weight = np.array([3.38, 0.48, 1.35, 465.0, 36.33, 27.66, 1.04, 11.7, 514.0, 62.0])
brain_weight = np.array([44.5, 15.5, 8.1, 423.0, 119.5, 115.0, 5.5, 50.0, 460.0, 132.0])

# Logarithmic Transformation to linearize exponential body-brain relationship
log_body = np.log(body_weight).reshape(-1, 1)
log_brain = np.log(brain_weight)

log_model = LinearRegression().fit(log_body, log_brain)
log_preds = log_model.predict(log_body)

print(f"Log-Log Regression Model: ln(Brain) = {log_model.intercept_:.4f} + {log_model.coef_[0]:.4f} * ln(Body)")
print(f"R^2 Performance Score in Log Space: {log_model.score(log_body, log_brain):.4f}")
```

---

## 6. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Closed-Form OLS Normal Equations Computation
**Problem**: Compute weight vector $\hat{\mathbf{w}} = (X^T X)^{-1} X^T y$ for $X = [[1, 1], [1, 2], [1, 3]]$ and $y = [3, 5, 7]$.
```python
# Solution
import numpy as np

X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([3, 5, 7])

w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Question 1 -> Intercept w0: {w_hat[0]:.4f}, Slope w1: {w_hat[1]:.4f}") # w0=1.0, w1=2.0
```

### Question 2: Sigmoid Probability & Log-Odds Transformation
**Problem**: For feature vector $\mathbf{x} = [1, 2]$ and weights $\mathbf{w} = [0.5, 1.5], b = -2.0$, calculate logit $z$, sigmoid probability $\hat{p}$, and odds ratio.
```python
# Solution
import math

x = np.array([1, 2])
w = np.array([0.5, 1.5])
b = -2.0

z = np.dot(w, x) + b # 0.5 + 3.0 - 2.0 = 1.5
p_hat = 1.0 / (1.0 + math.exp(-z))
odds = p_hat / (1.0 - p_hat)

print(f"Question 2 -> Logit z: {z:.2f}, Probability p: {p_hat:.4f}, Odds: {odds:.4f}")
```

### Question 3: Binary Cross-Entropy Loss Calculation
**Problem**: Compute cross-entropy loss for $y = 1, \hat{p} = 0.90$ vs $y = 0, \hat{p} = 0.70$.
```python
# Solution
loss_pos = -(1 * math.log(0.90)) # 0.1054
loss_neg = -(1 * math.log(1 - 0.70)) # 1.2040
print(f"Question 3 -> Loss (y=1, p=0.9): {loss_pos:.4f}, Loss (y=0, p=0.7): {loss_neg:.4f}")
```

### Question 4: Ridge Closed-Form Weight Shrinkage
**Problem**: Given $X^T X = \begin{bmatrix} 10 & 0 \\ 0 & 10 \end{bmatrix}$ and $X^T y = \begin{bmatrix} 20 \\ 30 \end{bmatrix}$, compute OLS weights vs Ridge weights with $n \lambda = 5$.
```python
# Solution
XtX = np.array([[10, 0], [0, 10]])
Xty = np.array([20, 30])
n_lambda = 5

w_ols = np.linalg.inv(XtX) @ Xty # [2.0, 3.0]
w_ridge = np.linalg.inv(XtX + n_lambda * np.eye(2)) @ Xty # [20/15, 30/15] = [1.3333, 2.0000]

print(f"Question 4 -> OLS Weights: {w_ols}, Ridge Weights (Shrunk): {w_ridge}")
```

### Question 5: Logistic Regression Decision Boundary Location
**Problem**: Find decision boundary location $x^*$ where $P(Y=1 \mid x) = 0.5$ for model $z = 0.8 x - 4.0$.
```python
# Solution
# 0.8 * x - 4.0 = 0 -> x = 5.0
w1, b = 0.8, -4.0
x_star = -b / w1
print(f"Question 5 -> Decision Boundary Location x*: {x_star:.2f}") # 5.00
```
