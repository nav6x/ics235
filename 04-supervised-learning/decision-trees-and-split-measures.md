# Decision Trees, CART Algorithm & Impurity Evaluation Split Measures

[← Back to Course README](../README.md)

- [1. Classification & Regression Trees (CART) Architecture](#1-classification-regression-trees-cart-architecture)
- [2. Impurity Measures for Split Evaluation](#2-impurity-measures-for-split-evaluation)
  - [2.1 Shannon Entropy & Information Gain](#21-shannon-entropy-information-gain)
  - [2.2 Gini Impurity Index](#22-gini-impurity-index)
  - [2.3 Gain Ratio & C4.5 Split Normalization](#23-gain-ratio-c45-split-normalization)
  - [2.4 Variance Reduction (Regression Trees)](#24-variance-reduction-regression-trees)
- [3. Tree Pruning & Cost-Complexity Optimization](#3-tree-pruning-cost-complexity-optimization)
- [4. From-Scratch Python Decision Tree & Scikit-Learn Implementation](#4-from-scratch-python-decision-tree-scikit-learn-implementation)
- [5. End-to-End Industry Case Study: Bank Customer Credit Assessment](#5-end-to-end-industry-case-study-bank-customer-credit-assessment)
- [6. 5 Solved Analytical & Numerical Practice Questions](#6-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Gini Impurity Calculation](#question-1-gini-impurity-calculation)
  - [Question 2: Shannon Entropy Calculation](#question-2-shannon-entropy-calculation)
  - [Question 3: Gini Gain from Split](#question-3-gini-gain-from-split)
  - [Question 4: C4.5 Split Information & Gain Ratio](#question-4-c45-split-information-gain-ratio)
  - [Question 5: Cost-Complexity Pruning Alpha Path](#question-5-cost-complexity-pruning-alpha-path)
> **Topic**: Decision Trees, CART Algorithm & Impurity Evaluation Split Measures: 1. Classification & Regression Trees (CART) Architecture, 2. Impurity Measures for Split Evaluation, 3. Tree Pruning & Cost-Complexity Optimization, 4. From-Scratch Python Decision Tree & Scikit-Learn Implementation

---

## 1. Classification & Regression Trees (CART) Architecture

Decision Trees perform recursive binary partitioning of feature space $\mathcal{X}$ into axis-aligned rectangular hyper-regions $\mathcal{R}_m$, assigning a constant prediction value $c_m$ to each region.

---

## 2. Impurity Measures for Split Evaluation

### 2.1 Shannon Entropy & Information Gain
For node $m$ containing class proportions $p_{mk} = \frac{1}{N_m} \sum_{\mathbf{x}_i \in \mathcal{R}_m} \mathbb{I}(y_i = k)$ for $k \in \{1, \dots, K\}$:

$$H(m) = -\sum_{k=1}^K p_{mk} \log_2(p_{mk})$$

Information Gain achieved by splitting node $m$ into left child $m_L$ and right child $m_R$ on feature $j$ at threshold $t$:

$$IG(m, j, t) = H(m) - \left( \frac{N_{m_L}}{N_m} H(m_L) + \frac{N_{m_R}}{N_m} H(m_R) \right)$$

### 2.2 Gini Impurity Index
Measures total variance across class indicators:

$$G(m) = 1 - \sum_{k=1}^K p_{mk}^2$$

Gini Reduction Gain:

$$\Delta G = G(m) - \left( \frac{N_{m_L}}{N_m} G(m_L) + \frac{N_{m_R}}{N_m} G(m_R) \right)$$

### 2.3 Gain Ratio & C4.5 Split Normalization
Normalizes Information Gain by Split Information to prevent bias toward high-cardinality features:

$$\text{SplitInfo}_A(m) = -\sum_{v=1}^V \frac{N_{m_v}}{N_m} \log_2\left( \frac{N_{m_v}}{N_m} \right)$$

$$\text{GainRatio}(m, A) = \frac{IG(m, A)}{\text{SplitInfo}_A(m)}$$

### 2.4 Variance Reduction (Regression Trees)
For continuous target $y$, impurity is measured by node sample variance (MSE):

$$\text{MSE}(m) = \frac{1}{N_m} \sum_{i \in \mathcal{R}_m} (y_i - \bar{y}_m)^2$$

Variance Reduction Gain:

$$\Delta \text{Var} = \text{MSE}(m) - \left( \frac{N_{m_L}}{N_m} \text{MSE}(m_L) + \frac{N_{m_R}}{N_m} \text{MSE}(m_R) \right)$$

---

## 3. Tree Pruning & Cost-Complexity Optimization

Cost-Complexity Pruning minimizes objective parameterized by $\alpha \ge 0$:

$$R_\alpha(T) = R(T) + \alpha |T|$$

Where $R(T) = \sum_{m=1}^{|T|} N_m Q_m(T)$ is total tree misclassification error and $|T|$ is number of terminal leaf nodes.

---

## 4. From-Scratch Python Decision Tree & Scikit-Learn Implementation

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

# Custom Gini Split Finder
def compute_best_gini_split(X, y):
    n_samples, n_features = X.shape
    best_gini_gain = -1.0
    best_split = None
    
    def gini(labels):
        if len(labels) == 0: return 0.0
        p = np.bincount(labels) / len(labels)
        return 1.0 - np.sum(p ** 2)
        
    parent_gini = gini(y)
    
    for feature_idx in range(n_features):
        thresholds = np.unique(X[:, feature_idx])
        for t in thresholds:
            left_mask = X[:, feature_idx] <= t
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
                
            g_left = gini(y[left_mask])
            g_right = gini(y[right_mask])
            
            n_l, n_r = np.sum(left_mask), np.sum(right_mask)
            gain = parent_gini - ((n_l / n_samples) * g_left + (n_r / n_samples) * g_right)
            
            if gain > best_gini_gain:
                best_gini_gain = gain
                best_split = (feature_idx, t)
                
    return best_split, best_gini_gain

# Execute Split Check on Iris
iris = load_iris()
split_info, gain = compute_best_gini_split(iris.data, iris.target)
print(f"Optimal Initial Gini Split Feature Index: {split_info[0]}, Threshold: {split_info[1]}, Gain: {gain:.4f}")
```

---

## 5. End-to-End Industry Case Study: Bank Customer Credit Assessment

```python
# Bank Credit Assessment Decision Tree Pipeline
df_credit_tree = pd.DataFrame({
    'credit_score': np.random.normal(680, 40, 300),
    'income': np.random.normal(75000, 15000, 300),
    'debt_ratio': np.random.uniform(0.1, 0.6, 300),
    'approved': np.random.choice([0, 1], size=300, p=[0.3, 0.7])
})

clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, ccp_alpha=0.01, random_state=42)
clf_tree.fit(df_credit_tree[['credit_score', 'income', 'debt_ratio']], df_credit_tree['approved'])

print("Trained Decision Tree Rules:")
print(export_text(clf_tree, feature_names=['credit_score', 'income', 'debt_ratio']))
```

---

## 6. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Gini Impurity Calculation
**Problem**: Node contains 30 Class A, 20 Class B, and 50 Class C samples. Compute Gini Impurity $G(m)$.
```python
# Solution
import numpy as np

p_a, p_b, p_c = 30/100, 20/100, 50/100
gini_val = 1.0 - (p_a**2 + p_b**2 + p_c**2)
print(f"Question 1 -> Gini Impurity: {gini_val:.4f}") # 1 - (0.09 + 0.04 + 0.25) = 0.6200
```

### Question 2: Shannon Entropy Calculation
**Problem**: Compute Shannon Entropy $H(m)$ in bits for node with class proportions $p = [0.75, 0.25]$.
```python
# Solution
import math

p1, p2 = 0.75, 0.25
entropy_val = -(p1 * math.log2(p1) + p2 * math.log2(p2))
print(f"Question 2 -> Shannon Entropy: {entropy_val:.4f} bits") # ~0.8113 bits
```

### Question 3: Gini Gain from Split
**Problem**: Parent node $S$ (Gini=0.50, $N=100$) splits into $S_L$ (Gini=0.20, $N_L=60$) and $S_R$ (Gini=0.10, $N_R=40$). Compute Gini Gain.
```python
# Solution
g_p = 0.50
g_l, n_l = 0.20, 60
g_r, n_r = 0.10, 40
n_tot = 100

g_gain = g_p - ((n_l / n_tot) * g_l + (n_r / n_tot) * g_r)
print(f"Question 3 -> Gini Gain: {g_gain:.4f}") # 0.50 - (0.12 + 0.04) = 0.3400
```

### Question 4: C4.5 Split Information & Gain Ratio
**Problem**: Feature split produces $N_1 = 50, N_2 = 50$ ($N=100$) with Information Gain $IG = 0.40$. Compute SplitInfo and GainRatio.
```python
# Solution
n1, n2, n_tot = 50, 50, 100
split_info = -( (n1/n_tot)*math.log2(n1/n_tot) + (n2/n_tot)*math.log2(n2/n_tot) ) # 1.0 bit
ig = 0.40
gain_ratio = ig / split_info

print(f"Question 4 -> SplitInfo: {split_info:.4f}, GainRatio: {gain_ratio:.4f}") # 0.4000
```

### Question 5: Cost-Complexity Pruning Alpha Path
**Problem**: Extract effective pruning alphas using `cost_complexity_pruning_path`.
```python
# Solution
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

X_bc, y_bc = load_breast_cancer(return_X_y=True)
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_bc, y_bc)
alphas = path.ccp_alphas
print(f"Question 5 -> Effective Alphas Count: {len(alphas)}, Max Alpha: {alphas[-1]:.4f}")
```
