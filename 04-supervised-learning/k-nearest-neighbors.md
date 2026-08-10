# K-Nearest Neighbors (KNN) Non-Parametric Classifier & Regressor

[← Back to Course README](../README.md)

- [1. Non-Parametric Voting & Lazy Learning Principles](#1-non-parametric-voting-lazy-learning-principles)
- [2. Distance Metric Taxonomy](#2-distance-metric-taxonomy)
  - [2.1 Minkowski Distance Metric ($L_p$ Norm)](#21-minkowski-distance-metric-lp-norm)
  - [2.2 Cosine Distance & Mahalanobis Distance](#22-cosine-distance-mahalanobis-distance)
- [3. Selection of K, Scaling Sensitivity & Curse of Dimensionality](#3-selection-of-k-scaling-sensitivity-curse-of-dimensionality)
- [4. From-Scratch Python KNN & Scikit-Learn Implementation](#4-from-scratch-python-knn-scikit-learn-implementation)
- [5. End-to-End Industry Case Study: Real Estate Valuation Engine](#5-end-to-end-industry-case-study-real-estate-valuation-engine)
- [6. 5 Solved Analytical & Numerical Practice Questions](#6-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Euclidean vs Manhattan Distance Calculation](#question-1-euclidean-vs-manhattan-distance-calculation)
  - [Question 2: Cosine Similarity & Distance Calculation](#question-2-cosine-similarity-distance-calculation)
  - [Question 3: Inverse Distance-Weighted Voting](#question-3-inverse-distance-weighted-voting)
  - [Question 4: Mahalanobis Distance Calculation](#question-4-mahalanobis-distance-calculation)
  - [Question 5: Optimal K Cross-Validation Grid Search](#question-5-optimal-k-cross-validation-grid-search)
> **Topic**: K-Nearest Neighbors (KNN) Non-Parametric Classifier & Regressor: 1. Non-Parametric Voting & Lazy Learning Principles, 2. Distance Metric Taxonomy, 3. Selection of K, Scaling Sensitivity & Curse of Dimensionality, 4. From-Scratch Python KNN & Scikit-Learn Implementation

---

## 1. Non-Parametric Voting & Lazy Learning Principles

KNN is an instance-based non-parametric classifier. Given query point $\mathbf{x}_0$, KNN identifies the set $\mathcal{N}_K(\mathbf{x}_0)$ of $K$ nearest training points and assigns majority class label (or distance-weighted vote):

$$\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_K(\mathbf{x}_0)} w_i \mathbb{I}(y_i = c)$$

Where distance weight $w_i = \frac{1}{d(\mathbf{x}_0, \mathbf{x}_i) + \epsilon}$.

---

## 2. Distance Metric Taxonomy

### 2.1 Minkowski Distance Metric ($L_p$ Norm)
For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$:

$$d_p(\mathbf{u}, \mathbf{v}) = \left( \sum_{j=1}^d |u_j - v_j|^p \right)^{1/p}$$

* **Manhattan Distance ($p=1$, $L_1$ Norm)**: $d_1(\mathbf{u}, \mathbf{v}) = \sum |u_j - v_j|$
* **Euclidean Distance ($p=2$, $L_2$ Norm)**: $d_2(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_j - v_j)^2}$
* **Chebyshev Distance ($p=\infty$, $L_\infty$ Norm)**: $d_\infty(\mathbf{u}, \mathbf{v}) = \max_j |u_j - v_j|$

### 2.2 Cosine Distance & Mahalanobis Distance
* **Cosine Distance**: $d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$
* **Mahalanobis Distance** (Accounts for feature covariance $\Sigma$):

$$d_{\Sigma}(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^T \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$

---

## 3. Selection of K, Scaling Sensitivity & Curse of Dimensionality

* **Small $K$ ($K=1$)**: Complex boundary, high variance, sensitive to noise.
* **Large $K$**: Smooth boundary, high bias, over-simplifies class structure.
* **Scaling Sensitivity**: Features with large raw scales dominate distance metrics; Z-score standardization is mandatory before fitting KNN.

---

## 4. From-Scratch Python KNN & Scikit-Learn Implementation

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class FromScratchKNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
        
    def predict(self, X):
        X_test = np.array(X)
        dists = cdist(X_test, self.X_train, metric=self.metric)
        preds = []
        
        for idx in range(len(X_test)):
            k_indices = np.argsort(dists[idx])[:self.k]
            k_labels = self.y_train[k_indices]
            preds.append(np.bincount(k_labels).argmax())
            
        return np.array(preds)

# Execute Comparison
X_knn = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_knn = np.array([0, 0, 0, 1, 1, 1])

custom_knn = FromScratchKNNClassifier(k=3).fit(X_knn, y_knn)
print(f"Custom KNN Prediction for [4, 4]: {custom_knn.predict([[4, 4]])[0]}")
```

---

## 5. End-to-End Industry Case Study: Real Estate Valuation Engine

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Synthetic Real Estate Valuation Dataset
np.random.seed(42)
n_homes = 200

sqft = np.random.normal(2000, 500, n_homes)
bedrooms = np.random.randint(1, 6, n_homes)
price = 150 * sqft + 20000 * bedrooms + np.random.normal(0, 15000, n_homes)

X_re = pd.DataFrame({'sqft': sqft, 'bedrooms': bedrooms})
y_re = price

re_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance'))
re_pipeline.fit(X_re, y_re)

print(f"Real Estate Valuation KNN R^2 Score: {re_pipeline.score(X_re, y_re):.4f}")
```

---

## 6. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Euclidean vs Manhattan Distance Calculation
**Problem**: Vectors $\mathbf{u} = [2, 5]$ and $\mathbf{v} = [5, 9]$. Compute Euclidean ($L_2$) and Manhattan ($L_1$) distances.
```python
# Solution
import numpy as np

u = np.array([2, 5])
v = np.array([5, 9])

d_euc = np.linalg.norm(u - v) # sqrt(9 + 16) = 5.0
d_man = np.sum(np.abs(u - v)) # 3 + 4 = 7.0

print(f"Question 1 -> Euclidean Distance: {d_euc:.4f}, Manhattan Distance: {d_man:.4f}")
```

### Question 2: Cosine Similarity & Distance Calculation
**Problem**: Compute Cosine Similarity and Cosine Distance between $\mathbf{u} = [1, 2]$ and $\mathbf{v} = [2, 4]$.
```python
# Solution
u = np.array([1, 2])
v = np.array([2, 4])

cos_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
cos_dist = 1.0 - cos_sim

print(f"Question 2 -> Cosine Similarity: {cos_sim:.4f}, Cosine Distance: {cos_dist:.4f}") # Sim=1.0, Dist=0.0
```

### Question 3: Inverse Distance-Weighted Voting
**Problem**: 3 nearest neighbors have labels `[0, 1, 1]` with distances `[0.5, 1.0, 2.0]`. Calculate inverse distance-weighted votes for Class 0 vs Class 1.
```python
# Solution
labels = [0, 1, 1]
dists = [0.5, 1.0, 2.0]

weights = [1.0 / d for d in dists] # [2.0, 1.0, 0.5]
vote_0 = sum(w for w, l in zip(weights, labels) if l == 0) # 2.0
vote_1 = sum(w for w, l in zip(weights, labels) if l == 1) # 1.5

print(f"Question 3 -> Vote Class 0: {vote_0:.2f}, Vote Class 1: {vote_1:.2f} -> Class: {0 if vote_0 > vote_1 else 1}")
```

### Question 4: Mahalanobis Distance Calculation
**Problem**: Given sample covariance matrix $\Sigma = \begin{bmatrix} 4 & 0 \\ 0 & 9 \end{bmatrix}$, compute Mahalanobis distance between $\mathbf{u} = [0, 0]$ and $\mathbf{v} = [2, 3]$.
```python
# Solution
sigma = np.array([[4, 0], [0, 9]])
u = np.array([0, 0])
v = np.array([2, 3])

diff = u - v
inv_sigma = np.linalg.inv(sigma)
d_mah = np.sqrt(diff.T @ inv_sigma @ diff)

print(f"Question 4 -> Mahalanobis Distance: {d_mah:.4f}") # sqrt(4/4 + 9/9) = sqrt(2) = 1.4142
```

### Question 5: Optimal K Cross-Validation Grid Search
**Problem**: Grid search optimal $K \in [1, 10]$ for KNN classifier using 5-fold cross-validation.
```python
# Solution
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

X_i, y_i = load_iris(return_X_y=True)
best_k, best_score = 1, 0.0

for k in range(1, 11):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_i, y_i, cv=5).mean()
    if score > best_score:
        best_k, best_score = k, score

print(f"Question 5 -> Optimal K: {best_k} (CV Score: {best_score:.4f})")
```
