# K-Means Clustering & Gaussian Mixture Models (EM Algorithm)

[← Back to Course README](../README.md)

- [1. Principles of Unsupervised Data Engineering](#1-principles-of-unsupervised-data-engineering)
- [2. K-Means Clustering Algorithm](#2-k-means-clustering-algorithm)
  - [2.1 Mathematical Formulation & WCSS Inertia](#21-mathematical-formulation-wcss-inertia)
  - [2.2 K-Means++ Smart Initialization](#22-k-means-smart-initialization)
- [3. Gaussian Mixture Models (GMM)](#3-gaussian-mixture-models-gmm)
- [4. Expectation-Maximization (EM) Algorithm Derivation](#4-expectation-maximization-em-algorithm-derivation)
  - [4.1 E-Step (Expectation) Responsibilities](#41-e-step-expectation-responsibilities)
  - [4.2 M-Step (Maximization) Parameter Updates](#42-m-step-maximization-parameter-updates)
- [5. From-Scratch Python K-Means & GMM Suite](#5-from-scratch-python-k-means-gmm-suite)
- [6. End-to-End Industry Case Study: Customer Behavioral Segmentation](#6-end-to-end-industry-case-study-customer-behavioral-segmentation)
- [7. 5 Solved Analytical & Numerical Practice Questions](#7-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Single Iteration Centroid Update](#question-1-single-iteration-centroid-update)
  - [Question 2: WCSS Inertia Calculation](#question-2-wcss-inertia-calculation)
  - [Question 3: GMM E-Step Responsibility Normalization](#question-3-gmm-e-step-responsibility-normalization)
  - [Question 4: Silhouette Score Calculation](#question-4-silhouette-score-calculation)
  - [Question 5: GMM M-Step Mean Update](#question-5-gmm-m-step-mean-update)
> **Topic**: K-Means Clustering & Gaussian Mixture Models (EM Algorithm): 1. Principles of Unsupervised Data Engineering, 2. K-Means Clustering Algorithm, 3. Gaussian Mixture Models (GMM), 4. Expectation-Maximization (EM) Algorithm Derivation

---

## 1. Principles of Unsupervised Data Engineering

Unsupervised learning explores latent structures and probability density patterns in unlabeled feature space $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^n$.

---

## 2. K-Means Clustering Algorithm

### 2.1 Mathematical Formulation & WCSS Inertia
Partitions $n$ samples into $K$ disjoint clusters $\mathcal{C} = \{C_1, \dots, C_K\}$ minimizing Within-Cluster Sum of Squares (WCSS Inertia):

$$J_{\text{WCSS}}(\boldsymbol{\mu}) = \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

Where $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_i \in C_k} \mathbf{x}_i$ is cluster centroid.

### 2.2 K-Means++ Smart Initialization
1. Select first centroid $\boldsymbol{\mu}_1$ uniformly at random from data points.
2. Select next centroid $\boldsymbol{\mu}_k$ with probability proportional to squared Euclidean distance to nearest existing centroid:

$$P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_{j=1}^n D(\mathbf{x}_j)^2}$$

---

## 3. Gaussian Mixture Models (GMM)

Models data probability density as linear combination of $K$ Gaussian components:

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

Where $\sum_{k=1}^K \pi_k = 1$ and $\pi_k \ge 0$.

---

## 4. Expectation-Maximization (EM) Algorithm Derivation

### 4.1 E-Step (Expectation) Responsibilities
Calculates posterior probability (responsibility) $\gamma_{ik} = P(z_i = k \mid \mathbf{x}_i)$ that sample $i$ belongs to cluster $k$:

$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

### 4.2 M-Step (Maximization) Parameter Updates
Updates parameters using effective cluster weight $N_k = \sum_{i=1}^n \gamma_{ik}$:

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} \mathbf{x}_i$$

$$\boldsymbol{\Sigma}_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^T$$

$$\pi_k^{\text{new}} = \frac{N_k}{n}$$

---

## 5. From-Scratch Python K-Means & GMM Suite

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Custom K-Means Loop
class FromScratchKMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        
    def fit(self, X):
        np.random.seed(42)
        n_samples = len(X)
        # Random initial centroids
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iter):
            # Assign
            dists = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(dists, axis=1)
            
            # Update
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
            
        self.inertia_ = np.sum((X - self.centroids[self.labels_])**2)
        return self

# Execute Comparison
X_b, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
custom_km = FromScratchKMeans(k=3).fit(X_b)
sklearn_km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_b)

print(f"Custom K-Means Inertia:  {custom_km.inertia_:.2f}")
print(f"Sklearn K-Means Inertia: {sklearn_km.inertia_:.2f}")
```

---

## 6. End-to-End Industry Case Study: Customer Behavioral Segmentation

```python
# GMM Customer Soft-Clustering Pipeline
df_cust = pd.DataFrame({
    'annual_income': np.random.normal(60000, 15000, 400),
    'spending_score': np.random.normal(50, 20, 400)
})

gmm_cust = GaussianMixture(n_components=3, random_state=42).fit(df_cust)
soft_membership = gmm_cust.predict_proba(df_cust)
print("Sample Customer Soft Membership Probabilities:", soft_membership[:3])
```

---

## 7. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Single Iteration Centroid Update
**Problem**: Samples in Cluster 1 are `[[2, 4], [4, 6], [6, 8]]`. Compute updated centroid $\boldsymbol{\mu}_1$.
```python
# Solution
import numpy as np
pts = np.array([[2, 4], [4, 6], [6, 8]])
mu_1 = np.mean(pts, axis=0)
print(f"Question 1 -> Updated Centroid: {mu_1}") # [4.0, 6.0]
```

### Question 2: WCSS Inertia Calculation
**Problem**: Cluster centroid is `[0, 0]`. Points are `[[1, 1], [-1, -1], [2, 0]]`. Calculate WCSS inertia.
```python
# Solution
pts = np.array([[1, 1], [-1, -1], [2, 0]])
mu = np.array([0, 0])
wcss = np.sum((pts - mu)**2)
print(f"Question 2 -> WCSS Inertia: {wcss:.2f}") # (2 + 2 + 4) = 8.00
```

### Question 3: GMM E-Step Responsibility Normalization
**Problem**: Component likelihood density values $p_1 = 0.40$ and $p_2 = 0.60$ with prior mixture weights $\pi_1 = 0.50, \pi_2 = 0.50$. Compute normalized responsibility $\gamma_1$.
```python
# Solution
pi1, pi2 = 0.50, 0.50
p1, p2 = 0.40, 0.60

unnorm1 = pi1 * p1 # 0.20
unnorm2 = pi2 * p2 # 0.30

gamma1 = unnorm1 / (unnorm1 + unnorm2)
print(f"Question 3 -> Normalized Responsibility gamma1: {gamma1:.4f}") # 0.20/0.50 = 0.4000
```

### Question 4: Silhouette Score Calculation
**Problem**: For sample $i$, mean intra-cluster distance $a_i = 2.0$ and mean nearest-cluster distance $b_i = 5.0$. Compute Silhouette coefficient $s_i$.
```python
# Solution
a_i, b_i = 2.0, 5.0
s_i = (b_i - a_i) / max(a_i, b_i)
print(f"Question 4 -> Silhouette Coefficient s_i: {s_i:.4f}") # (5 - 2)/5 = 0.6000
```

### Question 5: GMM M-Step Mean Update
**Problem**: Responsibilities $\gamma_{i1} = [0.8, 0.2]$ for points $\mathbf{x}_1 = [1, 2]$ and $\mathbf{x}_2 = [3, 4]$. Compute updated mean $\boldsymbol{\mu}_1$.
```python
# Solution
gammas = np.array([0.8, 0.2])
xs = np.array([[1, 2], [3, 4]])

N1 = np.sum(gammas) # 1.0
mu1_new = np.sum(gammas[:, np.newaxis] * xs, axis=0) / N1
print(f"Question 5 -> M-Step Updated Mean mu1: {mu1_new}") # ([0.8, 1.6] + [0.6, 0.8])/1.0 = [1.4, 2.4]
```
