# Hierarchical Clustering & Linkage Criteria

[← Back to Course README](../README.md)

- [1. Hierarchical Paradigms: Agglomerative vs Divisive](#1-hierarchical-paradigms-agglomerative-vs-divisive)
- [2. Linkage Criteria Taxonomy](#2-linkage-criteria-taxonomy)
- [3. Dendrogram Structure & Distance Cutoffs](#3-dendrogram-structure-distance-cutoffs)
- [4. Python Implementation & Scikit-Learn Suite](#4-python-implementation-scikit-learn-suite)
- [5. Industry Case Study & Solved Exercises](#5-industry-case-study-solved-exercises)
  - [Question 1: Single Linkage Computation](#question-1-single-linkage-computation)
  - [Question 2: Complete Linkage Computation](#question-2-complete-linkage-computation)
  - [Question 3: Ward's Variance Increase](#question-3-wards-variance-increase)
  - [Question 4: Cophenetic Correlation Coefficient](#question-4-cophenetic-correlation-coefficient)
  - [Question 5: Distance Threshold Clustering](#question-5-distance-threshold-clustering)
> **Topic**: Hierarchical Clustering & Linkage Criteria: 1. Hierarchical Paradigms: Agglomerative vs Divisive, 2. Linkage Criteria Taxonomy, 3. Dendrogram Structure & Distance Cutoffs, 4. Python Implementation & Scikit-Learn Suite

---

## 1. Hierarchical Paradigms: Agglomerative vs Divisive

1. **Agglomerative (Bottom-Up)**: Starts with each sample as an isolated cluster ($n$ clusters). Iteratively merges closest cluster pair.
2. **Divisive (Top-Down)**: Starts with all samples in single cluster. Iteratively splits clusters.

---

## 2. Linkage Criteria Taxonomy

For clusters $A$ and $B$:
* **Single Linkage**: $d(A, B) = \min_{\mathbf{u} \in A, \mathbf{v} \in B} d(\mathbf{u}, \mathbf{v})$
* **Complete Linkage**: $d(A, B) = \max_{\mathbf{u} \in A, \mathbf{v} \in B} d(\mathbf{u}, \mathbf{v})$
* **Average Linkage**: $d(A, B) = \frac{1}{|A||B|} \sum_{\mathbf{u} \in A} \sum_{\mathbf{v} \in B} d(\mathbf{u}, \mathbf{v})$
* **Ward's Minimum Variance Linkage**: Minimizes variance increase:

$$\Delta \text{Var} = \frac{|A||B|}{|A| + |B|} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$

---

## 3. Dendrogram Structure & Distance Cutoffs

Dendrograms plot hierarchical tree merge heights. Horizontal line cuts at distance threshold $h$ produce discrete clusters.

---

## 4. Python Implementation & Scikit-Learn Suite

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np

X = np.array([[1, 2], [2, 3], [5, 8], [8, 8], [1, 0], [9, 10]])
Z = linkage(X, method='ward')

agg = AgglomerativeClustering(n_clusters=2, linkage='ward').fit(X)
print(f"Agglomerative Cluster Labels: {agg.labels_}")
```

---

## 5. Industry Case Study & Solved Exercises

### Question 1: Single Linkage Computation
**Problem**: Cluster A=`[[0, 0]]`, Cluster B=`[[3, 4], [0, 5]]`. Compute single linkage distance.
```python
# Solution
pts_A = np.array([[0, 0]])
pts_B = np.array([[3, 4], [0, 5]])
dists = [np.linalg.norm(pts_A[0] - b) for b in pts_B] # [5.0, 5.0]
print(f"Question 1 -> Single Linkage Distance: {min(dists):.4f}") # 5.0000
```

### Question 2: Complete Linkage Computation
**Problem**: Cluster A=`[[0, 0]]`, Cluster B=`[[3, 4], [0, 5]]`. Compute complete linkage distance.
```python
# Solution
print(f"Question 2 -> Complete Linkage Distance: {max(dists):.4f}") # 5.0000
```

### Question 3: Ward's Variance Increase
**Problem**: Cluster A ($N_A=3, \boldsymbol{\mu}_A=[0, 0]$), Cluster B ($N_B=2, \boldsymbol{\mu}_B=[5, 0]$). Compute Ward variance increase.
```python
# Solution
na, nb = 3, 2
mu_a = np.array([0, 0])
mu_b = np.array([5, 0])

delta_var = ((na * nb) / (na + nb)) * np.sum((mu_a - mu_b)**2)
print(f"Question 3 -> Ward Variance Increase: {delta_var:.2f}") # (6/5) * 25 = 30.00
```

### Question 4: Cophenetic Correlation Coefficient
**Problem**: Compute cophenetic correlation using `scipy.cluster.hierarchy.cophenet`.
```python
# Solution
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

coph_c, _ = cophenet(Z, pdist(X))
print(f"Question 4 -> Cophenetic Correlation: {coph_c:.4f}")
```

### Question 5: Distance Threshold Clustering
**Problem**: Fit AgglomerativeClustering with `distance_threshold=6.0`.
```python
# Solution
agg_thresh = AgglomerativeClustering(n_clusters=None, distance_threshold=6.0, linkage='ward').fit(X)
print(f"Question 5 -> Formed Clusters Count: {agg_thresh.n_clusters_}")
```
