# Exploratory Data Analysis: Univariate, Bivariate & Multivariate Systems

[← Back to Course README](../README.md)

- [1. Principles of Exploratory Data Analysis](#1-principles-of-exploratory-data-analysis)
- [2. Univariate Analysis & Distribution Metrics](#2-univariate-analysis-distribution-metrics)
  - [2.1 Measures of Central Tendency & Dispersion](#21-measures-of-central-tendency-dispersion)
  - [2.2 Skewness & Asymmetry Metrics](#22-skewness-asymmetry-metrics)
  - [2.3 Kurtosis & Tail-Fatness Metrics](#23-kurtosis-tail-fatness-metrics)
- [3. Bivariate Analysis Dynamics](#3-bivariate-analysis-dynamics)
  - [3.1 Continuous vs Continuous Interactions](#31-continuous-vs-continuous-interactions)
  - [3.2 Categorical vs Continuous Interactions](#32-categorical-vs-continuous-interactions)
  - [3.3 Categorical vs Categorical Contingency Analysis](#33-categorical-vs-categorical-contingency-analysis)
- [4. Multivariate Interaction Topologies](#4-multivariate-interaction-topologies)
- [5. Python Mechanics: Automated EDA Suite & Seaborn Visualizations](#5-python-mechanics-automated-eda-suite-seaborn-visualizations)
- [6. End-to-End Industry Case Study: Customer Churn Feature Profiling](#6-end-to-end-industry-case-study-customer-churn-feature-profiling)
- [7. 5 Solved Analytical & Numerical Practice Questions](#7-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Sample Skewness Calculation](#question-1-sample-skewness-calculation)
  - [Question 2: Fisher Excess Kurtosis Calculation](#question-2-fisher-excess-kurtosis-calculation)
  - [Question 3: Interquartile Range (IQR) & Quantiles](#question-3-interquartile-range-iqr-quantiles)
  - [Question 4: Expected Frequency Matrix in Crosstab](#question-4-expected-frequency-matrix-in-crosstab)
  - [Question 5: Bivariate Mean Comparison](#question-5-bivariate-mean-comparison)
> **Topic**: Exploratory Data Analysis: Univariate, Bivariate & Multivariate Systems: 1. Principles of Exploratory Data Analysis, 2. Univariate Analysis & Distribution Metrics, 3. Bivariate Analysis Dynamics, 4. Multivariate Interaction Topologies

---

## 1. Principles of Exploratory Data Analysis

Exploratory Data Analysis (EDA), formalized by John Tukey (1977), uses statistical summaries and graphical visualizations to uncover underlying distribution geometry, detect anomalies, check mathematical assumptions, and select predictive features.

---

## 2. Univariate Analysis & Distribution Metrics

### 2.1 Measures of Central Tendency & Dispersion
* **Mean**: $\bar{x} = \frac{1}{n} \sum x_i$
* **Median**: Value splitting ordered dataset in half (Robust to outliers).
* **Variance**: $s^2 = \frac{1}{n-1} \sum (x_i - \bar{x})^2$
* **Standard Deviation**: $s = \sqrt{s^2}$

### 2.2 Skewness & Asymmetry Metrics
Quantifies distributional asymmetry relative to normal distribution:

$$\text{Skewness} = \frac{\mathbb{E}[(X - \mu)^3]}{\sigma^3} = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^3}{\left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^{3/2}}$$

* $\text{Skew} > 0$: Right-skewed (positive tail extends right; Mean > Median).
* $\text{Skew} = 0$: Symmetric distribution.
* $\text{Skew} < 0$: Left-skewed (negative tail extends left; Mean < Median).

### 2.3 Kurtosis & Tail-Fatness Metrics
Measures tail heaviness and propensity for extreme outliers:

$$\text{Kurtosis} = \frac{\mathbb{E}[(X - \mu)^4]}{\sigma^4} = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^4}{\left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^2}$$

* **Excess Kurtosis** = $\text{Kurtosis} - 3.0$.
* **Mesokurtic ($\text{Excess} = 0$)**: Standard Normal Distribution.
* **Leptokurtic ($\text{Excess} > 0$)**: Heavy tails and sharp peak (High outlier risk).
* **Platykurtic ($\text{Excess} < 0$)**: Light tails and flat peak.

---

## 3. Bivariate Analysis Dynamics

### 3.1 Continuous vs Continuous Interactions
Analyzes linear/non-linear dependencies using scatter plots, joint plots, and Pearson correlation coefficients.

### 3.2 Categorical vs Continuous Interactions
Evaluates group distribution differences using Box Plots, Violin Plots, and ANOVA $F$-tests.

### 3.3 Categorical vs Categorical Contingency Analysis
Constructs $r \times c$ contingency matrices and tests independence via Chi-Square test:

$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

---

## 4. Multivariate Interaction Topologies

Explores multi-variable relationships using correlation heatmaps, 3D scatter plots, pairplots with class hue encodings, and parallel coordinates.

---

## 5. Python Mechanics: Automated EDA Suite & Seaborn Visualizations

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Generate synthetic multivariate dataset
np.random.seed(42)
n = 300

df_eda = pd.DataFrame({
    'age': np.random.normal(40, 10, n),
    'income': np.random.lognormal(mean=10.5, sigma=0.75, size=n),
    'spending_score': np.random.uniform(1, 100, n),
    'segment': np.random.choice(['Low', 'Medium', 'High'], size=n)
})

# Compute Univariate Statistical Metrics
stats_summary = pd.DataFrame({
    'Mean': df_eda[['age', 'income', 'spending_score']].mean(),
    'Std': df_eda[['age', 'income', 'spending_score']].std(),
    'Skewness': df_eda[['age', 'income', 'spending_score']].apply(skew),
    'Excess_Kurtosis': df_eda[['age', 'income', 'spending_score']].apply(lambda x: kurtosis(x, fisher=True))
})

print("Univariate EDA Statistical Metrics:")
print(stats_summary)
```

---

## 6. End-to-End Industry Case Study: Customer Churn Feature Profiling

```python
# Churn Exploratory Data Analysis Pipeline
df_churn = pd.DataFrame({
    'tenure': np.random.exponential(scale=24, size=500),
    'monthly_charges': np.random.normal(70, 20, size=500),
    'contract_type': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], size=500),
    'churn': np.random.choice([0, 1], size=500, p=[0.75, 0.25])
})

# Bivariate Contingency Table: Contract Type vs Churn
contingency_churn = pd.crosstab(df_churn['contract_type'], df_churn['churn'], normalize='index')
print("Normalized Churn Rate by Contract Type:")
print(contingency_churn * 100)
```

---

## 7. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Sample Skewness Calculation
**Problem**: Calculate sample skewness for sample $X = [2, 4, 5, 6, 25]$.
```python
# Solution
import numpy as np
from scipy.stats import skew

data = np.array([2, 4, 5, 6, 25])
sk_val = skew(data, bias=False)
print(f"Question 1 -> Sample Skewness: {sk_val:.4f} (Positively/Right Skewed)") # ~2.08
```

### Question 2: Fisher Excess Kurtosis Calculation
**Problem**: Calculate Fisher excess kurtosis for sample $X = [10, 12, 11, 13, 12, 50]$.
```python
# Solution
from scipy.stats import kurtosis

data_k = np.array([10, 12, 11, 13, 12, 50])
kurt_val = kurtosis(data_k, fisher=True, bias=False)
print(f"Question 2 -> Excess Kurtosis: {kurt_val:.4f} (Leptokurtic tail)")
```

### Question 3: Interquartile Range (IQR) & Quantiles
**Problem**: Compute $Q_1$, $Q_3$, and IQR for continuous array `[10, 15, 25, 30, 42, 50, 65, 80]`.
```python
# Solution
import pandas as pd

s = pd.Series([10, 15, 25, 30, 42, 50, 65, 80])
q1 = s.quantile(0.25)
q3 = s.quantile(0.75)
iqr = q3 - q1
print(f"Question 3 -> Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}") # Q1=22.50, Q3=53.75, IQR=31.25
```

### Question 4: Expected Frequency Matrix in Crosstab
**Problem**: For $2 \times 2$ observed table `[[40, 10], [20, 30]]`, calculate Expected Frequency matrix $E$.
```python
# Solution
obs = np.array([[40, 10], [20, 30]])
row_sums = obs.sum(axis=1)
col_sums = obs.sum(axis=0)
total = obs.sum()

E = np.outer(row_sums, col_sums) / total
print(f"Question 4 -> Expected Frequency Matrix E:")
print(E)
```

### Question 5: Bivariate Mean Comparison
**Problem**: Compute group mean and variance of `salary` grouped by `experience_level` (`Junior` vs `Senior`).
```python
# Solution
df_sal = pd.DataFrame({
    'experience_level': ['Junior', 'Junior', 'Junior', 'Senior', 'Senior', 'Senior'],
    'salary': [50000, 55000, 52000, 110000, 125000, 130000]
})
summary = df_sal.groupby('experience_level')['salary'].agg(['mean', 'var'])
print("Question 5 -> Salary Group Statistics:")
print(summary)
```
