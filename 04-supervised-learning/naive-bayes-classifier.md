# Naive Bayes Probabilistic Classifiers & MAP Decision Theory

[← Back to Course README](../README.md)

- [1. Maximum A Posteriori (MAP) Decision Theory](#1-maximum-a-posteriori-map-decision-theory)
- [2. Naive Conditional Independence Assumption](#2-naive-conditional-independence-assumption)
- [3. Likelihood Distributions & Variants](#3-likelihood-distributions-variants)
  - [3.1 Gaussian Naive Bayes (Continuous Features)](#31-gaussian-naive-bayes-continuous-features)
  - [3.2 Multinomial Naive Bayes (Word Counts)](#32-multinomial-naive-bayes-word-counts)
  - [3.3 Bernoulli Naive Bayes (Binary Indicators)](#33-bernoulli-naive-bayes-binary-indicators)
- [4. Laplace (Additive) & Lidstone Smoothing](#4-laplace-additive-lidstone-smoothing)
- [5. Python Mechanics: From-Scratch Gaussian NB & Scikit-Learn Suite](#5-python-mechanics-from-scratch-gaussian-nb-scikit-learn-suite)
- [6. End-to-End Industry Case Study: NLP Email Spam Filter](#6-end-to-end-industry-case-study-nlp-email-spam-filter)
- [7. 5 Solved Analytical & Numerical Practice Questions](#7-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Prior Probabilities & Bayes Theorem](#question-1-prior-probabilities-bayes-theorem)
  - [Question 2: Laplace Smoothed Probability Calculation](#question-2-laplace-smoothed-probability-calculation)
  - [Question 3: Gaussian Probability Density Function](#question-3-gaussian-probability-density-function)
  - [Question 4: Multinomial Log-Likelihood Product Sum](#question-4-multinomial-log-likelihood-product-sum)
  - [Question 5: Bernoulli Naive Bayes Likelihood Calculation](#question-5-bernoulli-naive-bayes-likelihood-calculation)
> **Topic**: Naive Bayes Probabilistic Classifiers & MAP Decision Theory: 1. Maximum A Posteriori (MAP) Decision Theory, 2. Naive Conditional Independence Assumption, 3. Likelihood Distributions & Variants, 4. Laplace (Additive) & Lidstone Smoothing

---

## 1. Maximum A Posteriori (MAP) Decision Theory

Assigns sample $\mathbf{x} = (x_1, \dots, x_d)$ to class $c$ maximizing posterior probability:

$$P(Y=c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid Y=c) P(Y=c)}{P(\mathbf{x})}$$

Since evidence $P(\mathbf{x})$ is constant across classes:

$$\hat{y}_{\text{MAP}} = \arg\max_{c \in \{1, \dots, K\}} P(Y=c) P(\mathbf{x} \mid Y=c)$$

---

## 2. Naive Conditional Independence Assumption

Assumes features $x_1, \dots, x_d$ are conditionally independent given class label $c$:

$$P(\mathbf{x} \mid Y=c) = \prod_{j=1}^d P(x_j \mid Y=c)$$

$$\hat{y}_{\text{MAP}} = \arg\max_{c} \left[ P(Y=c) \prod_{j=1}^d P(x_j \mid Y=c) \right]$$

In log-probability space to prevent numerical underflow:

$$\hat{y}_{\text{MAP}} = \arg\max_{c} \left[ \ln P(Y=c) + \sum_{j=1}^d \ln P(x_j \mid Y=c) \right]$$

---

## 3. Likelihood Distributions & Variants

### 3.1 Gaussian Naive Bayes (Continuous Features)
Models continuous feature $x_j$ in class $c$ as Gaussian $\mathcal{N}(\mu_{cj}, \sigma_{cj}^2)$:

$$P(x_j \mid Y=c) = \frac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left( -\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} \right)$$

### 3.2 Multinomial Naive Bayes (Word Counts)
Models discrete word frequency vectors $\mathbf{x}$ over vocabulary size $V$:

$$P(\mathbf{x} \mid Y=c) = \frac{(\sum x_j)!}{\prod x_j!} \prod_{j=1}^V \theta_{cj}^{x_j}$$

### 3.3 Bernoulli Naive Bayes (Binary Indicators)
Models binary feature presence/absence indicators $x_j \in \{0, 1\}$:

$$P(\mathbf{x} \mid Y=c) = \prod_{j=1}^d \theta_{cj}^{x_j} (1 - \theta_{cj})^{1 - x_j}$$

---

## 4. Laplace (Additive) & Lidstone Smoothing

Prevents zero-probability propagation for unseen features by adding smoothing parameter $\alpha > 0$ (Lidstone: $0 < \alpha < 1$; Laplace: $\alpha = 1$):

$$\hat{\theta}_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha V}$$

Where $N_{cj}$ is count of feature $j$ in class $c$, $N_c$ is total feature counts in class $c$, and $V$ is vocabulary size.

---

## 5. Python Mechanics: From-Scratch Gaussian NB & Scikit-Learn Suite

```python
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_iris

class FromScratchGaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = {}
        self.vars_ = {}
        self.priors_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            self.means_[c] = np.mean(X_c, axis=0)
            self.vars_[c] = np.var(X_c, axis=0) + 1e-9
            self.priors_[c] = len(X_c) / len(X)
            
        return self
        
    def predict(self, X):
        preds = []
        for x in X:
            posteriors = {}
            for c in self.classes_:
                prior = np.log(self.priors_[c])
                gaussian_pdf = -0.5 * np.sum(np.log(2 * np.pi * self.vars_[c])) - 0.5 * np.sum(((x - self.means_[c])**2) / self.vars_[c])
                posteriors[c] = prior + gaussian_pdf
            preds.append(max(posteriors, key=posteriors.get))
        return np.array(preds)

# Execute Comparison
X_i, y_i = load_iris(return_X_y=True)
custom_gnb = FromScratchGaussianNB().fit(X_i, y_i)
sklearn_gnb = GaussianNB().fit(X_i, y_i)

print(f"Custom GNB Accuracy:  {np.mean(custom_gnb.predict(X_i) == y_i):.4f}")
print(f"Sklearn GNB Accuracy: {sklearn_gnb.score(X_i, y_i):.4f}")
```

---

## 6. End-to-End Industry Case Study: NLP Email Spam Filter

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

emails = [
    "Cheap loans available now, click here",
    "Meeting agenda for project sprint review",
    "Earn money fast with crypto investments",
    "Quarterly financial earnings report attached",
    "Win a free iPhone by taking survey"
]
labels = [1, 0, 1, 0, 1] # 1: Spam, 0: Ham

spam_pipeline = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))
spam_pipeline.fit(emails, labels)

test_email = ["Project review meeting schedule"]
pred_spam = spam_pipeline.predict(test_email)
print(f"Test Email Spam Prediction: {pred_spam[0]} ({'Spam' if pred_spam[0]==1 else 'Ham'})")
```

---

## 7. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Prior Probabilities & Bayes Theorem
**Problem**: Given $P(\text{Spam}) = 0.30$, $P(\text{Ham}) = 0.70$, $P(\text{"crypto"} \mid \text{Spam}) = 0.80$, and $P(\text{"crypto"} \mid \text{Ham}) = 0.05$, compute posterior probability $P(\text{Spam} \mid \text{"crypto"})$.
```python
# Solution
p_s, p_h = 0.30, 0.70
p_w_s, p_w_h = 0.80, 0.05

p_w = p_w_s * p_s + p_w_h * p_h # 0.24 + 0.035 = 0.275
p_s_w = (p_w_s * p_s) / p_w
print(f"Question 1 -> P(Spam|crypto): {p_s_w:.4f}") # 0.24 / 0.275 = 0.8727
```

### Question 2: Laplace Smoothed Probability Calculation
**Problem**: Word "bonus" appears 0 times in Ham class ($N_{\text{Ham}} = 400$ total words). Compute smoothed probability with $\alpha = 1.0$ and vocabulary size $V = 100$.
```python
# Solution
n_w, n_class, alpha, V = 0, 400, 1.0, 100
theta_laplace = (n_w + alpha) / (n_class + alpha * V)
print(f"Question 2 -> Laplace Smoothed Prob: {theta_laplace:.4f}") # 1 / 500 = 0.0020
```

### Question 3: Gaussian Probability Density Function
**Problem**: Compute Gaussian PDF for $x = 12$ given class mean $\mu = 10$ and variance $\sigma^2 = 4$.
```python
# Solution
import math

x, mu, var = 12.0, 10.0, 4.0
pdf_val = (1.0 / math.sqrt(2.0 * math.pi * var)) * math.exp(-((x - mu)**2) / (2.0 * var))
print(f"Question 3 -> Gaussian PDF value: {pdf_val:.4f}") # ~0.1210
```

### Question 4: Multinomial Log-Likelihood Product Sum
**Problem**: For class $c$, prior $P(c) = 0.5$ and feature log-likelihoods $\ln P(x_1|c) = -1.2, \ln P(x_2|c) = -2.3$. Compute log posterior score.
```python
# Solution
prior = 0.5
log_prior = math.log(prior)
log_likes = [-1.2, -2.3]

log_posterior = log_prior + sum(log_likes)
print(f"Question 4 -> Log Posterior Score: {log_posterior:.4f}") # -0.6931 - 3.5 = -4.1931
```

### Question 5: Bernoulli Naive Bayes Likelihood Calculation
**Problem**: Calculate Bernoulli likelihood for binary vector $\mathbf{x} = [1, 0]$ with parameter probabilities $\theta = [0.8, 0.3]$.
```python
# Solution
theta = [0.8, 0.3]
x = [1, 0]

lik = (theta[0]**x[0] * (1 - theta[0])**(1 - x[0])) * (theta[1]**x[1] * (1 - theta[1])**(1 - x[1]))
print(f"Question 5 -> Bernoulli Likelihood: {lik:.4f}") # 0.8 * 0.7 = 0.5600
```
