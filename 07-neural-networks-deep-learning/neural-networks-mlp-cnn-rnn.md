# Neural Networks, Backpropagation & Deep Learning Architectures

[← Back to Course README](../README.md)

- [1. Single-Layer Neural Network (Perceptron)](#1-single-layer-neural-network-perceptron)
- [2. Multilayer Perceptron (MLP) & Activation Functions](#2-multilayer-perceptron-mlp-activation-functions)
  - [Activation Functions](#activation-functions)
- [3. Backpropagation Learning Algorithm Derivation](#3-backpropagation-learning-algorithm-derivation)
- [4. Convolutional Neural Networks (CNN)](#4-convolutional-neural-networks-cnn)
  - [4.1 Conv2D Spatial Convolution & Receptive Fields](#41-conv2d-spatial-convolution-receptive-fields)
  - [4.2 Output Spatial Dimension Formula](#42-output-spatial-dimension-formula)
- [5. Recurrent Neural Networks (RNN & LSTM)](#5-recurrent-neural-networks-rnn-lstm)
- [6. PyTorch & Scikit-Learn Deep Learning Implementation](#6-pytorch-scikit-learn-deep-learning-implementation)
- [7. 5 Solved Analytical & Numerical Practice Questions](#7-5-solved-analytical-numerical-practice-questions)
  - [Question 1: Perceptron Weight Update Step](#question-1-perceptron-weight-update-step)
  - [Question 2: CNN Spatial Output Dimension Calculation](#question-2-cnn-spatial-output-dimension-calculation)
  - [Question 3: Softmax Probabilities Computation](#question-3-softmax-probabilities-computation)
  - [Question 4: ReLU Derivative Evaluation](#question-4-relu-derivative-evaluation)
  - [Question 5: MLP Hidden Layer Forward Pass](#question-5-mlp-hidden-layer-forward-pass)
> **Topic**: Neural Networks, Backpropagation & Deep Learning Architectures: 1. Single-Layer Neural Network (Perceptron), 2. Multilayer Perceptron (MLP) & Activation Functions, 3. Backpropagation Learning Algorithm Derivation, 4. Convolutional Neural Networks (CNN)

---

## 1. Single-Layer Neural Network (Perceptron)

Computes weighted sum of inputs followed by step activation function:

$$\hat{y} = f(\mathbf{w}^T \mathbf{x} + b) = \text{sgn}\left( \sum_{j=1}^d w_j x_j + b \right)$$

---

## 2. Multilayer Perceptron (MLP) & Activation Functions

MLPs resolve non-linearly separable problems by stacking hidden layers:

$$\mathbf{h} = g(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}), \quad \mathbf{\hat{y}} = \sigma(W^{(2)} \mathbf{h} + \mathbf{b}^{(2)})$$

### Activation Functions
* **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
* **ReLU**: $\text{ReLU}(z) = \max(0, z)$
* **Softmax**: $\text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$

---

## 3. Backpropagation Learning Algorithm Derivation

Using chain rule to compute loss gradients $\frac{\partial L}{\partial W^{(l)}}$ for weight updates:

$$\boldsymbol{\delta}^{(2)} = \nabla_{\mathbf{a}^{(2)}} L \odot \sigma'(\mathbf{a}^{(2)})$$

$$\boldsymbol{\delta}^{(1)} = \left( (W^{(2)})^T \boldsymbol{\delta}^{(2)} \right) \odot g'(\mathbf{a}^{(1)})$$

$$\frac{\partial L}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T$$

---

## 4. Convolutional Neural Networks (CNN)

### 4.1 Conv2D Spatial Convolution & Receptive Fields
Applies spatial filters $K \in \mathbb{R}^{F \times F}$ over input channels to produce feature maps.

### 4.2 Output Spatial Dimension Formula
For input width $W$, filter size $F$, padding $P$, and stride $S$:

$$O = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1$$

---

## 5. Recurrent Neural Networks (RNN & LSTM)

Processes sequential data $\mathbf{x}_1, \dots, \mathbf{x}_T$ with recurrent hidden state $\mathbf{h}_t$:

$$\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

---

## 6. PyTorch & Scikit-Learn Deep Learning Implementation

```python
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

# PyTorch MLP Definition
class SimpleMLP(nn.Module):
    def __init__(self, in_features, hidden, out_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_classes)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Sklearn MLP Training
X_d, y_d = load_digits(return_X_y=True)
mlp_sk = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42).fit(X_d, y_d)
print(f"MLPClassifier Accuracy on Digits Dataset: {mlp_sk.score(X_d, y_d):.4f}")
```

---

## 7. 5 Solved Analytical & Numerical Practice Questions

### Question 1: Perceptron Weight Update Step
**Problem**: Weight $\mathbf{w} = [0, 0], b = 0$, learning rate $\eta = 1.0$. Misclassified point $\mathbf{x} = [2, 3]$ with true label $y = 1$. Compute updated weights.
```python
# Solution
import numpy as np

w = np.array([0.0, 0.0])
b = 0.0
x = np.array([2.0, 3.0])
y = 1.0
eta = 1.0

w_new = w + eta * y * x # [2.0, 3.0]
b_new = b + eta * y # 1.0

print(f"Question 1 -> Updated Weight: {w_new}, Updated Bias: {b_new:.1f}")
```

### Question 2: CNN Spatial Output Dimension Calculation
**Problem**: Input spatial dimension $W = 64$, filter size $F = 3$, padding $P = 1$, and stride $S = 2$. Compute output dimension $O$.
```python
# Solution
import math

W, F, P, S = 64, 3, 1, 2
O = math.floor((W - F + 2*P) / S) + 1 # floor((64 - 3 + 2)/2) + 1 = floor(63/2) + 1 = 31 + 1 = 32
print(f"Question 2 -> CNN Output Spatial Dimension O: {O}")
```

### Question 3: Softmax Probabilities Computation
**Problem**: Raw logits vector $z = [2.0, 1.0, 0.1]$. Compute Softmax probabilities.
```python
# Solution
z = np.array([2.0, 1.0, 0.1])
exp_z = np.exp(z)
softmax_p = exp_z / np.sum(exp_z)
print(f"Question 3 -> Softmax Probabilities: {np.round(softmax_p, 4)}") # [0.659, 0.2424, 0.0986]
```

### Question 4: ReLU Derivative Evaluation
**Problem**: Evaluate derivative $\text{ReLU}'(z)$ for $z = 4.0$ vs $z = -1.5$.
```python
# Solution
z_pos, z_neg = 4.0, -1.5
d_pos = 1.0 if z_pos > 0 else 0.0
d_neg = 1.0 if z_neg > 0 else 0.0
print(f"Question 4 -> ReLU'(4.0): {d_pos:.1f}, ReLU'(-1.5): {d_neg:.1f}") # 1.0, 0.0
```

### Question 5: MLP Hidden Layer Forward Pass
**Problem**: For input $\mathbf{x} = [1, 1]$, weights $W = \begin{bmatrix} 0.5 & 0.5 \\ 0.2 & 0.8 \end{bmatrix}$, and zero bias, compute ReLU output.
```python
# Solution
x = np.array([1, 1])
W = np.array([[0.5, 0.5], [0.2, 0.8]])
h_raw = W @ x # [1.0, 1.0]
h_act = np.maximum(0, h_raw)
print(f"Question 5 -> MLP Hidden Layer Output: {h_act}")
```
