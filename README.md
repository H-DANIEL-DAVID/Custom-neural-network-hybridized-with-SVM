# ⚡ Dynamic Multi-Layer SVM Engine (Pure NumPy)

A deep neural network architecture with a Maximum-Margin (SVM) classification head built completely from scratch using **NumPy**. This engine features arbitrary hidden-layer depth, dynamic weight allocation, **Nesterov Accelerated Gradient (NAG)** momentum, and subgradient backpropagation for Hinge Loss without relying on external autograd frameworks like PyTorch or TensorFlow.

---

## 🛠️ Key Features

* **Zero-Framework Autograd:** Full backpropagation algorithm, weight updates, and loss gradients derived and vectorized in pure NumPy.
* **Dynamic Depth Allocation:** Instantiate any $L$-layer deep network by passing input dimensions, hidden unit counts, and layer depth.
* **Nesterov Accelerated Gradient (NAG):** Implements lookahead weight evaluation ($W_{lookahead} = W - \gamma V$) for faster convergence around local minima compared to standard momentum.
* **Maximum-Margin Loss Head:** Replaces standard Cross-Entropy/Softmax output with an SVM Hinge Loss formulation L = max(0,1 - y*(Wx + B)+ $L_2$ regularization penalty.
* **He/Kaiming Weight Initialization:** Prevents vanishing/exploding gradients in deep hidden layers.
* **LeakyReLU Activations:** Prevents dying ReLU neurons using a $0.01$ slope leak for negative values.
* **Mini batching** For faster gradient descent with improved accuracy.

---

## 📐 Mathematical Formulation

### 1. Nesterov Lookahead Step
Before computing gradients during forward-propagation, weights and biases compute lookahead states:
$$W_{lookahead} = W - \gamma \cdot V_W$$
$$B_{lookahead} = B - \gamma \cdot V_B$$

### 2. Output & Hinge Loss Subgradient
For binary targets $Y \in \{-1, +1\}$ and final linear activation $h$:
$$\text{Margin} = Y \cdot h$$
$$\mathcal{L}_{hinge} = \max(0, 1 - Y \cdot h)$$

The subgradient $\delta$ passing back from the top layer is:
$$\delta_{output} = \mathbb{I}(1 - Y \cdot h > 0) \odot (-Y)$$

### 3. Layer Gradient & $L_2$ Regularization
For layer weights $W_m$ and mini-batch size $N_{batch}$:
$$\nabla W_m = \frac{1}{N_{batch}} \left( Z_{m-1}^T \delta_m + \frac{1}{C} W_m \right)$$
$$V_W \leftarrow \gamma \cdot V_W + \eta \cdot \nabla W_m$$
$$W_m \leftarrow W_m - V_W$$

---

## 🚀 Quickstart & Usage

### Prerequisites
Make sure you have `numpy` and `matplotlib` installed:
```bash
pip install numpy matplotlib
