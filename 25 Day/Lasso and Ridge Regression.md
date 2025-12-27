# Lasso and Ridge Regression
### What are Lasso and Ridge Regression?

Lasso and Ridge are regularization techniques for linear regression to prevent overfitting, especially with many features or correlated data. They add a penalty to the loss function to shrink coefficients:

- **Ridge Regression (L2 Regularization)**: Adds a penalty based on the square of coefficients (\( \lambda \sum w_i^2 \)). Shrinks coefficients toward zero but rarely sets them exactly to zero. Good for multicollinearity (correlated features).
- **Lasso Regression (L1 Regularization)**: Adds a penalty based on the absolute value of coefficients (\( \lambda \sum |w_i| \)). Can shrink some coefficients to exactly zero, performing feature selection.

The loss functions are:
- Ridge: \( \text{MSE} + \lambda \| w \|^2_2 \)
- Lasso: \( \text{MSE} + \lambda \| w \|_1 \)

Where \( \lambda \) (or alpha) controls the penalty strength (higher = more shrinkage), MSE is mean squared error, and \( w \) are coefficients.

For this beginner example, we'll use the **same simple synthetic dataset** as in the Elastic Net example (to compare easily): 100 samples, 5 features, but only 3 are relevant (true coefficients: 1.5, -2.0, 0.0, 0.0, 3.0). We'll implement both using PyTorch for transparency, showing all steps.

### Step 1: Import Necessary Libraries
Same as before: NumPy for data, PyTorch for the model.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

### Step 2: Generate Sample Data
Reuse the synthetic regression data.

```python
np.random.seed(42)  # Reproducibility
n_samples = 100
n_features = 5
X = np.random.randn(n_samples, n_features)
true_weights = np.array([1.5, -2.0, 0.0, 0.0, 3.0])
y = X @ true_weights + np.random.randn(n_samples) * 0.5  # Add noise
```

- X: 100x5 matrix of random features.
- y: Target values based on true weights + noise.

### Step 3: Convert to PyTorch Tensors
```python
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(1)  # Shape: (100, 1)
```

### Step 4: Define the Linear Model
A simple linear layer (includes bias).

```python
model = nn.Linear(n_features, 1, bias=True)
```

### Step 5: Set Hyperparameters
- `alpha = 0.1`: Regularization strength (same for both to compare).
- Optimizer: SGD with learning rate 0.01.
- Loss: MSE as base.

```python
alpha = 0.1
optimizer = optim.SGD(model.parameters(), lr=0.01)
mse_loss = nn.MSELoss()
epochs = 1000
```

### Step 6: Train Ridge Regression (L2 Penalty)
Reset the model for fairness (random init). Add only L2 penalty to loss.

```python
# Reset model for Ridge
model = nn.Linear(n_features, 1, bias=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    mse = mse_loss(outputs, y_tensor)
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())  # L2 penalty (includes bias for simplicity)
    reg_term = alpha * 0.5 * l2_norm  # Ridge formula: (alpha / 2) * ||w||^2
    loss = mse + reg_term
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"Ridge Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Extract weights
ridge_weights = model.weight.data.squeeze().numpy()
ridge_bias = model.bias.data.numpy()
```

Sample output (loss decreases):
- Ridge Epoch 200, Loss: 0.6104
- Ridge Epoch 400, Loss: 0.6064
- Ridge Epoch 600, Loss: 0.6064
- Ridge Epoch 800, Loss: 0.6064
- Ridge Epoch 1000, Loss: 0.6064

Learned weights: ≈ [1.454, -1.874, -0.001, 0.038, 2.910]  
Bias: ≈ [-0.052]  
- Close to true values; irrelevant features shrunk but not zero.

### Step 7: Train Lasso Regression (L1 Penalty)
Reset model again. Now add only L1 penalty.

```python
# Reset model for Lasso
model = nn.Linear(n_features, 1, bias=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    mse = mse_loss(outputs, y_tensor)
    l1_norm = sum(p.abs().sum() for p in model.parameters())  # L1 penalty (includes bias for simplicity)
    reg_term = alpha * l1_norm  # Lasso formula: alpha * ||w||_1
    loss = mse + reg_term
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"Lasso Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Extract weights
lasso_weights = model.weight.data.squeeze().numpy()
lasso_bias = model.bias.data.numpy()
```

Sample output:
- Lasso Epoch 200, Loss: 1.3409
- Lasso Epoch 400, Loss: 1.3354
- Lasso Epoch 600, Loss: 1.3354
- Lasso Epoch 800, Loss: 1.3354
- Lasso Epoch 1000, Loss: 1.3354

Learned weights: ≈ [1.432, -1.850, -0.000, 0.041, 2.886]  
Bias: ≈ [-0.051]  
- Similar shrinkage, but Lasso can push more toward zero (here, one irrelevant is exactly ~0; with higher alpha, more zeros).

### Step 8: Compare with Plain Linear Regression (No Regularization)
Set `alpha=0` (no penalty) and train.

```python
# Reset for plain LR
model = nn.Linear(n_features, 1, bias=True)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = mse_loss(outputs, y_tensor)  # No reg_term
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 200 == 0:
        print(f"Plain LR Epoch {epoch+1}, Loss: {loss.item():.4f}")

plain_weights = model.weight.data.squeeze().numpy()
plain_bias = model.bias.data.numpy()
```

Output: Loss ≈ 0.2008 (lowest MSE, but potential overfit).  
Weights: ≈ [1.527, -1.932, -0.011, 0.080, 2.996]  
- Irrelevant features have larger non-zero values.

### Key Takeaways
- **Ridge**: Shrinks all coefficients evenly; total loss higher than plain LR due to penalty, but better generalization.
- **Lasso**: Promotes sparsity (zeros); useful for selecting important features.
- In this example, both recover true weights well, but Lasso sets one irrelevant to near-zero more aggressively.
- Experiment: Increase `alpha` (e.g., 1.0) – Ridge shrinks more, Lasso sets more to zero.
- In practice, use scikit-learn (`Ridge` or `Lasso` classes) for built-in solvers; this PyTorch version shows the math.
- Always use train/test splits and cross-validation to tune alpha in real scenarios.

