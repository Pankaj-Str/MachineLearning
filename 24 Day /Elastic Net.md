
### What is Elastic Net in Machine Learning?

Elastic Net is a regularization technique used in linear regression models to prevent overfitting and perform feature selection. It combines the strengths of two other regularization methods:
- **Lasso (L1 regularization)**: Adds a penalty equal to the absolute value of the coefficients, which can shrink some coefficients to exactly zero, effectively selecting a subset of features.
- **Ridge (L2 regularization)**: Adds a penalty equal to the square of the coefficients, which shrinks coefficients but rarely sets them to zero, helping with multicollinearity (when features are highly correlated).

Elastic Net uses a mix of both penalties, controlled by a parameter called `rho` (or `l1_ratio`). The formula for the loss function in Elastic Net Regression is:

\[
\text{Loss} = \frac{1}{2n} \| y - Xw \|^2 + \alpha \rho \| w \|_1 + \frac{\alpha (1 - \rho)}{2} \| w \|^2_2
\]

Where:
- \( \| y - Xw \|^2 \) is the mean squared error (MSE).
- \( \alpha \) is the overall regularization strength.
- \( \rho \) controls the balance between L1 and L2 (e.g., \(\rho = 1\) is pure Lasso, \(\rho = 0\) is pure Ridge).
- \( w \) are the model weights (coefficients).

This is useful when you have many features, some irrelevant or correlated, as it balances shrinkage and sparsity.

For this beginner example, we'll use synthetic data and implement Elastic Net using PyTorch (a deep learning library that can handle custom losses). We'll train a simple linear model with the Elastic Net penalty added to the loss.

### Step 1: Import Necessary Libraries
We need NumPy for data generation, and PyTorch for the model, optimizer, and loss.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
```

This imports successfully in our environment.

### Step 2: Generate Sample Data
We'll create a synthetic regression dataset with 100 samples and 5 features. The true relationship uses only 3 of the features (coefficients: 1.5, -2.0, 3.0), while the other two are irrelevant (coefficients: 0.0). We add some noise for realism.

```python
np.random.seed(42)  # For reproducibility
n_samples = 100
n_features = 5
X = np.random.randn(n_samples, n_features)
true_weights = np.array([1.5, -2.0, 0.0, 0.0, 3.0])
y = X @ true_weights + np.random.randn(n_samples) * 0.5
```

Output:
- X shape: (100, 5)
- y shape: (100,)
- True weights: [ 1.5 -2.   0.   0.   3. ]

### Step 3: Convert Data to PyTorch Tensors
PyTorch works with tensors, so we convert our NumPy arrays.

```python
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(1)  # Reshape y to (100, 1)
```

Output:
- X_tensor shape: torch.Size([100, 5])
- y_tensor shape: torch.Size([100, 1])

### Step 4: Define the Linear Model
We use PyTorch's `nn.Linear` for a simple linear regression model (with bias/intercept).

```python
model = nn.Linear(n_features, 1, bias=True)
```

This creates a model with random initial weights.

### Step 5: Set Hyperparameters
- `alpha = 0.1`: Regularization strength (higher values mean more shrinkage).
- `rho = 0.5`: Equal mix of L1 and L2 penalties.
- Learning rate for optimizer: 0.01.
- Epochs: 1000 (training iterations).

We'll also define the optimizer (SGD) and base MSE loss.

```python
alpha = 0.1
rho = 0.5
optimizer = optim.SGD(model.parameters(), lr=0.01)
mse_loss = nn.MSELoss()
```

### Step 6: Train the Model with Elastic Net Loss
In the training loop:
- Compute predictions.
- Calculate MSE.
- Add L1 and L2 penalties (note: for simplicity, we apply regularization to all parameters, including bias; in practice, bias is often excluded).
- Compute total loss = MSE + regularization term.
- Backpropagate and update weights.

```python
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Reset gradients
    outputs = model(X_tensor)  # Forward pass
    mse = mse_loss(outputs, y_tensor)  # Base loss
    l1_norm = sum(p.abs().sum() for p in model.parameters())  # L1 penalty
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())  # L2 penalty
    reg_term = alpha * rho * l1_norm + alpha * (1 - rho) * 0.5 * l2_norm
    loss = mse + reg_term  # Total Elastic Net loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

Sample training output (loss decreases and stabilizes):
- Epoch 200, Loss: 0.8919
- Epoch 400, Loss: 0.8864
- Epoch 600, Loss: 0.8864
- Epoch 800, Loss: 0.8864
- Epoch 1000, Loss: 0.8864

The final MSE (without reg term) was approximately 0.2217, and the reg term was 0.6646.

### Step 7: View Learned Weights
After training, extract the model's weights and bias.

```python
learned_weights = model.weight.data.squeeze().numpy()
bias = model.bias.data.numpy()
```

Example output:
- Learned weights: [ 1.4448142  -1.8656795  -0.00039757  0.03958775  2.8996177 ]
- Bias: [-0.05246321]

Compare to true weights [1.5, -2.0, 0.0, 0.0, 3.0]:
- The model recovers the important coefficients reasonably well.
- Irrelevant features (3rd and 4th) are shrunk close to zero due to regularization.

### Comparison: Without Regularization (Plain Linear Regression)
If we set `alpha = 0.0` and retrain (same code, just no reg term), the results are:
- Final Loss (MSE only): 0.2008
- Learned weights: [ 1.5271012  -1.9322481  -0.01064606  0.0803731   2.996379  ]
- Bias: [-0.07306219]

Without regularization, MSE is lower (better fit on training data), but irrelevant coefficients are larger (less shrinkage), which could lead to overfitting on new data.

### Key Takeaways for Beginners
- Elastic Net helps when you have many features: it shrinks less important ones (L2) and can set some to zero (L1).
- Tune `alpha` and `rho` based on your data (e.g., higher `alpha` for more regularization, higher `rho` for more sparsity).
- In practice, use libraries like scikit-learn's `ElasticNet` for easier implementation, but this PyTorch version shows the mechanics.
- Always split data into train/test sets for real evaluations (omitted here for simplicity).

