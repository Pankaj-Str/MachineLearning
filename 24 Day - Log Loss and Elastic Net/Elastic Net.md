
### What is Elastic Net in Machine Learning?

Elastic Net is a regularization technique used in linear regression models to prevent overfitting and perform feature selection. It combines the strengths of two other regularization methods:
- **Lasso (L1 regularization)**: Adds a penalty equal to the absolute value of the coefficients, which can shrink some coefficients to exactly zero, effectively selecting a subset of features.
- **Ridge (L2 regularization)**: Adds a penalty equal to the square of the coefficients, which shrinks coefficients but rarely sets them to zero, helping with multicollinearity (when features are highly correlated).

Elastic Net uses a mix of both penalties, controlled by a parameter called `rho` (or `l1_ratio`). The formula for the loss function in Elastic Net Regression is:

<img width="750" height="279" alt="Screenshot 2025-12-27 at 1 56 55 PM" src="https://github.com/user-attachments/assets/aed162b5-2250-4630-ba1a-39b7139841a8" />


This is useful when you have many features, some irrelevant or correlated, as it balances shrinkage and sparsity.

For this beginner example, we'll use synthetic data and implement Elastic Net using PyTorch (a deep learning library that can handle custom losses). We'll train a simple linear model with the Elastic Net penalty added to the loss.


---
### Lasso and Ridge Regression Using scikit-learn (sklearn)

scikit-learn (often imported as `sklearn`) is the most popular Python library for machine learning. It provides simple, ready-to-use classes for **LinearRegression**, **Ridge**, and **Lasso**, making implementation much easier than building from scratch with PyTorch.

We'll use the same simple synthetic dataset:
- 100 samples
- 5 features
- True coefficients: [1.5, -2.0, 0.0, 0.0, 3.0] (only 3 features are relevant)

### Step 1: Import Necessary Libraries

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
```

### Step 2: Generate Synthetic Data

```python
np.random.seed(42)  # For reproducibility
n_samples = 100
n_features = 5

X = np.random.randn(n_samples, n_features)
true_weights = np.array([1.5, -2.0, 0.0, 0.0, 3.0])
y = X @ true_weights + np.random.randn(n_samples) * 0.5  # Add noise
```

- `X` shape: (100, 5)
- `y` shape: (100,)

### Step 3: Plain Linear Regression (No Regularization)

```python
plain_model = LinearRegression()
plain_model.fit(X, y)

print("Plain Linear Regression:")
print("Coefficients:", plain_model.coef_)
print("Intercept:", plain_model.intercept_)
plain_mse = mean_squared_error(y, plain_model.predict(X))
print("Train MSE:", plain_mse)
print()
```

**Output:**
```
Plain Linear Regression:
Coefficients: [ 1.52710641 -1.93224448 -0.01064705  0.08037361  2.99638427]
Intercept: -0.07306236831451784
Train MSE: 0.20083099645358918
```

- Best fit on training data (lowest MSE).
- Irrelevant features have small but non-zero coefficients.

### Step 4: Ridge Regression (L2 Penalty)

```python
ridge_model = Ridge(alpha=1.0)  # alpha = regularization strength (λ)
ridge_model.fit(X, y)

print("Ridge Regression (alpha=1.0):")
print("Coefficients:", ridge_model.coef_)
print("Intercept:", ridge_model.intercept_)
ridge_mse = mean_squared_error(y, ridge_model.predict(X))
print("Train MSE:", ridge_mse)
print()
```

**Output:**
```
Ridge Regression (alpha=1.0):
Coefficients: [ 1.50700617 -1.91337163 -0.0106447   0.07746111  2.96727158]
Intercept: -0.07597844035385559
Train MSE: 0.20236112515627294
```

**Observations:**
- All coefficients are **shrunk** slightly toward zero compared to plain regression.
- Irrelevant coefficients are smaller but **not zero**.
- Slightly higher MSE due to bias introduced by regularization.

### Step 5: Lasso Regression (L1 Penalty)

```python
lasso_model = Lasso(alpha=0.1)  # Try different alpha values
lasso_model.fit(X, y)

print("Lasso Regression (alpha=0.1):")
print("Coefficients:", lasso_model.coef_)
print("Intercept:", lasso_model.intercept_)
lasso_mse = mean_squared_error(y, lasso_model.predict(X))
print("Train MSE:", lasso_mse)
```

**Output:**
```
Lasso Regression (alpha=0.1):
Coefficients: [ 1.39389613 -1.8395825  -0.          0.          2.89055553]
Intercept: -0.07826978806709867
Train MSE: 0.23895503595198925
```

**Observations:**
- Irrelevant coefficients are **exactly zero** → automatic **feature selection**!
- Relevant coefficients are shrunk more than in Ridge.
- Higher MSE than Ridge/plain, but model is simpler and more interpretable.

### Key Differences Summary

| Model              | Coefficient Shrinkage | Sets Coefficients to Zero? | Best For                          |
|-------------------|------------------------|-----------------------------|-----------------------------------|
| Plain Linear      | No                     | No                          | Small, clean datasets             |
| **Ridge**         | Yes (evenly)           | Rarely                      | Many features, multicollinearity  |
| **Lasso**         | Yes (stronger on small) | Yes                         | Feature selection, sparse models  |

### Tips for Beginners
- **Tune `alpha`**:
  - Small alpha (e.g., 0.001) → close to plain regression.
  - Large alpha (e.g., 10) → more shrinkage, possibly underfitting.
- Use **cross-validation** to find best alpha:
  ```python
  from sklearn.linear_model import RidgeCV, LassoCV
  
  ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, y)
  print("Best alpha:", ridge_cv.alpha_)
  ```
- Always split data into train/test in real projects:
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  ```

This sklearn approach is much simpler and faster than manual implementation. Copy and run the full code to experiment — try changing `alpha` to see how coefficients change!
