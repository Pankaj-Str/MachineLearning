## What is Multiple Linear Regression?

**Multiple Linear Regression** is an extension of **Linear Regression** where we use **two or more input features** to predict **one continuous output**.

### Simple Linear Regression

Only **one input** is used:

```text
House Size  ──────► House Price
```

Example:

```text
Price = b₀ + b₁ × Size
```

---

### Multiple Linear Regression

Here we use **multiple inputs**:

```text
House Size ───────┐
Bedrooms ─────────┤
Location ─────────┤──► House Price
Age ──────────────┘
```

For example, to predict a house price, we might use:

* `Size`
* `Bedrooms`
* `Age of House`
* `Distance from City`

The formula becomes:

**ŷ = b₀ + b₁x₁ + b₂x₂ + b₃x₃ + ... + bₙxₙ**

Where:

* **ŷ** = predicted output
* **b₀** = intercept
* **x₁, x₂, x₃...** = input features
* **b₁, b₂, b₃...** = coefficients

### Simple Example

Suppose our model learns:

```text
Price = 10 + 0.05 × Size + 5 × Bedrooms
```

For a house:

```text
Size = 1000 sq.ft
Bedrooms = 3
```

Then:

```text
Price = 10 + (0.05 × 1000) + (5 × 3)

      = 10 + 50 + 15

      = 75
```

So the predicted price is **75 units**.

### Python Example

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# Data
data = {
    'Size': [500, 700, 900, 1100, 1300],
    'Bedrooms': [1, 2, 2, 3, 3],
    'Price': [25, 35, 45, 55, 65]
}

df = pd.DataFrame(data)

# Input features
X = df[['Size', 'Bedrooms']]

# Target
y = df['Price']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price
prediction = model.predict([[1000, 3]])

print("Predicted Price:", prediction[0])
```

### Easy way to remember

> **Simple Linear Regression:** 1 input → 1 output
> **Multiple Linear Regression:** Multiple inputs → 1 output

**Example:**

```text
              ┌── Size
              ├── Bedrooms
              ├── Age
              └── Location
                     ↓
              Multiple Linear
                 Regression
                     ↓
               House Price
```
