## What is Linear Regression in Machine Learning?

**Linear Regression** is a **Supervised Machine Learning algorithm** used to predict a **continuous numerical value**.

Simple words mein:

> **Linear Regression data ke beech relationship find karke future value predict karta hai.**

### Simple Example: House Price

Suppose we have this data:

| House Size |    Price |
| ---------: | -------: |
|  500 sq.ft | ₹25 Lakh |
|  700 sq.ft | ₹35 Lakh |
|  900 sq.ft | ₹45 Lakh |
| 1100 sq.ft | ₹55 Lakh |
| 1300 sq.ft | ₹65 Lakh |

Machine Learning model in data ko dekhega aur **Size aur Price ke beech relationship** learn karega.

Then if we give:

```text
House Size = 1000 sq.ft
```

Model might predict:

```text
Predicted Price ≈ ₹50 Lakh
```

### Visual idea


<img width="1780" height="884" alt="ChatGPT Image Sep 24, 2026, 08_03_41 AM" src="https://github.com/user-attachments/assets/16cabe18-d152-476f-accc-a0e5b6e01a85" />



Model basically data points ke through ek **best-fit straight line** find karta hai.

### Linear Regression Formula

The basic formula is:

**ŷ = b₀ + b₁x**

Where:

* **ŷ** → predicted value
* **x** → input feature
* **b₀** → intercept
* **b₁** → coefficient/slope

For example:

```text
Price = 5 + 0.05 × Area
```

If:

```text
Area = 1000
```

then:

```text
Price = 5 + 0.05 × 1000
      = 55
```

So predicted price = **55 units**.

### Real-world applications

Linear Regression can be used for:

* House price prediction
* Salary prediction
* Sales forecasting
* Revenue prediction
* Temperature prediction
* Demand forecasting

### Important point

Linear Regression is generally used when the **output is continuous**.

```text
Input                  Output
────────────────────────────────
House Size       →     Price
Experience       →     Salary
Advertising      →     Sales
Area             →     Rent
```

### One-line definition for your lecture

> **Linear Regression is a supervised machine learning algorithm that learns the relationship between input and continuous output variables and uses that relationship to make predictions.**

----

# Linear Regression code using Python and `scikit-learn`:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 8, 10])

# Create model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make prediction
prediction = model.predict([[6]])

print("Predicted value:", prediction[0])

# Model parameters
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])
```

### Simple explanation

```python
model = LinearRegression()
```

Creates the Linear Regression model.

```python
model.fit(X, y)
```

The model **learns the relationship** between `X` and `y`.

```python
model.predict([[6]])
```

Now we give a **new value `6`**, and the model predicts its corresponding `y`.

### With graph

If you want to show students how the **data points + regression line** look:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 8, 10])

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Plot data points
plt.scatter(X, y)

# Plot regression line
plt.plot(X, y_pred)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")

plt.show()
```

The **dots are actual data**, and the **straight line is the line learned by Linear Regression**.

