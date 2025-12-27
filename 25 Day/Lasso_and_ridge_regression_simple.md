# Lasso and Ridge Regression

## Lasso and Ridge Regression – **Very Simple (Beginner Friendly)**

![Image](https://dataaspirant.com/wp-content/uploads/2020/11/5-Statistics-of-lasso-regression-1024x643.png)

![Image](https://towardsdatascience.com/wp-content/uploads/2019/01/1KeutQ9gUGBhoKbVC3iM9lQ.png)

![Image](https://miro.medium.com/1%2AnrWncnoJ4V_BkzEf1pd4MA.png)

![Image](https://i.imgur.com/j6P8EPq.png)

We will explain **everything using ONE simple example**, no math fear 🙂

---

## Step 1: Simple Problem (Single Feature)

Imagine this dataset:

| Study Hours (X) | Exam Marks (Y) |
| --------------- | -------------- |
| 1               | 35             |
| 2               | 40             |
| 3               | 50             |
| 4               | 55             |
| 5               | 95             |

👉 Notice:
Last value (**5 → 95**) is very high → **outlier**
This can cause **overfitting** in normal Linear Regression.

---

## Step 2: Normal Linear Regression (Problem)

Linear Regression tries to fit a line:

[
y = wx + b
]

Because of the outlier:

* Model bends too much
* Weight **w becomes very large**
* Poor generalization

📌 **Problem**: Overfitting

This is where **regularization** comes in.

---

## Step 3: Ridge Regression (L2 Regularization)

### What Ridge does

> Ridge **reduces the size of coefficients**, but **never makes them zero**

### Ridge formula idea

[
\text{Loss} = \text{MSE} + \alpha \sum w^2
]

### Effect

* Penalizes **large weights**
* Smooths the line
* Handles **outliers better**

📌 **Result**

* Weight becomes smaller
* All features remain in the model

---

### Simple Python Example (Ridge)

```python
from sklearn.linear_model import Ridge

X = [[1], [2], [3], [4], [5]]
y = [35, 40, 50, 55, 95]

model = Ridge(alpha=1.0)
model.fit(X, y)

print("Ridge weight:", model.coef_)
```

✔ Weight is **reduced**, not removed

---

## Step 4: Lasso Regression (L1 Regularization)

### What Lasso does

> Lasso can **reduce coefficients to exactly ZERO**

### Lasso formula idea

[
\text{Loss} = \text{MSE} + \alpha \sum |w|
]

### Effect

* Shrinks weights
* Can **remove features**
* Performs **feature selection**

📌 Even in simple cases, Lasso tries to keep model **simple**

---

### Simple Python Example (Lasso)

```python
from sklearn.linear_model import Lasso

X = [[1], [2], [3], [4], [5]]
y = [35, 40, 50, 55, 95]

model = Lasso(alpha=1.0)
model.fit(X, y)

print("Lasso weight:", model.coef_)
```

✔ If a feature is useless → weight becomes **0**

---

## Step 5: Key Difference (One Line)

| Regression | What it does                 |
| ---------- | ---------------------------- |
| Linear     | Fits best line (can overfit) |
| Ridge      | Shrinks weights (no zero)    |
| Lasso      | Shrinks + removes features   |

---

## Step 6: When to Use What?

### Use **Ridge**

* When all features are important
* When data has **outliers**
* When features are correlated

### Use **Lasso**

* When you want **feature selection**
* When dataset has many unnecessary features

---

## Final Beginner Summary (Exam Ready)

> **Ridge Regression** reduces the size of coefficients to prevent overfitting but keeps all features.
> **Lasso Regression** reduces coefficients and can set some of them to zero, performing feature selection.


