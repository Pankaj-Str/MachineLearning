# Model Optimization

---

# Hyperparameter Tuning in Machine Learning

### (GridSearchCV and RandomizedSearchCV Explained with Python Examples)

---

## What Is Hyperparameter Tuning in Machine Learning?

In machine learning, **hyperparameters** are the settings that are defined **before training the model**.

Examples:

* Decision Tree → `max_depth`
* Random Forest → `n_estimators`
* Support Vector Machine (SVM) → `C`, `gamma`

👉 **Hyperparameter Tuning** is the process of finding the **best combination of these settings** so that the model gives the **highest possible performance**.

---

## Parameters vs Hyperparameters

| Type            | Description                                                     |
| --------------- | --------------------------------------------------------------- |
| Parameters      | Learned automatically during training (weights, coefficients)   |
| Hyperparameters | Set manually before training (depth, learning rate, estimators) |

---

## Why Hyperparameter Tuning Is Important

Hyperparameter tuning helps to:

* Improve model accuracy
* Reduce overfitting and underfitting
* Build a well-generalized model
* Select the best model configuration

---

## Popular Hyperparameter Tuning Techniques

---

## 1. GridSearchCV

**GridSearchCV** tries **all possible combinations** of the given hyperparameters.

### How GridSearchCV Works

* You provide a list of parameter values
* The model is trained using **every combination**
* Cross-validation is applied for each combination
* The combination with the **best score** is selected

### Pros and Cons

* ✅ Very accurate
* ❌ Slow when the number of parameters is large

---

### Python Example: GridSearchCV (Random Forest + Iris Dataset)

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load dataset
X, y = load_iris(return_X_y=True)

# Model
model = RandomForestClassifier(random_state=42)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}

# GridSearchCV
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best Accuracy:", grid.best_score_)
```

---

## 2. RandomizedSearchCV

**RandomizedSearchCV** does not try all combinations.
Instead, it selects **random combinations** from the given parameter ranges.

### How RandomizedSearchCV Works

* You provide ranges of hyperparameters
* A fixed number of random combinations are tested
* Faster than GridSearchCV
* Ideal for large datasets and large search spaces

### Pros and Cons

* ✅ Faster execution
* ✅ Works well with large datasets
* ❌ Does not test every possible combination

---

### Python Example: RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=5,
    cv=5,
    scoring='accuracy',
    random_state=42
)

random_search.fit(X, y)

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)
```

---

## GridSearchCV vs RandomizedSearchCV

| Feature                | GridSearchCV   | RandomizedSearchCV |
| ---------------------- | -------------- | ------------------ |
| Speed                  | Slow           | Fast               |
| Accuracy               | Very High      | High               |
| Parameter Combinations | All            | Random             |
| Best Use Case          | Small datasets | Large datasets     |

---

## Real-Life Analogy

Think of buying a smartphone:

* **GridSearchCV** → Checking every model, every price, every feature
* **RandomizedSearchCV** → Checking a few selected best options

---

## One-Line Definition (SEO-Friendly)

> **Hyperparameter Tuning** is the process of optimizing model settings using techniques like GridSearchCV and RandomizedSearchCV to achieve the best performance in machine learning.

---

## Conclusion

Hyperparameter tuning is a **critical step** in building high-performance machine learning models.
While **GridSearchCV** provides the most accurate results, **RandomizedSearchCV** is faster and more practical for large datasets.

Using the right tuning technique ensures that your model performs well not only on training data but also on unseen real-world data.

---

