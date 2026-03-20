# 1. What is SMOTE?

SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE is used when your dataset is **imbalanced**, meaning one class has much more data than the other.

## Why is it needed?

If a dataset looks like this:

* 980 = Not Spam
* 20 = Spam

A model may simply predict “Not Spam” all the time and still get high accuracy, but it will fail to detect spam.

## What SMOTE does:

* It creates **synthetic (artificial) data points** for the minority class
* It does not duplicate data, instead it generates new samples using nearby points

## Simple idea:

Instead of copying minority data, SMOTE creates new “similar” data points between existing ones.

---

# 2. What is Hyperparameter Tuning?

Hyperparameter tuning is the process of finding the **best settings for a machine learning model**.

Example using K-Nearest Neighbors:

* `n_neighbors` → number of neighbors
* `weights` → how neighbors influence prediction

These settings are not learned automatically; we must choose them.

## Goal:

Find the combination of hyperparameters that gives the best performance.

---

# 3. New Example: Email Spam Detection

## Problem:

Build a model to classify emails as:

* Spam (1)
* Not Spam (0)

Dataset:

* Not Spam = 950
* Spam = 50

This is clearly imbalanced.

---

# 4. Complete Python Example (Using KNN)

## Step 1: Import Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
```

---

## Step 2: Create Imbalanced Dataset

```python
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    weights=[0.95, 0.05],  # imbalanced
    random_state=42
)
```

---

## Step 3: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Step 4: Apply SMOTE

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

Now both classes are balanced.

---

## Step 5: Hyperparameter Tuning

```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

model = KNeighborsClassifier()

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring='f1'
)

grid.fit(X_train_smote, y_train_smote)
```

---

## Step 6: Best Parameters

```python
print("Best Parameters:", grid.best_params_)
```

---

## Step 7: Evaluation

```python
y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))
```

---

# 5. What is Happening in This Example?

### Before SMOTE:

* Model sees very few spam examples
* Poor performance on spam detection

### After SMOTE:

* Dataset becomes balanced
* Model learns both classes properly

### After Hyperparameter Tuning:

* Best KNN settings are selected
* Model performance improves further

---

# 6. Key Difference

| Concept               | Purpose                   |
| --------------------- | ------------------------- |
| SMOTE                 | Fix class imbalance       |
| Hyperparameter Tuning | Improve model performance |

---

# 7. Important Rules

* Always apply SMOTE **only on training data**
* Never apply it to test data
* Use evaluation metrics like **F1-score** instead of accuracy
* Combine SMOTE + tuning for best results

---

# 8. When to Use Both Together?

Use both when:

* Your dataset is imbalanced
* You want better prediction performance
* You are working on real-world problems like:

  * Fraud detection
  * Spam filtering
  * Medical diagnosis

---

