# SMOTE + Hyperparameter Tuning

---

# 1. Problem Statement

We will build a **Credit Card Fraud Detection Model**.

Dataset situation:

* 990 = Normal transactions
* 10 = Fraud transactions

This is an **imbalanced dataset**, so we will:

1. Use SMOTE to balance data
2. Use hyperparameter tuning to improve model performance

---

# 2. Step-by-Step Implementation

---

## Step 1: Install Required Libraries

```bash
pip install scikit-learn imbalanced-learn pandas
```

---

## Step 2: Import Libraries

```python
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
```

---

## Step 3: Create Imbalanced Dataset

```python
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    weights=[0.99, 0.01],  # Highly imbalanced
    random_state=42
)

print("Class distribution before SMOTE:")
print(pd.Series(y).value_counts())
```

---

## Step 4: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Step 5: Apply SMOTE (Important Step)

```python
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())
```

Now the dataset becomes balanced.

---

## Step 6: Define Model

We use Random Forest.

```python
model = RandomForestClassifier(random_state=42)
```

---

## Step 7: Hyperparameter Tuning

```python
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid.fit(X_train_smote, y_train_smote)
```

---

## Step 8: Best Parameters

```python
print("Best Parameters:")
print(grid.best_params_)
```

---

## Step 9: Train Final Model

```python
best_model = grid.best_estimator_
```

---

## Step 10: Make Predictions

```python
y_pred = best_model.predict(X_test)
```

---

## Step 11: Evaluate Model

```python
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

# 3. Output Understanding

### Classification Report

* Precision → How accurate predictions are
* Recall → How many fraud cases detected
* F1-score → Balance of precision and recall

### Confusion Matrix

|               | Predicted Normal | Predicted Fraud |
| ------------- | ---------------- | --------------- |
| Actual Normal | Correct          | Wrong           |
| Actual Fraud  | Missed           | Correct         |

---

# 4. Full Workflow Summary

1. Create or load dataset
2. Split into training and testing
3. Apply SMOTE on training data
4. Define model
5. Perform hyperparameter tuning
6. Select best model
7. Evaluate on test data

---

# 5. Key Points to Remember

* SMOTE is used to balance imbalanced datasets
* Hyperparameter tuning improves model performance
* Always apply SMOTE **only on training data**
* Use F1-score instead of accuracy for imbalanced problems

---

# 6. Real-World Applications

* Fraud detection
* Medical diagnosis
* Spam detection
* Loan default prediction

---

