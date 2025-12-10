# ------------------------------
# 1) Import Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
sns.set()

# ------------------------------
# 2) Load Dataset
# ------------------------------
url = "https://raw.githubusercontent.com/Pankaj-Str/Complete-Python-Mastery/refs/heads/main/53%20DataSet/Loan_default.csv"
df = pd.read_csv(url)

print("Dataset Loaded Successfully!")
print(df.head())
print("\nColumns:", df.columns.tolist())

# ------------------------------
# 3) Drop columns that cannot be used
# ------------------------------
if "LoanID" in df.columns:
    df.drop("LoanID", axis=1, inplace=True)   # Remove text ID column

# ------------------------------
# 4) Data Cleaning (Drop missing values)
# ------------------------------
df = df.dropna()

# ------------------------------
# 5) Convert Yes/No columns → 1/0
# ------------------------------
binary_cols = ["HasMortgage", "HasDependents", "HasCoSigner"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# ------------------------------
# 6) Encode categorical text columns
# ------------------------------
categorical_cols = ["Education", "EmploymentType", "MaritalStatus", "LoanPurpose"]

for col in categorical_cols:
    df[col] = df[col].astype("category").cat.codes

print("\nCleaned Data:")
print(df.head())

# ------------------------------
# 7) Features and Target
# ------------------------------
target_col = "Default"

X = df.drop(target_col, axis=1)
y = df[target_col]

# ------------------------------
# 8) Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\nTrain/Test Split Done!")

# ------------------------------
# 9) Random Forest Model
# ------------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n RANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# ------------------------------
# 10) XGBoost Model
# ------------------------------
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("\n XGBOOST RESULTS")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))

# ------------------------------
# 11) Confusion Matrices
# ------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12,5))

sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt="d", ax=ax[0])
ax[0].set_title("Random Forest Confusion Matrix")

sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt="d", ax=ax[1])
ax[1].set_title("XGBoost Confusion Matrix")

plt.show()

# ------------------------------
# 12) Feature Importance (XGBoost)
# ------------------------------
plt.figure(figsize=(8,5))
importance = xgb.feature_importances_
plt.barh(X.columns, importance)
plt.title("XGBoost Feature Importance")
plt.show()
