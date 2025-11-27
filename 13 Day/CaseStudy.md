# Hierarchical Clustering Example with Credit Card Dataset (Bank Churners)

This example demonstrates hierarchical clustering using the **Credit Card Customers (Bank Churners)** dataset from Kaggle, provided in CSV format (`BankChurners.csv`). The code is designed for beginners, performing agglomerative hierarchical clustering, visualizing a dendrogram, and plotting the resulting clusters. It addresses the previous `KeyError` issue by correctly handling the dataset’s structure, specifically dropping the `CLIENTNUM` column and selecting appropriate numerical features for clustering.

## Dataset Description
- **Source**: Credit Card Customers (Bank Churners) dataset from Kaggle.
- **Download Link**: [Credit Card Customers on Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
- **File Name**: `BankChurners.csv`
- **Features**: 23 attributes, including:
  - `CLIENTNUM`: Customer ID (non-numerical, to be dropped).
  - `Customer_Age`: Age of the customer.
  - `Credit_Limit`: Credit limit of the card.
  - `Total_Revolving_Bal`: Total revolving balance on the card.
  - `Total_Trans_Amt`: Total transaction amount (proxy for purchases).
  - `Total_Trans_Ct`: Total transaction count.
  - Other numerical features (e.g., `Months_on_book`, `Dependent_count`).
  - Categorical features (e.g., `Gender`, `Education_Level`, to be excluded).
- **Instances**: 10,127 customers.
- **Purpose**: Cluster customers based on credit card usage patterns (e.g., high spenders, low balance users).
- **Note**: You may need to sign in to Kaggle to download the dataset. Save `BankChurners.csv` in the same directory as your script or update the file path.

## Instructions
1. Download `BankChurners.csv` from the Kaggle link above.
2. Place the file in the same directory as your Python script or provide the correct file path.
3. Install required libraries (see Prerequisites below).

## Prerequisites
Install required libraries:
```bash
pip install scikit-learn pandas matplotlib scipy numpy
```

## Complete Python Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Load the Credit Card dataset.py dataset from CSV
# Replace 'BankChurners.csv' with the path to your downloaded file
data = pd.read_csv('BankChurners.csv')

# Step 2: Inspect column names to verify dataset structure
print("Column names in the dataset:", data.columns.tolist())

# Step 3: Preprocess the data
# Drop non-numerical column (CLIENTNUM) and exclude categorical columns
data = data.drop('CLIENTNUM', axis=1)

# Select numerical columns for clustering
numerical_columns = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Total_Trans_Amt', 'Total_Trans_Ct'
]
data = data[numerical_columns]

# Handle missing values by filling with column means
data = data.fillna(data.mean())

# Select features for clustering (Total_Revolving_Bal, Total_Trans_Amt, Credit_Limit)
X = data[['Total_Revolving_Bal', 'Total_Trans_Amt', 'Credit_Limit']].values

# Standardize the features to ensure equal weighting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Compute the linkage matrix for hierarchical clustering
# Using Ward's method for linkage
linkage_matrix = linkage(X_scaled, method='ward')

# Step 5: Plot the dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=3)  # Show top 3 levels for clarity
plt.title('Dendrogram for Hierarchical Clustering (Bank Churners Dataset)')
plt.xlabel('Data Points (or Clusters)')
plt.ylabel('Distance')
plt.show()

# Step 6: Perform agglomerative clustering
# Choose 3 clusters based on the dendrogram
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(X_scaled)

# Step 7: Visualize the clusters (using Total_Revolving_Bal and Total_Trans_Amt)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title('Hierarchical Clustering Results (Bank Churners Dataset)')
plt.xlabel('Total Revolving Balance ($)')
plt.ylabel('Total Transaction Amount ($)')
plt.show()

# Step 8: Print number of points in each cluster
print("Number of points in each cluster:", np.bincount(labels))
```

## Explanation of the Code
1. **Dataset Loading**: Loads `BankChurners.csv` using `pandas.read_csv`. Ensure the file is in the correct directory or update the path.
2. **Inspect Columns**: Prints column names to confirm the dataset’s structure (e.g., `CLIENTNUM`, `Customer_Age`, `Credit_Limit`).
3. **Preprocessing**:
   - Drops the `CLIENTNUM` column (customer ID, non-numerical).
   - Selects 10 numerical columns to avoid errors from categorical features (e.g., `Gender`, `Education_Level`).
   - Fills missing values with column means.
   - Uses `Total_Revolving_Bal` (balance), `Total_Trans_Amt` (purchases proxy), and `Credit_Limit` for clustering.
   - Standardizes features using `StandardScaler` to ensure equal contribution to clustering.
4. **Linkage Matrix**: Computes the hierarchy using Ward’s method, which minimizes variance during merges.
5. **Dendrogram**: Visualizes the clustering process, truncated to the top 3 levels for clarity due to the large dataset (10,127 customers).
6. **Clustering**: Assigns customers to 3 clusters, chosen based on the dendrogram and typical clustering goals.
7. **Visualization**: Plots a scatter plot of customers, colored by cluster, using `Total_Revolving_Bal` and `Total_Trans_Amt`.
8. **Output**: Prints the number of customers in each cluster.

## Expected Output
- **Console Output (Column Names)**: A list of columns, e.g., `['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', ..., 'Credit_Limit', ...]`.
- **Dendrogram**: A tree-like plot showing how customers are merged, with significant distance jumps indicating cluster boundaries.
- **Scatter Plot**: A 2D plot of customers, colored by cluster, showing groups based on revolving balance and transaction amount (e.g., high spenders, low balance users).
- **Console Output (Clusters)**: Cluster sizes, e.g., `[4000, 3500, 2627]` (exact numbers depend on the clustering).

## Troubleshooting
- **File Not Found**: Ensure `BankChurners.csv` is in the same directory as the script or provide the full path (e.g., `/path/to/BankChurners.csv`).
- **KeyError**: If a `KeyError` occurs for any column, recheck `data.columns` to verify column names. Update `numerical_columns` to match available numerical features.
- **Missing Values**: The code fills missing values with means. To drop rows with missing values instead, replace `data.fillna(data.mean())` with `data.dropna()`.
- **Categorical Columns**: If clustering fails due to non-numerical data, ensure only numerical columns are included in `numerical_columns`.

## Notes for Beginners
- **Dataset Exploration**: Use `print(data.head())` or `data.describe()` to explore the dataset’s structure and values.
- **Feature Selection**: The code uses three features (`Total_Revolving_Bal`, `Total_Trans_Amt`, `Credit_Limit`) for simplicity. To include more features, update `X = data[['Total_Revolving_Bal', 'Total_Trans_Amt', 'Credit_Limit']].values` with additional columns (e.g., `Customer_Age`, `Total_Trans_Ct`).
- **Dendrogram Interpretation**: Look for large vertical gaps in the dendrogram to choose the number of clusters (3 is used here, but you can try 2, 4, etc.).
- **Visualization Limitation**: The scatter plot uses two features for 2D visualization. For high-dimensional data, consider PCA for better visualization.
- **Experimentation**: Try other linkage methods (`single`, `complete`, `average`) or adjust the number of clusters to explore different results.



