# Principal Component Analysis (PCA) Explained | Machine Learning

## Introduction

**Principal Component Analysis (PCA)** is a **dimensionality reduction** technique often used in Machine Learning. It helps simplify large datasets while retaining the most important patterns. PCA is especially useful for:

- Reducing the number of features in a dataset
- Removing multicollinearity
- Visualizing high-dimensional data
- Improving model performance by reducing noise

---

## What is PCA?

PCA transforms a high-dimensional dataset into a lower-dimensional space. It does this by creating **new features** called **principal components**, which are **linear combinations** of the original features.

### Key Concepts:

- **Principal Components**: New axes or directions that capture maximum variance.
- **Variance**: Measure of information. PCA keeps the components with the highest variance.
- **Orthogonality**: Principal components are orthogonal (uncorrelated).

---

## When to Use PCA?

- You have many features and want to simplify the dataset
- The features are correlated
- You want to visualize data in 2D or 3D
- To reduce overfitting

---

## Step-by-Step Explanation of PCA

Let’s break PCA into clear steps.

### Step 1: Standardize the Dataset

PCA is affected by scale. So, we standardize the data (mean = 0, std = 1).

### Step 2: Calculate the Covariance Matrix

This shows how features vary with respect to each other.

### Step 3: Compute Eigenvalues and Eigenvectors

Eigenvectors define the direction of the new feature space (principal components), and eigenvalues show the magnitude (importance).

### Step 4: Select Top k Components

Choose the top `k` eigenvectors corresponding to the largest `k` eigenvalues to keep the most important information.

### Step 5: Transform the Data

Project the original data onto the new feature space using the selected eigenvectors.

---

## PCA Python Implementation (Step-by-Step)

We'll use the **Iris dataset** from Scikit-learn.

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

---

### Step 2: Load and Prepare the Data

```python
iris = load_iris()
X = iris.data
y = iris.target
features = iris.feature_names

df = pd.DataFrame(X, columns=features)
df['target'] = y
df.head()
```

---

### Step 3: Standardize the Features

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### Step 4: Apply PCA

We’ll reduce to 2 components for visualization.

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

---

### Step 5: Check Explained Variance Ratio

```python
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", sum(pca.explained_variance_ratio_))
```

This tells how much information is retained in the reduced dimensions.

---

### Step 6: Visualize the Results

```python
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis')
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA on Iris Dataset")
plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names)
plt.grid(True)
plt.show()
```

---

## Interpreting the Output

- The PCA reduced the data from 4 dimensions to 2.
- The scatter plot shows how well-separated the classes are in the reduced space.
- You can keep more components (`n_components`) to retain more variance.

---

## Use Cases of PCA in Real Life

- Face recognition: Reduce image data while keeping key features
- Finance: Simplify large stock datasets
- Genomics: Reduce gene expression data
- Noise reduction: Keep the signal, remove noise

---

## Final Thoughts

- PCA is unsupervised and does not use target labels
- It’s a linear method—won’t capture nonlinear relationships
- Always standardize data before applying PCA
- Use explained variance to decide how many components to keep

---

## Bonus: PCA in Scikit-Learn (Custom n_components)

```python
# Keep enough components to retain 95% variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"Number of components to retain 95% variance: {pca_95.n_components_}")
```

---

