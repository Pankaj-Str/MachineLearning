### 1. Principal Component Analysis (PCA) Step by Step

#### Step-by-Step Explanation

1. **Standardize the Data**
    - Ensure each feature has a mean of zero and a standard deviation of one. This step is crucial because PCA is affected by the scale of the data.
  
2. **Compute the Covariance Matrix**
    - The covariance matrix helps to understand how the features of the data relate to each other.

3. **Compute the Eigenvalues and Eigenvectors**
    - Eigenvalues and eigenvectors of the covariance matrix are computed. The eigenvectors represent the directions of the new feature space, and the eigenvalues represent the magnitude of variance in these directions.

4. **Sort Eigenvalues and Eigenvectors**
    - Sort the eigenvalues in descending order and choose the top k eigenvalues. The corresponding eigenvectors will form the principal components.

5. **Transform the Data**
    - Project the data onto the selected principal components to obtain the reduced-dimensional representation.

#### Practical Example in Python

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Sample data
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# Step 1: Standardize the Data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Step 2: Compute the Covariance Matrix
cov_matrix = np.cov(X_standardized.T)

# Step 3: Compute the Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 4: Sort Eigenvalues and Eigenvectors
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Step 5: Transform the Data
k = 1  # Number of principal components
eigenvector_subset = sorted_eigenvectors[:, 0:k]
X_reduced = np.dot(X_standardized, eigenvector_subset)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_standardized[:, 0], X_standardized[:, 1], label='Original Data')
plt.scatter(X_reduced, np.zeros_like(X_reduced), label='PCA Transformed Data')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA Example')
plt.show()
```

### 2. Kernel Principal Component Analysis (Kernel PCA) Step by Step

#### Step-by-Step Explanation

1. **Standardize the Data**
    - Just like in standard PCA, standardize the data to have a mean of zero and a standard deviation of one.

2. **Compute the Kernel Matrix**
    - Instead of the covariance matrix, compute the kernel matrix using a kernel function (e.g., Gaussian kernel, polynomial kernel).

3. **Center the Kernel Matrix**
    - Center the kernel matrix by subtracting the mean of each column and each row.

4. **Compute the Eigenvalues and Eigenvectors**
    - Compute the eigenvalues and eigenvectors of the centered kernel matrix.

5. **Sort Eigenvalues and Eigenvectors**
    - Sort the eigenvalues in descending order and choose the top k eigenvalues and their corresponding eigenvectors.

6. **Transform the Data**
    - Project the data onto the selected eigenvectors to obtain the reduced-dimensional representation.

#### Practical Example in Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# Sample data
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# Step 1: Standardize the Data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Step 2: Compute the Kernel Matrix using RBF kernel
gamma = 15  # Hyperparameter for the RBF kernel
kernel_matrix = np.exp(-gamma * np.linalg.norm(X_standardized[:, np.newaxis] - X_standardized[np.newaxis, :], axis=2)**2)

# Step 3: Center the Kernel Matrix
N = kernel_matrix.shape[0]
one_n = np.ones((N, N)) / N
K_centered = kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) + one_n.dot(kernel_matrix).dot(one_n)

# Step 4: Compute the Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

# Step 5: Sort Eigenvalues and Eigenvectors
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Step 6: Transform the Data
k = 1  # Number of principal components
eigenvector_subset = sorted_eigenvectors[:, 0:k]
X_kpca = K_centered.dot(eigenvector_subset)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_standardized[:, 0], X_standardized[:, 1], label='Original Data')
plt.scatter(X_kpca, np.zeros_like(X_kpca), label='Kernel PCA Transformed Data')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Kernel PCA Example')
plt.show()
```

### Summary
- **PCA**: Useful for linear dimensionality reduction by maximizing variance.
- **Kernel PCA**: Extends PCA to handle non-linear relationships by using kernel functions.
