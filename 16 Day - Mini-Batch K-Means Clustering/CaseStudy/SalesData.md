### Mini-Batch K-Means Clustering - Case Study

**Objective:** To demonstrate the process and application of Mini-Batch K-Means clustering, a variation of the K-Means algorithm designed for large datasets, using a sample dataset.

**Dataset:** The hypothetical "sales_data" dataset containing the following features:
- Transaction ID
- Customer Age
- Annual Income
- Transaction Amount

### Steps:

1. **Data Preparation:**
   - Create the dataset.
   - Explore the dataset to understand its structure and distribution.

2. **Data Preprocessing:**
   - Handle any missing values (if any).
   - Standardize the features.

3. **Mini-Batch K-Means Clustering:**
   - Determine the optimal number of clusters using the Elbow method.
   - Perform Mini-Batch K-Means clustering with the optimal number of clusters.
   
4. **Cluster Analysis:**
   - Assign data points to clusters.
   - Analyze the clusters to understand their characteristics.

5. **Interpretation and Conclusion:**
   - Interpret the results and provide insights.
   - Discuss the advantages and limitations of Mini-Batch K-Means clustering.

### Step-by-Step Implementation

#### Step 1: Data Preparation

Let's first create the hypothetical dataset.

```python
import pandas as pd
import numpy as np

# Create a hypothetical "sales_data" dataset
np.random.seed(0)
num_transactions = 1000
data = {
    'Transaction ID': [f'T{i+1:04d}' for i in range(num_transactions)],
    'Customer Age': np.random.randint(18, 70, num_transactions),
    'Annual Income': np.random.randint(20000, 150000, num_transactions),
    'Transaction Amount': np.random.randint(10, 1000, num_transactions),
}
sales_data = pd.DataFrame(data)

# Save the dataset to a CSV file
file_path = '/mnt/data/sales_data.csv'
sales_data.to_csv(file_path, index=False)



### Step-by-Step Implementation

#### Step 1: Data Preparation

```python
import pandas as pd

# Load the dataset
file_path = 'path_to_your_dataset/sales_data.csv'
sales_data = pd.read_csv(file_path)

# Display the first few rows of the dataset
sales_data.head()
```

#### Step 2: Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# Check for missing values
sales_data.isnull().sum()

# Standardize the features (excluding the 'Transaction ID' column)
features = sales_data.columns[1:]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sales_data[features])

# Convert the scaled data back to a DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=features)
```

#### Step 3: Mini-Batch K-Means Clustering

```python
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    mbkmeans = MiniBatchKMeans(n_clusters=i, init='k-means++', max_iter=300, batch_size=100, random_state=0)
    mbkmeans.fit(scaled_data)
    wcss.append(mbkmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 7))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the elbow plot, assume the optimal number of clusters is 3

# Perform Mini-Batch K-Means clustering with the optimal number of clusters
mbkmeans = MiniBatchKMeans(n_clusters=3, init='k-means++', max_iter=300, batch_size=100, random_state=0)
sales_data['Cluster'] = mbkmeans.fit_predict(scaled_data)
```

#### Step 4: Cluster Analysis

```python
# Analyze the clusters
cluster_summary = sales_data.groupby('Cluster').mean()

import ace_tools as tools; tools.display_dataframe_to_user(name="Cluster Summary", dataframe=cluster_summary)
```

#### Step 5: Interpretation and Conclusion

```python
# Interpretation and conclusion
# Example interpretation:
# Cluster 0: Young customers with moderate income and high transaction amounts
# Cluster 1: Older customers with high income and low transaction amounts
# Cluster 2: Middle-aged customers with low income and moderate transaction amounts

# Advantages and limitations
# Advantages: Efficient, works well with large datasets, reduces computation time
# Limitations: Sensitive to initial cluster centers, may not work well with clusters of different shapes and sizes
```

### Conclusion

1. **Cluster Characteristics:**
   - Analyze the characteristics of each cluster based on the mean values of the features.

2. **Interpretation:**
   - Interpret the clusters to understand the behavior and demographics of the customers in each cluster.

3. **Advantages and Limitations:**
   - Mini-Batch K-Means clustering is efficient and works well with large datasets.
   - However, it is sensitive to the initial cluster centers and may not work well with clusters of different shapes and sizes.
