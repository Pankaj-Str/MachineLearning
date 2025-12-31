
# **Hierarchical Clustering**

*Agglomerative Clustering + Dendrogram using Mall Customers Dataset*

---

# **What is Hierarchical Clustering?**

Hierarchical clustering is an **unsupervised machine learning algorithm** that builds clusters in a tree-like structure called a **dendrogram**.

It has two types:

### 1️⃣ **Agglomerative Clustering (Bottom-Up)**

Start with each point as a cluster → merge the closest clusters.

### 2️⃣ **Divisive Clustering (Top-Down)**

Start with one big cluster → split recursively.
*(Rarely used in practice.)*

**Agglomerative Clustering** is the most widely used and is supported in scikit-learn.

---

# **Why Use Hierarchical Clustering?**

1. Does not require specifying number of clusters (dendrogram suggests it)
2. Works well for small-medium datasets
3. Visual clustering structure via dendrogram

---

# **Full Python Example — Hierarchical Clustering**

We will use the **Mall Customers Dataset**, selecting:

* Annual Income
* Spending Score

Perfect for 2D visualization.

---

# **Complete Code Example**

```python
# -----------------------------------
# 1) Import Libraries
# -----------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

sns.set()

# -----------------------------------
# 2) Load Dataset
# -----------------------------------
url = "https://raw.githubusercontent.com/Pankaj-Str/Complete-Python-Mastery/refs/heads/main/53%20DataSet/Mall_Customers.csv"
df = pd.read_csv(url)

print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------------
# 3) Select Features for Clustering
# -----------------------------------
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# -----------------------------------
# 4) Plot Dendrogram
# -----------------------------------
plt.figure(figsize=(10, 6))
linkage_matrix = linkage(X, method="ward")
dendrogram(linkage_matrix)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# -----------------------------------
# 5) Apply Agglomerative Clustering
# -----------------------------------
# NOTE: 'affinity' is removed in latest sklearn, metric is automatic for 'ward'.
hier_cluster = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

labels = hier_cluster.fit_predict(X)
df["Cluster"] = labels

# -----------------------------------
# 6) Visualize Final Clusters
# -----------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"],
    cmap="viridis",
    s=70,
    edgecolor="black"
)

plt.title("Hierarchical Clustering (Agglomerative)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# -----------------------------------
# 7) Display Cluster Counts
# -----------------------------------
print("\nCluster Counts:")
print(df["Cluster"].value_counts())

```

---

# **Explanation of Outputs**

## **1️⃣ Dendrogram**

You will see a branching tree diagram.

Cut the dendrogram at a distance where it forms **clear non-intersecting clusters**.
For the Mall dataset, around **4–5 clusters** appears visually optimal.

---

## **2️⃣ Cluster Visualization**

Scatter plot shows:

* High-income & high-spending customers
* Low-income & high-spending customers
* Moderate spenders
* Low-spending groups
* Outliers, if present

---

## **3️⃣ Cluster Count Example**

```
0    45 customers  
1    55 customers  
2    40 customers  
3    35 customers  
4    25 customers  
```

Each cluster represents a **customer segment**.

---

#  **Understanding Hyperparameters**

| Parameter              | Meaning                                     |
| ---------------------- | ------------------------------------------- |
| `n_clusters`           | Number of clusters (chosen from dendrogram) |
| `affinity='euclidean'` | Distance metric                             |
| `linkage='ward'`       | Minimizes variance inside clusters          |

### Most common linkage methods:

* **Ward (recommended)**
* Complete
* Average
* Single

---

# **When to Use Hierarchical Clustering?**

1. Small datasets (≤ 5000 rows)
2. When you want visual understanding
3. Customer segmentation
4. Gene expression datasets
5. Outlier analysis

---

# **Why Hierarchical Clustering is Useful**

* No need to specify number of clusters initially
* Dendrogram gives deep insight
* Works well for non-linear structures
* Very interpretable

---
