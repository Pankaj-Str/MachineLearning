## 📘 LDA (Linear Discriminant Analysis) in Machine Learning



![Image](https://images.prismic.io/thedecisionlab/Z9RfGDiBA97GigXC_LDAgraph2.png?auto=format%2Ccompress)



![Image](https://sebastianraschka.com/images/blog/2014/linear-discriminant-analysis/lda_1.png)

---

## 1. What is LDA?

**Linear Discriminant Analysis (LDA)** is a **supervised machine learning algorithm** used for:

* **Classification**
* **Dimensionality Reduction**

Unlike unsupervised methods, LDA **uses class labels** to learn directions that best separate different classes.

---

## 2. Core Intuition (Simple Explanation)

Assume you have data points from different classes plotted in space.
LDA tries to find a **line or plane** such that:

* Data points from **different classes are far apart**
* Data points within the **same class are close together**

This direction maximizes class separability.

---

## 3. Why Do We Use LDA?

| Problem                    | How LDA Helps       |
| -------------------------- | ------------------- |
| High-dimensional features  | Reduces dimensions  |
| Overlapping classes        | Improves separation |
| Classification performance | Increases accuracy  |
| Model efficiency           | Faster training     |

---

## 4. LDA vs PCA (Very Important for Beginners)

| Feature           | PCA                 | LDA                       |
| ----------------- | ------------------- | ------------------------- |
| Learning type     | Unsupervised        | Supervised                |
| Uses class labels | No                  | Yes                       |
| Objective         | Maximize variance   | Maximize class separation |
| Best use          | Feature compression | Classification            |

---

## 5. Mathematical Idea (Beginner-Friendly)

LDA is based on **variance analysis** using two scatter matrices:

### 1. Within-Class Scatter Matrix (Sw)

Measures how spread out the samples are **inside each class**
Goal: **Minimize this**

### 2. Between-Class Scatter Matrix (Sb)

Measures how far apart the class means are
Goal: **Maximize this**

### Objective Function

[
\text{Maximize } \frac{\text{Between-Class Variance}}{\text{Within-Class Variance}}
]

---

## 6. Step-by-Step Working of LDA

### Step 1: Compute Class Means

Calculate the mean vector for each class.

### Step 2: Compute Overall Mean

Find the mean of the entire dataset.

### Step 3: Compute Scatter Matrices

* Within-class scatter matrix (Sw)
* Between-class scatter matrix (Sb)

### Step 4: Solve Eigenvalue Problem

[
S_w^{-1} S_b
]

### Step 5: Select Top Eigenvectors

Eigenvectors with highest eigenvalues form new features.

### Step 6: Project Data

Original data is transformed into a lower-dimensional space.

---

## 7. Important Rule (Frequently Asked)

**Maximum number of LDA components = Number of classes − 1**

| Number of Classes | Max LDA Dimensions |
| ----------------- | ------------------ |
| 2                 | 1                  |
| 3                 | 2                  |
| 4                 | 3                  |

---

## 8. Simple Conceptual Example

### Dataset (Two Features, Two Classes)

| Height | Weight | Class |
| ------ | ------ | ----- |
| 160    | 55     | A     |
| 165    | 58     | A     |
| 180    | 75     | B     |
| 185    | 80     | B     |

LDA finds a **linear combination**:
[
Z = w_1 \times Height + w_2 \times Weight
]

This new feature maximizes separation between Class A and Class B.

---

## 9. Python Example Using scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Visualization
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDA Projection")
plt.show()
```

---

## 10. Real-World Applications of LDA

* Face recognition systems
* Medical diagnosis
* Spam detection
* Customer segmentation
* Text classification

---

## 11. Advantages of LDA

* Improves class separability
* Reduces overfitting
* Works well with small datasets
* Computationally efficient

---

## 12. Limitations of LDA

* Assumes normal data distribution
* Assumes equal covariance among classes
* Not suitable for non-linear data

---

## 13. When Should You Use LDA?

**Use LDA when:**

* You have labeled data
* Classes are roughly linearly separable
* You want dimensionality reduction for classification

**Avoid LDA when:**

* Data has complex non-linear patterns
* Class distributions are highly skewed

---

## 14. One-Line Summary

**LDA finds the best directions that separate different classes while keeping same-class data compact.**

