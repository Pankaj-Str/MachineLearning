# Overfitting and Underfitting
**Overfitting** and **Underfitting** are two very common problems when we train machine learning models.

Think of it like this simple school analogy:

- **Underfitting** → Student who didn't study enough → makes many mistakes even on questions he already saw (poor on training data) and also on new questions (poor on test data).
- **Overfitting** → Student who memorized every single word in the textbook and notes → gets 100% on the practice test (excellent on training data) but almost fails in the final exam because he can't handle slightly different questions (poor on test/new data).
- **Good fit** → Student who understood the concepts → does well on practice questions and also on new exam questions.

### Step-by-step simple explanation

**Step 1: We have real-world data with some pattern + noise**  
(Example: House size vs Price – generally bigger house = higher price, but some houses are priced weirdly due to location/view/renovation)

**Step 2: We try to draw a line/curve to predict price from size**

We can choose different "complexity" levels:

- Very simple model (straight line) → Underfitting
- Just right model → Good fit
- Very complicated/wiggly model → Overfitting

Here are visual examples (polynomial curve fitting):


<img width="953" height="557" alt="Screenshot 2026-01-21 at 8 03 50 AM" src="https://github.com/user-attachments/assets/74111af7-b644-4972-b499-fb003acb8252" />


**Left image / low degree (degree 1)** → Underfitting (straight line can't capture the curve)  
**Middle image / good degree (around 3–4)** → Good fit (follows the real pattern nicely)  
**Right image / very high degree (15 or more)** → Overfitting (wiggles a lot to pass through every single training point including noise)

### Learning Curve View (very important to detect them)

We train model → look at error on training data and validation/test data as model gets more complex or trains longer.


<img width="2048" height="1536" alt="p2" src="https://github.com/user-attachments/assets/bc9bb402-28f8-4cd3-93f6-fc0b878f1c24" />


**Underfitting** (High bias):  
- Both training error and validation error → high  
- Gap between them is small

**Good fit**:  
- Training error low  
- Validation error also low  
- Small gap between them

**Overfitting** (High variance):  
- Training error → very low (almost 0)  
- Validation error → high  
- Big gap between training and validation error

### Quick Everyday Examples

| Situation                          | Underfitting example                          | Overfitting example                              | Ideal / Good fit                                 |
|------------------------------------|-----------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Exam preparation                   | Didn't study at all                           | Memorized exact questions & answers              | Understood concepts + practiced variations       |
| Learning to cook                   | Always burns food (can't follow basic recipe) | Can make only that one dish perfectly            | Can cook many dishes well, even with variations  |
| House price prediction model       | Uses only "number of rooms"                   | Uses 200 features including neighbor's car color | Uses 8–12 meaningful features                    |

### Summary Table (very easy to remember)

| Model type     | Training performance | New data performance | Bias   | Variance | Gap between train & validation error |
|----------------|----------------------|----------------------|--------|----------|--------------------------------------|
| Underfitting   | Bad                  | Bad                  | High   | Low      | Small                                |
| Good fit       | Good                 | Good                 | Low    | Low      | Small                                |
| Overfitting    | Excellent            | Bad / Poor           | Low    | High     | Very large                           |

**Goal in machine learning = find the sweet spot → Good fit (low bias + low variance)**



-----
## Overfitting and Underfitting Example Using Python (Scikit-Learn)

To understand **Overfitting and Underfitting**, we will use a **real dataset** from **Scikit-Learn** called the **Diabetes Dataset**.

Dataset: **Diabetes dataset**
Goal: Predict **disease progression** based on patient features.

We will create **three models**:

1. Underfitting model (too simple)
2. Good model (balanced)
3. Overfitting model (too complex)

---

# Step 1: Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
```

---

# Step 2: Load Real Dataset

```python
diabetes = load_diabetes()

X = diabetes.data[:, np.newaxis, 2]   # Using one feature
y = diabetes.target
```

Here:

* **X** → feature (independent variable)
* **y** → target (dependent variable)

---

# Step 3: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

This splits the dataset into:

* **Training data (80%)**
* **Testing data (20%)**

---

# Step 4: Create Models with Different Complexity

We will create polynomial models with different degrees.

* Degree **1** → Underfitting
* Degree **3** → Good fit
* Degree **15** → Overfitting

---

## Model 1: Underfitting (Simple Model)


```python
model_underfit = make_pipeline(
    PolynomialFeatures(1),
    LinearRegression()
)

model_underfit.fit(X_train, y_train)
```

This model is **too simple** and cannot capture the real pattern.

---

## Model 2: Good Fit

```python
model_good = make_pipeline(
    PolynomialFeatures(3),
    LinearRegression()
)

model_good.fit(X_train, y_train)
```

This model captures the pattern better without memorizing the data.

---

## Model 3: Overfitting (Very Complex Model)

```python
model_overfit = make_pipeline(
    PolynomialFeatures(15),
    LinearRegression()
)

model_overfit.fit(X_train, y_train)
```

This model becomes **too complex** and tries to fit every training point.

---

# Step 5: Evaluate the Models

```python
print("Underfit Train Score:", model_underfit.score(X_train, y_train))
print("Underfit Test Score:", model_underfit.score(X_test, y_test))

print("Good Model Train Score:", model_good.score(X_train, y_train))
print("Good Model Test Score:", model_good.score(X_test, y_test))

print("Overfit Train Score:", model_overfit.score(X_train, y_train))
print("Overfit Test Score:", model_overfit.score(X_test, y_test))
```

Expected behavior:

| Model        | Train Score | Test Score |
| ------------ | ----------- | ---------- |
| Underfitting | Low         | Low        |
| Good Model   | Medium      | Medium     |
| Overfitting  | Very High   | Low        |

---

# Step 6: Visualization

```python
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1,1)

plt.scatter(X, y, color="black")

plt.plot(X_plot, model_underfit.predict(X_plot), label="Underfit")
plt.plot(X_plot, model_good.predict(X_plot), label="Good Fit")
plt.plot(X_plot, model_overfit.predict(X_plot), label="Overfit")

plt.legend()
plt.show()
```

Graph explanation:

* **Straight line** → Underfitting
* **Smooth curve** → Good model
* **Very wavy curve** → Overfitting

---

# Real Meaning

### Underfitting

Model cannot learn the data pattern.

Example:
Trying to fit **complex data using a straight line**.

---

### Good Fit

Model learns the **real pattern of the data**.

Best case for machine learning.

---

### Overfitting

Model memorizes the **training data including noise**.

Works well on training data but **fails on new data**.

---

# Simple Summary

| Concept      | Meaning                      |
| ------------ | ---------------------------- |
| Underfitting | Model too simple             |
| Good Fit     | Model learns correct pattern |
| Overfitting  | Model memorizes data         |

---


