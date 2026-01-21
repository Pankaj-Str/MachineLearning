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
# Example 
**realistic example** using machine learning that shows **underfitting**, **overfitting**, and the **good (balanced) fit** — using simple Python code you can imagine or run yourself.

### Example Problem: Predicting house prices from size (in sq ft)

We have some made-up data:

- Small houses (1000–1500 sq ft) ≈ ₹40–60 lakh
- Medium houses (1500–2500 sq ft) ≈ ₹60–120 lakh
- Larger houses show a bit of curving pattern (not perfectly straight)

But real data always has some noise (outliers).

### Step 1 – We try 3 different models (using polynomial regression)

We use the same data but change only the **complexity** (polynomial degree):

- Degree 1 → very simple straight line → **Underfitting**
- Degree 3 → reasonable curve → **Good fit** (sweet spot)
- Degree 15 → very wiggly crazy line → **Overfitting**

Here are the classic visuals that everyone uses to understand this:




<img width="911" height="619" alt="Screenshot 2026-01-21 at 8 07 44 AM" src="https://github.com/user-attachments/assets/6fc3a2ca-e289-4de9-bda1-c957474452e5" />
<img width="816" height="615" alt="Screenshot 2026-01-21 at 8 07 23 AM" src="https://github.com/user-attachments/assets/dfc304ae-75d9-4ba9-a7ee-69f1db9217f7" />








**Left plot (degree 1)** → Underfitting: too simple, misses the real curve pattern  
**Middle plot (degree ~3)** → Good fit: follows the true pattern nicely, not too wiggly  
**Right plot (degree 15)** → Overfitting: tries to go through every single training dot (including noise), bad on new houses

### Step 2 – How do we actually detect it in practice? → Learning Curves

We look at **training error** vs **validation error** as model complexity increases.


<img width="879" height="628" alt="Screenshot 2026-01-21 at 8 08 09 AM" src="https://github.com/user-attachments/assets/031d735a-b1bd-45d5-bb8b-37da8ce07715" />

<img width="639" height="480" alt="Screenshot 2026-01-21 at 8 08 22 AM" src="https://github.com/user-attachments/assets/b9ec30f4-9bb7-42fb-b125-3784a3369ba4" />





**Interpretation of curves:**

- **Underfitting zone** (left side): both errors high → model too simple
- **Sweet spot** (middle): both errors low + small gap → best generalization
- **Overfitting zone** (right side): training error → almost 0, but validation error shoots up → model memorized training data + noise

### Step 3 – Super simple Python example (conceptual code)

```python
# Imagine we have data
X = house_sizes          # e.g. [1000, 1200, 1500, 1800, ..., 3000]
y = house_prices         # e.g. [45, 52, 68, 95, ..., 180] lakh

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------------------------
# Model 1: Underfitting (degree 1 = straight line)
model_under = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
model_under.fit(X_train, y_train)
print("Underfit → Test score:", model_under.score(X_test, y_test))   # → low ~0.65–0.75

# -------------------------------------------------
# Model 2: Good fit (usually degree 2 or 3 works well here)
model_good = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
model_good.fit(X_train, y_train)
print("Good fit → Test score:", model_good.score(X_test, y_test))   # → high ~0.92–0.96

# -------------------------------------------------
# Model 3: Overfitting (very high degree)
model_over = make_pipeline(PolynomialFeatures(degree=15), LinearRegression())
model_over.fit(X_train, y_train)
print("Overfit → Train score:", model_over.score(X_train, y_train))  # → almost 1.0 (0.99+)
print("Overfit → Test score:", model_over.score(X_test, y_test))     # → drops badly ~0.4–0.7
```

**What you see in real run:**
- Underfit: poor on both train and test
- Good: good on both train and test
- Overfit: excellent on train, poor on test

### Quick Memory Table (very useful)

| Model          | Train Error | Test Error | Gap (Train vs Test) | What happened?                     |
|----------------|-------------|------------|----------------------|------------------------------------|
| Underfitting   | High        | High       | Small                | Too simple, didn't learn pattern   |
| Good fit       | Low         | Low        | Small                | Learned pattern, generalizes well  |
| Overfitting    | Very Low    | High       | Very Large           | Memorized data + noise, bad on new |

**Moral of the story (very simple sentence):**

"Underfitting = didn't study enough"  
"Overfitting = cheated by memorizing exact questions"  
"Good fit = actually understood the subject"





