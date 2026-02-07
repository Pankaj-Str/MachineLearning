### Real-Life Example: Classifying Emails as Spam or Not Spam Using SVM

Let's take a simple real-life example of using SVM to classify emails as **spam** or **not spam**. This is a classic binary classification problem (two classes: spam vs. ham/non-spam).

To keep it beginner-friendly, we'll simplify things:
- We'll use just **2 features** (instead of hundreds in real email classifiers):
  - Feature 1: Number of "spammy" words (e.g., count of words like "free", "win", "urgent"). Let's say normalized to a scale of 0-5.
  - Feature 2: Number of links in the email (0-5).
- Spam emails tend to have higher values in both features.
- We'll create a small **synthetic dataset** (fake but realistic data) with 6 emails: 3 spam and 3 non-spam.
- The data is **linearly separable** (can be split by a straight line), so we don't need kernels yet.
- We'll implement a basic linear SVM from scratch using NumPy (no fancy libraries like scikit-learn needed). It uses gradient descent to minimize the hinge loss (a way to find the best separating line while maximizing the margin).
- In real life, you'd use libraries like scikit-learn for efficiency, but this shows the core idea.

I'll explain **step by step**, including the code. You can copy-paste the code into a Python environment (with NumPy and Matplotlib) to run it yourself. The code will train the SVM and plot the results for visualization.

#### Step 1: Understand the Problem and Data
- **Goal**: Draw a line (decision boundary) that separates spam points from non-spam points with the widest margin.
- **Data Points**:
  - Non-spam (label -1): Low spammy words and links, e.g., normal emails.
    - Email 1: 0 spammy words, 0 links → Not spam
    - Email 2: 1 spammy word, 0 links → Not spam
    - Email 3: 0 spammy words, 1 link → Not spam
  - Spam (label +1): High spammy words and links.
    - Email 4: 3 spammy words, 2 links → Spam
    - Email 5: 4 spammy words, 1 link → Spam
    - Email 6: 5 spammy words, 3 links → Spam
- In 2D plot: Non-spam points are near (0,0), spam near (3-5,1-3). A line like "spammy words > 2" could separate them, but SVM finds the optimal one.

#### Step 2: Prepare the Data in Code
We use NumPy arrays for features (X) and labels (y). Labels are -1 and +1 (common for SVM math).

```python
import numpy as np
import matplotlib.pyplot as plt

# Features: [spammy_words, num_links]
X = np.array([
    [0, 0],  # Not spam
    [1, 0],  # Not spam
    [0, 1],  # Not spam
    [3, 2],  # Spam
    [4, 1],  # Spam
    [5, 3]   # Spam
])

# Labels: -1 for not spam, +1 for spam
y = np.array([-1, -1, -1, 1, 1, 1])
```

#### Step 3: Add a Bias Term
SVM's decision boundary is w1*feature1 + w2*feature2 + b = 0 (where b is bias). We add a column of 1s to X to include bias in the weights.

```python
# Add bias column (all 1s)
X = np.c_[X, np.ones(X.shape[0])]
```

Now X looks like:
```
[[0 0 1]
 [1 0 1]
 [0 1 1]
 [3 2 1]
 [4 1 1]
 [5 3 1]]
```

#### Step 4: Train the SVM Model
We use gradient descent to optimize the weights (w). The update rule comes from minimizing:
- Hinge loss: max(0, 1 - y * (X dot w)) — penalizes points too close or on the wrong side.
- Plus regularization: lambda * ||w||^2 — to maximize margin (small w means wider margin).

Hyperparameters:
- Learning rate (lr): How big each update step is.
- Lambda (regularization): Controls margin vs. misclassification trade-off.
- Epochs: Number of training iterations.

```python
# Initialize weights (w0 for feature1, w1 for feature2, w2 for bias)
w = np.zeros(X.shape[1])

# Hyperparameters
lr = 0.01          # Learning rate
lambda_param = 0.01  # Regularization (small for wider margin)
epochs = 1000      # Training iterations

# Training loop (gradient descent)
for epoch in range(epochs):
    for i in range(len(y)):
        # Compute prediction
        pred = np.dot(X[i], w)
        
        # If point is correctly classified with margin >=1, only regularize
        if y[i] * pred >= 1:
            # Update: minimize ||w||^2
            w -= lr * (2 * lambda_param * w)
        else:
            # Update: hinge loss + regularization
            w -= lr * (2 * lambda_param * w - y[i] * X[i])

print("Trained weights:", w)
```

- What happens: The loop checks each point. If it's too close or wrong, adjust w to push the boundary. Over time, w stabilizes to give the max margin.
- After running, you might get weights like [0.5, 0.3, -1.2] (varies slightly with runs). This means boundary: 0.5*spammy + 0.3*links -1.2 = 0.

#### Step 5: Make Predictions
To classify a new email, compute y_pred = sign(X_new dot w). If +1, spam; if -1, not spam.

```python
# Function to predict
def predict(X_new, w):
    X_new = np.append(X_new, 1)  # Add bias
    return np.sign(np.dot(X_new, w))

# Test on a new email: 2 spammy words, 1 link (should be spam? Borderline)
new_email = np.array([2, 1])
pred = predict(new_email, w)
print("Prediction for new email:", "Spam" if pred == 1 else "Not Spam")
```

#### Step 6: Visualize the Results
Plot the points, decision boundary (w[0]*x + w[1]*y + w[2] = 0), and margins (=1 and =-1).

```python
# Plot data points
plt.scatter(X[:3, 0], X[:3, 1], color='blue', label='Not Spam (-1)')
plt.scatter(X[3:, 0], X[3:, 1], color='red', label='Spam (+1)')

# Plot decision boundary: w[0]*x + w[1]*y + w[2] = 0 → y = -(w[0]/w[1])*x - w[2]/w[1]
x_vals = np.array([min(X[:,0])-1, max(X[:,0])+1])
y_vals = -(w[0]/w[1]) * x_vals - (w[2]/w[1])
plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')

# Plot margins: same but =1 and =-1
y_margin_pos = -(w[0]/w[1]) * x_vals - (w[2]/w[1]) + (1 / np.sqrt(w[0]**2 + w[1]**2)) / (w[1]/np.sqrt(w[0]**2 + w[1]**2))
y_margin_neg = -(w[0]/w[1]) * x_vals - (w[2]/w[1]) - (1 / np.sqrt(w[0]**2 + w[1]**2)) / (w[1]/np.sqrt(w[0]**2 + w[1]**2))
plt.plot(x_vals, y_margin_pos, 'k--', label='Margin +1')
plt.plot(x_vals, y_margin_neg, 'k--', label='Margin -1')

plt.xlabel('Spammy Words')
plt.ylabel('Num Links')
plt.title('SVM for Spam Classification')
plt.legend()
plt.show()
```

- The solid line is the boundary. Dashed lines are margins.
- Support vectors: Points closest to the boundary (likely [1,0] and [3,2] in this data).

#### Full Code to Run Together
Put all the code snippets above into one file (e.g., svm_spam.py) and run it. You'll see the weights printed, a prediction, and a plot window pop up showing the separation.

#### What Happens in Real Life?
- Real spam filters (like Gmail's) use SVM or similar with **thousands of features** (word frequencies, sender IP, etc.) and RBF kernel for non-linear data.
- Data comes from labeled emails (millions!).
- Pros: SVM is accurate and handles high dimensions well.
- Cons: Slow on huge data (use optimized libraries).

------

# scikit-learn SVM** (much easier and more powerful than writing it from scratch).

### Goal
- Classify emails as **Spam** (-1) or **Ham** (+1)
- Use only **two simple features** (just for teaching)
  - Feature 1: number of suspicious words ("free", "win", "urgent", etc.)
  - Feature 2: email length in characters

### Step-by-step code example (copy-paste friendly)

```python
# Step 1: Import the libraries we need
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC                  # The SVM classifier
from sklearn.preprocessing import StandardScaler  # Important: scale features!
from sklearn.metrics import accuracy_score

print("Libraries loaded ✓")

# Step 2: Create our small toy dataset (6 emails)
# Format: [suspicious_word_count, email_length_chars]
X = np.array([
    [1,  520],   # normal email
    [2,  480],   # normal
    [1,  650],   # normal (a bit longer)
    [8,   120],  # spam - short + many bad words
    [12,   90],  # spam
    [6,   180],  # spam
])

y = np.array([1, 1, 1, -1, -1, -1])   # +1 = Ham,  -1 = Spam

print("Data ready ✓")
print("X shape:", X.shape)           # should be (6, 2)
print("y:", y)

# Step 3: VERY IMPORTANT - Scale the features!
# SVM is very sensitive to different scales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures after scaling (mean ≈ 0, std ≈ 1):")
print(np.round(X_scaled, 2))

# Step 4: Create and train the SVM model
# We use RBF kernel (the "magic" one most people start with)
model = SVC(
    kernel='rbf',       # 'linear', 'poly', 'rbf' are popular choices
    C=1.0,              # smaller C → smoother boundary, larger C → tries harder to classify correctly
    gamma='scale'       # auto-adjusts for rbf kernel
)

model.fit(X_scaled, y)

print("\nSVM model trained! ✓")

# Step 5: Make predictions on the training data (just to check)
y_pred = model.predict(X_scaled)

print("\nActual labels:   ", y)
print("Predicted labels:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred) * 100, "%")

# Step 6: Predict a new email!
new_email = np.array([[4, 300]])           # 4 suspicious words, 300 chars
new_email_scaled = scaler.transform(new_email)
prediction = model.predict(new_email_scaled)[0]

print("\nNew email prediction:")
print("Suspicious words: 4, Length: 300 chars →", 
      "HAM" if prediction == 1 else "SPAM")

# Optional: Visualize the decision boundary (only works in 2D)
def plot_decision_boundary():
    plt.figure(figsize=(7,5))
    
    # Plot points
    plt.scatter(X_scaled[y==1][:,0], X_scaled[y==1][:,1], 
                c='blue', label='Ham', s=100, edgecolors='k')
    plt.scatter(X_scaled[y==-1][:,0], X_scaled[y==-1][:,1], 
                c='red', label='Spam', s=100, edgecolors='k')
    
    # Create mesh to plot decision boundary
    x_min, x_max = X_scaled[:,0].min()-1, X_scaled[:,0].max()+1
    y_min, y_max = X_scaled[:,1].min()-1, X_scaled[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, levels=[-1,0,1], colors=['red','blue'])
    plt.contour(xx, yy, Z, colors='k', linewidths=1, linestyles=['--','-','--'])
    
    plt.xlabel("Suspicious words (scaled)")
    plt.ylabel("Email length (scaled)")
    plt.title("SVM Decision Boundary (RBF kernel)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Uncomment to see the plot (if you're running in Jupyter / VSCode / Colab)
# plot_decision_boundary()
```

### Quick Summary – What each part does

| Step | What it does                              | Why important?                                 |
|------|-------------------------------------------|------------------------------------------------|
| 1    | Import libraries                          | We need `SVC` from scikit-learn                |
| 2    | Create toy data                           | Small dataset for learning                     |
| 3    | Scale features                            | SVM performs badly if features have very different scales |
| 4    | Create & train model                      | This is where the learning happens             |
| 5    | Predict & check accuracy                  | See if it learned something                    |
| 6    | Predict on new email                      | Real-world use case                            |
| Bonus| Plot boundary                             | Helps understand what SVM actually learned     |

### Popular easy variations you can try

Change just **one line** and experiment:

```python
# Try linear kernel (straight line)
model = SVC(kernel='linear', C=1.0)

# Try very strict (tries to classify every training point correctly)
model = SVC(kernel='rbf', C=100.0, gamma='scale')

# Try smoother boundary
model = SVC(kernel='rbf', C=0.1, gamma='scale')
```

Run this code in any Python environment with scikit-learn installed (`pip install scikit-learn numpy matplotlib`).

