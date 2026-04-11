

# What is Naive Bayes Classifier (NBC) in Machine Learning?

Naive Bayes Classifier (NBC) is a very simple and powerful **classification algorithm** in machine learning. It belongs to **supervised learning**, which means it needs labeled training data (input features along with their correct output classes) to learn.

### Simple Explanation: What is Naive Bayes?

Imagine you receive an **email** and you want to decide whether it is **Spam** or **Not Spam** (Ham).

You look at the words inside the email like: “Free”, “Offer”, “Win”, “Dear”, etc.

Naive Bayes thinks like this:  
- If the words “Free” and “Win” appear, what is the probability that the email is Spam?  
- If the words “Dear” and “Friend” appear, what is the probability that the email is Not Spam?

It calculates the probabilities for each class and chooses the class with the **highest probability**.

It is called **“Naive”** because it makes one **simple (naive) assumption**:  
**All features (like every word) are independent of each other.**  
(In real life, words are related, but even with this assumption, the algorithm works surprisingly well!)

This algorithm is based on **Bayes’ Theorem**.

### What is Bayes’ Theorem? (Very Simple Way)

Bayes’ Theorem is a formula to calculate probability:

**P(A|B) = [P(B|A) × P(A)] / P(B)**

In machine learning, we use it like this:

**P(Class | Features) = [P(Features | Class) × P(Class)] / P(Features)**

- **P(Class)** → Prior Probability (how often this class appeared in the training data)
- **P(Features | Class)** → Likelihood (if this class is true, how likely these features are)
- **P(Features)** → Evidence (this is constant, so we can ignore it during comparison)

Finally, we choose the class that has the **highest posterior probability**.

**Simplified Formula (with Naive Assumption):**

P(y | x₁, x₂, ..., xₙ) ∝ P(y) × P(x₁ | y) × P(x₂ | y) × ... × P(xₙ | y)

(Here, y = class, x = features)

### Real-Life Example (Classic Fruit Example)

Suppose you have to classify a fruit as **Apple**, **Banana**, or **Orange**.

Features: Color (Red/Yellow), Shape (Round/Long), Taste (Sweet/Sour).

From training data, you have calculated probabilities:

- P(Apple) = 0.4 (prior probability)
- If it is Apple, then P(Red | Apple) = 0.9
- P(Round | Apple) = 0.8
- P(Sweet | Apple) = 0.7

Now a new fruit comes: **Red + Round + Sweet**

You calculate the probability for Apple (and similarly for Banana and Orange).

Whichever class gives the highest probability, you assign the fruit to that class.

This process is very fast because you just need to multiply the probabilities.

### Different Types of Naive Bayes

Depending on the type of data, we use different versions:

1. **Gaussian Naive Bayes**  
   - Used when features are **continuous** (numbers like height, weight, temperature).  
   - It assumes that the features follow a **Normal (Gaussian) distribution** (bell curve).  
   - Example: Iris flower dataset (petal length, sepal width, etc.)

2. **Multinomial Naive Bayes**  
   - Used when features are **counts** (like how many times a word appears in a document).  
   - Most commonly used for **Text Classification** (spam detection, news categorization).

3. **Bernoulli Naive Bayes**  
   - Used when features are **binary** (0 or 1) – whether a word is present or not.  
   - It only checks the presence or absence of a word, not the count.

### How Does Naive Bayes Get Trained? (Step by Step)

1. Collect training data (features + labels).
2. Calculate **prior probability** for each class (how many samples belong to each class).
3. Calculate **conditional probability** for each feature given the class: P(feature | class).
4. When new data comes, multiply all the probabilities and pick the class with the highest value.

### Advantages (Why is it Popular?)

- Very **fast** to train and predict.
- Works well even with **less training data**.
- Excellent for **high-dimensional data** (thousands of features, very common in text data).
- Simple and easy to implement.
- Not heavily affected by irrelevant features.

### Disadvantages (Limitations)

- The **naive assumption** (features are independent) is not always true in real life.
- If a feature never appeared in training data for a class, its probability becomes **zero** (this is fixed using **Laplace Smoothing**).
- For continuous data, if the Gaussian assumption is wrong, accuracy may drop.

### Where is Naive Bayes Used? (Applications)

- **Spam Email Detection** (most famous use)
- Sentiment Analysis (positive/negative reviews)
- Text Classification (categorizing news articles)
- Medical Diagnosis (predicting disease from symptoms)
- Recommendation Systems
- Weather Prediction (sometimes)

### Simple Python Code Example (Using Scikit-learn)

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load Iris dataset (example)
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()   # Gaussian version
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

For text data, use `MultinomialNB()` instead.

**Tip:** For text classification, first convert text into numbers using CountVectorizer or TF-IDF, then apply MultinomialNB.

---

