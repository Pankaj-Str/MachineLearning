# Overfitting** and **Underfitting
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
















**Left image / low degree (degree 1)** → Underfitting (straight line can't capture the curve)  
**Middle image / good degree (around 3–4)** → Good fit (follows the real pattern nicely)  
**Right image / very high degree (15 or more)** → Overfitting (wiggles a lot to pass through every single training point including noise)

### Learning Curve View (very important to detect them)

We train model → look at error on training data and validation/test data as model gets more complex or trains longer.








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
