# Handling Missing Data

# Exploratory Data Analysis (EDA)

Using the real **UCI Automobile Dataset** (1985 Auto Imports data, 205 cars).

### Why Do We Care About Missing Data?
In real datasets like car information, some values are often missing (shown as **?** or blank).  
If we ignore them or delete too many rows, we lose important information.  
**Imputation** means "filling" the missing spots intelligently so our analysis (like average price, horsepower vs price relationship) remains reliable.

**Real Dataset Example**:  
The UCI Automobile dataset has details like:  
- `make` (e.g., toyota, bmw)  
- `horsepower`  
- `price`  
- `normalized-losses` (insurance risk/loss score)  
- `num-of-doors`, `bore`, `stroke`, etc.

**Missing values in this dataset**:
- `normalized-losses`: 41 missing (about 20%)
- `price`: 4 missing
- `horsepower`: 2 missing
- `peak-rpm`: 2 missing
- `bore` and `stroke`: 4 each
- `num-of-doors`: 2 missing

We'll use **Python + pandas** to show this (you can run the same code in Jupyter Notebook or Google Colab).

### Step 1: Load the Data and Find Missing Values
```python
import pandas as pd
import numpy as np

# Load the dataset (no header in raw file)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 
           'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 
           'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 
           'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 
           'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

df = pd.read_csv(url, names=columns, na_values='?')  # '?' becomes NaN (missing)

print(df.shape)                    # (205, 26)
print(df.isnull().sum())           # Shows count of missing values per column
```

**Output example**:
```
normalized-losses    41
num-of-doors          2
bore                  4
stroke                4
horsepower            2
peak-rpm              2
price                 4
...
```

We see missing data clearly now.

### Common Imputation Techniques (Explained Simply with Examples)

#### 1. **Mean Imputation** (Best for numerical data with normal distribution)
Replace missing value with the **average** of the column.

**Simple Analogy**: If some students' test scores are missing, fill with the class average.

**Example in Automobile Data** (for `horsepower`):
```python
# Before: some horsepower missing
print(df['horsepower'].mean())   # e.g., around 104

# Impute with mean
df['horsepower_mean'] = df['horsepower'].fillna(df['horsepower'].mean())

# Or in one line
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
```

**When to use**: When data is symmetric (not too skewed).  
**Caution**: Can reduce variation slightly.

#### 2. **Median Imputation** (Robust to outliers)
Replace with the **middle value** when data is sorted.

**Simple Analogy**: If car prices are missing, use the middle price instead of average (average can be pulled up by very expensive cars like Ferrari).

**Example** (good for `price` or `normalized-losses`):
```python
median_price = df['price'].median()
df['price'].fillna(median_price, inplace=True)
```

**When to use**: When there are extreme high/low values (outliers), like luxury cars.

#### 3. **Mode Imputation** (Best for categorical data)
Replace with the **most frequent** value.

**Simple Analogy**: If a car's number of doors is missing, fill with the most common (e.g., "four" doors).

**Example** (for `num-of-doors`):
```python
mode_doors = df['num-of-doors'].mode()[0]   # e.g., 'four'
df['num-of-doors'].fillna(mode_doors, inplace=True)
```

**When to use**: For categories like fuel-type, body-style, doors.

#### 4. **Group-wise Imputation** (Smarter – Recommended for EDA)
Fill based on similar cars (e.g., same `make` or `body-style`).

**Why better?** A Toyota's missing horsepower should be filled with average Toyota horsepower, not overall average.

**Real Example**:
```python
# Fill normalized-losses with mean of same 'make'
df['normalized-losses'] = df.groupby('make')['normalized-losses'].transform(lambda x: x.fillna(x.mean()))

# Alternative: fill price with median of same body-style
df['price'] = df.groupby('body-style')['price'].transform(lambda x: x.fillna(x.median()))
```

This keeps the data more realistic.

#### 5. **Forward/Backward Fill** (for sequential data – less common here)
```python
df['price'].fillna(method='ffill', inplace=True)  # Use previous value
# or 'bfill' for next value
```

Useful if data is time-ordered, but not ideal for car specs.

#### 6. **Advanced: KNN Imputation** (Uses similar rows)
It finds "similar" cars based on other features and takes their average.

```python
from sklearn.impute import KNNImputer

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])
imputer = KNNImputer(n_neighbors=5)
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
```

**When to use**: When you have many features and want higher accuracy (but slower).

### Quick Comparison Table (Easy to Remember)

| Technique       | Best For              | Example Column       | Pros                          | Cons                          |
|-----------------|-----------------------|----------------------|-------------------------------|-------------------------------|
| Mean           | Symmetric numbers    | horsepower          | Simple, fast                 | Affected by outliers         |
| Median         | Skewed numbers       | price, curb-weight  | Robust to outliers           | Ignores relationships        |
| Mode           | Categories           | num-of-doors        | Works for text               | Can over-represent common value |
| Group Mean/Median | Related groups      | normalized-losses   | More accurate                | Needs a grouping column      |
| KNN            | Complex data         | bore, stroke        | Considers all features       | Computationally heavier      |

### Practical Tips for EDA on Automobile Data
1. Always check `df.isnull().sum()` and `df.isnull().mean() * 100` (percentage).
2. Visualize missing data:
   ```python
   import seaborn as sns
   import matplotlib.pyplot as plt
   sns.heatmap(df.isnull(), cbar=False)
   plt.show()
   ```
3. After imputation, re-check summary statistics (`df.describe()`) to see if they changed too much.
4. Decide based on **why** data is missing (e.g., some old cars don't report normalized-losses).
5. If too many missing (>50% in a column), sometimes better to drop the column.

### Simple Code Summary (All in One)
```python
# Replace ? with NaN
df.replace('?', np.nan, inplace=True)

# Simple imputation
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)
df['price'].fillna(df['price'].median(), inplace=True)
df['num-of-doors'].fillna(df['num-of-doors'].mode()[0], inplace=True)

# Better: group-based
df['normalized-losses'] = df.groupby('make')['normalized-losses'].transform(lambda x: x.fillna(x.mean()))
```

------

# What is an Outlier?

An **outlier** is a data point that is very different from all the other data.

**Simple example**:  
In a class, most students score between 40 to 80 marks.  
But one student scores 950 marks.  
That 950 is an **outlier** — it does not belong with the others.

### Outliers in Car Data (Automobile Dataset)

In the car dataset:
- Most cars have a **price** between $5,000 and $25,000.
- But a few luxury cars (like Porsche, Jaguar, BMW) cost $40,000 or more.
- These very expensive cars are **outliers**.

If we don’t handle them, our average price, graphs, and conclusions will become wrong.

So in EDA, we must **find** outliers and then **handle** them.

---

### Step-by-Step: Outlier Detection and Handling

#### Step 1: Find the Outliers (Detection)

There are two easy ways:

**Way 1: Use a Box Plot (Easiest to see)**  
A box plot shows normal data in a box.  
Any points outside the “whiskers” are outliers (shown as dots).

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['price'])
plt.title("Outliers in Car Prices")
plt.show()
```

You will see many dots above the box — these are the outliers.

**Way 2: Use IQR Method (Most Common)**  
IQR = Middle 50% of the data.

Steps:
1. Find Q1 (25% value)
2. Find Q3 (75% value)
3. Calculate IQR = Q3 - Q1
4. Anything below (Q1 - 1.5×IQR) or above (Q3 + 1.5×IQR) is an outlier.

```python
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['price'] < lower) | (df['price'] > upper)]

print("Number of outliers in price:", len(outliers))
print("Expensive outlier cars:", outliers['make'].values)
```

In this car data, you will usually find 15–20 outliers in the price column (mostly luxury cars).

---

#### Step 2: Handle the Outliers (What to do with them?)

You have 4 simple options:

| Method              | What it does                                      | When to use it                              | Simple Code Example |
|---------------------|---------------------------------------------------|---------------------------------------------|---------------------|
| 1. Remove           | Delete the entire row                             | Very few outliers                           | `df = df[(df['price'] >= lower) & (df['price'] <= upper)]` |
| 2. Capping          | Change extreme values to the upper/lower limit    | When outliers are real (like luxury cars)   | `df['price'] = np.where(df['price'] > upper, upper, df['price'])` |
| 3. Transform        | Use log to make data smaller                      | When data is very skewed                    | `df['price_log'] = np.log(df['price'])` |
| 4. Keep as it is    | Do nothing                                        | When outliers are important for business    | No code needed |

**Best choice for this car data**: **Capping**

```python
# Capping (Recommended)
df['price_capped'] = np.clip(df['price'], lower, upper)
```

This means: Any price higher than the upper limit is brought down to the upper limit.  
Any price lower than the lower limit is brought up to the lower limit.

---

### Full Easy Code Summary

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = ['symboling','normalized-losses','make', ... ]  # same as before
df = pd.read_csv(url, names=columns, na_values='?')

# Fix price column
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'].fillna(df['price'].median(), inplace=True)

# Step 1: Find outliers using IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("Outliers found:", len(df[(df['price'] < lower) | (df['price'] > upper)]))

# Step 2: Handle by Capping
df['price_capped'] = np.clip(df['price'], lower, upper)

# Check before and after
print("Highest price before:", df['price'].max())
print("Highest price after capping:", df['price_capped'].max())
```

---



# What is Categorical Data?
Categorical data is data that contains **words** or **categories** instead of numbers.

**Examples from Car Dataset:**
- `make` → toyota, bmw, honda, mazda...
- `fuel-type` → gas, diesel
- `body-style` → sedan, hatchback, wagon, convertible
- `num-of-doors` → two, four
- `drive-wheels` → fwd, rwd, 4wd
- `engine-type` → ohc, ohcf, dohc...

**Problem:**  
Machine Learning models can only understand **numbers**, not words.  
So we must **convert** these words into numbers.  
This process is called **Encoding**.

---

### Two Main Encoding Techniques

#### 1. **Label Encoding** (Simple Number Assignment)

**What it does:**  
It gives each category a unique number.

**Example:**
| fuel-type | After Label Encoding |
|-----------|----------------------|
| gas       | 0                    |
| diesel    | 1                    |

**Another Example (num-of-doors):**
| num-of-doors | Label Encoded |
|--------------|---------------|
| two          | 0             |
| four         | 1             |

**Simple Code:**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['fuel-type_encoded'] = le.fit_transform(df['fuel-type'])
df['body-style_encoded'] = le.fit_transform(df['body-style'])

print(df[['fuel-type', 'fuel-type_encoded']].head())
```

**When to use Label Encoding:**
- When categories have **order** (like small < medium < large)
- When the column has only **2 categories** (binary)
- For tree-based models (like Decision Tree, Random Forest)

**Problem with Label Encoding:**
- The model may think **4 is bigger than 0**, even if there is no real order.
- Example: It may think "wagon (3)" is bigger than "sedan (0)" — which is wrong.

---

#### 2. **One-Hot Encoding** (Best and Most Popular)

**What it does:**  
It creates **new columns** for each category.  
Puts **1** if the category is present, **0** otherwise.

**Simple Example (fuel-type):**

| fuel-type | fuel-type_gas | fuel-type_diesel |
|-----------|---------------|------------------|
| gas       | 1             | 0                |
| diesel    | 0             | 1                |
| gas       | 1             | 0                |

**Another Example (body-style with 4 types):**

| body-style  | body_sedan | body_hatchback | body_wagon | body_convertible |
|-------------|------------|----------------|------------|------------------|
| sedan       | 1          | 0              | 0          | 0                |
| hatchback   | 0          | 1              | 0          | 0                |
| wagon       | 0          | 0              | 1          | 0                |

**Simple Code (Using pandas - easiest way):**
```python
# One-Hot Encoding using pandas
df_encoded = pd.get_dummies(df, columns=['fuel-type', 'body-style', 'drive-wheels'])

print(df_encoded.head())
```

**Using Scikit-learn:**
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids dummy variable trap

encoded = ohe.fit_transform(df[['fuel-type', 'body-style']])

# Convert to DataFrame
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())
```

---

### Quick Comparison (Easy to Remember)

| Feature                  | Label Encoding                  | One-Hot Encoding                          |
|--------------------------|---------------------------------|-------------------------------------------|
| How it works             | Converts to numbers (0,1,2...) | Creates new columns with 0 and 1         |
| Best for                 | Ordered data, 2 categories     | Most categorical columns                 |
| Creates new columns      | No                              | Yes (one per category)                   |
| Risk of wrong order      | Yes                             | No                                       |
| Good for Linear Models   | Not good                        | Very good                                |
| Good for Tree Models     | Okay                            | Good                                     |
| Memory usage             | Low                             | Higher (if many categories)              |

**Golden Rule in Practice:**
- Use **One-Hot Encoding** for most columns (safer)
- Use **Label Encoding** only when categories have natural order or very few unique values

---

### Real Example with Automobile Data

```python
import pandas as pd

# One-Hot Encoding (Recommended)
df_encoded = pd.get_dummies(df, columns=['make', 'fuel-type', 'body-style', 
                                         'drive-wheels', 'engine-type'])

print("Original columns:", df.shape[1])
print("After One-Hot Encoding:", df_encoded.shape[1])
```

You will see the number of columns increases a lot because new columns are created for each car brand and body style.

---

Here's a **super simple English** explanation of **Data Scaling & Normalization** using the same **Automobile Dataset**.

### What is Data Scaling / Normalization?

**Scaling** means making all the numbers in different columns **similar in range**.

**Why do we need it?**  
Different columns have very different ranges:

| Column          | Range                  | Problem |
|-----------------|------------------------|--------|
| `price`         | 5,000 to 45,000        | Very big numbers |
| `horsepower`    | 48 to 288              | Medium |
| `engine-size`   | 61 to 326              | Small |
| `city-mpg`      | 13 to 49               | Small |

If we don’t scale, columns with big numbers (like price) will dominate the model.  
Small columns (like mpg) will be ignored.  
So we **scale** all features to bring them to a similar range.

This is very important for:
- Linear Regression
- KNN, SVM, Neural Networks
- PCA, Clustering

Tree models (Random Forest, XGBoost) usually **don’t need** scaling.

---

### Two Most Popular Methods

#### 1. **Min-Max Scaling** (Also called Normalization)

**What it does:**  
It squeezes all values between **0 and 1**.

Formula:  
**New Value = (X - Min) / (Max - Min)**

**Simple Example (price column):**

| Original Price | After Min-Max Scaling |
|----------------|-----------------------|
| 5,000          | 0.00                  |
| 15,000         | 0.33                  |
| 30,000         | 0.78                  |
| 45,000         | 1.00                  |

**Code (Very Easy):**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Scale only numeric columns
numeric_cols = ['price', 'horsepower', 'engine-size', 'curb-weight', 'city-mpg']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print(df[numeric_cols].head())
```

**Result:** All values will now be between **0 and 1**.

**When to use Min-Max Scaling:**
- When you want data strictly between 0 and 1
- When your data does **not** have many outliers
- Good for image data or Neural Networks

**Disadvantage:** Sensitive to outliers (one very expensive car can squash all other values).

---

#### 2. **Standardization** (Z-Score Scaling) – Most Commonly Used

**What it does:**  
It makes the data have **mean = 0** and **standard deviation = 1**.

Formula:  
**New Value = (X - Mean) / Standard Deviation**

**Simple Example:**

| Original Horsepower | After Standardization |
|---------------------|-----------------------|
| 100                 | -0.45                 |
| 150                 | 0.85                  |
| 200                 | 2.15                  |

Now the data is centered around 0. Negative and positive values are normal.

**Code:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numeric_cols = ['price', 'horsepower', 'engine-size', 'curb-weight', 'wheel-base']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("After Standardization:")
print("Mean should be close to 0:", df[numeric_cols].mean())
print("Std should be close to 1:", df[numeric_cols].std())
```

**When to use Standardization:**
- When your data has **outliers**
- Most popular choice in Machine Learning
- Works well with algorithms that assume normal distribution

**Advantage:** Less affected by outliers compared to Min-Max.

---

### Quick Comparison (Easy to Remember)

| Feature                    | Min-Max Scaling              | Standardization (Z-score)       |
|----------------------------|------------------------------|---------------------------------|
| New Range                  | 0 to 1                       | Mean=0, Std=1                   |
| Formula                    | (X - Min)/(Max - Min)        | (X - Mean)/Std                  |
| Sensitive to Outliers      | Yes (very sensitive)         | Less sensitive                  |
| Best For                   | Neural Networks, Images      | Most ML algorithms              |
| Values can be negative?    | No                           | Yes                             |
| Easy to understand         | Very easy                    | Slightly harder                 |

**Golden Rule for Beginners:**
- Use **Standardization** in most cases.
- Use **Min-Max** when you specifically need 0 to 1 range.

---

### Full Easy Code Example (Automobile Dataset)

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors',
           'body-style','drive-wheels','engine-location','wheel-base','length','width','height',
           'curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system',
           'bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']

df = pd.read_csv(url, names=columns, na_values='?')

# Quick cleaning
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['price'].fillna(df['price'].median(), inplace=True)
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

# Select numeric columns for scaling
numeric_cols = ['wheel-base', 'length', 'width', 'curb-weight', 
                'engine-size', 'horsepower', 'city-mpg', 'price']

# === Min-Max Scaling ===
minmax_scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[numeric_cols] = minmax_scaler.fit_transform(df[numeric_cols])

# === Standardization ===
std_scaler = StandardScaler()
df_std = df.copy()
df_std[numeric_cols] = std_scaler.fit_transform(df[numeric_cols])

print("Before Scaling - Price range:", df['price'].min(), "to", df['price'].max())
print("After Min-Max - Price range:", df_minmax['price'].min(), "to", df_minmax['price'].max())
print("After Standardization - Price mean:", round(df_std['price'].mean(), 2))
```

---


