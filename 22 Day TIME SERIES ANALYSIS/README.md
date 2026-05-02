# Time Series Analysis

**Introduction to Time Series Analysis**

Time Series Analysis is a statistical technique that deals with time series data, or data that is observed sequentially over time. This type of analysis is crucial in various fields such as finance, economics, environmental science, and more, for understanding past behavior and predicting future values.

**Key Topics in Time Series Analysis:**

1. **Understanding Time Series Data**
2. **Time Series Components**
3. **Stationarity**
4. **Autocorrelation and Partial Autocorrelation**
5. **Time Series Models**
6. **Model Evaluation**
7. **Seasonal Decomposition**
8. **Advanced Topics**

---

### 1. Understanding Time Series Data

Time series data is a sequence of data points typically measured at successive points in time, spaced at uniform time intervals. Key characteristics include trend, seasonality, and cyclic patterns.

**Example:**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Example time series data
data = {
    'Date': pd.date_range(start='1/1/2020', periods=12, freq='M'),
    'Value': [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118]
}
df = pd.DataFrame(data)

# Plotting the data
plt.plot(df['Date'], df['Value'])
plt.title('Monthly Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```

---

### 2. Time Series Components

Time series data can be decomposed into several components:

- **Trend:** The long-term progression of the series.
- **Seasonality:** Regular pattern of fluctuations within a specific period.
- **Cyclic Patterns:** Long-term cycles or oscillations.
- **Irregular Components:** Random noise or residuals.

**Example:**

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df['Value'], model='multiplicative', period=12)
result.plot()
plt.show()
```

---

### 3. Stationarity

A time series is stationary if its properties do not depend on the time at which the series is observed. Testing for stationarity typically involves checking the mean, variance, and autocorrelation.

**Example:**

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

---

### 4. Autocorrelation and Partial Autocorrelation

- **Autocorrelation Function (ACF):** Measures the correlation between observations of a time series separated by k time units.
- **Partial Autocorrelation Function (PACF):** Measures the correlation between observations of a time series separated by k time units, after removing the correlations explained by all shorter lags.

**Example:**

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['Value'])
plot_pacf(df['Value'])
plt.show()
```

---

### 5. Time Series Models

Several models are used for time series forecasting:

- **AR (Autoregressive) Model:** A model that uses the dependent relationship between an observation and some number of lagged observations.
- **MA (Moving Average) Model:** A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.
- **ARMA (Autoregressive Moving Average) Model:** A combination of AR and MA models.
- **ARIMA (Autoregressive Integrated Moving Average) Model:** An extension of ARMA that also includes differencing to make the time series stationary.

**Example:**

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['Value'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```

---

### 6. Model Evaluation

Evaluating the performance of a time series model is crucial. Common metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

**Example:**

```python
from sklearn.metrics import mean_squared_error

predictions = model_fit.forecast(steps=12)
mse = mean_squared_error(df['Value'], predictions)
rmse = mse ** 0.5
print('RMSE:', rmse)
```

---

### 7. Seasonal Decomposition

Seasonal decomposition involves breaking down the time series into trend, seasonal, and residual components.

**Example:**

```python
decomposition = seasonal_decompose(df['Value'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(df['Value'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
```

---

### 8. Advanced Topics

- **Vector Autoregression (VAR)**
- **Seasonal ARIMA (SARIMA)**
- **Prophet Forecasting**
- **GARCH Models**

**Example: Using Facebook's Prophet for forecasting:**

```python
from fbprophet import Prophet

prophet_df = df.rename(columns={'Date': 'ds', 'Value': 'y'})
model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

model.plot(forecast)
plt.show()
```
# Example - Complete
```python
# ================================================
# Beginner Time Series Analysis Example in Python
# ================================================

# Step 1: Install libraries (run once if needed)
# !pip install pandas numpy matplotlib seaborn statsmodels

# Step 2: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("✅ Libraries imported successfully!")
```

```python
# Step 3: Create a Sample Time Series Dataset
# We'll create synthetic monthly sales data with trend + seasonality + noise

np.random.seed(42)  # For reproducibility

# Date range: 3 years of monthly data
dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='ME')

# Create synthetic sales data
base_trend = np.linspace(100, 200, len(dates))           # Upward trend
seasonality = 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)  # Yearly seasonality
noise = np.random.normal(0, 15, len(dates))              # Random noise

sales = base_trend + seasonality + noise

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Sales': sales
})

# Set Date as index (very important for time series)
df.set_index('Date', inplace=True)

print("✅ Dataset created!")
print(df.head())
print(f"\nShape: {df.shape}")
```

```python
# Step 4: Basic Visualization
plt.figure(figsize=(14, 7))

plt.plot(df.index, df['Sales'], marker='o', linestyle='-', color='blue', label='Monthly Sales')
plt.title('Monthly Sales Data (2022-2024)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Quick statistics
print(df['Sales'].describe())
```

```python
# Step 5: Resampling (Change frequency)
# Monthly to Quarterly
quarterly = df['Sales'].resample('QE').mean()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Sales'], label='Monthly', alpha=0.6)
plt.plot(quarterly.index, quarterly, marker='o', linewidth=3, label='Quarterly Average')
plt.title('Monthly vs Quarterly Sales')
plt.legend()
plt.grid(True)
plt.show()
```

```python
# Step 6: Rolling Statistics (Moving Average)
df['MA_3'] = df['Sales'].rolling(window=3).mean()   # 3-month moving average
df['MA_6'] = df['Sales'].rolling(window=6).mean()   # 6-month moving average

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Sales'], label='Original Sales', alpha=0.7)
plt.plot(df.index, df['MA_3'], label='3-Month MA', linewidth=2)
plt.plot(df.index, df['MA_6'], label='6-Month MA', linewidth=2)
plt.title('Sales with Moving Averages')
plt.legend()
plt.grid(True)
plt.show()
```

```python
# Step 7: Check Stationarity (Augmented Dickey-Fuller Test)

def check_stationarity(timeseries):
    print("=== Augmented Dickey-Fuller Test ===")
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    
    if result[1] <= 0.05:
        print("✅ Stationary (Reject null hypothesis)")
    else:
        print("❌ Non-stationary (Fail to reject null hypothesis)")
    
    print(f'Critical Values: {result[4]}')

check_stationarity(df['Sales'])
```

```python
# Step 8: Time Series Decomposition
decomposition = seasonal_decompose(df['Sales'], model='additive', period=12)

fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.suptitle('Time Series Decomposition', fontsize=16)
plt.tight_layout()
plt.show()

print("Decomposition helps us separate Trend + Seasonality + Residual")
```

```python
# Step 9: Simple Forecasting with ARIMA
# For beginners: ARIMA(p,d,q)
# p = autoregression, d = differencing, q = moving average

# Train on first 80% data
train_size = int(len(df) * 0.8)
train = df['Sales'][:train_size]
test = df['Sales'][train_size:]

# Fit ARIMA model (you can tune these parameters)
model = ARIMA(train, order=(2, 1, 2))  # Example order
model_fit = model.fit()

print(model_fit.summary().tables[1])  # Show coefficients
```

```python
# Step 10: Make Predictions
forecast = model_fit.forecast(steps=len(test))

# Plot actual vs forecast
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Test Data', color='blue')
plt.plot(test.index, forecast, label='Forecast', color='red', linestyle='--')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.grid(True)
plt.show()

# Calculate error
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(test, forecast)
print(f"Mean Absolute Error: {mae:.2f}")
```

### Key Takeaways for Beginners:

1. **Always set Date as index** with `df.set_index('Date')`
2. **Visualize first** — time series is all about patterns over time
3. **Check Stationarity** before modeling
4. **Decomposition** reveals hidden patterns (trend + seasonality)
5. Start simple (Moving Average → ARIMA → Prophet/LSTM later)

### Next Steps to Learn:
- Try real datasets (`pandas_datareader`, Kaggle sales data)
- Use `auto_arima` from `pmdarima`
- Try Facebook Prophet for easier seasonal modeling
- Learn ACF/PACF plots for parameter selection



