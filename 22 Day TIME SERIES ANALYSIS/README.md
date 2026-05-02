### Time Series Analysis

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

----

# Complete example 

'''python 

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
'''

