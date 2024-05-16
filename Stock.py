import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load historical stock price data from a CSV file (replace 'AAPL.csv' with your actual file path)
data = pd.read_csv('AAPL.csv')

# Check for 'Date' Column (optional, for informative message)
if 'Date' not in data.columns:
    print("Warning: 'Date' column not found. Please check your data.")

# Data Cleaning Steps:

# 1. Handle Missing Values (if any)
# Check for missing values
print(data.isnull().sum())

# Choose an appropriate method to handle missing values (e.g., imputation, removal)
# For example, to remove rows with missing values:
# data.dropna(inplace=True)

# 2. Handle Outliers (if any)
# Identify outliers (e.g., using Z-scores or IQR)
# Choose an appropriate method to handle outliers (e.g., winsorizing, removal)
# For example, to winsorize outliers:
# data['Close'] = np.where(data['Close'] < lower_threshold, lower_threshold, data['Close'])
# data['Close'] = np.where(data['Close'] > upper_threshold, upper_threshold, data['Close'])

# 3. Handle Data Format Inconsistencies (if any)
# Check data types and formats (e.g., ensure numeric columns are numeric)
# Use conversion functions (e.g., pd.to_numeric) to address inconsistencies
# For example:
# data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Access closing prices using the index (assuming 'Close' is the closing price column)
df_indexed = data.set_index('Date')['Close']

# Perform seasonal decomposition using the indexed data
decomposition = seasonal_decompose(df_indexed, model='additive', period=30)

# Extract the trend, seasonal, and residual components
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Print some information about the decomposition (optional)
print(f"Trend: \n{trend.head()}")
print(f"Seasonal: \n{seasonal.head()}")
print(f"Residual: \n{residual.head()}")

# Plot the components
plt.figure(figsize=(12, 6))
decomposition.plot()
plt.show()

# Plot individual components
plt.figure(figsize=(8, 5))
trend.plot(title="Trend Component")
plt.show()

plt.figure(figsize=(8, 5))
seasonal.plot(title="Seasonal Component")
plt.show()

plt.figure(figsize=(8, 5))
residual.plot(title="Residuals")
plt.show()

# Analyze the plot visually for seasonality patterns, randomness, and trend characteristics
# ...

# Now proceed with your forecasting model development and reporting.
# Remember to document your steps and choices thoroughly.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Assuming df_indexed contains your indexed stock price data
train_size = int(len(df_indexed) * 0.8)
train, test = df_indexed[:train_size], df_indexed[train_size:]

# Define ARIMA model parameters (replace with appropriate values)
p, d, q = 1, 1, 1  # Example values; adjust based on your analysis

# Fit ARIMA model
model = ARIMA(train, order=(p, d, q))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))

# Plot actual vs. forecasted
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.fill_between(test.index, forecast - 1.96 * rmse, forecast + 1.96 * rmse, color='gray', alpha=0.2)
plt.title(f"ARIMA Forecast (RMSE: {rmse:.2f})")
plt.legend()
plt.show()

# Report your findings and insights
