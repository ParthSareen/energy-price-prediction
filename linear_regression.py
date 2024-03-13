import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
data = pd.read_csv('data/cleaned_data.csv')
data['hour_number'] = pd.to_datetime(data['hour_number'], format='%Y%m%d-%H', errors='coerce')
data.drop(columns=['to_drop'], inplace=True)

data = data.sort_values(by='hour_number')
data = data.set_index('hour_number')
data['time'] = np.arange(len(data.index))
print(data.head())

# Splitting the data
X = data[['demand', 'capacity', 'time']]
y = data['price']

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Split into training and test sets
train_size = int(0.8 * len(data))
train, test = data.iloc[:train_size], data.iloc[train_size:]

X_train, y_train = train[['demand', 'capacity', 'time']], train['price']
X_test, y_test = test[['demand', 'capacity', 'time']], test['price']

# Normalize our data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_normalized, y_train)
print('Linear Regression Coefficients: \n', model.coef_)

train_predict = model.predict(X_train_normalized)
test_predict = model.predict(X_test_normalized)

# Evaluate our model
# RMSE
train_mse = mean_squared_error(y_train, train_predict)
test_mse = mean_squared_error(y_test, test_predict)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')

# R-Squared
train_r2 = r2_score(y_train, train_predict)
test_r2 = r2_score(y_test, test_predict)
print(f'Train R-Squared: {train_r2:.2f}')
print(f'Test R-Squared: {test_r2:.2f}')

# Concatenate the predicted values for training and testing sets
predicted_values = np.concatenate((train_predict, test_predict))

# Create date range for the predicted values starting from the end of training data
start_date = data.index[-len(test_predict)]
date_range = np.arange(start_date, start_date + pd.Timedelta(hours=len(predicted_values)), dtype='datetime64[h]')

# Convert y values to numpy array
y_values = data['price'].to_numpy()

# Plot the predicted and actual values against date
plt.figure(figsize=(10, 6))
plt.plot(date_range, predicted_values, label='Predicted Price', color='red', linestyle='--', alpha=0.6)
plt.plot(date_range, y_values, label='Actual Price', color='blue', alpha=0.6)
plt.title('Energy Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot only the test part of the predicted and actual values against date
plt.figure(figsize=(10, 6))
plt.plot(date_range[-len(test_predict):], test_predict, label='Predicted Price (Test)', color='red', linestyle='--', alpha=0.6)
plt.plot(date_range[-len(test_predict):], y_values[-len(test_predict):], label='Actual Price (Test)', color='blue', alpha=0.6)
plt.title('Energy Price Prediction (Test Data)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
