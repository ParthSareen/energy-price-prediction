import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Plot baseline and predictions
plt.figure(figsize=(10,6))
plt.plot(y, label='Actual Price')
plt.plot(np.concatenate((train_predict, test_predict)), label='Predicted Price')
plt.title('Energy Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

