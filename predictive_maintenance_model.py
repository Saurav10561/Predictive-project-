
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load dataset (assuming CSV format)
data = pd.read_csv('equipment_failure_data.csv')

# Data preprocessing
data['Failure_Date'] = pd.to_datetime(data['Failure_Date'])
data['Days_Since_Last_Failure'] = data['Failure_Date'].diff().dt.days.fillna(0)
data['Cumulative_Failures'] = np.cumsum(data['Failure'])

# Feature selection
X = data[['Usage_Hours', 'Maintenance_Schedule', 'Days_Since_Last_Failure']]
y = data['Failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression model for prediction
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Time series forecasting visualization
plt.plot(data['Failure_Date'], data['Cumulative_Failures'])
plt.title('Cumulative Equipment Failures Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Failures')
plt.show()

print(f'Model Mean Squared Error: {mse}')
