# ev_forecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("data/ev_data_sample.csv")

print("Initial shape:", data.shape)
print(data.dtypes)
print("\nMissing values:\n", data.isnull().sum())

# Convert 'Date' to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Sort by Date
data = data.sort_values(by="Date")

# Feature engineering
data["Monthly Growth"] = data["Electric Vehicle (EV) Total"].pct_change().fillna(0)
data["Lag1"] = data["Electric Vehicle (EV) Total"].shift(1).fillna(method='bfill')
data["RollingMean"] = data["Electric Vehicle (EV) Total"].rolling(3).mean().fillna(method='bfill')

# Prepare input/output
features = ["Lag1", "RollingMean", "Monthly Growth"]
X = data[features]
y = data["Electric Vehicle (EV) Total"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict & evaluate
y_pred = model.predict(X)
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ev_model.pkl")
print("\n✅ Model saved at: model/ev_model.pkl")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(data["Date"], y, label="Actual", marker="o")
plt.plot(data["Date"], y_pred, label="Predicted", linestyle="--")
plt.title("EV Adoption Trend")
plt.xlabel("Date")
plt.ylabel("EV Total")
plt.legend()
plt.tight_layout()
plt.savefig("ev_forecast_plot.png")
plt.show()
