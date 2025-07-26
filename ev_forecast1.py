# ev_forecast.py

# --------------------------
# EV Adoption Forecasting
# --------------------------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
DATA_PATH = os.path.join("data", "ev_data_sample1.csv")
df = pd.read_csv(DATA_PATH)

# 2. Initial Checks
print("Initial shape:", df.shape)
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# 3. Data Preprocessing
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df[df['Date'].notnull()]

df['County'] = df['County'].fillna('Unknown')
df['State'] = df['State'].fillna('Unknown')

# Convert numeric columns
cols_to_clean = [
    'Battery Electric Vehicles (BEVs)',
    'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Electric Vehicle (EV) Total',
    'Non-Electric Vehicle Total',
    'Total Vehicles'
]
for col in cols_to_clean:
    df[col] = df[col].astype(str).str.replace(',', '').astype(int)

# Handle outliers
Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound,
                                           np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound,
                                                    df['Percent Electric Vehicles']))

# 4. Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

le_county = LabelEncoder()
le_state = LabelEncoder()
le_use = LabelEncoder()

df['County_encoded'] = le_county.fit_transform(df['County'])
df['State_encoded'] = le_state.fit_transform(df['State'])
df['Vehicle_Use_encoded'] = le_use.fit_transform(df['Vehicle Primary Use'])

# 5. Define features and target
features = [
    'Year', 'Month', 'Quarter',
    'County_encoded', 'State_encoded', 'Vehicle_Use_encoded',
    'Battery Electric Vehicles (BEVs)',
    'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Non-Electric Vehicle Total'
]
target = 'Electric Vehicle (EV) Total'

X = df[features]
y = df[target]

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 9. Save Model
MODEL_PATH = os.path.join("model", "ev_model.pkl")
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\n Model saved at: {MODEL_PATH}")
