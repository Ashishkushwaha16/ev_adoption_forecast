import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("EV Adoption Forecast – Washington State (King County)")

# Load dataset
try:
    data = pd.read_csv("data/ev_data_samplefinal.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please ensure 'ev_data_samplefinal.csv' exists in the 'data' folder.")
    st.stop()

# Rename columns for easier access (optional but clean)
data = data.rename(columns={"Electric Vehicle Total": "EV_Registrations"})

# Sidebar year selection
start_year = st.sidebar.slider("Start Year", 2023, 2030, 2025)
end_year = st.sidebar.slider("End Year", 2031, 2040, 2035)

# Prepare data for training
X = data[["Year"]]
y = data["EV_Registrations"]

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Forecast
future_years = pd.DataFrame({"Year": range(start_year, end_year + 1)})
future_preds = model.predict(future_years)

# Forecast results
forecast_df = future_years.copy()
forecast_df["Predicted_Registrations"] = future_preds.astype(int)

# Display DataFrame
st.subheader("Forecasted EV Registrations")
st.dataframe(forecast_df)

# Plot
st.subheader("Trend Visualization")
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(data["Year"], data["EV_Registrations"], color="blue", label="Historical")
ax.plot(future_years["Year"], future_preds, color="orange", linestyle="--", label="Forecast")
ax.set_xlabel("Year")
ax.set_ylabel("EV Registrations")
ax.set_title("EV Registration Forecast (King County)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
