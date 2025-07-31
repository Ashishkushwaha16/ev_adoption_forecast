import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load CSV
data = pd.read_csv("data/ev_data_samplefinal.csv")

# Rename for convenience
data = data.rename(columns={"Electric Vehicle Total": "EV_Registrations"})

# Train model
X = data[["Year"]]
y = data["EV_Registrations"]

model = LinearRegression()
model.fit(X, y)

# Predict for future years
future_years = pd.DataFrame({"Year": range(2025, 2036)})
future_preds = model.predict(future_years)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(data["Year"], data["EV_Registrations"], color="blue", label="Historical")
plt.plot(future_years["Year"], future_preds, color="red", linestyle="--", label="Forecast")
plt.title("EV Registrations Forecast (King County)")
plt.xlabel("Year")
plt.ylabel("EV Registrations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
