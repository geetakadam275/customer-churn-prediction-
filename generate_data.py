"""Step 1: create a sample customer dataset.

If you have a real dataset (for example the Kaggle "Telco Customer Churn" CSV),
skip this step and just save it as data/churn_data.csv.
The rest of the project works with any CSV that has a Yes/No "Churn" column.
"""
import numpy as np
import pandas as pd

from config import DATA_PATH

rng = np.random.default_rng(42)
n = 3000

tenure = rng.integers(1, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
payment = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    n, p=[0.35, 0.20, 0.22, 0.23],
)
gender = rng.choice(["Male", "Female"], n)
senior = rng.choice([0, 1], n, p=[0.84, 0.16])
support_calls = rng.poisson(1.5, n)

base_price = {"DSL": 45, "Fiber optic": 80, "No": 20}
monthly = np.array([base_price[i] for i in internet]) + rng.normal(0, 8, n)
monthly = monthly.clip(18, 120).round(2)
total = (monthly * tenure).round(2)

# Churn probability: higher for new, month-to-month, high-bill, many-support-calls customers
logit = (
    -1.6
    + 1.3 * (contract == "Month-to-month")
    - 1.1 * (contract == "Two year")
    + 0.6 * (internet == "Fiber optic")
    + 0.015 * (monthly - 65)
    - 0.035 * tenure
    + 0.35 * support_calls
    + 0.4 * (payment == "Electronic check")
    + 0.3 * senior
)
prob = 1 / (1 + np.exp(-logit))
churn = np.where(rng.random(n) < prob, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(1, n + 1)],
    "gender": gender,
    "SeniorCitizen": senior,
    "tenure": tenure,
    "Contract": contract,
    "PaymentMethod": payment,
    "InternetService": internet,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "SupportCalls": support_calls,
    "Churn": churn,
})

# Add a few data-quality problems so preprocessing has something to fix
df.loc[rng.choice(n, 25, replace=False), "TotalCharges"] = np.nan
df = pd.concat([df, df.sample(20, random_state=1)], ignore_index=True)  # duplicates

DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_PATH, index=False)
print(f"Saved {len(df)} rows to {DATA_PATH}")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
