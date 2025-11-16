# laptop_price_predictor.py

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("laptop_data.csv")
target = "Price"
df = df.dropna(subset=[target])  # remove rows without price


# ---------------------------
# 2. Feature Engineering
# ---------------------------
def extract_number(s):
    if pd.isna(s): return np.nan
    s = str(s).replace(",", "")
    match = re.search(r"(\d+(\.\d+)?)", s)
    return float(match.group(1)) if match else np.nan

df["ram_gb"] = df["Ram"].apply(extract_number)
df["storage_gb"] = df["Memory"].apply(extract_number)
df.loc[df["Memory"].str.contains("TB", case=False, na=False), "storage_gb"] *= 1000
df["screen_inch"] = df["Inches"].apply(extract_number)
df["weight_kg"] = df["Weight"].apply(extract_number)
df.loc[df["weight_kg"] > 100, "weight_kg"] /= 1000  # grams → kg

features = ["ram_gb", "storage_gb", "screen_inch", "weight_kg", "Company", "Cpu", "Gpu", "OpSys"]
X, y = df[features], df[target]


# ---------------------------
# 3. Preprocessing & Model
# ---------------------------
numeric = ["ram_gb", "storage_gb", "screen_inch", "weight_kg"]
categorical = ["Company", "Cpu", "Gpu", "OpSys"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric),
    ("cat", categorical_transformer, categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])


# ---------------------------
# 4. Train/Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}, R²: {r2_score(y_test, preds):.4f}")


# ---------------------------
# 5. Save Model
# ---------------------------
joblib.dump(model, "laptop_price_model.joblib")
print("✅ Model saved as laptop_price_model.joblib")


# ---------------------------
# 6. Predict from User Input8


# ---------------------------
def predict_from_input():
    print("\n🔹 Enter Laptop Specs 🔹")
    ram = float(input("RAM (GB): "))
    storage = float(input("Storage (GB): "))
    screen = float(input("Screen size (inches): "))
    weight = float(input("Weight (kg): "))
    company = input("Company (e.g., Dell, HP, Lenovo): ")
    cpu = input("CPU (e.g., Intel Core i5, AMD Ryzen 5): ")
    gpu = input("GPU (e.g., Nvidia GTX 1650, AMD Radeon): ")
    opsys = input("Operating System (e.g., Windows 10, macOS): ")

    user_df = pd.DataFrame([{
        "ram_gb": ram,
        "storage_gb": storage,
        "screen_inch": screen,
        "weight_kg": weight,
        "Company": company,
        "Cpu": cpu,
        "Gpu": gpu,
        "OpSys": opsys
    }])

    price = model.predict(user_df)[0]
    print(f"\n💻 Predicted Laptop Price: ₹ {price:,.2f}")


# Run prediction
predict_from_input()

