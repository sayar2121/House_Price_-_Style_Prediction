# train_all_models.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Import the new function for RMSE and the other metrics
from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error

print("🚀 Starting combined model training script...")

# --- 1. Load Dataset ---
DATA_PATH = "data/train.csv"
try:
    data = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully from 'data/train.csv'.")
except FileNotFoundError:
    print(f"❌ Error: '{DATA_PATH}' not found. Make sure 'train.csv' is in the 'data' folder.")
    exit()

# --- 2. Feature Selection & Preprocessing ---
features = [
    "LotArea", "OverallQual", "YearBuilt", "GrLivArea",
    "FullBath", "BedroomAbvGr", "GarageCars"
]
# Handle potential missing values that could affect both models
if 'GarageCars' in features:
    data['GarageCars'] = data['GarageCars'].fillna(0)

X = data[features]
print(f"✅ Features selected for both models: {features}")

# --- 3. Price Prediction Model (Regressor) ---
print("\n--- Training Price Prediction Model ---")
price_target = "SalePrice"
y_price = data[price_target]

X_train, X_test, y_price_train, y_price_test = train_test_split(
    X, y_price, test_size=0.2, random_state=42
)

print("⏳ Training RandomForestRegressor...")
price_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
price_model.fit(X_train, y_price_train)
print("✅ Price model training complete.")

# Evaluate the price model
y_price_pred = price_model.predict(X_test)
r2 = r2_score(y_price_test, y_price_pred)
# FIX: Use the new root_mean_squared_error function
rmse = root_mean_squared_error(y_price_test, y_price_pred)
print("📊 Price Model Performance:")
print(f"   - R² Score: {r2:.4f}")
print(f"   - RMSE: ${rmse:,.2f}")

# --- 4. House Style Prediction Model (Classifier) ---
print("\n--- Training House Style Prediction Model ---")
style_target = "HouseStyle"
y_style = data[style_target]

# We can reuse the same split for the features (X)
_, _, y_style_train, y_style_test = train_test_split(
    X, y_style, test_size=0.2, random_state=42
)

print("⏳ Training RandomForestClassifier...")
style_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
style_model.fit(X_train, y_style_train)
print("✅ Style model training complete.")

# Evaluate the style model
y_style_pred = style_model.predict(X_test)
accuracy = accuracy_score(y_style_test, y_style_pred)
print("📊 Style Model Performance:")
print(f"   - Accuracy: {accuracy:.2%}")

# --- 5. Save Both Models ---
output_dir = "app"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

price_model_path = os.path.join(output_dir, "house_price_model.pkl")
joblib.dump(price_model, price_model_path)
print(f"\n✅ Price model saved to: {price_model_path}")

style_model_path = os.path.join(output_dir, "house_style_model.pkl")
joblib.dump(style_model, style_model_path)
print(f"✅ Style model saved to: {style_model_path}")
