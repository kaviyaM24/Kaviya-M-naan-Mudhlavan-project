import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("house_prices.csv")  # Your dataset here

# Preprocess
df = df.dropna()
X = df.drop("price", axis=1)  # Features
y = df["price"]  # Target

# One-hot encode categorical features
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")

# Save model and columns
joblib.dump(model, "model.joblib")
joblib.dump(X.columns.tolist(), "columns.joblib")￼Enter
