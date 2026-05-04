import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
import os
import joblib

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/training_data.csv")

X = df.drop("processing_seconds", axis=1)
y = df["processing_seconds"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. MLflow Setup
# -----------------------------
mlflow.set_experiment("payflow-processing-seconds")

results = []
trained_models = {}

# -----------------------------
# 3. Train Models
# -----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log parameters
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Set tag
        mlflow.set_tag("team", "ml_engineering")

        # Log model
        mlflow.sklearn.log_model(model, name)

        # Save results
        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        trained_models[name] = model

# -----------------------------
# 4. Select Best Model
# -----------------------------
best = min(results, key=lambda x: x["mae"])
best_model_name = best["name"]
best_model = trained_models[best_model_name]

# -----------------------------
# 5. Save Best Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")

# -----------------------------
# 6. Save JSON Output
# -----------------------------
output = {
    "experiment_name": "payflow-processing-seconds",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

# -----------------------------
# 7. Done
# -----------------------------
print("✅ Step 1 completed!")
print(f"🏆 Best Model: {best_model_name}")
print(f"📉 MAE: {best['mae']}")
print("💾 Model saved at models/model.pkl")