import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json
import os

# Load dataset
df = pd.read_csv("data/training_data.csv")

X = df.drop("processing_seconds", axis=1)
y = df["processing_seconds"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid
param_grid = {
    "n_estimators": [50, 150, 250],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 3, 5]
}

mlflow.set_experiment("payflow-processing-seconds")

# Start parent run
with mlflow.start_run(run_name="tuning-payflow"):

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # Create output
    output = {
        "search_type": "grid",
        "n_folds": 3,
        "total_trials": len(grid.cv_results_["params"]),
        "best_params": grid.best_params_,
        "best_mae": mae,
        "best_cv_mae": -grid.best_score_,
        "parent_run_name": "tuning-payflow"
    }

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Save JSON
with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Step 2 completed and file saved!")