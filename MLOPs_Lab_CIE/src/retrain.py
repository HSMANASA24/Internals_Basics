import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import json

# Load data
train = pd.read_csv("data/training_data.csv")
new = pd.read_csv("data/new_data.csv")

# Combine
combined = pd.concat([train, new], ignore_index=True)

X = combined.drop("processing_seconds", axis=1)
y = combined["processing_seconds"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Retrain model (same type as best: LinearRegression)
model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, preds)

# Champion MAE from Task 1 (use your exact value)
champion_mae = 0.35523679545340303

improvement = champion_mae - retrained_mae

action = "promoted" if improvement >= 1.0 else "kept_champion"

output = {
    "original_data_rows": len(train),
    "new_data_rows": len(new),
    "combined_data_rows": len(combined),
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": 1.0,
    "action": action,
    "comparison_metric": "mae"
}

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Step 4 completed!")