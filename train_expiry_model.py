import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model(data_path, output_path):
    df = pd.read_csv(data_path)

    feature_cols = [
        "item_type",
        "item_name",
        "avg_temp_c",
        "humidity_pct",
        "storage_type",
        "location",
        "season",
    ]
    target_col = "expiry_days"

    categorical_cols = ["item_type", "item_name", "storage_type", "location", "season"]
    numeric_cols = ["avg_temp_c", "humidity_pct"]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    val_pred = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, val_pred)
    rmse = mean_squared_error(y_val, val_pred) ** 0.5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    print(f"Saved model to: {output_path}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="produce_expiry_dataset.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--output",
        default="models/expiry_model.joblib",
        help="Path to save trained model",
    )
    args = parser.parse_args()

    train_model(Path(args.data), Path(args.output))
