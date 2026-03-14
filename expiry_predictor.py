from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "models" / "expiry_model.joblib"


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Expiry model not found at {MODEL_PATH}. Train it first with train_expiry_model.py"
        )
    return joblib.load(MODEL_PATH)


def predict_expiry_days(
    item_name,
    item_type,
    avg_temp_c,
    humidity_pct,
    storage_type,
    location,
    season,
):
    model = _load_model()
    row = pd.DataFrame(
        [
            {
                "item_type": item_type,
                "item_name": item_name,
                "avg_temp_c": float(avg_temp_c),
                "humidity_pct": float(humidity_pct),
                "storage_type": storage_type,
                "location": location,
                "season": season,
            }
        ]
    )
    pred = float(model.predict(row)[0])
    return max(1, int(round(pred)))
