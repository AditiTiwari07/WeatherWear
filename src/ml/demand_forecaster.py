import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from config.config import (
    MERGED_DATASET,
    MODELS_DIR,
    MODEL_FORECASTER_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)

FEATURES = [
    "avg_temperature",
    "avg_precipitation",
    "avg_windspeed",
    "avg_humidity",
    "month",
    "season",
    "city_encoded",
]

TARGET = "review_count"

def load_data():
    print("Loading merged dataset...")
    df = pd.read_parquet(MERGED_DATASET)
    print(f"Total rows: {len(df):,}")

    df = df.dropna(subset=[
        "avg_temperature", "avg_precipitation",
        "avg_windspeed", "avg_humidity",
        "month", "season", "review_count", "city"
    ])
    print(f"Rows after dropping nulls: {len(df):,}")

    le_city = LabelEncoder()
    df["city_encoded"] = le_city.fit_transform(df["city"])
    print(f"Cities encoded: {list(le_city.classes_)}")

    joblib.dump(le_city, MODELS_DIR / "city_encoder.pkl")
    print("City encoder saved.")

    return df

def train_forecaster(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train):,} rows")
    print(f"Test:  {len(X_test):,} rows")

    print("\nTraining XGBoost demand forecaster...")
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\nRMSE:  {rmse:.2f}")
    print(f"MAE:   {mae:.2f}")
    print(f"R²:    {r2:.4f}")

    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat}: {imp:.4f}")

    print("\nSample Predictions vs Actual:")
    results = pd.DataFrame({
        "actual":    y_test.values[:10],
        "predicted": y_pred[:10].round(0),
    })
    print(results.to_string(index=False))

    return model

def save_model(model):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FORECASTER_PATH)
    print(f"\nForecaster saved -> {MODEL_FORECASTER_PATH}")

def run():
    print("=" * 50)
    print("WEATHERWEAR — DEMAND FORECASTER")
    print("=" * 50)
    df = load_data()
    model = train_forecaster(df)
    save_model(model)
    print("\nPhase 5b complete.")

if __name__ == "__main__":
    run()