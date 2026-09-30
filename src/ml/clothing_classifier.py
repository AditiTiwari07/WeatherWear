import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from config.config import (
    MERGED_DATASET,
    MODELS_DIR,
    MODEL_CLASSIFIER_PATH,
    LABEL_ENCODER_PATH,
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
    "avg_rating",
    "review_count",
    "review_demand_score",
]

TARGET = "clothing_category"

def load_data():
    print("Loading merged dataset...")
    df = pd.read_parquet(MERGED_DATASET)
    print(f"Total rows: {len(df):,}")

    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"Rows after dropping nulls: {len(df):,}")
    return df

def train_classifier(df):
    X = df[FEATURES]
    y = df[TARGET]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {list(le.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
    print(f"Train: {len(X_train):,} rows")
    print(f"Test:  {len(X_test):,} rows")

    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAccuracy:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Score:       {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nFeature Importances:")
    for feat, imp in sorted(
        zip(FEATURES, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    ):
        print(f"  {feat}: {imp:.4f}")

    return clf, le

def save_models(clf, le):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_CLASSIFIER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"\nClassifier saved -> {MODEL_CLASSIFIER_PATH}")
    print(f"Label encoder saved -> {LABEL_ENCODER_PATH}")

def run():
    print("=" * 50)
    print("WEATHERWEAR — CLOTHING CLASSIFIER")
    print("=" * 50)
    df = load_data()
    clf, le = train_classifier(df)
    save_models(clf, le)
    print("\nPhase 5a complete.")

if __name__ == "__main__":
    run()