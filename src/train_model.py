
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


FEATURES = [
    "hour",
    "failed_attempts",
    "new_device",
    "new_country",
    "distance_km",
    "login_success",
]


def main():
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "auth_events.csv"
    model_dir = project_root / "models"
    model_path = model_dir / "isolation_forest.joblib"

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Run src/generate_data.py first."
        )

    data = pd.read_csv(data_path)

    X = data[FEATURES]
    y = data["is_anomaly"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )

    # Model yalnızca giriş özelliklerini kullanarak öğrenir.
    model.fit(X_train)

    # Isolation Forest: -1 = anomaly, 1 = normal
    raw_predictions = model.predict(X_test)
    predictions = (raw_predictions == -1).astype(int)

    print("=== Model Evaluation ===")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES},
        model_path,
    )

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()