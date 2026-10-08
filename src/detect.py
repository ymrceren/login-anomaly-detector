
from pathlib import Path

import joblib
import pandas as pd

def predict_event(model, features, event_values):
    event = pd.DataFrame(
        [event_values],
        columns=features,
    )

    prediction = model.predict(event)[0]
    score = model.decision_function(event)[0]

    return prediction, score

def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "isolation_forest.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model not found. Run src/train_model.py first."
        )

    saved = joblib.load(model_path)
    model = saved["model"]
    features = saved["features"]

    print("Enter login event details.")

    try:
        hour = int(input("Login hour (0-23): "))
        failed_attempts = int(input("Failed attempts: "))
        new_device = int(input("New device? (0/1): "))
        new_country = int(input("New country? (0/1): "))
        distance_km = float(input("Distance from usual location (km): "))
        login_success = int(input("Login successful? (0/1): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    if not 0 <= hour <= 23:
        print("Hour must be between 0 and 23.")
        return

    if failed_attempts < 0 or distance_km < 0:
        print("Attempts and distance cannot be negative.")
        return

    if new_device not in (0, 1) or new_country not in (0, 1):
        print("Device and country values must be 0 or 1.")
        return

    if login_success not in (0, 1):
        print("Login success must be 0 or 1.")
        return

    
    event_values = [
        hour,
        failed_attempts,
        new_device,
        new_country,
        distance_km,
        login_success,
    ]

    prediction, score = predict_event(
        model,
        features,
        event_values,
    )

    print("\n=== Detection Result ===")

    if prediction == -1:
        print("Anomaly candidate: review this login event.")
    else:
        print("No anomaly detected by the model.")

    print(f"Anomaly score: {score:.4f}")
    print("This result is an indicator, not proof of an attack.")


if __name__ == "__main__":
    main()