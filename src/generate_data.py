
import random
from pathlib import Path

import pandas as pd


def generate_login_data(number_of_records=1000):
    random.seed(42)
    records = []

    for _ in range(number_of_records):
        # Normal giriş davranışları
        hour = random.choices(
            range(24),
            weights=[
                1, 1, 1, 1, 1, 2,
                4, 7, 9, 9, 8, 8,
                8, 8, 8, 9, 9, 8,
                7, 5, 4, 3, 2, 1
            ]
        )[0]

        failed_attempts = random.choices(
            [0, 1, 2, 3, 5, 8, 12],
            weights=[65, 18, 8, 4, 3, 1, 1]
        )[0]

        new_device = random.choices(
            [0, 1], weights=[92, 8]
        )[0]

        new_country = random.choices(
            [0, 1], weights=[98, 2]
        )[0]

        distance_km = random.expovariate(1 / 30)
        login_success = random.choices(
            [0, 1], weights=[10, 90]
        )[0]

        # Bazı kayıtlara kontrollü anomaliler ekle
        is_anomaly = 0

        if random.random() < 0.05:
            hour = random.choice([0, 1, 2, 3, 4])
            failed_attempts = random.randint(6, 15)
            new_device = 1
            new_country = random.choice([0, 1])
            distance_km = random.uniform(500, 5000)
            login_success = random.choice([0, 1])
            is_anomaly = 1

        records.append({
            "hour": hour,
            "failed_attempts": failed_attempts,
            "new_device": new_device,
            "new_country": new_country,
            "distance_km": round(distance_km, 2),
            "login_success": login_success,
            "is_anomaly": is_anomaly
        })

    return pd.DataFrame(records)


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "auth_events.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_login_data()
    data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Total records: {len(data)}")
    print(f"Known anomalies: {data['is_anomaly'].sum()}")
    print("\nFirst five records:")
    print(data.head())


if __name__ == "__main__":
    main()