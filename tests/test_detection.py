
import math

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.detect import predict_event


FEATURES = [
    "hour",
    "failed_attempts",
    "new_device",
    "new_country",
    "distance_km",
    "login_success",
]


def build_test_model():
    # Test için örnek ve normal davranışlara benzeyen veriler
    data = pd.DataFrame(
        [
            [8, 0, 0, 0, 5, 1],
            [9, 1, 0, 0, 10, 1],
            [10, 0, 0, 0, 3, 1],
            [14, 1, 0, 0, 20, 1],
            [16, 0, 0, 0, 8, 1],
            [18, 2, 1, 0, 30, 1],
            [11, 0, 0, 0, 6, 1],
            [13, 1, 0, 0, 12, 1],
        ],
        columns=FEATURES,
    )

    model = IsolationForest(
        n_estimators=50,
        contamination=0.1,
        random_state=42,
    )

    model.fit(data)
    return model


def test_prediction_returns_valid_label_and_score():
    model = build_test_model()

    prediction, score = predict_event(
        model,
        FEATURES,
        [3, 10, 1, 1, 2000, 1],
    )

    assert prediction in (-1, 1)
    assert math.isfinite(float(score))


def test_prediction_score_is_numeric():
    model = build_test_model()

    prediction, score = predict_event(
        model,
        FEATURES,
        [9, 0, 0, 0, 5, 1],
    )

    assert isinstance(float(score), float)
    assert prediction in (-1, 1)