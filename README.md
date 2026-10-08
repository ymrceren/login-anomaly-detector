
# ML-Based Login Anomaly Detection

## Overview

This project detects potentially unusual login events using
machine learning. It uses the Isolation Forest algorithm to
identify login activity that differs from typical patterns.

The project uses synthetic login data and is intended for
educational purposes.

## Features

- Generates synthetic login event data
- Analyzes login behavior using machine learning
- Examines login hour, failed attempts, device changes,
  country changes, location distance, and login status
- Predicts whether a login event may be anomalous
- Reports model evaluation metrics
- Includes automated tests using pytest

## Technologies

- Python
- pandas
- scikit-learn
- joblib
- pytest

## Project Structure

```text
login-anomaly-detector/
├── data/
│   └── auth_events.csv
├── models/
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   └── detect.py
├── tests/
│   └── test_detection.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository and open the project directory.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset

```bash
python src/generate_data.py
```

### 2. Train and evaluate the model

```bash
python src/train_model.py
```

### 3. Analyze a login event

```bash
python src/detect.py
```

Enter the requested login event details when prompted.

### 4. Run automated tests

```bash
python -m pytest -v
```

## How It Works

The Isolation Forest algorithm is trained using login-related
features to identify unusual patterns.

The `is_anomaly` column in the synthetic dataset is used for
evaluation only and is not included among the model features.

## Limitations

- The dataset is synthetic and does not represent real users.
- An anomaly is not proof of a cyberattack.
- False positives and missed anomalies are possible.
- Real-world use would require validation on suitable data,
  privacy safeguards, and additional security controls.

## Disclaimer

This project is for educational and portfolio purposes only.
It does not access real user accounts or authenticate users.