import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/isolation_forest_v1.joblib"

# Load bundle once at startup
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]

MODEL_VERSION = "isolation_forest_v1"

def score(features: np.ndarray) -> float:
    """
    Returns anomaly score (higher = more anomalous)
    """
    # Convert incoming array → DataFrame with correct columns
    X = pd.DataFrame(features, columns=FEATURE_NAMES)

    return -model.score_samples(X)[0]

print("Expected features:", FEATURE_NAMES)
