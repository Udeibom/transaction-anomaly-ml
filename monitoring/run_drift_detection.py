import pandas as pd
import numpy as np
from pathlib import Path

from monitoring.feature_drift import feature_drift_ks
from monitoring.alert_health import alert_rate_drift
from monitoring.drift_decision import overall_drift_decision

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

SCORES_PATH = DATA_DIR / "iforest_scores.csv"
FEATURES_PATH = DATA_DIR / "features.parquet"
REF_FEATURES_PATH = DATA_DIR / "reference_features.parquet"

for path in [SCORES_PATH, FEATURES_PATH, REF_FEATURES_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
scores = pd.read_csv(SCORES_PATH)
features = pd.read_parquet(FEATURES_PATH)
ref_features = pd.read_parquet(REF_FEATURES_PATH)

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
if "anomaly_score" not in scores.columns:
    raise ValueError("Missing 'anomaly_score' column in scores")

# ------------------------------------------------------------------
# Baselines (assumed stable)
# ------------------------------------------------------------------
BASE_MEAN = scores["anomaly_score"].mean()
BASE_STD = scores["anomaly_score"].std()
BASE_ALERT_RATE = 0.01  # operating point

# ------------------------------------------------------------------
# Current statistics
# ------------------------------------------------------------------
current_mean = scores["anomaly_score"].mean()
current_alert_rate = (
    scores["anomaly_score"]
    >= np.percentile(scores["anomaly_score"], 99)
).mean()

# ------------------------------------------------------------------
# Score drift detection
# ------------------------------------------------------------------
score_drift = abs(current_mean - BASE_MEAN) > 3 * BASE_STD

# ------------------------------------------------------------------
# Feature drift detection (SCHEMA SAFE)
# ------------------------------------------------------------------
feature_drifts = {}

# Only compare features that exist in BOTH datasets
common_features = sorted(
    set(ref_features.columns).intersection(features.columns)
)

if not common_features:
    print("⚠️ No common features between reference and current data")
else:
    for feature in common_features:
        try:
            ref_col = ref_features[feature].dropna()
            cur_col = features[feature].dropna()

            # Skip degenerate cases
            if ref_col.empty or cur_col.empty:
                print(f"⚠️ Skipping '{feature}' (empty data)")
                continue

            result = feature_drift_ks(ref_col, cur_col)
            feature_drifts[feature] = result["drift_detected"]

        except Exception as e:
            print(f"⚠️ Drift check failed for '{feature}': {e}")

# ------------------------------------------------------------------
# Alert-rate drift
# ------------------------------------------------------------------
alert_drift = alert_rate_drift(
    BASE_ALERT_RATE,
    current_alert_rate
)

# ------------------------------------------------------------------
# Final drift decision
# ------------------------------------------------------------------
drift_detected = overall_drift_decision(
    score_drift=score_drift,
    feature_drifts=feature_drifts,
    alert_drift=alert_drift
)

# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
print("Score drift:", score_drift)
print(" Feature drifts:", feature_drifts)
print(" Alert-rate drift:", alert_drift)
print(" DRIFT DETECTED:", drift_detected)
