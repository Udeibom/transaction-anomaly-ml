from pathlib import Path
import pandas as pd
import numpy as np

from monitoring.metrics import compute_score_metrics, compute_alert_rate
from monitoring.drift import ks_drift
from monitoring.logger import log_metrics

# -----------------------------
# Resolve project root safely
# -----------------------------
BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

# -----------------------------
# Define paths
# -----------------------------
IFOREST_SCORES_PATH = BASE_DIR / "data" / "processed" / "iforest_scores.csv"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features.parquet"
REF_FEATURES_PATH = BASE_DIR / "data" / "processed" / "reference_features.parquet"

METRICS_LOG_PATH = "data/monitoring/metrics_log.csv"  # relative (logger resolves)

# -----------------------------
# Safety checks
# -----------------------------
assert IFOREST_SCORES_PATH.exists(), f"Missing file: {IFOREST_SCORES_PATH}"
assert FEATURES_PATH.exists(), f"Missing file: {FEATURES_PATH}"
assert REF_FEATURES_PATH.exists(), f"Missing file: {REF_FEATURES_PATH}"

# -----------------------------
# Load data
# -----------------------------
scores = pd.read_csv(IFOREST_SCORES_PATH)
features = pd.read_parquet(FEATURES_PATH)

# Reference window (e.g., first 7 days)
ref_features = pd.read_parquet(REF_FEATURES_PATH)

# -----------------------------
# Threshold (Day 11 strategy)
# -----------------------------
THRESHOLD = np.percentile(scores["anomaly_score"], 99)

metrics = {}

# -----------------------------
# Score distribution metrics
# -----------------------------
metrics.update(
    compute_score_metrics(scores["anomaly_score"])
)

# -----------------------------
# Alert rate
# -----------------------------
metrics["alert_rate"] = compute_alert_rate(
    scores["anomaly_score"],
    THRESHOLD
)

# -----------------------------
# Feature drift detection
# Example: transaction amount
# -----------------------------
drift_result = ks_drift(
    ref_features["log_amt"],
    features["log_amt"]
)

metrics["amount_ks_stat"] = drift_result["ks_stat"]
metrics["amount_drift"] = drift_result["drift_detected"]

# -----------------------------
# Log metrics
# -----------------------------
log_metrics(metrics, METRICS_LOG_PATH)
