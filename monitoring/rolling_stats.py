import numpy as np
import pandas as pd

def rolling_score_stats(scores: pd.Series, window: int):
    return scores.rolling(window).agg(
        mean="mean",
        std="std",
        p95=lambda x: np.percentile(x, 95),
        p99=lambda x: np.percentile(x, 99),
    )

def score_drift_detected(
    baseline_mean,
    baseline_std,
    current_mean,
    n_std=3
):
    return abs(current_mean - baseline_mean) > n_std * baseline_std

