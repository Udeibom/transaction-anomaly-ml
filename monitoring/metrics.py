import numpy as np
import pandas as pd

def compute_score_metrics(scores: pd.Series):
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "p95_score": np.percentile(scores, 95),
        "p99_score": np.percentile(scores, 99),
    }

def compute_alert_rate(scores: pd.Series, threshold: float):
    return (scores >= threshold).mean()
