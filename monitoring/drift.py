import pandas as pd
from scipy.stats import ks_2samp

def ks_drift(reference: pd.Series, current: pd.Series):
    stat, p_value = ks_2samp(reference, current)
    return {
        "ks_stat": stat,
        "p_value": p_value,
        "drift_detected": p_value < 0.05
    }
