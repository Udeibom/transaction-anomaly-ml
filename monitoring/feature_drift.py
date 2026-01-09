import pandas as pd
from scipy.stats import ks_2samp

def feature_drift_ks(reference: pd.Series, current: pd.Series, alpha=0.05):
    stat, p_value = ks_2samp(reference, current)

    return {
        "ks_stat": stat,
        "p_value": p_value,
        "drift_detected": p_value < alpha
    }
