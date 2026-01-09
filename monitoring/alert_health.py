def alert_rate_drift(
    baseline_rate,
    current_rate,
    tolerance=0.002
):
    """
    tolerance = acceptable absolute deviation (e.g. ±0.2%)
    """
    return abs(current_rate - baseline_rate) > tolerance
