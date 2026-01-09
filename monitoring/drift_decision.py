def overall_drift_decision(
    score_drift: bool,
    feature_drifts: dict,
    alert_drift: bool,
    min_signals=2
):
    signals = [
        score_drift,
        alert_drift,
        any(feature_drifts.values())
    ]

    return sum(signals) >= min_signals
