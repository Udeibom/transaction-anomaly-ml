import uuid
from api.logging_utils import log_event

def log_request(features, model_version: str) -> str:
    """
    Log incoming request metadata.
    Returns a request_id for correlation.
    """
    request_id = str(uuid.uuid4())

    log_event(
        "requests.jsonl",
        {
            "request_id": request_id,
            "model_version": model_version,
            "num_features": len(features)
        }
    )

    return request_id


def log_prediction(request_id: str, score: float, threshold: float, is_anomaly: bool):
    """
    Log model prediction output.
    """
    log_event(
        "predictions.jsonl",
        {
            "request_id": request_id,
            "anomaly_score": score,
            "threshold": threshold,
            "is_anomaly": is_anomaly
        }
    )


def log_alert(request_id: str, score: float):
    """
    Log only anomalous transactions.
    """
    log_event(
        "alerts.jsonl",
        {
            "request_id": request_id,
            "anomaly_score": score
        }
    )
