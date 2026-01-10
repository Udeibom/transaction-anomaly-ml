from fastapi import FastAPI
import numpy as np

from api.schema import TransactionFeatures, ScoreResponse
from api.model_loader import score, MODEL_VERSION

# Static threshold from Day 11
ANOMALY_THRESHOLD = 3.5  # example value; replace with saved percentile

app = FastAPI(
    title="Transaction Anomaly Detection API",
    version="1.0"
)

from api.observability import (
    log_request,
    log_prediction,
    log_alert
)

@app.post("/score_transaction", response_model=ScoreResponse)
def score_transaction(payload: TransactionFeatures):
    # Convert payload to numpy
    features = np.array(payload.features)

    # Log request metadata
    request_id = log_request(
        features=features.tolist(),
        model_version=MODEL_VERSION
    )

    # Score transaction
    anomaly_score = score(features.reshape(1, -1))
    is_anomaly = anomaly_score >= ANOMALY_THRESHOLD

    # Log prediction
    log_prediction(
        request_id=request_id,
        score=float(anomaly_score),
        threshold=ANOMALY_THRESHOLD,
        is_anomaly=bool(is_anomaly)
    )

    # Log alert only if anomalous
    if is_anomaly:
        log_alert(
            request_id=request_id,
            score=float(anomaly_score)
        )

    return ScoreResponse(
        anomaly_score=float(anomaly_score),
        is_anomaly=bool(is_anomaly),
        threshold=ANOMALY_THRESHOLD,
        model_version=MODEL_VERSION
    )
