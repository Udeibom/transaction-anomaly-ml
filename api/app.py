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

@app.post("/score_transaction", response_model=ScoreResponse)
def score_transaction(payload: TransactionFeatures):
    features = np.array(payload.features).reshape(1, -1)

    anomaly_score = score(features)
    is_anomaly = anomaly_score >= ANOMALY_THRESHOLD

    return ScoreResponse(
        anomaly_score=float(anomaly_score),
        is_anomaly=bool(is_anomaly),
        threshold=ANOMALY_THRESHOLD,
        model_version=MODEL_VERSION
    )
