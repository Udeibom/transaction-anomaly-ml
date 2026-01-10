from pydantic import BaseModel
from typing import List

class TransactionFeatures(BaseModel):
    features: List[float]

class ScoreResponse(BaseModel):
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    model_version: str
