from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class TickerInfo(BaseModel):
    ticker: str
    name: str | None = None
    currency: str = "BRL"


class PricePoint(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class SentimentScore(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    positives: int = 0
    negatives: int = 0
    neutrals: int = 0


class PredictionPoint(BaseModel):
    date: date
    predicted_close: float
    direction: str  # ALTA | BAIXA | NEUTRO


class PredictionResponse(BaseModel):
    ticker: str
    last_price: float
    predictions: list[PredictionPoint]
    sentiment: SentimentScore | None = None
    direction_accuracy: float | None = None


class HorizonPrediction(BaseModel):
    horizon: Literal["D1", "W1", "M1"]
    horizon_days: int
    target_date: date
    base_close: float
    predicted_close: float
    predicted_pct: float
    direction: str  # ALTA | BAIXA | NEUTRO
    explanation: str | None = None


class MultiHorizonResponse(BaseModel):
    ticker: str
    last_price: float
    horizons: list[HorizonPrediction]
    sentiment: SentimentScore | None = None
    predictions: list[PredictionPoint] = Field(default_factory=list)
    direction_accuracy: float | None = None


class PredictionHistoryItem(BaseModel):
    id: int
    ticker: str
    horizon_days: int
    created_at: datetime
    target_date: date
    base_close: float
    predicted_close: float
    predicted_pct: float
    actual_close: float | None = None
    error_pct: float | None = None
    resolved: bool = False
    explanation: str | None = None


class HistoryResponse(BaseModel):
    ticker: str
    items: list[PredictionHistoryItem]


class ReconcileResponse(BaseModel):
    checked: int
    resolved: int


class TrainingRequest(BaseModel):
    ticker: str
    epochs: int = Field(default=50, ge=1, le=500)
    sequence_length: int = Field(default=5, ge=3, le=60)


class ModelRunItem(BaseModel):
    id: int
    ticker: str
    status: str
    epochs: int | None = None
    loss: float | None = None
    direction_accuracy: float | None = None
    artifact_path: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class ModelRunsResponse(BaseModel):
    ticker: str
    items: list[ModelRunItem]
