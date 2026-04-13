"""REST routes for tickers, predictions, sentiment and training."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.api.errors import DomainError, NotFoundError
from app.api.schemas import (
    HistoryResponse,
    ModelRunItem,
    ModelRunsResponse,
    MultiHorizonResponse,
    PredictionHistoryItem,
    ReconcileResponse,
    SentimentScore,
    TickerInfo,
)
from app.domain import InvalidTickerError, TickerSymbol
from app.infra.dependencies import (
    model_runs_repo,
    prediction_repo,
    prediction_service,
    reconciliation_service,
    sentiment_service,
)
from app.repositories.model_runs_repository import ModelRunsRepo
from app.repositories.predictions_repository import PredictionsRepo
from app.services.prediction_service import (
    PredictionService,
    ReconciliationService,
)
from app.services.sentiment_service import SentimentService

router = APIRouter()


def _validate_ticker(ticker: str) -> str:
    """Normaliza e valida via ``TickerSymbol`` do dominio, traduzindo erro para 400."""
    try:
        return str(TickerSymbol.from_raw(ticker))
    except InvalidTickerError as exc:
        raise DomainError("invalid ticker", status_code=400) from exc


@router.get("/tickers", response_model=list[TickerInfo], tags=["tickers"])
async def list_tickers() -> list[TickerInfo]:
    return [
        TickerInfo(ticker="PETR4", name="Petrobras PN"),
        TickerInfo(ticker="VALE3", name="Vale ON"),
        TickerInfo(ticker="ITUB4", name="Itaú Unibanco PN"),
    ]


@router.get(
    "/predict/{ticker}",
    response_model=MultiHorizonResponse,
    tags=["predictions"],
)
async def predict(
    request: Request,
    ticker: str,
    days: int = Query(default=5, ge=1, le=30),
    service: PredictionService = Depends(prediction_service),
) -> MultiHorizonResponse:
    ticker = _validate_ticker(ticker)
    try:
        response = await service.predict_with_horizons(ticker)
    except ValueError as exc:
        raise NotFoundError(str(exc)) from exc
    # Respect requested ``days`` for the inline preview, keeping the contract.
    response.predictions = response.predictions[: max(1, min(days, len(response.predictions)))]
    return response


@router.get(
    "/predictions/{ticker}/history",
    response_model=HistoryResponse,
    tags=["predictions"],
)
async def prediction_history(
    ticker: str,
    limit: int = Query(default=60, ge=1, le=500),
    repo: PredictionsRepo = Depends(prediction_repo),
) -> HistoryResponse:
    ticker = _validate_ticker(ticker)
    rows = await repo.list_history(ticker, limit=limit)
    items = [
        PredictionHistoryItem(
            id=row.id,
            ticker=row.ticker_symbol,
            horizon_days=row.horizon_days,
            created_at=row.created_at,
            target_date=row.target_date,
            base_close=row.base_close,
            predicted_close=row.predicted_close,
            predicted_pct=row.predicted_pct,
            actual_close=row.actual_close,
            error_pct=row.error_pct,
            resolved=row.actual_close is not None,
            explanation=row.explanation,
        )
        for row in rows
    ]
    return HistoryResponse(ticker=ticker, items=items)


ALLOWED_HORIZONS: frozenset[int] = frozenset({1, 7, 30})


@router.get(
    "/predictions/{ticker}/horizon/{horizon_days}/explanation",
    tags=["predictions"],
)
async def prediction_explanation(
    ticker: str,
    horizon_days: int,
    service: PredictionService = Depends(prediction_service),
) -> dict[str, object]:
    ticker = _validate_ticker(ticker)
    if horizon_days not in ALLOWED_HORIZONS:
        raise DomainError("invalid horizon_days", status_code=400)
    text = await service.get_explanation(ticker, horizon_days)
    if not text:
        raise NotFoundError(
            f"no explanation stored for {ticker}/{horizon_days}"
        )
    return {
        "ticker": ticker,
        "horizon_days": horizon_days,
        "explanation": text,
    }


@router.post(
    "/predictions/reconcile",
    response_model=ReconcileResponse,
    tags=["predictions"],
)
async def reconcile(
    service: ReconciliationService = Depends(reconciliation_service),
) -> ReconcileResponse:
    checked, resolved = await service.reconcile()
    return ReconcileResponse(checked=checked, resolved=resolved)


@router.get("/sentiment/{ticker}", response_model=SentimentScore, tags=["sentiment"])
async def sentiment(
    ticker: str,
    service: SentimentService = Depends(sentiment_service),
) -> SentimentScore:
    ticker = _validate_ticker(ticker)
    return await service.aggregate(ticker)


@router.get(
    "/models/{ticker}/runs",
    response_model=ModelRunsResponse,
    tags=["training"],
)
async def model_runs(
    ticker: str,
    limit: int = Query(default=20, ge=1, le=100),
    repo: ModelRunsRepo = Depends(model_runs_repo),
) -> ModelRunsResponse:
    ticker = _validate_ticker(ticker)
    rows = await repo.list_recent(ticker, limit=limit)
    items = [
        ModelRunItem(
            id=row.id,
            ticker=row.ticker_symbol,
            status=row.status,
            epochs=row.epochs,
            loss=row.loss,
            direction_accuracy=row.direction_accuracy,
            artifact_path=row.artifact_path,
            created_at=row.created_at,
            finished_at=row.finished_at,
        )
        for row in rows
    ]
    return ModelRunsResponse(ticker=ticker, items=items)


@router.post("/train/{ticker}", tags=["training"])
async def train(ticker: str) -> dict[str, str]:
    ticker = _validate_ticker(ticker)
    from app.infra.tasks import train_model

    try:
        train_model.delay(ticker)  # type: ignore[attr-defined]
        return {"status": "queued", "ticker": ticker}
    except Exception:  # noqa: BLE001 - broker unavailable in some dev setups
        return {"status": "queue_unavailable", "ticker": ticker}
