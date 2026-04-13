"""Prediction service: orchestrates data fetch + feature build + LSTM predict.

Dependencies (price repo, model loader, sentiment service, predictions repo)
are injected via the constructor. When ``MODEL_FALLBACK_STUB=true`` and no
trained artifact is found for a ticker, we fall back to a deterministic
pseudo-prediction so the stack still works for a `docker compose up` demo
without training.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from app.api.schemas import (
    HorizonPrediction,
    MultiHorizonResponse,
    PredictionPoint,
    PredictionResponse,
    SentimentScore,
)
from app.ml.explanation import (
    ExplanationGenerator,
    ExplanationInput,
)
from app.repositories.predictions_repository import (
    PredictionsRepo,
    PredictionUpsert,
)

NEUTRAL_THRESHOLD_PCT: float = 0.5
HORIZON_DEFINITIONS: tuple[tuple[str, int], ...] = (
    ("D1", 1),
    ("W1", 7),
    ("M1", 30),
)
HORIZON_LABELS: dict[int, str] = {1: "Amanha", 7: "+7 dias", 30: "+30 dias"}
ROLLOUT_STEPS: int = 30
MIN_HISTORY_DAYS: int = 120


class _PricesRepo(Protocol):
    async def get_history(self, ticker: str) -> list[dict[str, Any]]: ...


class _OhlcvRepo(Protocol):
    async def ensure_ticker(self, symbol: str, name: str | None = None) -> int: ...

    async def upsert_many(
        self, ticker_id: int, rows: Any
    ) -> int: ...

    async def count(self, ticker_id: int) -> int: ...

    async def list_history(
        self, ticker_id: int, limit: int = 1000
    ) -> list[dict[str, Any]]: ...


class _SentimentAggregator(Protocol):
    async def aggregate(self, ticker: str) -> SentimentScore: ...


class PredictionService:
    """Produce prediction responses and persist them for reconciliation."""

    def __init__(
        self,
        *,
        prices_repo: _PricesRepo | None = None,
        sentiment_service: _SentimentAggregator | None = None,
        predictions_repo: PredictionsRepo | None = None,
        ohlcv_repo: _OhlcvRepo | None = None,
        explanation_generator: ExplanationGenerator | None = None,
        models_dir: str | Path = "/app/models",
        fallback_stub: bool = True,
    ) -> None:
        self._prices_repo = prices_repo
        self._sentiment = sentiment_service
        self._predictions_repo = predictions_repo
        self._ohlcv_repo = ohlcv_repo
        self._explainer = explanation_generator
        self._models_dir = Path(models_dir)
        self._fallback_stub = fallback_stub

    async def predict(self, ticker: str, days: int = 5) -> PredictionResponse:
        """Backwards-compatible entry: returns first ``days`` rollout points."""
        multi = await self.predict_with_horizons(ticker)
        points = multi.predictions[:days]
        return PredictionResponse(
            ticker=multi.ticker,
            last_price=multi.last_price,
            predictions=points,
            sentiment=multi.sentiment,
            direction_accuracy=multi.direction_accuracy,
        )

    async def predict_with_horizons(self, ticker: str) -> MultiHorizonResponse:
        ticker = ticker.upper().strip()
        rollout, last_price = await self._compute_rollout(ticker)
        today = date.today()

        sentiment = await self._safe_sentiment(ticker)
        horizons = self._build_horizons(
            last_price, rollout, today, ticker=ticker, sentiment=sentiment
        )
        preview = self._build_preview_points(rollout, today, limit=5)

        await self._persist_horizons(ticker, horizons)

        return MultiHorizonResponse(
            ticker=ticker,
            last_price=round(last_price, 2),
            horizons=horizons,
            predictions=preview,
            sentiment=sentiment,
            direction_accuracy=None,
        )

    # --- internals -----------------------------------------------------

    async def _compute_rollout(self, ticker: str) -> tuple[list[float], float]:
        if self._prices_repo is None:
            return self._stub_rollout(ticker)

        history = await self._load_history_db_first(ticker)

        model_path = self._models_dir / f"{ticker}.keras"
        if not history or not model_path.exists():
            if self._fallback_stub:
                return self._stub_rollout(ticker, history=history)
            raise ValueError(f"no data or trained model for {ticker}")

        return await self._real_rollout(history, model_path)

    async def _load_history_db_first(self, ticker: str) -> list[dict[str, Any]]:
        """Try the local OHLCV table first; pull from BrAPI when scarce.

        Keeps the previous behaviour as a fallback chain, so demos with
        no DB access still work via the live BrAPI -> Yahoo path.
        """
        history: list[dict[str, Any]] = []

        if self._ohlcv_repo is not None:
            ticker_id = await self._safe_ensure_ticker(ticker)
            if ticker_id is not None:
                history = await self._safe_db_history(ticker_id)
                if len(history) >= MIN_HISTORY_DAYS:
                    return history

        live: list[dict[str, Any]] = []
        if self._prices_repo is not None:
            try:
                live = await self._prices_repo.get_history(ticker)
            except Exception:  # noqa: BLE001
                live = []

        if live and self._ohlcv_repo is not None:
            await self._safe_persist(ticker, live)
            ticker_id = await self._safe_ensure_ticker(ticker)
            if ticker_id is not None:
                refreshed = await self._safe_db_history(ticker_id)
                if refreshed:
                    return refreshed

        return live or history

    async def _safe_ensure_ticker(self, ticker: str) -> int | None:
        if self._ohlcv_repo is None:
            return None
        try:
            return await self._ohlcv_repo.ensure_ticker(ticker)
        except Exception:  # noqa: BLE001
            return None

    async def _safe_db_history(self, ticker_id: int) -> list[dict[str, Any]]:
        if self._ohlcv_repo is None:
            return []
        try:
            return await self._ohlcv_repo.list_history(ticker_id, limit=2000)
        except Exception:  # noqa: BLE001
            return []

    async def _safe_persist(
        self, ticker: str, rows: list[dict[str, Any]]
    ) -> None:
        if self._ohlcv_repo is None:
            return
        try:
            ticker_id = await self._ohlcv_repo.ensure_ticker(ticker)
            await self._ohlcv_repo.upsert_many(ticker_id, rows)
        except Exception:  # noqa: BLE001
            return

    async def _real_rollout(
        self, history: list[dict[str, Any]], model_path: Path
    ) -> tuple[list[float], float]:
        import pandas as pd

        from app.ml.lstm_model import LSTMPricePredictor

        df = pd.DataFrame(history)
        predictor = LSTMPricePredictor()
        predictor.load(model_path)
        preds = predictor.predict(df, days=ROLLOUT_STEPS)
        last_price = float(df["close"].iloc[-1])
        return [float(v) for v in preds], last_price

    def _stub_rollout(
        self, ticker: str, history: list[dict[str, Any]] | None = None
    ) -> tuple[list[float], float]:
        rng = random.Random(ticker)
        if history:
            try:
                last_price = float(history[-1]["close"])
            except (KeyError, TypeError, ValueError):
                last_price = round(25 + rng.random() * 25, 2)
        else:
            last_price = round(25 + rng.random() * 25, 2)

        rollout: list[float] = []
        price = last_price
        for _ in range(ROLLOUT_STEPS):
            drift = (rng.random() - 0.5) * 0.02
            price = round(price * (1 + drift), 2)
            rollout.append(price)
        return rollout, last_price

    def _build_horizons(
        self,
        base: float,
        rollout: list[float],
        today: date,
        *,
        ticker: str = "",
        sentiment: SentimentScore | None = None,
    ) -> list[HorizonPrediction]:
        horizons: list[HorizonPrediction] = []
        for label, horizon_days in HORIZON_DEFINITIONS:
            idx = min(horizon_days - 1, len(rollout) - 1)
            predicted = float(rollout[idx])
            pct = _signed_pct(predicted, base)
            direction = _direction_for(pct)
            horizon = HorizonPrediction(
                horizon=label,  # type: ignore[arg-type]
                horizon_days=horizon_days,
                target_date=today + timedelta(days=horizon_days),
                base_close=round(base, 2),
                predicted_close=round(predicted, 2),
                predicted_pct=round(pct, 2),
                direction=direction,
                explanation=self._generate_explanation(
                    ticker=ticker,
                    horizon_days=horizon_days,
                    base=base,
                    predicted=predicted,
                    pct=pct,
                    direction=direction,
                    sentiment=sentiment,
                ),
            )
            horizons.append(horizon)
        return horizons

    def _generate_explanation(
        self,
        *,
        ticker: str,
        horizon_days: int,
        base: float,
        predicted: float,
        pct: float,
        direction: str,
        sentiment: SentimentScore | None,
    ) -> str | None:
        if self._explainer is None:
            return None
        payload = ExplanationInput(
            ticker=ticker,
            horizon_label=HORIZON_LABELS.get(horizon_days, f"+{horizon_days} dias"),
            horizon_days=horizon_days,
            base_close=round(base, 2),
            predicted_close=round(predicted, 2),
            predicted_pct=round(pct, 2),
            direction=direction,
            sentiment_score=(sentiment.score if sentiment is not None else None),
            sentiment_positives=(sentiment.positives if sentiment is not None else 0),
            sentiment_negatives=(sentiment.negatives if sentiment is not None else 0),
            sentiment_neutrals=(sentiment.neutrals if sentiment is not None else 0),
            macro_score=None,
            macro_top_keywords=(),
        )
        try:
            return self._explainer.generate(payload)
        except Exception:  # noqa: BLE001
            return None

    async def get_explanation(
        self, ticker: str, horizon_days: int
    ) -> str | None:
        if self._predictions_repo is None:
            return None
        record = await self._predictions_repo.latest_for(
            ticker.upper().strip(), horizon_days
        )
        if record is None:
            return None
        return record.explanation

    def _build_preview_points(
        self, rollout: list[float], today: date, limit: int
    ) -> list[PredictionPoint]:
        points: list[PredictionPoint] = []
        prev: float | None = None
        for i, value in enumerate(rollout[:limit], start=1):
            if prev is None:
                pct = 0.0
            else:
                pct = _signed_pct(value, prev)
            points.append(
                PredictionPoint(
                    date=today + timedelta(days=i),
                    predicted_close=round(float(value), 2),
                    direction=_direction_for(pct),
                )
            )
            prev = value
        return points

    async def _safe_sentiment(self, ticker: str) -> SentimentScore | None:
        if self._sentiment is None:
            return None
        try:
            return await self._sentiment.aggregate(ticker)
        except Exception:  # noqa: BLE001
            return None

    async def _persist_horizons(
        self, ticker: str, horizons: list[HorizonPrediction]
    ) -> None:
        if self._predictions_repo is None:
            return
        for h in horizons:
            try:
                await self._predictions_repo.upsert(
                    PredictionUpsert(
                        ticker=ticker,
                        horizon_days=h.horizon_days,
                        target_date=h.target_date,
                        base_close=h.base_close,
                        predicted_close=h.predicted_close,
                        predicted_pct=h.predicted_pct,
                        direction=h.direction,
                        explanation=h.explanation,
                    )
                )
            except Exception:  # noqa: BLE001 - persistence must never break predict
                continue


def _signed_pct(value: float, base: float) -> float:
    if base == 0:
        return 0.0
    return (value - base) / base * 100.0


def _direction_for(pct: float) -> str:
    if abs(pct) < NEUTRAL_THRESHOLD_PCT:
        return "NEUTRO"
    return "ALTA" if pct > 0 else "BAIXA"


class ReconciliationService:
    """Fill ``actual_close`` on predictions whose ``target_date`` has passed."""

    def __init__(
        self,
        *,
        predictions_repo: PredictionsRepo,
        prices_repo: _PricesRepo,
    ) -> None:
        self._repo = predictions_repo
        self._prices = prices_repo

    async def reconcile(self) -> tuple[int, int]:
        pending = await self._repo.list_unresolved()
        resolved = 0
        by_ticker: dict[str, list[Any]] = {}
        for record in pending:
            by_ticker.setdefault(record.ticker_symbol, []).append(record)

        for ticker, records in by_ticker.items():
            try:
                history = await self._prices.get_history(ticker)
            except Exception:  # noqa: BLE001
                history = []
            if not history:
                continue
            lookup = self._index_by_date(history)
            for record in records:
                actual = lookup.get(record.target_date.isoformat())
                if actual is None:
                    continue
                base = record.base_close or 1.0
                error_pct = (actual - record.predicted_close) / base * 100.0
                await self._repo.update_actual(
                    record_id=record.id,
                    actual_close=round(float(actual), 4),
                    error_pct=round(float(error_pct), 4),
                    resolved_at=datetime.now(tz=timezone.utc),
                )
                resolved += 1
        return len(pending), resolved

    @staticmethod
    def _index_by_date(history: list[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in history:
            raw = row.get("date")
            if raw is None:
                continue
            key = str(raw)[:10]
            try:
                out[key] = float(row.get("close", 0.0))
            except (TypeError, ValueError):
                continue
        return out
