"""Pytest fixtures - isolate the FastAPI app from external services.

We override ``prediction_service`` and ``sentiment_service`` dependency
providers with lightweight stubs (no BrAPI, no yfinance, no Anthropic)
so integration tests stay fast and deterministic.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from typing import Any

import pytest

from app.db.models import PredictionRecord
from app.ml.explanation import HeuristicExplanationGenerator
from app.infra.dependencies import (
    model_runs_repo,
    prediction_repo,
    prediction_service,
    reconciliation_service,
    sentiment_service,
)
from app.main import app
from app.repositories.predictions_repository import (
    PredictionUpsert,
    PredictionsRepo,
)
from app.services.prediction_service import (
    PredictionService,
    ReconciliationService,
)
from app.services.sentiment_service import SentimentService


class InMemoryPredictionsRepo:
    """In-memory PredictionsRepo used by tests."""

    def __init__(self) -> None:
        self._rows: list[PredictionRecord] = []
        self._next_id: int = 1

    async def upsert(self, item: PredictionUpsert) -> PredictionRecord:
        for row in self._rows:
            if (
                row.ticker_symbol == item.ticker
                and row.horizon_days == item.horizon_days
                and row.target_date == item.target_date
            ):
                row.base_close = item.base_close
                row.predicted_close = item.predicted_close
                row.predicted_pct = item.predicted_pct
                row.direction = item.direction
                if item.explanation is not None:
                    row.explanation = item.explanation
                return row
        record = PredictionRecord(
            ticker_symbol=item.ticker,
            horizon_days=item.horizon_days,
            target_date=item.target_date,
            base_close=item.base_close,
            predicted_close=item.predicted_close,
            predicted_pct=item.predicted_pct,
            direction=item.direction,
            explanation=item.explanation,
        )
        record.id = self._next_id
        record.created_at = datetime.utcnow()
        self._next_id += 1
        self._rows.append(record)
        return record

    async def latest_for(
        self, ticker: str, horizon_days: int
    ) -> PredictionRecord | None:
        rows = [
            r
            for r in self._rows
            if r.ticker_symbol == ticker.upper().strip()
            and r.horizon_days == horizon_days
        ]
        if not rows:
            return None
        rows.sort(key=lambda r: (r.target_date, r.id), reverse=True)
        return rows[0]

    async def list_history(
        self, ticker: str, limit: int = 60
    ) -> list[PredictionRecord]:
        rows = [r for r in self._rows if r.ticker_symbol == ticker.upper().strip()]
        rows.sort(key=lambda r: (r.target_date, r.id), reverse=True)
        return rows[:limit]

    async def list_unresolved(
        self, until: date | None = None
    ) -> list[PredictionRecord]:
        cutoff = until or date.today()
        return [
            r
            for r in self._rows
            if r.actual_close is None and r.target_date <= cutoff
        ]

    async def update_actual(
        self,
        record_id: int,
        actual_close: float,
        error_pct: float,
        resolved_at: datetime,
    ) -> None:
        for row in self._rows:
            if row.id == record_id:
                row.actual_close = actual_close
                row.error_pct = error_pct
                row.resolved_at = resolved_at
                return


_shared_repo = InMemoryPredictionsRepo()


def _stub_predictions_repo() -> PredictionsRepo:
    return _shared_repo


def _stub_prediction_service() -> PredictionService:
    return PredictionService(
        prices_repo=None,
        sentiment_service=None,
        predictions_repo=_shared_repo,
        explanation_generator=HeuristicExplanationGenerator(),
        fallback_stub=True,
    )


def _stub_sentiment_service() -> SentimentService:
    return SentimentService(news_source=None, analyzer=None, fallback_stub=True)


class _EmptyPricesRepo:
    async def get_history(self, ticker: str) -> list[dict[str, Any]]:
        return []


def _stub_reconciliation_service() -> ReconciliationService:
    return ReconciliationService(
        predictions_repo=_shared_repo,
        prices_repo=_EmptyPricesRepo(),
    )


class _StubModelRunsRepo:
    async def start(self, _: Any) -> int:
        return 1

    async def finish(self, _run_id: int, _item: Any) -> None:
        return None

    async def list_recent(self, ticker: str, limit: int = 20) -> list[Any]:
        return []


def _stub_model_runs_repo() -> _StubModelRunsRepo:
    return _StubModelRunsRepo()


@pytest.fixture(autouse=True)
def _override_dependencies() -> Iterator[None]:
    app.dependency_overrides[prediction_service] = _stub_prediction_service
    app.dependency_overrides[sentiment_service] = _stub_sentiment_service
    app.dependency_overrides[prediction_repo] = _stub_predictions_repo
    app.dependency_overrides[reconciliation_service] = _stub_reconciliation_service
    app.dependency_overrides[model_runs_repo] = _stub_model_runs_repo
    try:
        yield
    finally:
        app.dependency_overrides.clear()
