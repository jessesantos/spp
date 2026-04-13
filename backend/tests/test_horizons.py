"""Tests for the multi-horizon prediction path using the stub rollout."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import pytest

from app.ml.explanation import (
    HeuristicExplanationGenerator,
    MAX_WORDS,
    MIN_WORDS,
)
from app.repositories.predictions_repository import PredictionUpsert
from app.services.prediction_service import (
    PredictionService,
    ReconciliationService,
)
from tests.conftest import InMemoryPredictionsRepo


class _FakePricesRepo:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def get_history(self, ticker: str) -> list[dict[str, Any]]:
        return list(self._rows)


@pytest.mark.asyncio
async def test_predict_with_horizons_returns_three_horizons() -> None:
    repo = InMemoryPredictionsRepo()
    service = PredictionService(
        prices_repo=None,
        sentiment_service=None,
        predictions_repo=repo,
        explanation_generator=HeuristicExplanationGenerator(),
        fallback_stub=True,
    )

    response = await service.predict_with_horizons("PETR4")

    assert response.ticker == "PETR4"
    assert len(response.horizons) == 3
    labels = [h.horizon for h in response.horizons]
    assert labels == ["D1", "W1", "M1"]
    days = [h.horizon_days for h in response.horizons]
    assert days == [1, 7, 30]

    today = date.today()
    for h in response.horizons:
        assert h.target_date == today + timedelta(days=h.horizon_days)
        assert h.direction in {"ALTA", "BAIXA", "NEUTRO"}
        assert h.base_close == response.last_price
        expected_pct = round(
            (h.predicted_close - h.base_close) / h.base_close * 100.0, 2
        )
        assert h.predicted_pct == pytest.approx(expected_pct, abs=0.02)
        assert h.explanation is not None
        words = len(re.findall(r"\S+", h.explanation))
        assert MIN_WORDS <= words <= MAX_WORDS


@pytest.mark.asyncio
async def test_predict_with_horizons_is_idempotent_on_repo() -> None:
    repo = InMemoryPredictionsRepo()
    service = PredictionService(
        prices_repo=None,
        sentiment_service=None,
        predictions_repo=repo,
        fallback_stub=True,
    )

    await service.predict_with_horizons("VALE3")
    await service.predict_with_horizons("VALE3")

    rows = await repo.list_history("VALE3", limit=100)
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_neutral_threshold_applied_on_small_variation() -> None:
    service = PredictionService(
        prices_repo=None,
        sentiment_service=None,
        predictions_repo=None,
        fallback_stub=True,
    )
    rollout = [100.001, 100.002, 100.003]
    horizons = service._build_horizons(100.0, rollout + [100.0] * 30, date.today())
    assert all(h.direction == "NEUTRO" for h in horizons)


@pytest.mark.asyncio
async def test_reconciliation_fills_actual_close() -> None:
    repo = InMemoryPredictionsRepo()
    yesterday = date.today() - timedelta(days=1)
    await repo.upsert(
        PredictionUpsert(
            ticker="PETR4",
            horizon_days=1,
            target_date=yesterday,
            base_close=30.0,
            predicted_close=31.0,
            predicted_pct=3.33,
            direction="ALTA",
        )
    )

    prices = _FakePricesRepo(
        [{"date": yesterday.isoformat(), "close": 30.50}]
    )
    svc = ReconciliationService(predictions_repo=repo, prices_repo=prices)

    checked, resolved = await svc.reconcile()

    assert checked == 1
    assert resolved == 1
    rows = await repo.list_history("PETR4")
    assert rows[0].actual_close == pytest.approx(30.5, abs=0.01)
    assert rows[0].error_pct is not None
    assert isinstance(rows[0].resolved_at, datetime)
