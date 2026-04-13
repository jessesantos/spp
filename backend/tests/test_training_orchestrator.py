"""TrainingOrchestrator tests with in-memory fakes (no TF, no BrAPI)."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base
from app.ml.training_orchestrator import TrainingOrchestrator
from app.repositories.model_runs_repository import SqlAlchemyModelRunsRepo
from app.repositories.ohlcv_repository import SqlAlchemyOhlcvRepo


class _PricesStub:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def get_history(self, ticker: str) -> list[dict[str, Any]]:
        return list(self._rows)


class _PricesEmpty:
    async def get_history(self, ticker: str) -> list[dict[str, Any]]:
        return []


@pytest.fixture()
async def repos() -> tuple[SqlAlchemyOhlcvRepo, SqlAlchemyModelRunsRepo]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    return SqlAlchemyOhlcvRepo(maker), SqlAlchemyModelRunsRepo(maker)


@pytest.mark.asyncio
async def test_train_marks_failed_when_no_data(
    repos: tuple[SqlAlchemyOhlcvRepo, SqlAlchemyModelRunsRepo],
) -> None:
    ohlcv, runs = repos
    orchestrator = TrainingOrchestrator(
        prices_repo=_PricesEmpty(),
        ohlcv_repo=ohlcv,
        model_runs_repo=runs,
    )

    result = await orchestrator.train("PETR4", epochs=1)

    assert result.status == "failed"
    assert result.error and "no OHLCV" in result.error
    history = await runs.list_recent("PETR4")
    assert len(history) == 1
    assert history[0].status == "failed"


@pytest.mark.asyncio
async def test_train_marks_failed_when_rows_below_minimum(
    repos: tuple[SqlAlchemyOhlcvRepo, SqlAlchemyModelRunsRepo],
) -> None:
    ohlcv, runs = repos
    rows = [
        {
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.0 + i * 0.01,
            "volume": 1000,
        }
        for i in range(10)
    ]
    orchestrator = TrainingOrchestrator(
        prices_repo=_PricesStub(rows),
        ohlcv_repo=ohlcv,
        model_runs_repo=runs,
    )

    result = await orchestrator.train("VALE3", epochs=1)

    assert result.status == "failed"
    assert "insufficient" in (result.error or "")
    history = await runs.list_recent("VALE3")
    assert history[0].status == "failed"
