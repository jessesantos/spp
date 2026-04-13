"""SqlAlchemyPredictionsRepo tests against in-memory SQLite."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base
from app.repositories.predictions_repository import (
    PredictionUpsert,
    SqlAlchemyPredictionsRepo,
)


@pytest.fixture()
async def repo() -> SqlAlchemyPredictionsRepo:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    return SqlAlchemyPredictionsRepo(sessionmaker=maker)


@pytest.mark.asyncio
async def test_upsert_is_idempotent(repo: SqlAlchemyPredictionsRepo) -> None:
    target = date.today() + timedelta(days=1)
    item = PredictionUpsert(
        ticker="PETR4",
        horizon_days=1,
        target_date=target,
        base_close=30.0,
        predicted_close=31.0,
        predicted_pct=3.33,
        direction="ALTA",
    )

    first = await repo.upsert(item)
    second = await repo.upsert(
        PredictionUpsert(
            ticker="PETR4",
            horizon_days=1,
            target_date=target,
            base_close=30.0,
            predicted_close=32.0,
            predicted_pct=6.66,
            direction="ALTA",
        )
    )

    assert first.id == second.id
    history = await repo.list_history("PETR4", limit=50)
    assert len(history) == 1
    assert history[0].predicted_close == pytest.approx(32.0)


@pytest.mark.asyncio
async def test_list_history_ordered_by_target_desc(
    repo: SqlAlchemyPredictionsRepo,
) -> None:
    base = date.today()
    for horizon, offset in ((1, 1), (7, 7), (30, 30)):
        await repo.upsert(
            PredictionUpsert(
                ticker="VALE3",
                horizon_days=horizon,
                target_date=base + timedelta(days=offset),
                base_close=50.0,
                predicted_close=50.5,
                predicted_pct=1.0,
                direction="ALTA",
            )
        )

    rows = await repo.list_history("VALE3", limit=10)
    assert [r.horizon_days for r in rows] == [30, 7, 1]


@pytest.mark.asyncio
async def test_list_unresolved_and_update_actual(
    repo: SqlAlchemyPredictionsRepo,
) -> None:
    past = date.today() - timedelta(days=1)
    await repo.upsert(
        PredictionUpsert(
            ticker="ITUB4",
            horizon_days=1,
            target_date=past,
            base_close=20.0,
            predicted_close=21.0,
            predicted_pct=5.0,
            direction="ALTA",
        )
    )
    pending = await repo.list_unresolved()
    assert len(pending) == 1

    await repo.update_actual(
        record_id=pending[0].id,
        actual_close=20.5,
        error_pct=-2.5,
        resolved_at=datetime.now(tz=timezone.utc),
    )

    still_pending = await repo.list_unresolved()
    assert still_pending == []
