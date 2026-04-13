"""SqlAlchemyOhlcvRepo tests against in-memory SQLite."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.models import Base
from app.repositories.ohlcv_repository import SqlAlchemyOhlcvRepo


@pytest.fixture()
async def repo() -> SqlAlchemyOhlcvRepo:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(bind=engine, expire_on_commit=False)
    return SqlAlchemyOhlcvRepo(sessionmaker=maker)


def _row(d: date, close: float = 10.0) -> dict[str, object]:
    return {
        "date": d.isoformat(),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1000,
    }


@pytest.mark.asyncio
async def test_ensure_ticker_is_idempotent(repo: SqlAlchemyOhlcvRepo) -> None:
    first = await repo.ensure_ticker("PETR4", name="Petrobras")
    second = await repo.ensure_ticker("PETR4")
    assert first == second


@pytest.mark.asyncio
async def test_upsert_many_is_idempotent_by_trade_date(
    repo: SqlAlchemyOhlcvRepo,
) -> None:
    ticker_id = await repo.ensure_ticker("VALE3")
    today = date.today()
    rows = [_row(today - timedelta(days=i)) for i in range(3)]

    inserted_first = await repo.upsert_many(ticker_id, rows)
    inserted_second = await repo.upsert_many(ticker_id, rows)

    assert inserted_first == 3
    assert inserted_second == 0
    assert await repo.count(ticker_id) == 3


@pytest.mark.asyncio
async def test_latest_trade_date_and_list_history(
    repo: SqlAlchemyOhlcvRepo,
) -> None:
    ticker_id = await repo.ensure_ticker("ITUB4")
    today = date.today()
    await repo.upsert_many(
        ticker_id,
        [
            _row(today - timedelta(days=2), close=20.0),
            _row(today - timedelta(days=1), close=21.0),
            _row(today, close=22.0),
        ],
    )

    assert await repo.latest_trade_date(ticker_id) == today
    history = await repo.list_history(ticker_id, limit=10)
    assert [row["close"] for row in history] == [20.0, 21.0, 22.0]
