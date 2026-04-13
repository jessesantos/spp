"""OHLCV repository: ticker bootstrap, idempotent upsert, history queries.

Protocol + SQLAlchemy 2.0 async implementation. Idempotency is enforced by
the unique constraint on ``(ticker_id, trade_date)`` (Alembic 0003).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Protocol

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import OHLCV, Ticker


@dataclass(frozen=True)
class OhlcvRow:
    """Value object describing one OHLCV bar."""

    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


class OhlcvRepo(Protocol):
    async def ensure_ticker(self, symbol: str, name: str | None = None) -> int: ...

    async def upsert_many(
        self, ticker_id: int, rows: Iterable[dict[str, Any]]
    ) -> int: ...

    async def count(self, ticker_id: int) -> int: ...

    async def latest_trade_date(self, ticker_id: int) -> date | None: ...

    async def list_history(
        self, ticker_id: int, limit: int = 1000
    ) -> list[dict[str, Any]]: ...


class SqlAlchemyOhlcvRepo:
    """Async SQLAlchemy implementation of :class:`OhlcvRepo`."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def ensure_ticker(self, symbol: str, name: str | None = None) -> int:
        symbol = symbol.upper().strip()
        async with self._sessionmaker() as session:
            async with session.begin():
                stmt = select(Ticker).where(Ticker.symbol == symbol)
                result = await session.execute(stmt)
                existing = result.scalar_one_or_none()
                if existing is not None:
                    if name and not existing.name:
                        existing.name = name
                    return int(existing.id)
                ticker = Ticker(symbol=symbol, name=name)
                session.add(ticker)
                await session.flush()
                return int(ticker.id)

    async def upsert_many(
        self, ticker_id: int, rows: Iterable[dict[str, Any]]
    ) -> int:
        normalized = [_normalize(row) for row in rows]
        normalized = [row for row in normalized if row is not None]
        if not normalized:
            return 0

        async with self._sessionmaker() as session:
            async with session.begin():
                existing_dates = await self._existing_dates(
                    session, ticker_id, [row.trade_date for row in normalized]
                )
                inserted = 0
                for row in normalized:
                    if row.trade_date in existing_dates:
                        continue
                    session.add(
                        OHLCV(
                            ticker_id=ticker_id,
                            trade_date=row.trade_date,
                            open=row.open,
                            high=row.high,
                            low=row.low,
                            close=row.close,
                            volume=row.volume,
                        )
                    )
                    existing_dates.add(row.trade_date)
                    inserted += 1
                await session.flush()
                return inserted

    async def count(self, ticker_id: int) -> int:
        stmt = select(func.count(OHLCV.id)).where(OHLCV.ticker_id == ticker_id)
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            return int(result.scalar_one() or 0)

    async def latest_trade_date(self, ticker_id: int) -> date | None:
        stmt = select(func.max(OHLCV.trade_date)).where(OHLCV.ticker_id == ticker_id)
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            value = result.scalar_one_or_none()
            if value is None:
                return None
            if isinstance(value, datetime):
                return value.date()
            return value

    async def list_history(
        self, ticker_id: int, limit: int = 1000
    ) -> list[dict[str, Any]]:
        stmt = (
            select(OHLCV)
            .where(OHLCV.ticker_id == ticker_id)
            .order_by(OHLCV.trade_date.asc())
            .limit(max(1, int(limit)))
        )
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
        return [
            {
                "date": _as_date(row.trade_date).isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": int(row.volume),
            }
            for row in rows
        ]

    @staticmethod
    async def _existing_dates(
        session: AsyncSession, ticker_id: int, candidates: list[date]
    ) -> set[date]:
        if not candidates:
            return set()
        stmt = select(OHLCV.trade_date).where(
            OHLCV.ticker_id == ticker_id,
            OHLCV.trade_date.in_(candidates),
        )
        result = await session.execute(stmt)
        return {_as_date(value) for value in result.scalars().all()}


def _normalize(raw: dict[str, Any]) -> OhlcvRow | None:
    trade = raw.get("date") or raw.get("trade_date")
    if trade is None:
        return None
    try:
        trade_date = _coerce_date(trade)
        return OhlcvRow(
            trade_date=trade_date,
            open=float(raw.get("open", 0.0)),
            high=float(raw.get("high", 0.0)),
            low=float(raw.get("low", 0.0)),
            close=float(raw.get("close", 0.0)),
            volume=int(raw.get("volume", 0) or 0),
        )
    except (TypeError, ValueError):
        return None


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value)).date()
    text = str(value)[:10]
    return date.fromisoformat(text)


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return _coerce_date(value)
