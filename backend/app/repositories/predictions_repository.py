"""Predictions repository: upsert, list history, reconcile actuals.

Protocol + SQLAlchemy async implementation. Idempotency is guaranteed by
the unique constraint on (ticker_symbol, horizon_days, target_date).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.db.models import PredictionRecord


@dataclass(frozen=True)
class PredictionUpsert:
    """Value object describing a prediction row to persist."""

    ticker: str
    horizon_days: int
    target_date: date
    base_close: float
    predicted_close: float
    predicted_pct: float
    direction: str
    explanation: str | None = None


class PredictionsRepo(Protocol):
    async def upsert(self, item: PredictionUpsert) -> PredictionRecord: ...

    async def list_history(
        self, ticker: str, limit: int = 60
    ) -> list[PredictionRecord]: ...

    async def list_unresolved(
        self, until: date | None = None
    ) -> list[PredictionRecord]: ...

    async def latest_for(
        self, ticker: str, horizon_days: int
    ) -> PredictionRecord | None: ...

    async def update_actual(
        self,
        record_id: int,
        actual_close: float,
        error_pct: float,
        resolved_at: datetime,
    ) -> None: ...


class SqlAlchemyPredictionsRepo:
    """Async SQLAlchemy implementation of PredictionsRepo."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def upsert(self, item: PredictionUpsert) -> PredictionRecord:
        async with self._sessionmaker() as session:
            async with session.begin():
                existing = await self._find_existing(session, item)
                if existing is None:
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
                    session.add(record)
                    await session.flush()
                    return record
                existing.base_close = item.base_close
                existing.predicted_close = item.predicted_close
                existing.predicted_pct = item.predicted_pct
                existing.direction = item.direction
                if item.explanation is not None:
                    existing.explanation = item.explanation
                await session.flush()
                return existing

    async def list_history(
        self, ticker: str, limit: int = 60
    ) -> list[PredictionRecord]:
        ticker = ticker.upper().strip()
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.ticker_symbol == ticker)
            .order_by(PredictionRecord.target_date.desc(), PredictionRecord.id.desc())
            .limit(max(1, int(limit)))
        )
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def list_unresolved(
        self, until: date | None = None
    ) -> list[PredictionRecord]:
        cutoff = until or date.today()
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.actual_close.is_(None))
            .where(PredictionRecord.target_date <= cutoff)
            .order_by(PredictionRecord.target_date.asc())
        )
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def latest_for(
        self, ticker: str, horizon_days: int
    ) -> PredictionRecord | None:
        ticker = ticker.upper().strip()
        stmt = (
            select(PredictionRecord)
            .where(
                PredictionRecord.ticker_symbol == ticker,
                PredictionRecord.horizon_days == horizon_days,
            )
            .order_by(
                PredictionRecord.target_date.desc(),
                PredictionRecord.id.desc(),
            )
            .limit(1)
        )
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def update_actual(
        self,
        record_id: int,
        actual_close: float,
        error_pct: float,
        resolved_at: datetime,
    ) -> None:
        stmt = (
            update(PredictionRecord)
            .where(PredictionRecord.id == record_id)
            .values(
                actual_close=actual_close,
                error_pct=error_pct,
                resolved_at=resolved_at,
            )
        )
        async with self._sessionmaker() as session:
            async with session.begin():
                await session.execute(stmt)

    @staticmethod
    async def _find_existing(
        session: AsyncSession, item: PredictionUpsert
    ) -> PredictionRecord | None:
        stmt = select(PredictionRecord).where(
            PredictionRecord.ticker_symbol == item.ticker,
            PredictionRecord.horizon_days == item.horizon_days,
            PredictionRecord.target_date == item.target_date,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
