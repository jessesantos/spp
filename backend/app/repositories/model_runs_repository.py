"""Model runs repository: persistence of training executions.

Tracks the lifecycle of every LSTM training attempt so that the API can
expose history, the scheduler can detect failures, and operators can
audit when a model artifact was produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ModelRun


@dataclass(frozen=True)
class ModelRunStart:
    """Fields needed to open a training run row."""

    ticker: str
    epochs: int


@dataclass(frozen=True)
class ModelRunFinish:
    """Fields needed to close a training run row."""

    status: str  # done | failed
    loss: float | None = None
    direction_accuracy: float | None = None
    artifact_path: str | None = None


class ModelRunsRepo(Protocol):
    async def start(self, item: ModelRunStart) -> int: ...

    async def finish(self, run_id: int, item: ModelRunFinish) -> None: ...

    async def list_recent(self, ticker: str, limit: int = 20) -> list[ModelRun]: ...


class SqlAlchemyModelRunsRepo:
    """Async SQLAlchemy implementation of :class:`ModelRunsRepo`."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self._sessionmaker = sessionmaker

    async def start(self, item: ModelRunStart) -> int:
        async with self._sessionmaker() as session:
            async with session.begin():
                run = ModelRun(
                    ticker_symbol=item.ticker.upper().strip(),
                    status="running",
                    epochs=item.epochs,
                )
                session.add(run)
                await session.flush()
                return int(run.id)

    async def finish(self, run_id: int, item: ModelRunFinish) -> None:
        stmt = (
            update(ModelRun)
            .where(ModelRun.id == run_id)
            .values(
                status=item.status,
                loss=item.loss,
                direction_accuracy=item.direction_accuracy,
                artifact_path=item.artifact_path,
                finished_at=datetime.now(tz=timezone.utc),
            )
        )
        async with self._sessionmaker() as session:
            async with session.begin():
                await session.execute(stmt)

    async def list_recent(self, ticker: str, limit: int = 20) -> list[ModelRun]:
        ticker = ticker.upper().strip()
        stmt = (
            select(ModelRun)
            .where(ModelRun.ticker_symbol == ticker)
            .order_by(ModelRun.created_at.desc(), ModelRun.id.desc())
            .limit(max(1, int(limit)))
        )
        async with self._sessionmaker() as session:
            result = await session.execute(stmt)
            return list(result.scalars().all())

    @staticmethod
    async def _get(session: AsyncSession, run_id: int) -> ModelRun | None:
        stmt = select(ModelRun).where(ModelRun.id == run_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
