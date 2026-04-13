"""Repositorio de snapshots de mercados de previsao.

Persiste um registro agregado por (ticker, snapshot_date) com upsert
idempotente, expondo ``list_recent`` para leitura de feature historica.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.db.models import PredictionMarketSignalRecord


@dataclass(frozen=True)
class MarketSignalUpsert:
    ticker: str
    snapshot_date: date
    score: float
    confidence: float
    market_count: int
    topics: tuple[str, ...] = ()
    top_questions: tuple[str, ...] = ()


class PredictionMarketSignalsRepo(Protocol):
    async def upsert(self, item: MarketSignalUpsert) -> PredictionMarketSignalRecord: ...

    async def latest_for(
        self, ticker: str
    ) -> PredictionMarketSignalRecord | None: ...

    async def list_recent(
        self, ticker: str, limit: int = 30
    ) -> list[PredictionMarketSignalRecord]: ...


class SqlAlchemyPredictionMarketSignalsRepo:
    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def upsert(self, item: MarketSignalUpsert) -> PredictionMarketSignalRecord:
        topics_csv = ",".join(item.topics[:20]) if item.topics else None
        questions_json = (
            json.dumps(list(item.top_questions[:10])) if item.top_questions else None
        )
        async with self._sessionmaker() as session:
            stmt = (
                pg_insert(PredictionMarketSignalRecord)
                .values(
                    ticker_symbol=item.ticker.upper().strip(),
                    snapshot_date=item.snapshot_date,
                    score=float(item.score),
                    confidence=float(item.confidence),
                    market_count=int(item.market_count),
                    topics=topics_csv,
                    top_questions=questions_json,
                    created_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["ticker_symbol", "snapshot_date"],
                    set_={
                        "score": float(item.score),
                        "confidence": float(item.confidence),
                        "market_count": int(item.market_count),
                        "topics": topics_csv,
                        "top_questions": questions_json,
                    },
                )
                .returning(PredictionMarketSignalRecord)
            )
            result = await session.execute(stmt)
            row = result.scalar_one()
            await session.commit()
            return row

    async def latest_for(
        self, ticker: str
    ) -> PredictionMarketSignalRecord | None:
        async with self._sessionmaker() as session:
            stmt = (
                select(PredictionMarketSignalRecord)
                .where(PredictionMarketSignalRecord.ticker_symbol == ticker.upper().strip())
                .order_by(PredictionMarketSignalRecord.snapshot_date.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_recent(
        self, ticker: str, limit: int = 30
    ) -> list[PredictionMarketSignalRecord]:
        async with self._sessionmaker() as session:
            stmt = (
                select(PredictionMarketSignalRecord)
                .where(PredictionMarketSignalRecord.ticker_symbol == ticker.upper().strip())
                .order_by(PredictionMarketSignalRecord.snapshot_date.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())
