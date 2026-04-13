"""SQLAlchemy 2.0 ORM models for the SPP persistence layer."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for declarative models."""


class Ticker(Base):
    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(128))
    currency: Mapped[str] = mapped_column(String(8), default="BRL", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    ohlcv: Mapped[list["OHLCV"]] = relationship(back_populates="ticker", cascade="all,delete")


class OHLCV(Base):
    __tablename__ = "ohlcv"
    __table_args__ = (
        UniqueConstraint(
            "ticker_id", "trade_date", name="uq_ohlcv_ticker_trade_date"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_id: Mapped[int] = mapped_column(
        ForeignKey("tickers.id", ondelete="CASCADE"), index=True, nullable=False
    )
    trade_date: Mapped[datetime] = mapped_column(Date, nullable=False, index=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    ticker: Mapped[Ticker] = relationship(back_populates="ohlcv")


class NewsArticle(Base):
    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1024))
    source: Mapped[str | None] = mapped_column(String(256))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class SentimentCache(Base):
    __tablename__ = "sentiment_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cache_key: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    ticker_symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text)
    impact: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class PredictionRecord(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "ticker_symbol",
            "horizon_days",
            "target_date",
            name="uq_predictions_ticker_horizon_target",
        ),
        Index("ix_predictions_ticker_horizon", "ticker_symbol", "horizon_days"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    target_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    base_close: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    predicted_close: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    explanation: Mapped[str | None] = mapped_column(Text)
    actual_close: Mapped[float | None] = mapped_column(Float)
    error_pct: Mapped[float | None] = mapped_column(Float)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    epochs: Mapped[int | None] = mapped_column(Integer)
    loss: Mapped[float | None] = mapped_column(Float)
    direction_accuracy: Mapped[float | None] = mapped_column(Float)
    artifact_path: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PredictionMarketSignalRecord(Base):
    """Snapshot diario agregado de mercados de previsao (Kalshi/Polymarket) por ticker."""

    __tablename__ = "prediction_market_signals"
    __table_args__ = (
        UniqueConstraint(
            "ticker_symbol", "snapshot_date",
            name="uq_pm_signals_ticker_date",
        ),
        Index("ix_pm_signals_ticker", "ticker_symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker_symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    snapshot_date: Mapped[datetime] = mapped_column(Date, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    market_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    topics: Mapped[str | None] = mapped_column(Text)
    top_questions: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
