"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-13 00:00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tickers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=16), nullable=False, unique=True),
        sa.Column("name", sa.String(length=128), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=False, server_default="BRL"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_tickers_symbol", "tickers", ["symbol"], unique=True)

    op.create_table(
        "ohlcv",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "ticker_id",
            sa.Integer(),
            sa.ForeignKey("tickers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.create_index("ix_ohlcv_ticker_id", "ohlcv", ["ticker_id"])
    op.create_index("ix_ohlcv_trade_date", "ohlcv", ["trade_date"])

    op.create_table(
        "news_articles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticker_symbol", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("url", sa.String(length=1024), nullable=True),
        sa.Column("source", sa.String(length=256), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_news_ticker_symbol", "news_articles", ["ticker_symbol"])

    op.create_table(
        "sentiment_cache",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cache_key", sa.String(length=128), nullable=False, unique=True),
        sa.Column("ticker_symbol", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("impact", sa.String(length=16), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_sentiment_cache_key", "sentiment_cache", ["cache_key"], unique=True)
    op.create_index("ix_sentiment_ticker", "sentiment_cache", ["ticker_symbol"])

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticker_symbol", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("epochs", sa.Integer(), nullable=True),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("direction_accuracy", sa.Float(), nullable=True),
        sa.Column("artifact_path", sa.String(length=512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_model_runs_ticker", "model_runs", ["ticker_symbol"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticker_symbol", sa.String(length=16), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("predicted_close", sa.Float(), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model_run_id", sa.Integer(), sa.ForeignKey("model_runs.id"), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_predictions_ticker", "predictions", ["ticker_symbol"])


def downgrade() -> None:
    op.drop_index("ix_predictions_ticker", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_model_runs_ticker", table_name="model_runs")
    op.drop_table("model_runs")
    op.drop_index("ix_sentiment_ticker", table_name="sentiment_cache")
    op.drop_index("ix_sentiment_cache_key", table_name="sentiment_cache")
    op.drop_table("sentiment_cache")
    op.drop_index("ix_news_ticker_symbol", table_name="news_articles")
    op.drop_table("news_articles")
    op.drop_index("ix_ohlcv_trade_date", table_name="ohlcv")
    op.drop_index("ix_ohlcv_ticker_id", table_name="ohlcv")
    op.drop_table("ohlcv")
    op.drop_index("ix_tickers_symbol", table_name="tickers")
    op.drop_table("tickers")
