"""ohlcv unique (ticker_id, trade_date) and model_runs ticker index

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-13 12:00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_ohlcv_ticker_trade_date",
        "ohlcv",
        ["ticker_id", "trade_date"],
    )
    # ix_model_runs_ticker may already exist from 0001; create_index with
    # if_not_exists keeps the migration idempotent on Postgres.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_model_runs_ticker_symbol "
        "ON model_runs (ticker_symbol)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_model_runs_ticker_symbol")
    op.drop_constraint(
        "uq_ohlcv_ticker_trade_date", "ohlcv", type_="unique"
    )
