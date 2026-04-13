"""add prediction_market_signals table

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-13
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_market_signals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ticker_symbol", sa.String(length=16), nullable=False),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("market_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("topics", sa.Text(), nullable=True),
        sa.Column("top_questions", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint(
            "ticker_symbol", "snapshot_date", name="uq_pm_signals_ticker_date"
        ),
    )
    op.create_index(
        "ix_pm_signals_ticker",
        "prediction_market_signals",
        ["ticker_symbol"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_pm_signals_ticker", table_name="prediction_market_signals"
    )
    op.drop_table("prediction_market_signals")
