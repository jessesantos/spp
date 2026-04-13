"""multi-horizon prediction columns

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-13 00:00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column(
            "horizon_days",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )
    op.add_column(
        "predictions",
        sa.Column(
            "base_close",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
    )
    op.add_column(
        "predictions",
        sa.Column(
            "predicted_pct",
            sa.Float(),
            nullable=False,
            server_default="0.0",
        ),
    )
    op.add_column(
        "predictions",
        sa.Column("actual_close", sa.Float(), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("error_pct", sa.Float(), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index(
        "ix_predictions_ticker_horizon",
        "predictions",
        ["ticker_symbol", "horizon_days"],
    )
    op.create_unique_constraint(
        "uq_predictions_ticker_horizon_target",
        "predictions",
        ["ticker_symbol", "horizon_days", "target_date"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_predictions_ticker_horizon_target", "predictions", type_="unique"
    )
    op.drop_index("ix_predictions_ticker_horizon", table_name="predictions")
    op.drop_column("predictions", "resolved_at")
    op.drop_column("predictions", "error_pct")
    op.drop_column("predictions", "actual_close")
    op.drop_column("predictions", "predicted_pct")
    op.drop_column("predictions", "base_close")
    op.drop_column("predictions", "horizon_days")
