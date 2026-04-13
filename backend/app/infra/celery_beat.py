"""Celery beat schedule for the intelligent training pipeline.

The schedule is attached to :data:`celery_app` on import. Run with::

    celery -A app.infra.celery_app beat --loglevel=info

Two recurring jobs:

- ``daily_ohlcv_sync`` (22:00 America/Sao_Paulo): pull the OHLCV delta
  for every monitored ticker and persist via :class:`SqlAlchemyOhlcvRepo`.
- ``weekly_retrain`` (Sunday 23:00 America/Sao_Paulo): retrain every
  monitored ticker headlessly through :class:`TrainingOrchestrator`.
"""

from __future__ import annotations

from celery.schedules import crontab

from app.infra.celery_app import celery_app

celery_app.conf.beat_schedule = {
    "daily_ohlcv_sync": {
        "task": "spp.daily_ohlcv_sync",
        "schedule": crontab(hour=22, minute=0),
    },
    "weekly_retrain": {
        "task": "spp.weekly_retrain",
        "schedule": crontab(hour=23, minute=0, day_of_week="sun"),
    },
}
celery_app.conf.timezone = "America/Sao_Paulo"

# Ensure tasks are imported so beat can resolve their names when the
# scheduler starts in a separate process.
import app.infra.tasks  # noqa: E402,F401
