"""Celery tasks: training entry-point + scheduled jobs.

Heavy work runs synchronously inside the worker container. We bridge to
the async orchestrator with ``asyncio.run`` so the task remains a plain
sync function (Celery does not yet ship native async tasks).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from typing import Any

from app.infra.celery_app import celery_app

log = logging.getLogger("spp.tasks")

DEFAULT_EPOCHS: int = 50
SYNC_LOOKBACK_DAYS: int = 30


@celery_app.task(name="spp.train_model")
def train_model(ticker: str, epochs: int = DEFAULT_EPOCHS) -> dict[str, Any]:
    """Train an LSTM for ``ticker`` and persist a ModelRun row."""
    from app.infra.dependencies import build_training_orchestrator

    orchestrator = build_training_orchestrator()
    try:
        result = asyncio.run(orchestrator.train(ticker, epochs=epochs))
        return result.to_dict()
    except Exception as exc:  # noqa: BLE001 - report failure to Celery
        log.exception("train_model.failed", extra={"ticker": ticker})
        return {"status": "failed", "ticker": ticker, "error": str(exc)}


@celery_app.task(name="spp.daily_ohlcv_sync")
def daily_ohlcv_sync() -> dict[str, Any]:
    """Pull the recent delta of OHLCV for every monitored ticker."""
    return asyncio.run(_run_daily_sync())


@celery_app.task(name="spp.weekly_retrain")
def weekly_retrain(epochs: int = DEFAULT_EPOCHS) -> dict[str, Any]:
    """Retrain every monitored ticker headlessly."""
    return asyncio.run(_run_weekly_retrain(epochs))


async def _run_daily_sync() -> dict[str, Any]:
    from app.infra.dependencies import ohlcv_repo, price_repository

    repo = ohlcv_repo()
    prices = price_repository()
    summary: dict[str, Any] = {"updated": {}, "errors": {}}

    for ticker in _monitored_tickers():
        try:
            ticker_id = await repo.ensure_ticker(ticker)
            latest = await repo.latest_trade_date(ticker_id)
            cutoff = latest or (date.today() - timedelta(days=365))
            history = await prices.get_history(ticker)
            delta = [row for row in history if _row_after(row, cutoff)]
            inserted = await repo.upsert_many(ticker_id, delta) if delta else 0
            summary["updated"][ticker] = inserted
        except Exception as exc:  # noqa: BLE001
            log.warning("daily_sync.failed", extra={"ticker": ticker, "error": str(exc)})
            summary["errors"][ticker] = str(exc)
    return summary


async def _run_weekly_retrain(epochs: int) -> dict[str, Any]:
    from app.infra.dependencies import build_training_orchestrator, ohlcv_repo

    orchestrator = build_training_orchestrator()
    repo = ohlcv_repo()
    summary: dict[str, Any] = {"runs": {}}

    for ticker in _monitored_tickers():
        try:
            ticker_id = await repo.ensure_ticker(ticker)
            if await repo.count(ticker_id) <= 0:
                summary["runs"][ticker] = {"status": "skipped", "reason": "no_rows"}
                continue
            result = await orchestrator.train(ticker, epochs=epochs)
            summary["runs"][ticker] = result.to_dict()
        except Exception as exc:  # noqa: BLE001
            log.warning("weekly_retrain.failed", extra={"ticker": ticker, "error": str(exc)})
            summary["runs"][ticker] = {"status": "failed", "error": str(exc)}
    return summary


def _monitored_tickers() -> list[str]:
    """Mirrors the static list served by ``GET /api/tickers``."""
    return ["PETR4", "VALE3", "ITUB4"]


def _row_after(row: dict[str, Any], cutoff: date) -> bool:
    raw = row.get("date") or row.get("trade_date")
    if raw is None:
        return False
    try:
        text = str(raw)[:10]
        parsed = date.fromisoformat(text)
        return parsed > cutoff
    except (TypeError, ValueError):
        return False
