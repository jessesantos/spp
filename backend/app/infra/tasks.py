"""Celery tasks: training entry-point + scheduled jobs.

Heavy work runs synchronously inside the worker container. We bridge to
the async orchestrator with ``asyncio.run`` so the task remains a plain
sync function (Celery does not yet ship native async tasks).

Concorrencia: treinamentos concorrentes para o mesmo ticker sao
serializados por um file lock exclusivo em ``{MODELS_DIR}/{TICKER}.lock``
via ``fcntl.flock``. Evita que dois workers (ou um worker + o CLI manual)
sobrescrevam simultaneamente os artefatos ``.keras`` e ``.keras.aux.joblib``
ou gerem duas linhas de ``model_runs`` inconsistentes. O lock e liberado
automaticamente ao final da task (ou se o worker crashar).
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

from app.infra.celery_app import celery_app
from app.infra.config import settings

log = logging.getLogger("spp.tasks")

DEFAULT_EPOCHS: int = 50
SYNC_LOOKBACK_DAYS: int = 30
BACKFILL_LOOKBACK_DAYS: int = 1825  # 5 anos (ADR 0009)

# Allowlist estreito para nomes de arquivo de lock (defesa em profundidade).
# O ticker ja foi validado no `TickerSymbol` quando chegou pela API, mas
# train_model tambem pode ser disparado por Celery beat, onde a origem
# e interna - manter sanitizacao aqui bloqueia tentativas de path traversal
# caso um ticker malformado vaze para a fila.
_LOCK_SAFE = re.compile(r"[^A-Z0-9]")


@contextlib.contextmanager
def _ticker_training_lock(ticker: str) -> Iterator[None]:
    """Lock exclusivo de arquivo por ticker para serializar treinos concorrentes.

    Usa ``fcntl.flock(LOCK_EX)`` - bloqueia ate o lock estar disponivel.
    O file descriptor permanece aberto durante o treino; ao sair do
    ``with`` o OS libera o lock automaticamente mesmo em caso de crash.
    """
    safe_ticker = _LOCK_SAFE.sub("", ticker.upper())[:10] or "UNKNOWN"
    lock_dir = Path(settings.models_dir)
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - disco cheio ou permissao
        log.warning("training.lock_dir_failed", extra={"error": str(exc)})
    lock_path = lock_dir / f"{safe_ticker}.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@celery_app.task(name="spp.train_model")
def train_model(ticker: str, epochs: int = DEFAULT_EPOCHS) -> dict[str, Any]:
    """Train an LSTM for ``ticker`` and persist a ModelRun row.

    Serializado por ticker via ``_ticker_training_lock``: se outro worker
    ja esta treinando o mesmo ativo, a chamada bloqueia ate o primeiro
    terminar e so entao inicia (evita race na escrita do ``.keras`` +
    companion ``.aux.joblib`` e duplicidade em ``model_runs``).
    """
    from app.infra.dependencies import build_training_orchestrator

    orchestrator = build_training_orchestrator()
    try:
        with _ticker_training_lock(ticker):
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
            cutoff = latest or (date.today() - timedelta(days=BACKFILL_LOOKBACK_DAYS))
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
            # Mesmo no beat (serial por default), manter lock aqui protege
            # contra corrida com ``train_model`` disparado via API/manual.
            with _ticker_training_lock(ticker):
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
