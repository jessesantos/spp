"""TrainingOrchestrator: SOLID-friendly entry point for LSTM training.

Replaces the imperative CLI in :mod:`app.ml.train` with an injectable
class that is reusable from:

- the manual CLI (``python -m app.ml.train --ticker PETR4``)
- the Celery task ``spp.train_model``
- the weekly Celery beat schedule (``weekly_retrain``)

Heavy dependencies (TensorFlow, Anthropic) are resolved lazily inside
methods so that simply importing this module remains cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import pandas as pd

from app.repositories.model_runs_repository import (
    ModelRunFinish,
    ModelRunsRepo,
    ModelRunStart,
)
from app.repositories.ohlcv_repository import OhlcvRepo

log = logging.getLogger("spp.training")

MIN_TRAINING_ROWS: int = 60


class _PricesRepo(Protocol):
    async def get_history(self, ticker: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ModelRunResult:
    """Outcome of a training run, returned to callers (CLI, Celery)."""

    run_id: int | None
    ticker: str
    status: str
    loss: float | None = None
    direction_accuracy: float | None = None
    artifact_path: str | None = None
    rows: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ticker": self.ticker,
            "status": self.status,
            "loss": self.loss,
            "direction_accuracy": self.direction_accuracy,
            "artifact_path": self.artifact_path,
            "rows": self.rows,
            "error": self.error,
        }


class TrainingOrchestrator:
    """Coordinate price ingest, sentiment + macro enrichment and LSTM fit.

    Dependencies are injected so tests can substitute in-memory fakes
    without touching BrAPI, RSS feeds or the Anthropic API.
    """

    def __init__(
        self,
        *,
        prices_repo: _PricesRepo,
        ohlcv_repo: OhlcvRepo,
        model_runs_repo: ModelRunsRepo | None = None,
        sentiment_builder: Callable[[str], Awaitable[float]] | None = None,
        macro_builder: Callable[[], Awaitable[float]] | None = None,
        fx_builder: Callable[[str], Awaitable[float]] | None = None,
        market_builder: Callable[[str], Awaitable[float]] | None = None,
        models_dir: str | Path = "/app/models",
    ) -> None:
        self._prices = prices_repo
        self._ohlcv = ohlcv_repo
        self._runs = model_runs_repo
        self._sentiment_builder = sentiment_builder
        self._macro_builder = macro_builder
        self._fx_builder = fx_builder
        self._market_builder = market_builder
        self._models_dir = Path(models_dir)

    async def train(
        self,
        ticker: str,
        *,
        period: str = "3y",
        epochs: int = 50,
        sequence_length: int = 5,
        batch_size: int = 16,
    ) -> ModelRunResult:
        ticker = ticker.upper().strip()
        run_id = await self._open_run(ticker, epochs)

        try:
            df = await self._gather_dataset(ticker)
        except Exception as exc:  # noqa: BLE001 - surface as failed run
            return await self._fail(run_id, ticker, exc, rows=0)

        if len(df) < MIN_TRAINING_ROWS:
            return await self._fail(
                run_id,
                ticker,
                ValueError(
                    f"insufficient rows for {ticker}: {len(df)} < {MIN_TRAINING_ROWS}"
                ),
                rows=len(df),
            )

        sentiment = await self._safe_signal(self._sentiment_builder, ticker)
        macro = await self._safe_signal(self._macro_builder, None)
        fx = await self._safe_signal(self._fx_builder, ticker)
        market = await self._safe_signal(self._market_builder, ticker)
        df = self._attach_signal(df, "sentiment", sentiment)
        df = self._attach_signal(df, "macro_score", macro)
        df = self._attach_signal(df, "fx_score", fx)
        df = self._attach_signal(df, "market_signal_score", market)

        try:
            metrics, artifact = self._fit_and_save(
                df, ticker, epochs, sequence_length, batch_size
            )
        except Exception as exc:  # noqa: BLE001
            return await self._fail(run_id, ticker, exc, rows=len(df))

        result = ModelRunResult(
            run_id=run_id,
            ticker=ticker,
            status="done",
            loss=float(metrics.get("loss", 0.0)),
            direction_accuracy=metrics.get("direction_accuracy"),
            artifact_path=str(artifact),
            rows=len(df),
        )
        await self._close_run(result)
        return result

    # --- internals -----------------------------------------------------

    async def _gather_dataset(self, ticker: str) -> pd.DataFrame:
        rows = await self._prices.get_history(ticker)
        if not rows:
            ticker_id = await self._ohlcv.ensure_ticker(ticker)
            rows = await self._ohlcv.list_history(ticker_id, limit=5000)
        else:
            ticker_id = await self._ohlcv.ensure_ticker(ticker)
            await self._ohlcv.upsert_many(ticker_id, rows)

        if not rows:
            raise ValueError(f"no OHLCV available for {ticker}")

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df

    def _fit_and_save(
        self,
        df: pd.DataFrame,
        ticker: str,
        epochs: int,
        sequence_length: int,
        batch_size: int,
    ) -> tuple[dict[str, float], Path]:
        from app.ml.lstm_model import LSTMConfig, LSTMPricePredictor

        cfg = LSTMConfig(
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size,
        )
        predictor = LSTMPricePredictor(config=cfg)
        metrics = predictor.train(df)

        self._models_dir.mkdir(parents=True, exist_ok=True)
        artifact = self._models_dir / f"{ticker}.keras"
        predictor.save(artifact)
        log.info(
            "training.saved",
            extra={"ticker": ticker, "path": str(artifact), "metrics": metrics},
        )
        return metrics, artifact

    async def _safe_signal(
        self,
        builder: Callable[..., Awaitable[float]] | None,
        ticker: str | None,
    ) -> float:
        if builder is None:
            return 0.0
        try:
            if ticker is None:
                return float(await builder())  # type: ignore[misc]
            return float(await builder(ticker))
        except Exception as exc:  # noqa: BLE001
            log.warning("training.signal_failed", extra={"error": str(exc)})
            return 0.0

    @staticmethod
    def _attach_signal(df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
        if column not in df.columns:
            df[column] = float(value)
        return df

    async def _open_run(self, ticker: str, epochs: int) -> int | None:
        if self._runs is None:
            return None
        try:
            return await self._runs.start(ModelRunStart(ticker=ticker, epochs=epochs))
        except Exception as exc:  # noqa: BLE001
            log.warning("training.run_open_failed", extra={"error": str(exc)})
            return None

    async def _close_run(self, result: ModelRunResult) -> None:
        if self._runs is None or result.run_id is None:
            return
        try:
            await self._runs.finish(
                result.run_id,
                ModelRunFinish(
                    status=result.status,
                    loss=result.loss,
                    direction_accuracy=result.direction_accuracy,
                    artifact_path=result.artifact_path,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("training.run_close_failed", extra={"error": str(exc)})

    async def _fail(
        self,
        run_id: int | None,
        ticker: str,
        exc: BaseException,
        rows: int,
    ) -> ModelRunResult:
        log.warning(
            "training.failed",
            extra={"ticker": ticker, "error": str(exc), "rows": rows},
        )
        result = ModelRunResult(
            run_id=run_id,
            ticker=ticker,
            status="failed",
            rows=rows,
            error=str(exc),
        )
        await self._close_run(result)
        return result
