"""CLI de treinamento do LSTM (thin wrapper sobre :class:`TrainingOrchestrator`).

Uso::

    python -m app.ml.train --ticker PETR4
    python -m app.ml.train --ticker PETR4 --period 2y --epochs 80

A logica real (ingest, sentimento, macro, fit, persistencia de ModelRun)
vive em ``app.ml.training_orchestrator``. Este modulo apenas faz o wiring
para CLI: parse de args, construcao do orchestrator a partir do
composition root e logging do resultado.

Treinamento recomendado em GPU no Windows nativo (ver
``docs/TRAINING_WINDOWS.md``). Dentro do container so roda CPU.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from app.infra.config import settings
from app.ml.training_orchestrator import ModelRunResult

log = logging.getLogger("spp.train")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinar LSTM do SPP")
    parser.add_argument("--ticker", required=True, help="Ex.: PETR4, VALE3")
    parser.add_argument(
        "--period",
        default="3y",
        help="Janela de historico BrAPI (1mo, 3mo, 6mo, 1y, 2y, 3y, 5y). Padrao: 3y",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Pular analise de sentimento (util se nao houver ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--no-macro",
        action="store_true",
        help="Pular contexto macro/geopolitico global",
    )
    parser.add_argument(
        "--output-dir",
        default=settings.models_dir,
        help="Onde salvar o artefato .keras (padrao: config.models_dir)",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    from app.infra.dependencies import build_training_orchestrator

    ticker = args.ticker.upper().strip()
    if not ticker.isalnum() or not (1 <= len(ticker) <= 10):
        log.error("ticker.invalido", extra={"ticker": ticker})
        return 2

    orchestrator = build_training_orchestrator(
        with_sentiment=not args.no_sentiment,
        with_macro=not args.no_macro,
        models_dir=args.output_dir,
    )
    result: ModelRunResult = await orchestrator.train(
        ticker,
        period=args.period,
        epochs=args.epochs,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )
    if result.status == "done":
        print(f"OK: modelo salvo em {result.artifact_path}")
        return 0
    print(f"FALHA: {result.error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
