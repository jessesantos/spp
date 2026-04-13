"""Application-wide dependency wiring (composition root).

Creating concrete clients here keeps ``main.py`` small and lets tests
substitute fakes by overriding the corresponding FastAPI dependency.
"""

from __future__ import annotations

from functools import lru_cache

import httpx

from app.data.brapi import BrAPIClient
from app.data.news import RSSNewsClient
from app.data.prices_repository import BrAPIYahooRepository, PriceRepository
from app.data.yahoo import YahooClient
from app.db.session import get_sessionmaker
from app.infra.config import settings
from app.ml.explanation import (
    ClaudeExplanationGenerator,
    ExplanationGenerator,
    HeuristicExplanationGenerator,
)
from app.ml.training_orchestrator import TrainingOrchestrator
from app.repositories.model_runs_repository import (
    ModelRunsRepo,
    SqlAlchemyModelRunsRepo,
)
from app.repositories.ohlcv_repository import OhlcvRepo, SqlAlchemyOhlcvRepo
from app.repositories.predictions_repository import (
    PredictionsRepo,
    SqlAlchemyPredictionsRepo,
)
from app.services.prediction_service import (
    PredictionService,
    ReconciliationService,
)
from app.services.sentiment_service import SentimentService


@lru_cache
def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=10.0)


@lru_cache
def price_repository() -> PriceRepository:
    brapi = BrAPIClient(http_client=_http_client(), base_url=settings.brapi_base_url)
    yahoo = YahooClient()
    return BrAPIYahooRepository(brapi=brapi, yahoo=yahoo)


@lru_cache
def sentiment_service() -> SentimentService:
    news = RSSNewsClient(feeds=tuple(settings.rss_feeds))
    analyzer = _build_analyzer()
    return SentimentService(
        news_source=news if analyzer else None,
        analyzer=analyzer,
        fallback_stub=True,
    )


@lru_cache
def prediction_repo() -> PredictionsRepo:
    return SqlAlchemyPredictionsRepo(sessionmaker=get_sessionmaker())


@lru_cache
def ohlcv_repo() -> OhlcvRepo:
    return SqlAlchemyOhlcvRepo(sessionmaker=get_sessionmaker())


@lru_cache
def model_runs_repo() -> ModelRunsRepo:
    return SqlAlchemyModelRunsRepo(sessionmaker=get_sessionmaker())


@lru_cache
def prediction_service() -> PredictionService:
    return PredictionService(
        prices_repo=price_repository(),
        sentiment_service=sentiment_service(),
        predictions_repo=prediction_repo(),
        ohlcv_repo=ohlcv_repo(),
        explanation_generator=_build_explanation_generator(),
        models_dir=settings.models_dir,
        fallback_stub=settings.model_fallback_stub,
    )


def _build_explanation_generator() -> ExplanationGenerator:
    if settings.anthropic_api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            return ClaudeExplanationGenerator(
                client=client, model=settings.claude_model
            )
        except Exception:  # noqa: BLE001 - never crash on optional dep
            return HeuristicExplanationGenerator()
    return HeuristicExplanationGenerator()


@lru_cache
def reconciliation_service() -> ReconciliationService:
    return ReconciliationService(
        predictions_repo=prediction_repo(),
        prices_repo=price_repository(),
    )


def build_training_orchestrator(
    *,
    with_sentiment: bool = True,
    with_macro: bool = True,
    models_dir: str | None = None,
) -> TrainingOrchestrator:
    """Build a fresh :class:`TrainingOrchestrator` for CLI / Celery."""

    sentiment_builder = _make_sentiment_builder() if with_sentiment else None
    macro_builder = _make_macro_builder() if with_macro else None
    return TrainingOrchestrator(
        prices_repo=price_repository(),
        ohlcv_repo=ohlcv_repo(),
        model_runs_repo=model_runs_repo(),
        sentiment_builder=sentiment_builder,
        macro_builder=macro_builder,
        models_dir=models_dir or settings.models_dir,
    )


def _build_analyzer() -> object | None:
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic

        from app.ml.claude_sentiment import ClaudeSentimentAnalyzer

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return ClaudeSentimentAnalyzer(client=client, model=settings.claude_model)
    except Exception:  # noqa: BLE001 - never crash the app on optional dep
        return None


def _make_sentiment_builder():
    service = sentiment_service()

    async def _builder(ticker: str) -> float:
        score = await service.aggregate(ticker)
        return float(getattr(score, "score", 0.0))

    return _builder


def _make_macro_builder():
    async def _builder() -> float:
        from app.ml.claude_sentiment import ClaudeSentimentAnalyzer
        from app.ml.macro_context import (
            ClaudeMacroAnalyzer,
            MacroContextBuilder,
            RSSMacroNewsSource,
        )

        inner: ClaudeSentimentAnalyzer | None = None
        if settings.anthropic_api_key:
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                inner = ClaudeSentimentAnalyzer(
                    client=client, model=settings.claude_model
                )
            except Exception:  # noqa: BLE001
                inner = None

        builder = MacroContextBuilder(
            source=RSSMacroNewsSource(),
            analyzer=ClaudeMacroAnalyzer(inner=inner),
        )
        ctx = await builder.build(limit=30)
        return float(ctx.score)

    return _builder
