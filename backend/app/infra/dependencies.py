"""Application-wide dependency wiring (composition root).

Creating concrete clients here keeps ``main.py`` small and lets tests
substitute fakes by overriding the corresponding FastAPI dependency.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import httpx

from app.data.brapi import BrAPIClient
from app.data.news import RSSNewsClient
from app.data.prediction_markets import (
    KalshiClient,
    PolymarketClient,
    PredictionMarketAggregator,
)
from app.data.prices_repository import BrAPIYahooRepository, PriceRepository
from app.data.yahoo import YahooClient
from app.db.session import get_sessionmaker
from app.infra.config import settings
from app.ml.explanation import (
    ClaudeExplanationGenerator,
    ExplanationGenerator,
    HeuristicExplanationGenerator,
)
from app.ml.fx_impact import CurrencyImpactAnalyzer, YahooUsdBrlProvider
from app.ml.training_orchestrator import TrainingOrchestrator
from app.repositories.model_runs_repository import (
    ModelRunsRepo,
    SqlAlchemyModelRunsRepo,
)
from app.repositories.ohlcv_repository import OhlcvRepo, SqlAlchemyOhlcvRepo
from app.repositories.prediction_market_signals_repository import (
    PredictionMarketSignalsRepo,
    SqlAlchemyPredictionMarketSignalsRepo,
)
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
def prediction_market_signals_repo() -> PredictionMarketSignalsRepo:
    return SqlAlchemyPredictionMarketSignalsRepo(sessionmaker=get_sessionmaker())


@lru_cache
def prediction_market_aggregator() -> PredictionMarketAggregator:
    providers = [
        KalshiClient(
            http_client=_http_client(),
            base_url=settings.kalshi_base_url,
            api_key=settings.kalshi_api_key,
        ),
        PolymarketClient(
            http_client=_http_client(),
            base_url=settings.polymarket_base_url,
        ),
    ]
    return PredictionMarketAggregator(providers=providers)


@lru_cache
def currency_impact_analyzer() -> CurrencyImpactAnalyzer:
    yahoo = YahooClient()
    return CurrencyImpactAnalyzer(fx_provider=YahooUsdBrlProvider(yahoo))


@lru_cache
def _sentiment_skill_text() -> str | None:
    path = Path(settings.sentiment_skill_path)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


@lru_cache
def prediction_service() -> PredictionService:
    return PredictionService(
        prices_repo=price_repository(),
        sentiment_service=sentiment_service(),
        predictions_repo=prediction_repo(),
        ohlcv_repo=ohlcv_repo(),
        explanation_generator=_build_explanation_generator(),
        fx_analyzer=currency_impact_analyzer(),
        market_aggregator=(
            prediction_market_aggregator()
            if settings.prediction_markets_enabled
            else None
        ),
        models_dir=settings.models_dir,
        fallback_stub=settings.model_fallback_stub,
    )


def _build_explanation_generator() -> ExplanationGenerator:
    if settings.anthropic_api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            return ClaudeExplanationGenerator(
                client=client,
                model=settings.claude_model,
                system_prompt=_sentiment_skill_text(),
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
    with_fx: bool = True,
    with_markets: bool = True,
    models_dir: str | None = None,
) -> TrainingOrchestrator:
    """Build a fresh :class:`TrainingOrchestrator` for CLI / Celery."""

    sentiment_builder = _make_sentiment_builder() if with_sentiment else None
    macro_builder = _make_macro_builder() if with_macro else None
    fx_builder = _make_fx_builder() if with_fx else None
    market_builder = (
        _make_market_builder()
        if with_markets and settings.prediction_markets_enabled
        else None
    )
    return TrainingOrchestrator(
        prices_repo=price_repository(),
        ohlcv_repo=ohlcv_repo(),
        model_runs_repo=model_runs_repo(),
        sentiment_builder=sentiment_builder,
        macro_builder=macro_builder,
        fx_builder=fx_builder,
        market_builder=market_builder,
        models_dir=models_dir or settings.models_dir,
    )


def _build_analyzer() -> object | None:
    if not settings.anthropic_api_key:
        return None
    try:
        import anthropic

        from app.ml.claude_sentiment import ClaudeSentimentAnalyzer

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return ClaudeSentimentAnalyzer(
            client=client,
            model=settings.claude_model,
            system_prompt=_sentiment_skill_text(),
        )
    except Exception:  # noqa: BLE001 - never crash the app on optional dep
        return None


def _make_sentiment_builder():
    service = sentiment_service()

    async def _builder(ticker: str) -> float:
        score = await service.aggregate(ticker)
        return float(getattr(score, "score", 0.0))

    return _builder


def _make_fx_builder():
    analyzer = currency_impact_analyzer()
    prices = price_repository()

    async def _builder(ticker: str) -> float:
        try:
            history = await prices.get_history(ticker)
        except Exception:  # noqa: BLE001
            history = []
        impact = await analyzer.analyze(ticker, history)
        return float(impact.fx_score)

    return _builder


def _make_market_builder():
    aggregator = prediction_market_aggregator()

    async def _builder(ticker: str) -> float:
        signal = await aggregator.signal_for(ticker)
        return float(signal.score)

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
                    client=client,
                    model=settings.claude_model,
                    system_prompt=_sentiment_skill_text(),
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
