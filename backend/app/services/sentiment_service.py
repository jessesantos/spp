"""Sentiment service orchestrating RSS fetch + Claude analysis + aggregation."""

from __future__ import annotations

from typing import Any, Protocol

from app.api.schemas import SentimentScore
from app.ml.aggregator import aggregate_sentiment


class _NewsSource(Protocol):
    async def fetch(self, ticker: str) -> list[dict[str, Any]]: ...


class _Analyzer(Protocol):
    def analyze_batch(
        self, articles: list[dict[str, Any]], ticker: str
    ) -> list[Any]: ...


class SentimentService:
    """Aggregate per-ticker sentiment using an injected news source + analyzer.

    Both dependencies are optional to keep a working stub when the demo
    stack runs without an Anthropic API key.
    """

    def __init__(
        self,
        *,
        news_source: _NewsSource | None = None,
        analyzer: _Analyzer | None = None,
        fallback_stub: bool = True,
    ) -> None:
        self._news = news_source
        self._analyzer = analyzer
        self._fallback_stub = fallback_stub

    async def aggregate(self, ticker: str) -> SentimentScore:
        ticker = ticker.upper().strip()
        if self._news is None or self._analyzer is None:
            if self._fallback_stub:
                return self._stub_score()
            raise RuntimeError("sentiment pipeline is not wired")

        try:
            articles = await self._news.fetch(ticker)
        except Exception:  # noqa: BLE001
            articles = []

        if not articles:
            return self._stub_score() if self._fallback_stub else SentimentScore(
                score=0.0, confidence=0.0
            )

        results = self._analyzer.analyze_batch(articles, ticker)
        summary = aggregate_sentiment(results)
        return SentimentScore(
            score=float(summary["score"]),
            confidence=float(summary["confidence"]),
            positives=int(summary["positives"]),
            negatives=int(summary["negatives"]),
            neutrals=int(summary["neutrals"]),
        )

    @staticmethod
    def _stub_score() -> SentimentScore:
        return SentimentScore(
            score=0.15, confidence=0.6, positives=4, negatives=2, neutrals=4
        )
