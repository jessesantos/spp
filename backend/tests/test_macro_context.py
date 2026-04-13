"""Testes da camada de contexto macro global."""

from __future__ import annotations

from typing import Any

import pytest

from app.ml.claude_sentiment import SentimentResult
from app.ml.macro_context import (
    ClaudeMacroAnalyzer,
    MacroContextBuilder,
    _heuristic_score,
)


class _FakeSource:
    def __init__(self, articles: list[dict[str, Any]]) -> None:
        self._articles = articles

    async def fetch_macro(self, limit: int = 40) -> list[dict[str, Any]]:
        return list(self._articles[:limit])


class _FakeAnalyzer:
    def __init__(self, mapping: dict[str, SentimentResult]) -> None:
        self._mapping = mapping

    def analyze_macro(self, article: dict[str, Any]) -> SentimentResult:
        key = article.get("title", "")
        return self._mapping.get(key, SentimentResult(score=0.0, confidence=0.1))


def test_heuristic_marks_war_as_negative() -> None:
    article = {"title": "Russia-Ukraine war escalates", "summary": "sanctions imposed"}
    result = _heuristic_score(article)
    assert result.score < 0
    assert result.confidence > 0.2


def test_heuristic_marks_rate_cut_as_positive() -> None:
    article = {"title": "Fed rate cut sparks rally", "summary": "growth ahead"}
    result = _heuristic_score(article)
    assert result.score > 0


def test_heuristic_neutral_when_no_keywords() -> None:
    article = {"title": "Local weather report", "summary": "nothing financial"}
    result = _heuristic_score(article)
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_builder_aggregates_mixed_sentiment() -> None:
    articles = [
        {"title": "war", "summary": ""},
        {"title": "deal", "summary": ""},
        {"title": "crisis", "summary": ""},
    ]
    mapping = {
        "war": SentimentResult(score=-0.8, confidence=0.9, impact="alto", title="war"),
        "deal": SentimentResult(score=0.6, confidence=0.7, title="deal"),
        "crisis": SentimentResult(score=-0.5, confidence=0.8, title="crisis"),
    }
    builder = MacroContextBuilder(_FakeSource(articles), _FakeAnalyzer(mapping))
    ctx = await builder.build(limit=10)
    assert ctx.count == 3
    assert ctx.score < 0
    assert ctx.negatives == 2
    assert ctx.positives == 1
    assert "war" in ctx.high_impact_titles


@pytest.mark.asyncio
async def test_builder_empty_returns_neutral() -> None:
    builder = MacroContextBuilder(_FakeSource([]), _FakeAnalyzer({}))
    ctx = await builder.build()
    assert ctx.count == 0
    assert ctx.score == 0.0


def test_claude_macro_analyzer_falls_back_to_heuristic_without_inner() -> None:
    analyzer = ClaudeMacroAnalyzer(inner=None)
    result = analyzer.analyze_macro({"title": "war escalates", "summary": ""})
    assert result.score < 0
