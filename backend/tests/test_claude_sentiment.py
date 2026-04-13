"""Tests for ClaudeSentimentAnalyzer with a fully mocked Anthropic client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.ml.aggregator import aggregate_sentiment
from app.ml.claude_sentiment import ClaudeSentimentAnalyzer, SentimentResult


@dataclass
class _FakeBlock:
    text: str


@dataclass
class _FakeResponse:
    content: list[_FakeBlock]


class _FakeMessages:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_kwargs = kwargs
        return _FakeResponse(content=[_FakeBlock(text=self._reply)])


class _FakeClient:
    def __init__(self, reply: str) -> None:
        self.messages = _FakeMessages(reply)


def test_analyze_parses_valid_json_payload() -> None:
    reply = (
        '{"score": 0.7, "confidence": 0.8, "reasoning": "boa notícia",'
        ' "impact": "alto", "keywords": ["lucro"]}'
    )
    analyzer = ClaudeSentimentAnalyzer(client=_FakeClient(reply))
    result = analyzer.analyze({"title": "Petrobras lucra"}, "PETR4")
    assert isinstance(result, SentimentResult)
    assert result.score == pytest.approx(0.7)
    assert result.confidence == pytest.approx(0.8)
    assert result.impact == "alto"
    assert "lucro" in result.keywords


def test_analyze_strips_markdown_fences() -> None:
    reply = '```json\n{"score": -1, "confidence": 0.9}\n```'
    analyzer = ClaudeSentimentAnalyzer(client=_FakeClient(reply))
    result = analyzer.analyze({"title": "x", "summary": "y"}, "VALE3")
    assert result.score == -1.0
    assert result.confidence == pytest.approx(0.9)


def test_analyze_returns_neutral_on_parse_failure() -> None:
    analyzer = ClaudeSentimentAnalyzer(client=_FakeClient("not json at all"))
    result = analyzer.analyze({"title": "oops"}, "ITUB4")
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.reasoning == "parse_error"


def test_analyze_clips_out_of_range_values() -> None:
    reply = '{"score": 5, "confidence": 42}'
    analyzer = ClaudeSentimentAnalyzer(client=_FakeClient(reply))
    result = analyzer.analyze({"title": "x"}, "BBDC4")
    assert result.score == 1.0
    assert result.confidence == 1.0


def test_prompt_injects_article_in_tagged_delimiters() -> None:
    client = _FakeClient('{"score": 0, "confidence": 0}')
    analyzer = ClaudeSentimentAnalyzer(client=client)
    analyzer.analyze({"title": "hello"}, "PETR4")
    sent_prompt = client.messages.last_kwargs
    assert sent_prompt is not None
    content = sent_prompt["messages"][0]["content"]
    assert "<article>" in content
    assert "</article>" in content
    # prompt-injection guard phrasing
    assert "DADO" in content or "dado" in content.lower()


def test_aggregate_sentiment_weights_by_confidence() -> None:
    sents = [
        SentimentResult(score=1.0, confidence=1.0, impact="alto"),
        SentimentResult(score=-1.0, confidence=0.1, impact="baixo"),
    ]
    summary = aggregate_sentiment(sents)
    assert summary["score"] > 0  # the confident positive dominates
    assert summary["positives"] == 1
    assert summary["negatives"] == 1
    assert summary["neutrals"] == 0
    assert len(summary["high_impact"]) == 1


def test_aggregate_sentiment_empty_is_zero() -> None:
    summary = aggregate_sentiment([])
    assert summary["score"] == 0.0
    assert summary["confidence"] == 0.0
