"""Tests for the explanation generators."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from app.ml.explanation import (
    ClaudeExplanationGenerator,
    ExplanationInput,
    HeuristicExplanationGenerator,
    MAX_WORDS,
    MIN_WORDS,
    _clamp_words,
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _payload(direction: str = "ALTA") -> ExplanationInput:
    return ExplanationInput(
        ticker="PETR4",
        horizon_label="Amanha",
        horizon_days=1,
        base_close=30.0,
        predicted_close=30.90,
        predicted_pct=3.0,
        direction=direction,
        sentiment_score=0.25,
        sentiment_positives=4,
        sentiment_negatives=1,
        sentiment_neutrals=2,
        macro_score=0.1,
        macro_top_keywords=("copom", "juros"),
    )


def test_heuristic_returns_within_word_bounds() -> None:
    generator = HeuristicExplanationGenerator()
    for direction in ("ALTA", "BAIXA", "NEUTRO"):
        text = generator.generate(_payload(direction))
        count = _word_count(text)
        assert MIN_WORDS <= count <= MAX_WORDS, (
            f"{direction}: got {count} words"
        )


def test_heuristic_directions_produce_distinct_openings() -> None:
    generator = HeuristicExplanationGenerator()
    alta = generator.generate(_payload("ALTA"))
    baixa = generator.generate(_payload("BAIXA"))
    neutro = generator.generate(_payload("NEUTRO"))
    assert "tendencia de alta" in alta
    assert "tendencia de baixa" in baixa
    assert "nao detecta tendencia dominante" in neutro
    assert alta != baixa != neutro


def test_clamp_words_truncates_and_preserves_punctuation() -> None:
    long_text = " ".join(["palavra"] * (MAX_WORDS + 50))
    clamped = _clamp_words(long_text)
    assert _word_count(clamped) <= MAX_WORDS
    assert clamped.endswith((".", "!", "?"))


def test_clamp_words_pads_short_text_up_to_minimum() -> None:
    short = "texto curto apenas."
    clamped = _clamp_words(short)
    assert _word_count(clamped) >= MIN_WORDS
    assert clamped.endswith((".", "!", "?"))


class _FakeClient:
    def __init__(self, text: str, *, raise_exc: bool = False) -> None:
        self._text = text
        self._raise = raise_exc
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_: Any) -> Any:
        if self._raise:
            raise RuntimeError("api down")
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


def test_claude_generator_returns_llm_text_when_long_enough() -> None:
    long_text = " ".join(["palavra"] * (MIN_WORDS + 5)) + "."
    client = _FakeClient(long_text)
    generator = ClaudeExplanationGenerator(client=client)
    result = generator.generate(_payload())
    assert _word_count(result) >= MIN_WORDS
    assert "palavra" in result


def test_claude_generator_falls_back_when_too_short() -> None:
    client = _FakeClient("muito curto.")
    generator = ClaudeExplanationGenerator(client=client)
    result = generator.generate(_payload("BAIXA"))
    assert _word_count(result) >= MIN_WORDS
    assert "tendencia de baixa" in result


def test_claude_generator_falls_back_on_exception() -> None:
    client = _FakeClient("irrelevant", raise_exc=True)
    generator = ClaudeExplanationGenerator(client=client)
    result = generator.generate(_payload("NEUTRO"))
    assert _word_count(result) >= MIN_WORDS
    assert "nao detecta tendencia dominante" in result


@pytest.mark.parametrize("horizon_days", [1, 7, 30])
def test_heuristic_adjusts_block_per_horizon(horizon_days: int) -> None:
    payload = ExplanationInput(
        ticker="VALE3",
        horizon_label=f"+{horizon_days} dias",
        horizon_days=horizon_days,
        base_close=50.0,
        predicted_close=51.0,
        predicted_pct=2.0,
        direction="ALTA",
    )
    text = HeuristicExplanationGenerator().generate(payload)
    assert _word_count(text) >= MIN_WORDS
