"""Regression tests for the v3.1 additions: EWMA volatility + SKILL injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.ml.claude_sentiment import ClaudeSentimentAnalyzer
from app.ml.explanation import ClaudeExplanationGenerator, ExplanationInput
from app.ml.features import add_conditional_volatility, build_features


def _synth_df(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = np.cumsum(rng.normal(0, 1, rows)) + 50
    return pd.DataFrame(
        {
            "close": close,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "volume": rng.integers(1_000_000, 5_000_000, rows),
        }
    )


def test_add_conditional_volatility_returns_finite_non_negative() -> None:
    out = add_conditional_volatility(_synth_df())
    assert "cond_vol" in out.columns
    vals = out["cond_vol"].to_numpy()
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0.0)


def test_conditional_volatility_clusters_after_shock() -> None:
    df = _synth_df(80).copy()
    # inject a 10% shock at the midpoint - later vol should be above early vol
    df.loc[40, "close"] = df.loc[39, "close"] * 1.10
    out = add_conditional_volatility(df)
    early = out["cond_vol"].iloc[5:20].mean()
    late = out["cond_vol"].iloc[42:60].mean()
    assert late >= early


def test_build_features_includes_cond_vol() -> None:
    df = build_features(_synth_df())
    assert "cond_vol" in df.columns


def test_skill_markdown_exists_and_has_austrian_content() -> None:
    skill_path = Path(__file__).resolve().parents[1] / "app" / "ml" / "SKILL.md"
    assert skill_path.exists(), "SKILL.md missing from backend/app/ml/"
    text = skill_path.read_text(encoding="utf-8")
    for phrase in (
        "Escola Austriaca",
        "Mises",
        "malinvestment",
        "Graham",
        "ROIC",
        "reflexividade",
    ):
        assert phrase in text, f"SKILL.md missing mention of {phrase}"


class _RecordingAnthropic:
    """Captures what gets passed to messages.create()."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

        class _Messages:
            def __init__(inner_self) -> None:  # noqa: N805
                pass

            def create(inner_self, **kwargs: Any) -> Any:  # noqa: N805
                self.last_kwargs = kwargs
                return _StubResp(
                    '{"score": 0.0, "confidence": 0.1, "reasoning": "x"}'
                )

        self.messages = _Messages()


class _StubResp:
    def __init__(self, text: str) -> None:
        class _Block:
            pass

        block = _Block()
        block.text = text  # type: ignore[attr-defined]
        self.content = [block]


def test_claude_sentiment_includes_system_prompt_when_skill_provided() -> None:
    client = _RecordingAnthropic()
    analyzer = ClaudeSentimentAnalyzer(
        client=client,
        system_prompt="SKILL: analyse with Austrian school",
    )
    analyzer.analyze({"title": "news", "summary": "x"}, "PETR4")
    assert client.last_kwargs is not None
    assert client.last_kwargs.get("system") == "SKILL: analyse with Austrian school"


def test_claude_sentiment_omits_system_when_no_skill() -> None:
    client = _RecordingAnthropic()
    analyzer = ClaudeSentimentAnalyzer(client=client)
    analyzer.analyze({"title": "news", "summary": "x"}, "PETR4")
    assert client.last_kwargs is not None
    assert "system" not in client.last_kwargs


def test_claude_explanation_includes_system_prompt_when_skill_provided() -> None:
    client = _RecordingAnthropic()

    def _stub_create(**kwargs: Any) -> Any:
        client.last_kwargs = kwargs
        # return enough words to satisfy 100+ min
        words = " ".join(["palavra"] * 120)
        return _StubResp(words)

    client.messages.create = _stub_create  # type: ignore[assignment]

    generator = ClaudeExplanationGenerator(
        client=client,
        system_prompt="SKILL: explain with Austrian school",
    )
    payload = ExplanationInput(
        ticker="PETR4",
        horizon_label="Amanha",
        horizon_days=1,
        base_close=30.0,
        predicted_close=30.5,
        predicted_pct=1.67,
        direction="ALTA",
    )
    generator.generate(payload)
    assert client.last_kwargs is not None
    assert client.last_kwargs.get("system") == "SKILL: explain with Austrian school"
