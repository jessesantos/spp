"""Weighted aggregation of per-article sentiment into a single score."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from app.ml.claude_sentiment import SentimentResult


def _as_dict(item: SentimentResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, SentimentResult):
        return item.as_dict()
    return item


def aggregate_sentiment(
    sentiments: Iterable[SentimentResult | dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate sentiment items into a weighted summary.

    Weighting: each article's score contributes ``score * confidence``;
    the final score is the confidence-weighted mean, bounded to ``[-1, 1]``.
    """
    items = [_as_dict(s) for s in sentiments]
    if not items:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "positives": 0,
            "negatives": 0,
            "neutrals": 0,
            "high_impact": [],
        }

    weighted = sum(float(s["score"]) * float(s["confidence"]) for s in items)
    total_conf = sum(float(s["confidence"]) for s in items)
    score = weighted / total_conf if total_conf > 0 else 0.0
    score = max(-1.0, min(1.0, score))

    positives = sum(1 for s in items if float(s["score"]) > 0)
    negatives = sum(1 for s in items if float(s["score"]) < 0)
    neutrals = sum(1 for s in items if float(s["score"]) == 0)

    return {
        "score": score,
        "confidence": total_conf / len(items),
        "positives": positives,
        "negatives": negatives,
        "neutrals": neutrals,
        "high_impact": [s for s in items if str(s.get("impact", "")).lower() == "alto"],
    }
