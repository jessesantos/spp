"""Integration tests for /api/sentiment using the stub fallback path."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_sentiment_returns_valid_score() -> None:
    with TestClient(app) as client:
        response = client.get("/api/sentiment/PETR4")
    assert response.status_code == 200
    body = response.json()
    assert -1.0 <= body["score"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0


def test_sentiment_rejects_invalid_ticker() -> None:
    with TestClient(app) as client:
        response = client.get("/api/sentiment/!!!")
    assert response.status_code == 400
