"""Integration tests for /api/predict using the stub fallback path."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_predict_returns_payload_for_known_ticker() -> None:
    with TestClient(app) as client:
        response = client.get("/api/predict/PETR4?days=3")
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "PETR4"
    assert len(body["predictions"]) == 3
    for point in body["predictions"]:
        assert point["direction"] in {"ALTA", "BAIXA", "NEUTRO"}
        assert isinstance(point["predicted_close"], int | float)


def test_predict_rejects_invalid_ticker() -> None:
    with TestClient(app) as client:
        response = client.get("/api/predict/INVALID!!")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "DomainError"


def test_predict_rejects_out_of_range_days() -> None:
    with TestClient(app) as client:
        response = client.get("/api/predict/PETR4?days=999")
    assert response.status_code == 422


def test_predict_attaches_request_id_header() -> None:
    with TestClient(app) as client:
        response = client.get("/api/predict/PETR4")
    assert "x-request-id" in {k.lower() for k in response.headers}


def test_list_tickers_returns_array() -> None:
    with TestClient(app) as client:
        response = client.get("/api/tickers")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert all("ticker" in t for t in payload)
