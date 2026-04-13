"""Unit tests for Kalshi/Polymarket clients and aggregator.

Uses fake httpx clients to avoid hitting the live APIs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.data.prediction_markets import (
    KalshiClient,
    PolymarketClient,
    PredictionMarket,
    PredictionMarketAggregator,
    _parse_kalshi,
    _parse_polymarket,
    _score_markets,
)


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "err",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )

    def json(self) -> Any:
        return self._payload


class _FakeHttp:
    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.last_url: str | None = None
        self.last_params: dict[str, Any] | None = None

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> _FakeResponse:
        self.last_url = url
        self.last_params = params
        return _FakeResponse(self._payload)


class _ErrorHttp:
    async def get(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        import httpx

        raise httpx.ConnectError("boom")


def test_parse_kalshi_uses_last_price() -> None:
    raw = {
        "ticker": "FED-2026-CUT",
        "title": "Fed cuts rate in May",
        "last_price_dollars": "0.62",
        "volume_fp": "12500",
        "close_time": "2026-05-07T15:00:00Z",
    }
    market = _parse_kalshi(raw)
    assert market is not None
    assert market.source == "kalshi"
    assert market.market_id == "FED-2026-CUT"
    assert market.probability_yes == pytest.approx(0.62)
    assert market.volume == 12500.0


def test_parse_kalshi_falls_back_to_midpoint() -> None:
    raw = {
        "ticker": "X",
        "title": "Question",
        "yes_bid_dollars": "0.4",
        "yes_ask_dollars": "0.6",
    }
    market = _parse_kalshi(raw)
    assert market is not None
    assert market.probability_yes == pytest.approx(0.5)


def test_parse_kalshi_rejects_missing_fields() -> None:
    assert _parse_kalshi({}) is None
    assert _parse_kalshi({"ticker": "X"}) is None


def test_parse_polymarket_uses_last_trade() -> None:
    raw = {
        "id": "0xabc",
        "question": "Brent > $100 by Q3",
        "lastTradePrice": "0.37",
        "volume": "80000",
        "endDate": "2026-09-30T00:00:00Z",
        "slug": "brent-above-100",
    }
    market = _parse_polymarket(raw)
    assert market is not None
    assert market.source == "polymarket"
    assert market.probability_yes == pytest.approx(0.37)
    assert market.url is not None and "brent-above-100" in market.url


def test_parse_polymarket_uses_outcome_prices_string() -> None:
    raw = {
        "id": "0xdef",
        "question": "Recession in 2026",
        "outcomePrices": json.dumps(["0.28", "0.72"]),
    }
    market = _parse_polymarket(raw)
    assert market is not None
    assert market.probability_yes == pytest.approx(0.28)


@pytest.mark.asyncio
async def test_kalshi_client_filters_by_keywords() -> None:
    payload = {
        "markets": [
            {
                "ticker": "FED-MAY",
                "title": "Fed cuts rate in May",
                "last_price_dollars": "0.65",
                "volume_fp": "1000",
            },
            {
                "ticker": "OSCAR-2026",
                "title": "Best picture at Oscars",
                "last_price_dollars": "0.5",
                "volume_fp": "100",
            },
        ]
    }
    client = KalshiClient(http_client=_FakeHttp(payload))
    markets = await client.fetch_markets(keywords=("fed", "rate"))
    assert len(markets) == 1
    assert markets[0].market_id == "FED-MAY"


@pytest.mark.asyncio
async def test_kalshi_client_degrades_on_network_error() -> None:
    client = KalshiClient(http_client=_ErrorHttp())
    markets = await client.fetch_markets(keywords=("fed",))
    assert markets == []


@pytest.mark.asyncio
async def test_polymarket_client_handles_list_root_payload() -> None:
    payload = [
        {
            "id": "a",
            "question": "Oil rally above 100",
            "lastTradePrice": "0.33",
            "volume": "5000",
        }
    ]
    client = PolymarketClient(http_client=_FakeHttp(payload))
    markets = await client.fetch_markets(keywords=("oil",))
    assert len(markets) == 1


@pytest.mark.asyncio
async def test_aggregator_signal_for_petr4_uses_oil_topics() -> None:
    oil_bullish = [
        PredictionMarket(
            source="polymarket",
            market_id="a",
            question="Oil surges above $100 brent",
            probability_yes=0.7,
            volume=10_000,
            close_time=None,
        )
    ]

    class _StubProvider:
        async def fetch_markets(
            self, keywords: tuple[str, ...], limit: int = 20
        ) -> list[PredictionMarket]:
            return oil_bullish

    aggregator = PredictionMarketAggregator(providers=[_StubProvider()])
    signal = await aggregator.signal_for("PETR4")
    assert signal.score > 0
    assert signal.confidence > 0
    assert len(signal.markets) == 1
    assert signal.topics


@pytest.mark.asyncio
async def test_aggregator_empty_when_no_markets() -> None:
    class _EmptyProvider:
        async def fetch_markets(
            self, keywords: tuple[str, ...], limit: int = 20
        ) -> list[PredictionMarket]:
            return []

    aggregator = PredictionMarketAggregator(providers=[_EmptyProvider()])
    signal = await aggregator.signal_for("PETR4")
    assert signal.score == 0.0
    assert signal.confidence == 0.0
    assert signal.markets == []


def test_score_markets_bounded() -> None:
    markets = [
        PredictionMarket(
            source="kalshi",
            market_id="a",
            question="recession 2026",
            probability_yes=0.9,
            volume=1000,
            close_time=None,
        ),
        PredictionMarket(
            source="kalshi",
            market_id="b",
            question="oil surge brent",
            probability_yes=0.1,
            volume=1000,
            close_time=None,
        ),
    ]
    topics = [(1, ("oil", "brent")), (-1, ("recession",))]
    score, matched = _score_markets(markets, topics)
    assert -1.0 <= score <= 1.0
    assert matched
