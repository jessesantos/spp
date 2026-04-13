"""Clients para mercados de previsao (Kalshi, Polymarket) e agregacao.

Mercados de previsao precificam probabilidade de eventos (ex.: "Fed cuts
rates in May?", "Brent > $100 by Q3?") com capital real em jogo. A
ponderacao dessas probabilidades entra como feature ``market_signal_score``
no LSTM.

Arquitetura (SOLID):

- ``PredictionMarketProvider`` (Protocol) - contrato comum de leitura.
- ``KalshiClient`` - API publica de eventos macroeconomicos EUA.
- ``PolymarketClient`` - Gamma API publica (crypto/macro/geopolitica).
- ``PredictionMarketAggregator`` - filtra mercados relevantes para o
  ticker, cruza com mapeamento de topicos B3 e produz
  ``MarketSignal(ticker, score, details)``.

Seguranca:

- Zero autenticacao na leitura publica; ``KALSHI_API_KEY`` opcional
  (aumenta rate-limit, nao e requerido).
- SSRF: URLs base fixas por config; ticker / query parameters sao
  sanitizados via regex allowlist.
- Timeouts explicitos em todas as chamadas.
- Erros sempre degradam para lista vazia (nunca derrubam predict).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx


DEFAULT_KALSHI_BASE: str = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_POLYMARKET_BASE: str = "https://gamma-api.polymarket.com"

_SAFE_QUERY = re.compile(r"[^A-Za-z0-9\- ]")


def _safe_query(value: str) -> str:
    """Remove caracteres que poderiam escapar de uma query parameter."""
    return _SAFE_QUERY.sub("", str(value))[:64]


# Mapeamento ticker -> topicos/keywords relevantes em mercados de previsao.
# Cada entrada tem (sign, keywords) - sign orienta como o "yes" do mercado
# deve ser traduzido em sinal de alta/baixa para o ticker.
#
# Exemplo: "Fed cuts rate" -> bullish para risk assets -> PETR4/VALE3 alta
# (sinal +1). "US recession 2026" -> bearish -> VALE3 (exportadora) baixa.
TICKER_MARKET_TOPICS: dict[str, list[tuple[int, tuple[str, ...]]]] = {
    "PETR4": [
        (1, ("oil", "brent", "opec", "petroleum", "crude")),
        (-1, ("recession",)),
    ],
    "PETR3": [
        (1, ("oil", "brent", "opec", "petroleum", "crude")),
        (-1, ("recession",)),
    ],
    "VALE3": [
        (1, ("iron ore", "china gdp", "china growth", "commodity")),
        (-1, ("recession", "china recession")),
    ],
    "ITUB4": [
        (1, ("rate cut", "fed cut", "selic cut")),
        (-1, ("recession", "default", "banking crisis")),
    ],
    "BBDC4": [
        (1, ("rate cut", "fed cut", "selic cut")),
        (-1, ("recession", "banking crisis")),
    ],
    "BBAS3": [
        (1, ("rate cut", "fed cut", "selic cut")),
        (-1, ("recession",)),
    ],
    "MGLU3": [
        (1, ("rate cut", "selic cut", "consumer")),
        (-1, ("recession", "inflation")),
    ],
    "AMER3": [
        (1, ("rate cut", "consumer")),
        (-1, ("recession", "inflation")),
    ],
    "ELET3": [
        (1, ("rate cut",)),
        (-1, ("inflation", "drought")),
    ],
}

DEFAULT_TOPICS: list[tuple[int, tuple[str, ...]]] = [
    (1, ("rate cut", "fed cut")),
    (-1, ("recession", "war", "default")),
]


@dataclass(frozen=True)
class PredictionMarket:
    """Snapshot de um mercado de previsao em dado momento."""

    source: str  # "kalshi" | "polymarket"
    market_id: str
    question: str
    probability_yes: float  # [0, 1]
    volume: float
    close_time: datetime | None
    url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "market_id": self.market_id,
            "question": self.question,
            "probability_yes": self.probability_yes,
            "volume": self.volume,
            "close_time": (
                self.close_time.isoformat() if self.close_time is not None else None
            ),
            "url": self.url,
        }


@dataclass(frozen=True)
class MarketSignal:
    """Sinal agregado para um ticker, derivado de um conjunto de mercados."""

    ticker: str
    score: float  # [-1, +1]
    confidence: float  # [0, 1], cresce com volume / numero de mercados
    markets: list[PredictionMarket] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "score": self.score,
            "confidence": self.confidence,
            "markets": [m.as_dict() for m in self.markets],
            "topics": list(self.topics),
        }


class PredictionMarketProvider(Protocol):
    """Contrato de leitura de mercados de previsao."""

    async def fetch_markets(
        self, keywords: tuple[str, ...], limit: int = 20
    ) -> list[PredictionMarket]: ...


class KalshiClient:
    """Cliente da API publica do Kalshi (leitura)."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        base_url: str = DEFAULT_KALSHI_BASE,
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def fetch_markets(
        self, keywords: tuple[str, ...], limit: int = 20
    ) -> list[PredictionMarket]:
        url = f"{self._base_url}/markets"
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            resp = await self._http.get(
                url,
                params={"limit": min(int(limit), 100), "status": "open"},
                headers=headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return []

        markets_raw = data.get("markets") or []
        out: list[PredictionMarket] = []
        for raw in markets_raw:
            snapshot = _parse_kalshi(raw)
            if snapshot is None:
                continue
            if _matches_keywords(snapshot.question, keywords):
                out.append(snapshot)
        return out


class PolymarketClient:
    """Cliente do endpoint publico Gamma do Polymarket."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        base_url: str = DEFAULT_POLYMARKET_BASE,
        timeout: float = 10.0,
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def fetch_markets(
        self, keywords: tuple[str, ...], limit: int = 20
    ) -> list[PredictionMarket]:
        url = f"{self._base_url}/markets"
        try:
            resp = await self._http.get(
                url,
                params={
                    "limit": min(int(limit), 100),
                    "active": "true",
                    "closed": "false",
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return []

        markets_raw = data if isinstance(data, list) else data.get("markets", [])
        out: list[PredictionMarket] = []
        for raw in markets_raw:
            snapshot = _parse_polymarket(raw)
            if snapshot is None:
                continue
            if _matches_keywords(snapshot.question, keywords):
                out.append(snapshot)
        return out


class PredictionMarketAggregator:
    """Filtra mercados por topicos do ticker e produz ``MarketSignal``."""

    def __init__(self, providers: list[PredictionMarketProvider]) -> None:
        self._providers = providers

    async def signal_for(
        self,
        ticker: str,
        topics: list[tuple[int, tuple[str, ...]]] | None = None,
        per_provider_limit: int = 30,
    ) -> MarketSignal:
        ticker = ticker.upper().strip()
        topics = topics or TICKER_MARKET_TOPICS.get(ticker, DEFAULT_TOPICS)

        all_keywords: list[str] = []
        for _, kws in topics:
            all_keywords.extend(kws)
        keyword_tuple = tuple(all_keywords)

        collected: list[PredictionMarket] = []
        for provider in self._providers:
            try:
                found = await provider.fetch_markets(
                    keyword_tuple, limit=per_provider_limit
                )
            except Exception:  # noqa: BLE001 - nunca quebrar predict
                found = []
            collected.extend(found)

        if not collected:
            return MarketSignal(ticker=ticker, score=0.0, confidence=0.0)

        score, matched_topics = _score_markets(collected, topics)
        confidence = _confidence_for(collected)
        return MarketSignal(
            ticker=ticker,
            score=round(score, 4),
            confidence=round(confidence, 4),
            markets=collected,
            topics=sorted(set(matched_topics)),
        )


# --- parsers --------------------------------------------------------


def _parse_kalshi(raw: dict[str, Any]) -> PredictionMarket | None:
    ticker = raw.get("ticker")
    title = raw.get("title") or raw.get("subtitle")
    if not ticker or not title:
        return None
    last_price = _to_float(raw.get("last_price_dollars") or raw.get("last_price"))
    if last_price is None:
        bid = _to_float(raw.get("yes_bid_dollars") or raw.get("yes_bid"))
        ask = _to_float(raw.get("yes_ask_dollars") or raw.get("yes_ask"))
        if bid is not None and ask is not None:
            last_price = (bid + ask) / 2.0
    if last_price is None:
        return None
    probability = _clip01(_dollars_to_probability(last_price))
    volume = _to_float(raw.get("volume_fp") or raw.get("volume") or 0.0) or 0.0
    close_time = _parse_datetime(raw.get("close_time") or raw.get("expiration_time"))
    return PredictionMarket(
        source="kalshi",
        market_id=str(ticker),
        question=str(title),
        probability_yes=probability,
        volume=float(volume),
        close_time=close_time,
        url=f"https://kalshi.com/markets/{ticker}",
    )


def _parse_polymarket(raw: dict[str, Any]) -> PredictionMarket | None:
    market_id = raw.get("id") or raw.get("conditionId")
    question = raw.get("question")
    if not market_id or not question:
        return None
    price = _to_float(raw.get("lastTradePrice"))
    if price is None:
        prices = raw.get("outcomePrices")
        if isinstance(prices, str):
            try:
                import json as _json

                prices = _json.loads(prices)
            except (ValueError, TypeError):
                prices = None
        if isinstance(prices, list) and prices:
            price = _to_float(prices[0])
    if price is None:
        return None
    probability = _clip01(price)
    volume = _to_float(raw.get("volume") or 0.0) or 0.0
    close_time = _parse_datetime(raw.get("endDate") or raw.get("end_date_iso"))
    slug = raw.get("slug")
    url = f"https://polymarket.com/event/{slug}" if slug else None
    return PredictionMarket(
        source="polymarket",
        market_id=str(market_id),
        question=str(question),
        probability_yes=probability,
        volume=float(volume),
        close_time=close_time,
        url=url,
    )


# --- scoring --------------------------------------------------------


def _matches_keywords(text: str, keywords: tuple[str, ...]) -> bool:
    if not keywords:
        return True
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def _score_markets(
    markets: list[PredictionMarket],
    topics: list[tuple[int, tuple[str, ...]]],
) -> tuple[float, list[str]]:
    """Mapeia probabilidade "yes" de cada mercado para um sinal -1..+1 no ticker.

    Regra: probabilidade ``p`` e convertida em sinal centrado em zero
    ``(2 * p - 1)`` no intervalo [-1, +1]. Multiplicado pelo ``sign`` do
    topico (bullish ou bearish para o ticker). Resultado final e media
    ponderada por volume entre mercados correspondidos.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    matched_topics: list[str] = []
    for market in markets:
        for sign, keywords in topics:
            if not _matches_keywords(market.question, keywords):
                continue
            centered = 2.0 * market.probability_yes - 1.0
            weight = max(1.0, market.volume) ** 0.5  # sublinear em volume
            weighted_sum += sign * centered * weight
            total_weight += weight
            matched_topics.extend(keywords)
            break
    if total_weight == 0:
        return 0.0, matched_topics
    score = weighted_sum / total_weight
    return max(-1.0, min(1.0, score)), matched_topics


def _confidence_for(markets: list[PredictionMarket]) -> float:
    if not markets:
        return 0.0
    count_factor = min(len(markets) / 10.0, 1.0)
    vol_factor = min(sum(m.volume for m in markets) / 100_000.0, 1.0)
    return 0.5 * count_factor + 0.5 * vol_factor


# --- utils ----------------------------------------------------------


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _dollars_to_probability(dollars: float) -> float:
    """Kalshi cota em dolares por share; 0.65 = 65% de probabilidade."""
    if dollars > 1.5:
        # Em formato centavos (ex.: 65 em vez de 0.65), normaliza.
        return dollars / 100.0
    return dollars


def _parse_datetime(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        text = str(raw).replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None
