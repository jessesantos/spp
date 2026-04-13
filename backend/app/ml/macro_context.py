"""Analisador de contexto macro/geopolitico global.

Complementa o sentimento especifico de ticker, buscando em feeds
internacionais e nacionais fatos que costumam mover a Bolsa brasileira:
guerras, politica monetaria (Fed, BCE, BoE, BoJ, Copom), politica comercial
(tarifas, sancoes), commodities (petroleo, minerio, soja), eleicoes,
regulacao. Esses sinais sao agregados em um ``MacroContext`` que pode ser
injetado como feature adicional no treino do LSTM.

Design (SOLID):
- ``MacroNewsSource`` protocol abstrai de onde vem o texto (RSS, API, DB).
- ``MacroAnalyzer`` protocol abstrai quem pontua (Claude, heuristica, etc).
- ``MacroContextBuilder`` orquestra source -> analyzer -> aggregate.

Prompt-injection (OWASP LLM01): o texto das noticias e tratado como dado
dentro de tags ``<article>`` e o prompt instrui Claude a nao seguir
instrucoes embutidas.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.ml.claude_sentiment import ClaudeSentimentAnalyzer, SentimentResult

# Feeds globais cobrindo geopolitica, macroeconomia e commodities.
# Escolhidos por serem publicos, estaveis e com RSS sem auth.
INTERNATIONAL_FEEDS: tuple[str, ...] = (
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/topNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.content.dj-n.com/public/rss/RSSMarketsMain",
    "https://www.ft.com/rss/home",
    "https://apnews.com/hub/world-news.rss",
    "https://apnews.com/hub/business.rss",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.bloomberg.com/politics/news.rss",
)

# Feeds nacionais de macro e politica
NATIONAL_MACRO_FEEDS: tuple[str, ...] = (
    "https://valor.globo.com/rss/editoria/financas",
    "https://valor.globo.com/rss/editoria/brasil",
    "https://www.infomoney.com.br/mercados/feed/",
    "https://www.infomoney.com.br/economia/feed/",
    "https://g1.globo.com/rss/g1/economia/",
    "https://g1.globo.com/rss/g1/politica/",
)

# Palavras-chave que marcam uma noticia como relevante para o contexto macro.
MACRO_KEYWORDS: tuple[str, ...] = (
    # Geopolitica
    "WAR", "GUERRA", "SANCTION", "SANCAO", "SANCOES", "TARIFF", "TARIFA",
    "UKRAINE", "UCRANIA", "RUSSIA", "ISRAEL", "PALESTIN", "GAZA", "IRAN",
    "CHINA", "TAIWAN", "NATO", "OTAN", "EMBARGO",
    # Politica monetaria
    "FED", "FOMC", "RATE HIKE", "INTEREST RATE", "INFLATION", "INFLAC",
    "CPI", "PPI", "SELIC", "COPOM", "ECB", "BCE", "BOE", "BOJ",
    # Commodities
    "OIL", "PETROLEO", "BRENT", "WTI", "OPEC", "OPEP",
    "IRON ORE", "MINERIO", "SOJA", "SOYBEAN", "GOLD", "OURO",
    # Mercado
    "RECESSION", "RECESSAO", "CRISE", "DEFAULT", "DOWNGRADE", "UPGRADE",
    "TRADE WAR", "GUERRA COMERCIAL", "SUPPLY CHAIN",
)


@dataclass(frozen=True)
class MacroContext:
    """Resumo agregado do contexto macro relevante para predicao."""

    score: float  # -1.0 (muito negativo) a +1.0 (muito positivo)
    confidence: float  # 0.0 a 1.0
    count: int  # quantas noticias foram consideradas
    positives: int = 0
    negatives: int = 0
    neutrals: int = 0
    top_keywords: list[str] = field(default_factory=list)
    high_impact_titles: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "count": self.count,
            "positives": self.positives,
            "negatives": self.negatives,
            "neutrals": self.neutrals,
            "top_keywords": list(self.top_keywords),
            "high_impact_titles": list(self.high_impact_titles),
        }


class MacroNewsSource(Protocol):
    async def fetch_macro(self, limit: int = 40) -> list[dict[str, Any]]: ...


class MacroAnalyzer(Protocol):
    def analyze_macro(
        self, article: dict[str, Any]
    ) -> SentimentResult: ...


class RSSMacroNewsSource:
    """Coleta artigos dos feeds macro (internacionais + nacionais).

    Retorna apenas artigos que contem alguma ``MACRO_KEYWORDS`` no titulo
    ou sumario. Isso reduz custo de API Claude posteriormente.
    """

    def __init__(
        self,
        international_feeds: tuple[str, ...] = INTERNATIONAL_FEEDS,
        national_feeds: tuple[str, ...] = NATIONAL_MACRO_FEEDS,
        per_feed_limit: int = 10,
    ) -> None:
        self._feeds = international_feeds + national_feeds
        self._per_feed_limit = per_feed_limit

    async def fetch_macro(self, limit: int = 40) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._fetch_sync, limit)

    def _fetch_sync(self, limit: int) -> list[dict[str, Any]]:
        try:
            import feedparser
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("feedparser is required for macro news") from exc

        articles: list[dict[str, Any]] = []
        for feed_url in self._feeds:
            try:
                parsed = feedparser.parse(feed_url)
            except Exception:  # noqa: BLE001
                continue
            per_feed = 0
            for entry in getattr(parsed, "entries", []):
                title = str(getattr(entry, "title", ""))
                summary = str(getattr(entry, "summary", ""))
                blob = (title + " " + summary).upper()
                if not any(kw in blob for kw in MACRO_KEYWORDS):
                    continue
                articles.append(
                    {
                        "title": title,
                        "summary": summary,
                        "published": str(getattr(entry, "published", "")),
                        "source": feed_url,
                        "url": str(getattr(entry, "link", "")),
                    }
                )
                per_feed += 1
                if per_feed >= self._per_feed_limit or len(articles) >= limit:
                    break
            if len(articles) >= limit:
                break
        return articles


class ClaudeMacroAnalyzer:
    """Adapta ``ClaudeSentimentAnalyzer`` para contexto macro (sem ticker).

    Usa o mesmo contrato de retorno ``SentimentResult`` mas com um prompt
    macro. Se nao houver cliente Claude, cai em heuristica simples.
    """

    def __init__(self, inner: ClaudeSentimentAnalyzer | None = None) -> None:
        self._inner = inner

    def analyze_macro(self, article: dict[str, Any]) -> SentimentResult:
        if self._inner is not None:
            # Reusa o mesmo pipeline de prompt-injection guard do analyzer
            # de ticker, passando "GLOBAL" como pseudo-ticker.
            return self._inner.analyze(article, ticker="GLOBAL")
        return _heuristic_score(article)


def _heuristic_score(article: dict[str, Any]) -> SentimentResult:
    """Fallback sem LLM: scoring por palavras-chave positivas/negativas."""
    title = str(article.get("title", ""))
    summary = str(article.get("summary", ""))
    blob = (title + " " + summary).upper()

    negative_hits = sum(
        1 for kw in _NEGATIVE_HEURISTICS if kw in blob
    )
    positive_hits = sum(
        1 for kw in _POSITIVE_HEURISTICS if kw in blob
    )

    if negative_hits == 0 and positive_hits == 0:
        return SentimentResult(score=0.0, confidence=0.2, title=title)

    total = negative_hits + positive_hits
    score = (positive_hits - negative_hits) / max(total, 1)
    confidence = min(0.5 + 0.1 * total, 0.9)
    impact = "alto" if total >= 3 else "medio" if total == 2 else "baixo"
    return SentimentResult(
        score=score,
        confidence=confidence,
        reasoning="heuristica por palavras-chave",
        impact=impact,
        title=title,
    )


_NEGATIVE_HEURISTICS: tuple[str, ...] = (
    "WAR", "GUERRA", "CRISIS", "CRISE", "RECESSION", "RECESSAO",
    "INFLATION", "INFLAC", "SANCTION", "SANCAO", "DEFAULT",
    "DOWNGRADE", "TARIFF", "TARIFA", "EMBARGO", "ATTACK", "ATAQUE",
    "RATE HIKE", "FED HIKE", "TRADE WAR",
)

_POSITIVE_HEURISTICS: tuple[str, ...] = (
    "RATE CUT", "FED CUT", "GROWTH", "RECOVERY", "RECUPERACAO",
    "UPGRADE", "SURGE", "RALLY", "ALTA FORTE", "ACORDO", "DEAL",
    "PEACE", "PAZ", "TRUCE", "STIMULUS", "ESTIMULO",
)


class MacroContextBuilder:
    """Orquestra source + analyzer e agrega em ``MacroContext``."""

    def __init__(
        self,
        source: MacroNewsSource,
        analyzer: MacroAnalyzer,
        *,
        neutral_threshold: float = 0.15,
    ) -> None:
        self._source = source
        self._analyzer = analyzer
        self._neutral_threshold = neutral_threshold

    async def build(self, limit: int = 40) -> MacroContext:
        articles = await self._source.fetch_macro(limit=limit)
        if not articles:
            return MacroContext(score=0.0, confidence=0.0, count=0)

        results = [self._analyzer.analyze_macro(a) for a in articles]
        return self._aggregate(results)

    def _aggregate(self, results: list[SentimentResult]) -> MacroContext:
        if not results:
            return MacroContext(score=0.0, confidence=0.0, count=0)

        weighted = sum(r.score * max(r.confidence, 0.1) for r in results)
        total_conf = sum(max(r.confidence, 0.1) for r in results)
        score = weighted / total_conf if total_conf > 0 else 0.0

        positives = sum(1 for r in results if r.score > self._neutral_threshold)
        negatives = sum(1 for r in results if r.score < -self._neutral_threshold)
        neutrals = len(results) - positives - negatives
        confidence = sum(r.confidence for r in results) / len(results)

        keywords_flat: list[str] = []
        for r in results:
            keywords_flat.extend(r.keywords)
        top_keywords = _top_n(keywords_flat, n=8)

        high_impact = [
            r.title for r in results if r.impact.lower() == "alto" and r.title
        ][:5]

        return MacroContext(
            score=round(score, 4),
            confidence=round(confidence, 4),
            count=len(results),
            positives=positives,
            negatives=negatives,
            neutrals=neutrals,
            top_keywords=top_keywords,
            high_impact_titles=high_impact,
        )


def _top_n(items: list[str], n: int = 8) -> list[str]:
    counter: dict[str, int] = {}
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())[:48]
        if not key:
            continue
        counter[key] = counter.get(key, 0) + 1
    ranked = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:n]]
