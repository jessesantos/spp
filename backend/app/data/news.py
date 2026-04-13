"""RSS news aggregator for Brazilian financial feeds."""

from __future__ import annotations

import asyncio
from typing import Any

DEFAULT_FEEDS: tuple[str, ...] = (
    "https://www.infomoney.com.br/feed/",
    "https://www.moneytimes.com.br/feed/",
    "https://einvestidor.estadao.com.br/feed",
)


class RSSNewsClient:
    """Collects RSS news entries matching a ticker."""

    def __init__(self, feeds: tuple[str, ...] = DEFAULT_FEEDS, limit: int = 20) -> None:
        self._feeds = feeds
        self._limit = limit

    async def fetch(self, ticker: str) -> list[dict[str, Any]]:
        ticker_clean = ticker.replace(".SA", "").upper().strip()
        return await asyncio.to_thread(self._fetch_sync, ticker_clean)

    def _fetch_sync(self, ticker: str) -> list[dict[str, Any]]:
        try:
            import feedparser
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("feedparser is required for RSS news") from exc

        articles: list[dict[str, Any]] = []
        for feed_url in self._feeds:
            try:
                parsed = feedparser.parse(feed_url)
            except Exception:  # noqa: BLE001 - broken feed must not break batch
                continue
            for entry in getattr(parsed, "entries", []):
                title = str(getattr(entry, "title", ""))
                summary = str(getattr(entry, "summary", ""))
                if ticker in title.upper() or ticker in summary.upper():
                    articles.append(
                        {
                            "title": title,
                            "summary": summary,
                            "published": str(getattr(entry, "published", "")),
                            "source": feed_url,
                            "url": str(getattr(entry, "link", "")),
                        }
                    )
                if len(articles) >= self._limit:
                    return articles
        return articles
