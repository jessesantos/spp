"""Async BrAPI client for B3 quote data, with Redis caching and retries."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

_VALID_TICKER = re.compile(r"^[A-Z0-9]{1,10}$")


class BrAPIError(RuntimeError):
    """Raised when BrAPI returns an unusable payload after retries."""


class BrAPIClient:
    """Async client for the public BrAPI quote endpoint.

    Parameters are injected to keep this class pure and testable:
    pass a fake ``httpx.AsyncClient`` and a fake cache in unit tests.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        base_url: str = "https://brapi.dev/api",
        cache: Any | None = None,
        cache_ttl_seconds: int = 300,
        retries: int = 2,
        backoff_seconds: float = 0.5,
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._cache = cache
        self._cache_ttl = cache_ttl_seconds
        self._retries = retries
        self._backoff = backoff_seconds

    async def get_quote(self, ticker: str, range_: str = "5y") -> dict[str, Any]:
        ticker = ticker.upper().strip()
        if not _VALID_TICKER.match(ticker):
            raise ValueError(f"invalid ticker: {ticker!r}")

        cache_key = f"brapi:quote:{ticker}:{range_}"
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return cached  # type: ignore[no-any-return]

        url = f"{self._base_url}/quote/{ticker}"
        params = {"interval": "1d", "range": range_, "fundamental": "false", "dividends": "false"}

        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = await self._http.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results") or []
                if not results:
                    raise BrAPIError(f"empty results for {ticker}")
                payload: dict[str, Any] = results[0]
                await self._cache_set(cache_key, payload)
                return payload
            except (httpx.HTTPError, BrAPIError, ValueError) as exc:
                last_error = exc
                if attempt >= self._retries:
                    break
                await asyncio.sleep(self._backoff * (2**attempt))

        raise BrAPIError(f"BrAPI failed for {ticker}: {last_error}") from last_error

    async def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self._cache is None:
            return None
        try:
            raw = await self._cache.get(key)
        except Exception:  # noqa: BLE001 - cache failure must not break callers
            return None
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(raw)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return None

    async def _cache_set(self, key: str, payload: dict[str, Any]) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.set(key, json.dumps(payload), ex=self._cache_ttl)
        except Exception:  # noqa: BLE001
            return
