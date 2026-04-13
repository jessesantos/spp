"""Price repository with BrAPI → Yahoo fallback strategy."""

from __future__ import annotations

from typing import Any, Protocol

from app.data.brapi import BrAPIClient, BrAPIError
from app.data.yahoo import YahooClient


class PriceRepository(Protocol):
    async def get_history(self, ticker: str) -> list[dict[str, Any]]: ...


class BrAPIYahooRepository:
    """Primary: BrAPI historicalDataPrice. Fallback: yfinance."""

    def __init__(self, brapi: BrAPIClient, yahoo: YahooClient) -> None:
        self._brapi = brapi
        self._yahoo = yahoo

    async def get_history(self, ticker: str) -> list[dict[str, Any]]:
        try:
            payload = await self._brapi.get_quote(ticker)
            rows = self._extract_brapi_history(payload)
            if rows:
                return rows
        except (BrAPIError, ValueError):
            pass
        return await self._yahoo.get_history(ticker)

    @staticmethod
    def _extract_brapi_history(payload: dict[str, Any]) -> list[dict[str, Any]]:
        history = payload.get("historicalDataPrice") or []
        rows: list[dict[str, Any]] = []
        for item in history:
            date_raw = item.get("date")
            if date_raw is None:
                continue
            rows.append(
                {
                    "date": _as_iso_date(date_raw),
                    "open": float(item.get("open", 0.0)),
                    "high": float(item.get("high", 0.0)),
                    "low": float(item.get("low", 0.0)),
                    "close": float(item.get("close", 0.0)),
                    "volume": int(item.get("volume", 0)),
                }
            )
        return rows


def _as_iso_date(value: Any) -> str:
    # BrAPI returns unix timestamps (ints) or ISO strings depending on endpoint
    if isinstance(value, int | float):
        import datetime as _dt

        return _dt.datetime.fromtimestamp(float(value), tz=_dt.UTC).date().isoformat()
    return str(value)[:10]
