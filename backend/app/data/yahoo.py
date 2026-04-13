"""Yahoo Finance fallback adapter using yfinance (sync → asyncio.to_thread)."""

from __future__ import annotations

import asyncio
from typing import Any


class YahooClient:
    """Thin async wrapper over the yfinance library.

    ``yfinance`` is synchronous; we keep it off the event loop by
    running calls in a default thread-pool executor.
    """

    def __init__(self, suffix: str = ".SA") -> None:
        self._suffix = suffix

    async def get_history(self, ticker: str, period: str = "6mo") -> list[dict[str, Any]]:
        ticker = ticker.upper().strip()
        return await asyncio.to_thread(self._history_sync, ticker, period)

    def _history_sync(self, ticker: str, period: str) -> list[dict[str, Any]]:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("yfinance is required for Yahoo fallback") from exc

        symbol = ticker if ticker.endswith(self._suffix) else f"{ticker}{self._suffix}"
        frame = yf.Ticker(symbol).history(period=period)
        if frame.empty:
            return []
        frame = frame.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            rows.append(
                {
                    "date": row["date"].date().isoformat()
                    if hasattr(row["date"], "date")
                    else str(row["date"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                }
            )
        return rows
