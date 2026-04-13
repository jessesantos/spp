"""Unit tests for ``CurrencyImpactAnalyzer``."""

from __future__ import annotations

from typing import Any

import pytest

from app.ml.fx_impact import (
    SECTOR_EXPOSURE,
    CurrencyImpactAnalyzer,
    FxImpact,
    YahooUsdBrlProvider,
)


class _FakeFxProvider:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def get_usdbrl_history(self, days: int = 120) -> list[dict[str, Any]]:
        return self._rows


def _synth_rows(n: int, base: float = 5.0, drift: float = 0.005) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    price = base
    for i in range(n):
        price = price * (1 + drift)
        day = f"2026-01-{(i % 28) + 1:02d}"
        rows.append({"date": day, "close": round(price, 4)})
    return rows


@pytest.mark.asyncio
async def test_sector_exposure_known_tickers() -> None:
    analyzer = CurrencyImpactAnalyzer(fx_provider=_FakeFxProvider([]))
    assert analyzer.sector_exposure("PETR4") == SECTOR_EXPOSURE["PETR4"]
    assert analyzer.sector_exposure("MGLU3") == SECTOR_EXPOSURE["MGLU3"]


@pytest.mark.asyncio
async def test_sector_exposure_unknown_defaults_zero() -> None:
    analyzer = CurrencyImpactAnalyzer(fx_provider=_FakeFxProvider([]))
    assert analyzer.sector_exposure("XYZW1") == 0.0


@pytest.mark.asyncio
async def test_analyze_returns_fx_impact_for_exporter() -> None:
    fx_rows = _synth_rows(60, drift=0.005)
    ticker_rows = _synth_rows(60, drift=0.006)
    analyzer = CurrencyImpactAnalyzer(fx_provider=_FakeFxProvider(fx_rows))
    result = await analyzer.analyze("PETR4", ticker_rows)
    assert isinstance(result, FxImpact)
    assert result.ticker == "PETR4"
    assert result.exposure_label == "exportador"
    assert -1.0 <= result.fx_score <= 1.0
    assert "exportador" in result.keywords


@pytest.mark.asyncio
async def test_analyze_importer_negative_exposure() -> None:
    analyzer = CurrencyImpactAnalyzer(
        fx_provider=_FakeFxProvider(_synth_rows(40))
    )
    result = await analyzer.analyze("MGLU3", _synth_rows(40))
    assert result.exposure_label == "importador"
    assert result.sector_exposure < 0


@pytest.mark.asyncio
async def test_analyze_handles_empty_fx_history() -> None:
    analyzer = CurrencyImpactAnalyzer(fx_provider=_FakeFxProvider([]))
    result = await analyzer.analyze("PETR4", _synth_rows(10))
    assert result.correlation == 0.0
    assert result.sample_size == 0
    assert -1.0 <= result.fx_score <= 1.0


@pytest.mark.asyncio
async def test_analyze_handles_empty_ticker_history() -> None:
    analyzer = CurrencyImpactAnalyzer(
        fx_provider=_FakeFxProvider(_synth_rows(30))
    )
    result = await analyzer.analyze("VALE3", [])
    assert result.correlation == 0.0
    assert result.exposure_label == "exportador"


@pytest.mark.asyncio
async def test_fx_score_bounded() -> None:
    analyzer = CurrencyImpactAnalyzer(fx_provider=_FakeFxProvider(_synth_rows(30)))
    result = await analyzer.analyze("PETR4", _synth_rows(30))
    assert -1.0 <= result.fx_score <= 1.0
    assert -1.0 <= result.correlation <= 1.0


@pytest.mark.asyncio
async def test_analyze_marks_choque_cambial() -> None:
    # fx rows with > 3% move over the window
    fx_rows = _synth_rows(30, base=5.0, drift=0.005)
    analyzer = CurrencyImpactAnalyzer(
        fx_provider=_FakeFxProvider(fx_rows), window_days=20
    )
    result = await analyzer.analyze("PETR4", _synth_rows(30))
    assert result.usdbrl_change_pct is not None


@pytest.mark.asyncio
async def test_yahoo_usdbrl_provider_chooses_period() -> None:
    class _FakeYahoo:
        async def get_history(self, ticker: str, period: str) -> list[dict[str, Any]]:
            return [{"date": "2026-01-01", "close": 5.0, "period": period}]

    provider = YahooUsdBrlProvider(_FakeYahoo())
    rows = await provider.get_usdbrl_history(days=30)
    assert rows and rows[0]["period"] == "1mo"
    rows = await provider.get_usdbrl_history(days=100)
    assert rows[0]["period"] == "6mo"
    rows = await provider.get_usdbrl_history(days=500)
    assert rows[0]["period"] == "2y"
