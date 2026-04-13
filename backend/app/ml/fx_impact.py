"""Analisador de impacto cambial BRL/USD sobre tickers da B3.

Tres pilares:

1. **Correlacao historica** do ticker com USDBRL nos ultimos N dias.
2. **Heuristica setorial** - exportadoras (PETR4, VALE3) ganham com dolar
   alto, varejo importador (MGLU3, AMER3) sofre.
3. **Score composto** (-1 a +1) estavel, usado como feature ``fx_score``
   no LSTM.

Design:
- ``FxHistoryProvider`` (Protocol) abstrai a fonte do historico cambial.
- ``CurrencyImpactAnalyzer`` consome historico de ticker + USDBRL e
  devolve ``FxImpact``.
- Heuristica setorial fica em dict modulo-level, faciltando extensao.

Prompt-injection / OWASP: N/A (sem LLM neste modulo).
SSRF: fonte cambial e apenas yfinance/BrAPI via repos ja existentes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol


# Heuristica: classificacao por exposicao ao dolar.
# +1.0 = exportadora pura (receita USD, custo BRL) - dolar alto ajuda.
# -1.0 = importadora pura (custo USD, receita BRL) - dolar alto prejudica.
#  0.0 = neutro / domestico sem exposicao relevante.
SECTOR_EXPOSURE: dict[str, float] = {
    # Commodities / exportadoras
    "PETR4": 0.9,
    "PETR3": 0.9,
    "VALE3": 0.95,
    "SUZB3": 0.8,
    "KLBN11": 0.6,
    "EMBR3": 0.7,
    "JBSS3": 0.5,
    "MRFG3": 0.5,
    "BRFS3": 0.3,
    "CSNA3": 0.4,
    "GGBR4": 0.4,
    "USIM5": 0.4,
    # Financeiro (baixa exposicao direta, mas alta de dolar sinaliza aperto)
    "ITUB4": -0.1,
    "BBDC4": -0.1,
    "BBAS3": 0.0,
    "SANB11": -0.1,
    # Utilities / dominial
    "ELET3": -0.2,
    "ENGI11": -0.2,
    "CMIG4": -0.2,
    "SBSP3": -0.1,
    # Varejo / consumo domestico (importadoras)
    "MGLU3": -0.8,
    "AMER3": -0.7,
    "LREN3": -0.6,
    "VIIA3": -0.7,
    "ASAI3": -0.4,
    "PCAR3": -0.3,
    "PETZ3": -0.5,
    # Construcao / sensivel a juros (via dolar -> juros)
    "CYRE3": -0.3,
    "MRVE3": -0.3,
    # Aereas (importam combustivel e leasing em USD)
    "AZUL4": -0.8,
    "GOLL4": -0.8,
    # Telecom
    "VIVT3": -0.2,
    # Tech local
    "TOTS3": -0.2,
}

DEFAULT_SECTOR_EXPOSURE: float = 0.0
CORRELATION_WINDOW_DAYS: int = 90


@dataclass(frozen=True)
class FxImpact:
    """Resultado do analisador cambial."""

    ticker: str
    sector_exposure: float  # [-1, +1] heuristica
    correlation: float  # [-1, +1] Pearson ticker.close vs USDBRL.close
    fx_score: float  # score composto final (feature do LSTM)
    exposure_label: str  # "exportador" | "importador" | "neutro"
    sample_size: int
    usdbrl_last: float | None = None
    usdbrl_change_pct: float | None = None  # variacao N dias em %
    reasoning: str = ""
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "sector_exposure": self.sector_exposure,
            "correlation": self.correlation,
            "fx_score": self.fx_score,
            "exposure_label": self.exposure_label,
            "sample_size": self.sample_size,
            "usdbrl_last": self.usdbrl_last,
            "usdbrl_change_pct": self.usdbrl_change_pct,
            "reasoning": self.reasoning,
            "keywords": list(self.keywords),
        }


class FxHistoryProvider(Protocol):
    """Fornece historico do par USDBRL (close por data ISO)."""

    async def get_usdbrl_history(self, days: int = 120) -> list[dict[str, Any]]: ...


class CurrencyImpactAnalyzer:
    """Calcula impacto cambial de um ticker."""

    def __init__(
        self,
        fx_provider: FxHistoryProvider,
        *,
        window_days: int = CORRELATION_WINDOW_DAYS,
        sector_map: dict[str, float] | None = None,
    ) -> None:
        self._fx = fx_provider
        self._window = window_days
        self._sector = sector_map or SECTOR_EXPOSURE

    def sector_exposure(self, ticker: str) -> float:
        return float(self._sector.get(ticker.upper().strip(), DEFAULT_SECTOR_EXPOSURE))

    async def analyze(
        self,
        ticker: str,
        ticker_history: list[dict[str, Any]],
    ) -> FxImpact:
        """Produz o impacto cambial para o ticker, dado seu historico OHLCV."""
        ticker = ticker.upper().strip()
        exposure = self.sector_exposure(ticker)

        try:
            fx_rows = await self._fx.get_usdbrl_history(days=self._window + 10)
        except Exception:  # noqa: BLE001 - nunca derrubar predict por FX
            fx_rows = []

        correlation, sample = _pearson_aligned(ticker_history, fx_rows, self._window)
        usdbrl_last = _last_close(fx_rows)
        usdbrl_change = _change_pct(fx_rows, self._window)

        fx_score = _compose_score(exposure, correlation)
        label = _label_for(exposure)
        reasoning = _build_reasoning(exposure, correlation, usdbrl_change, label)
        keywords = _build_keywords(exposure, correlation, usdbrl_change)

        return FxImpact(
            ticker=ticker,
            sector_exposure=round(exposure, 3),
            correlation=round(correlation, 4),
            fx_score=round(fx_score, 4),
            exposure_label=label,
            sample_size=sample,
            usdbrl_last=(round(usdbrl_last, 4) if usdbrl_last is not None else None),
            usdbrl_change_pct=(
                round(usdbrl_change, 3) if usdbrl_change is not None else None
            ),
            reasoning=reasoning,
            keywords=keywords,
        )


# --- helpers ---------------------------------------------------------


def _pearson_aligned(
    ticker_rows: list[dict[str, Any]],
    fx_rows: list[dict[str, Any]],
    window: int,
) -> tuple[float, int]:
    t_by_date = _index_close_by_date(ticker_rows)
    f_by_date = _index_close_by_date(fx_rows)
    common_dates = sorted(set(t_by_date.keys()) & set(f_by_date.keys()))
    if len(common_dates) < 10:
        return 0.0, len(common_dates)
    common_dates = common_dates[-window:]

    t_closes = [t_by_date[d] for d in common_dates]
    f_closes = [f_by_date[d] for d in common_dates]

    t_ret = _pct_returns(t_closes)
    f_ret = _pct_returns(f_closes)
    if len(t_ret) < 5:
        return 0.0, len(t_ret)
    return _pearson(t_ret, f_ret), len(t_ret)


def _index_close_by_date(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows or []:
        raw = row.get("date") or row.get("trade_date")
        if raw is None:
            continue
        key = str(raw)[:10]
        try:
            out[key] = float(row.get("close", 0.0))
        except (TypeError, ValueError):
            continue
    return out


def _pct_returns(values: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev == 0 or not math.isfinite(prev):
            continue
        out.append((values[i] - prev) / prev)
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs = xs[-n:]
    ys = ys[-n:]
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return 0.0
    value = num / (denx * deny)
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _last_close(rows: list[dict[str, Any]]) -> float | None:
    for row in reversed(rows or []):
        try:
            return float(row.get("close", 0.0))
        except (TypeError, ValueError):
            continue
    return None


def _change_pct(rows: list[dict[str, Any]], window: int) -> float | None:
    if not rows:
        return None
    closes = [r.get("close") for r in rows if r.get("close") is not None]
    closes = [float(c) for c in closes[-window - 1 :] if c is not None]
    if len(closes) < 2:
        return None
    first, last = closes[0], closes[-1]
    if first == 0:
        return None
    return (last - first) / first * 100.0


def _compose_score(exposure: float, correlation: float) -> float:
    """Combina heuristica setorial e correlacao empirica.

    Peso 0.6 na heuristica (estrutural, pouco ruidosa) e 0.4 na correlacao
    (empirica, capta desvios de curto prazo). Sinal final ainda em [-1, 1].
    """
    score = 0.6 * exposure + 0.4 * correlation
    return max(-1.0, min(1.0, score))


def _label_for(exposure: float) -> str:
    if exposure >= 0.3:
        return "exportador"
    if exposure <= -0.3:
        return "importador"
    return "neutro"


def _build_reasoning(
    exposure: float,
    correlation: float,
    change_pct: float | None,
    label: str,
) -> str:
    parts: list[str] = []
    parts.append(
        f"Perfil estrutural: {label} (exposicao heuristica {exposure:+.2f})."
    )
    parts.append(
        f"Correlacao retornos com USDBRL nos ultimos {CORRELATION_WINDOW_DAYS} "
        f"dias: {correlation:+.2f}."
    )
    if change_pct is not None:
        if abs(change_pct) >= 3.0:
            movimento = "forte"
        elif abs(change_pct) >= 1.0:
            movimento = "moderado"
        else:
            movimento = "fraco"
        direcao = "alta" if change_pct >= 0 else "queda"
        parts.append(
            f"USDBRL teve {movimento} {direcao} de {change_pct:+.2f}% no periodo."
        )
    return " ".join(parts)


def _build_keywords(
    exposure: float, correlation: float, change_pct: float | None
) -> tuple[str, ...]:
    kws: list[str] = []
    if exposure >= 0.3:
        kws.append("exportador")
    elif exposure <= -0.3:
        kws.append("importador")
    if correlation >= 0.3:
        kws.append("correlacao_positiva_usd")
    elif correlation <= -0.3:
        kws.append("correlacao_negativa_usd")
    if change_pct is not None and abs(change_pct) >= 3.0:
        kws.append("choque_cambial")
    return tuple(kws)


class YahooUsdBrlProvider:
    """Fornece historico USDBRL via ``YahooClient`` existente.

    Encapsula conversao do yfinance symbol ``USDBRL=X`` no formato comum
    do SPP (lista de dicts com ``date``/``close``).
    """

    def __init__(self, yahoo_client: Any, symbol: str = "USDBRL=X") -> None:
        self._yahoo = yahoo_client
        self._symbol = symbol

    async def get_usdbrl_history(self, days: int = 120) -> list[dict[str, Any]]:
        period = _period_for_days(days)
        try:
            rows = await self._yahoo.get_history(self._symbol, period=period)
        except Exception:  # noqa: BLE001 - graceful fallback
            return []
        return rows or []


def _period_for_days(days: int) -> str:
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    if days <= 180:
        return "6mo"
    if days <= 365:
        return "1y"
    if days <= 730:
        return "2y"
    return "5y"
