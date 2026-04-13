"""TickerSymbol value object: centraliza validacao e normalizacao de tickers B3.

Substitui `_validate_ticker` espalhado nas rotas. Seguro por construcao:
apos instanciar, o ``value`` ja esta normalizado (uppercase, sem espacos)
e validado contra um allowlist estreito - bloqueia tentativas de SSRF ou
prompt injection via entrada do usuario.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Allowlist conservadora: letras maiusculas + digitos, 1 a 10 chars.
# Cobre todos os padroes B3 (PETR4, VALE3, ITUB4, BOVA11, BBSE3 etc).
_VALID_TICKER = re.compile(r"^[A-Z0-9]{1,10}$")


class InvalidTickerError(ValueError):
    """Levantado quando o ticker nao passa na validacao de formato."""


@dataclass(frozen=True, slots=True)
class TickerSymbol:
    """Representa um ticker B3 normalizado e validado.

    Prefira criar via ``TickerSymbol.from_raw(raw)`` que normaliza
    caixa/whitespace antes de validar. O construtor bruto espera o
    valor ja normalizado e e mais util em codigo interno que sabe
    que o input ja esta limpo.

    Exemplos::

        t = TickerSymbol.from_raw("  petr4 ")
        str(t)   # "PETR4"
        t.value  # "PETR4"

        TickerSymbol.from_raw("x")        # ValueError (muito curto? nao, 1 char eh ok)
        TickerSymbol.from_raw("petr-4")   # InvalidTickerError (tem hifen)
    """

    value: str

    def __post_init__(self) -> None:
        if not _VALID_TICKER.match(self.value):
            raise InvalidTickerError(
                f"invalid ticker: {self.value!r} "
                "(esperado 1-10 chars alfanumericos maiusculos)"
            )

    @classmethod
    def from_raw(cls, raw: str | "TickerSymbol") -> "TickerSymbol":
        """Normaliza (upper + strip) e constroi; idempotente com instancias existentes."""
        if isinstance(raw, TickerSymbol):
            return raw
        if not isinstance(raw, str):
            raise InvalidTickerError(f"ticker must be str, got {type(raw).__name__}")
        return cls(raw.upper().strip())

    def __str__(self) -> str:
        return self.value
