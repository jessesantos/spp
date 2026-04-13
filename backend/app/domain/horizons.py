"""Horizontes de previsao suportados (D1/W1/M1) - value objects e mapas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HorizonCode = Literal["D1", "W1", "M1"]


@dataclass(frozen=True)
class Horizon:
    """Descricao de um horizonte de previsao.

    ``code`` e o identificador estavel (usado em API), ``days`` e o
    offset em dias corridos a partir de hoje, ``label`` e o texto
    amigavel em pt-BR exibido no frontend/explicacao.
    """

    code: HorizonCode
    days: int
    label: str


D1 = Horizon(code="D1", days=1, label="Amanha")
W1 = Horizon(code="W1", days=7, label="+7 dias")
M1 = Horizon(code="M1", days=30, label="+30 dias")

ALL_HORIZONS: tuple[Horizon, ...] = (D1, W1, M1)

# Formas compativeis com call sites legados que consomem tuples/dicts.
HORIZON_DEFINITIONS: tuple[tuple[HorizonCode, int], ...] = tuple(
    (h.code, h.days) for h in ALL_HORIZONS
)

HORIZON_LABELS: dict[int, str] = {h.days: h.label for h in ALL_HORIZONS}


def horizon_label_for_days(days: int) -> str:
    """Retorna o label amigavel para o offset em dias.

    Fallback para ``+{N} dias`` quando o offset nao esta no conjunto
    padrao - mantem a saida legivel para horizontes customizados.
    """
    return HORIZON_LABELS.get(days, f"+{days} dias")
