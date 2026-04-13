"""Domain layer: value objects e entidades puras sem dependencia de framework.

Exporta primitivos usados em varias camadas (api/, services/, ml/) para
centralizar regras de negocio estaveis e invariantes.
"""

from app.domain.horizons import (
    HORIZON_DEFINITIONS,
    HORIZON_LABELS,
    Horizon,
    horizon_label_for_days,
)
from app.domain.ticker import InvalidTickerError, TickerSymbol

__all__ = [
    "HORIZON_DEFINITIONS",
    "HORIZON_LABELS",
    "Horizon",
    "InvalidTickerError",
    "TickerSymbol",
    "horizon_label_for_days",
]
