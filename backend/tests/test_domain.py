"""Tests for the domain layer: TickerSymbol value object + horizons constants."""

from __future__ import annotations

import pytest

from app.domain import (
    HORIZON_DEFINITIONS,
    HORIZON_LABELS,
    InvalidTickerError,
    TickerSymbol,
    horizon_label_for_days,
)
from app.domain.horizons import ALL_HORIZONS, D1, M1, W1


def test_ticker_from_raw_normalizes() -> None:
    assert str(TickerSymbol.from_raw("  petr4 ")) == "PETR4"
    assert str(TickerSymbol.from_raw("vale3")) == "VALE3"


def test_ticker_accepts_alphanumeric_upper() -> None:
    assert TickerSymbol("PETR4").value == "PETR4"
    assert TickerSymbol("BOVA11").value == "BOVA11"
    assert TickerSymbol("X1").value == "X1"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "petr-4",
        "PETR 4",
        "PETR4.SA",
        "../../etc/passwd",
        "PETR4;DROP",
        "A" * 11,
        "\x00PETR4",
    ],
)
def test_ticker_rejects_invalid(bad: str) -> None:
    with pytest.raises(InvalidTickerError):
        TickerSymbol.from_raw(bad)


def test_ticker_from_raw_idempotent_on_existing_instance() -> None:
    original = TickerSymbol("PETR4")
    assert TickerSymbol.from_raw(original) is original


def test_ticker_rejects_non_string_input() -> None:
    with pytest.raises(InvalidTickerError):
        TickerSymbol.from_raw(42)  # type: ignore[arg-type]


def test_horizon_definitions_match_all_horizons() -> None:
    assert HORIZON_DEFINITIONS == (("D1", 1), ("W1", 7), ("M1", 30))
    assert ALL_HORIZONS == (D1, W1, M1)


def test_horizon_labels_map_days_to_ptbr() -> None:
    assert HORIZON_LABELS == {1: "Amanha", 7: "+7 dias", 30: "+30 dias"}


def test_horizon_label_for_days_fallback() -> None:
    assert horizon_label_for_days(1) == "Amanha"
    assert horizon_label_for_days(7) == "+7 dias"
    assert horizon_label_for_days(30) == "+30 dias"
    # offset desconhecido cai no padrao
    assert horizon_label_for_days(14) == "+14 dias"
