"""Unit tests for pure feature-engineering functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    FeatureConfig,
    add_bollinger_bands,
    add_macd,
    add_moving_averages,
    add_returns,
    add_rsi,
    add_volatility,
    add_volume_features,
    build_features,
    select_feature_columns,
)


def _synth_df(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = np.cumsum(rng.normal(0, 1, rows)) + 50
    return pd.DataFrame(
        {
            "close": close,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "volume": rng.integers(1_000_000, 5_000_000, rows),
        }
    )


def test_build_features_adds_expected_columns() -> None:
    df = build_features(_synth_df())
    for col in (
        "ma_5",
        "ma_20",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "bb_percent",
        "volatility",
        "daily_return",
        "momentum",
        "volume_ratio",
    ):
        assert col in df.columns, f"missing {col}"


def test_build_features_no_nans() -> None:
    df = build_features(_synth_df())
    assert not df.isna().any().any()


def test_rsi_is_bounded_0_100() -> None:
    df = add_rsi(_synth_df(), period=14)
    assert df["rsi"].between(0, 100).all()


def test_moving_averages_match_rolling_mean() -> None:
    df = _synth_df(10)
    out = add_moving_averages(df.copy(), (5,))
    expected = df["close"].rolling(5, min_periods=1).mean()
    assert np.allclose(out["ma_5"].to_numpy(), expected.to_numpy())


def test_build_features_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        build_features(pd.DataFrame({"x": [1, 2]}))


def test_select_feature_columns_excludes_target() -> None:
    df = build_features(_synth_df())
    df["target_price"] = df["close"].shift(-1)
    cols = select_feature_columns(df)
    assert "target_price" not in cols
    assert "close" in cols


def test_feature_config_is_frozen() -> None:
    cfg = FeatureConfig()
    with pytest.raises(Exception):  # noqa: PT011 - dataclass FrozenInstanceError
        cfg.rsi_period = 5  # type: ignore[misc]


def test_bollinger_percent_well_defined_when_flat() -> None:
    df = pd.DataFrame({"close": [10.0] * 30, "volume": [1] * 30})
    out = add_bollinger_bands(df, period=20, std_dev=2.0)
    # flat series has zero std → bb_percent should be finite (no inf/nan)
    assert np.isfinite(out["bb_percent"]).all()


def test_volatility_non_negative() -> None:
    out = add_volatility(_synth_df(), window=5)
    assert (out["volatility"].dropna() >= 0).all()


def test_macd_histogram_is_difference() -> None:
    out = add_macd(_synth_df(), fast=12, slow=26, signal=9)
    assert np.allclose(out["macd_hist"], out["macd"] - out["macd_signal"])


def test_returns_finite() -> None:
    out = add_returns(_synth_df())
    assert np.isfinite(out["daily_return"]).all()


def test_volume_ratio_handles_zero() -> None:
    df = _synth_df()
    df.loc[0, "volume"] = 0
    out = add_volume_features(df, window=5)
    assert np.isfinite(out["volume_ratio"]).all()
