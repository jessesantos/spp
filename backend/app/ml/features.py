"""Technical-indicator feature engineering.

Pure functions on pandas DataFrames. Ported from the legacy
``feature_engine.py`` module. All outputs are deterministic for a given
input - no global state, no logging side-effects on hot paths.

A DataFrame passed in must contain at least the columns ``close`` and
``volume``; ``open``, ``high``, ``low``, ``date`` are preserved when
present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Hyperparameters for the feature pipeline."""

    ma_periods: tuple[int, ...] = (5, 10, 20, 50)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    volatility_window: int = 5
    momentum_period: int = 10
    volume_ma_window: int = 5
    excluded_columns: tuple[str, ...] = field(
        default=("date", "text", "clean_text", "target_price", "target_direction"),
    )


def add_moving_averages(df: pd.DataFrame, periods: tuple[int, ...]) -> pd.DataFrame:
    for period in periods:
        df[f"ma_{period}"] = df["close"].rolling(window=period, min_periods=1).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)
    return df


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.DataFrame:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int, std_dev: float) -> pd.DataFrame:
    sma = df["close"].rolling(window=period, min_periods=1).mean()
    std = df["close"].rolling(window=period, min_periods=1).std().fillna(0)
    df["bb_upper"] = sma + (std * std_dev)
    df["bb_middle"] = sma
    df["bb_lower"] = sma - (std * std_dev)
    width = (df["bb_upper"] - df["bb_lower"]).replace(0, 1e-10)
    df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, 1e-10)
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / width
    return df


def add_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    log_returns = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = log_returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["daily_return"] = (df["close"].pct_change() * 100).fillna(0)
    return df


def add_momentum(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df["momentum"] = df["close"].diff(period).fillna(0)
    return df


def add_volume_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df["volume_ma"] = df["volume"].rolling(window=window, min_periods=1).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"].replace(0, 1)
    return df


def add_conditional_volatility(
    df: pd.DataFrame, *, decay: float = 0.94, annualize: bool = True
) -> pd.DataFrame:
    """EWMA conditional volatility (RiskMetrics / GARCH-like).

    Implementa sigma_t^2 = decay * sigma_{t-1}^2 + (1 - decay) * r_{t-1}^2,
    lambda=0.94 (RiskMetrics padrao J.P. Morgan 1996) como aproximacao
    barata de GARCH(1,1) com alpha=0.06, beta=0.94. Captura clustering
    de volatilidade sem dependencia externa.

    Adiciona coluna ``cond_vol`` em forma anualizada (raiz de 252 pregoes).
    """
    log_returns = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    squared = log_returns.to_numpy(dtype=float) ** 2
    variance = np.zeros_like(squared)
    if len(squared) > 0:
        # Inicializa com media incondicional das primeiras 20 obs.
        init = squared[: min(20, len(squared))].mean() if len(squared) >= 2 else 0.0
        variance[0] = init
        for i in range(1, len(squared)):
            variance[i] = decay * variance[i - 1] + (1.0 - decay) * squared[i - 1]
    sigma = np.sqrt(np.maximum(variance, 0.0))
    if annualize:
        sigma = sigma * np.sqrt(252.0)
    df["cond_vol"] = sigma
    return df


def build_features(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute the full feature set used by the LSTM model.

    The input ``df`` must contain at least ``close`` and ``volume`` columns.
    Returns a *copy* of the DataFrame with indicator columns appended and
    NaNs forward-filled then zero-filled.
    """
    cfg = config or FeatureConfig()
    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'close' and 'volume' columns")

    out = df.copy()
    out = add_moving_averages(out, cfg.ma_periods)
    out = add_rsi(out, cfg.rsi_period)
    out = add_macd(out, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out = add_bollinger_bands(out, cfg.bollinger_period, cfg.bollinger_std)
    out = add_volatility(out, cfg.volatility_window)
    out = add_conditional_volatility(out)
    out = add_returns(out)
    out = add_momentum(out, cfg.momentum_period)
    out = add_volume_features(out, cfg.volume_ma_window)
    return out.ffill().fillna(0)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Shift close by -1 as next-day prediction target.

    Drops the final row which has no target. Pure; returns a new DataFrame.
    """
    out = df.copy()
    out["target_price"] = out["close"].shift(-1)
    out["target_direction"] = (out["target_price"] > out["close"]).astype(int)
    return out.iloc[:-1].copy()


def select_feature_columns(df: pd.DataFrame, config: FeatureConfig | None = None) -> list[str]:
    """Return the list of numeric feature columns suitable for the model."""
    cfg = config or FeatureConfig()
    numeric_kinds = {"i", "u", "f"}
    return [
        col
        for col in df.columns
        if col not in cfg.excluded_columns and df[col].dtype.kind in numeric_kinds
    ]
