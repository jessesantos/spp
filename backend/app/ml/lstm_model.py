"""LSTM price predictor wrapper.

Ports the Keras model architecture from legacy ``model.py`` into a class
that can be instantiated with explicit hyperparameters (no hidden
global state) and saved/loaded to disk in the native Keras v3 format.

TensorFlow / Keras imports are performed lazily inside methods so that
unrelated code paths (e.g. `/health`, `/api/tickers`) do not pay the
~2 s TensorFlow import cost, and unit tests that do not exercise the
model do not require TensorFlow at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from app.ml.features import FeatureConfig, build_features, select_feature_columns

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow import keras


@dataclass(frozen=True)
class LSTMConfig:
    sequence_length: int = 5
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-3
    dropout: float = 0.2
    units: tuple[int, int, int] = (128, 64, 32)
    dense_units: int = 16


class LSTMPricePredictor:
    """Wrapper around a Keras LSTM with fit/predict/save/load."""

    def __init__(
        self,
        config: LSTMConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ) -> None:
        self.config = config or LSTMConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.model: Any | None = None
        self._feature_columns: list[str] | None = None
        self._scaler_features: Any | None = None
        self._scaler_target: Any | None = None

    # --- public API ----------------------------------------------------

    def build(self, n_features: int) -> Any:
        from tensorflow import keras
        from tensorflow.keras import layers

        cfg = self.config
        u1, u2, u3 = cfg.units
        model = keras.Sequential(
            [
                layers.Input(shape=(cfg.sequence_length, n_features)),
                layers.LSTM(u1, return_sequences=True, dropout=cfg.dropout),
                layers.LSTM(u2, return_sequences=True, dropout=cfg.dropout),
                layers.LSTM(u3, dropout=cfg.dropout),
                layers.Dense(cfg.dense_units, activation="relu"),
                layers.Dropout(cfg.dropout),
                layers.Dense(1),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        self.model = model
        return model

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Fit the model on a price DataFrame with OHLCV columns."""
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow import keras

        features_df = build_features(df, self.feature_config)
        features_df["target_price"] = features_df["close"].shift(-1)
        features_df = features_df.iloc[:-1].copy()

        self._feature_columns = select_feature_columns(features_df, self.feature_config)
        x_raw = features_df[self._feature_columns].to_numpy(dtype=np.float32)
        y_raw = features_df["target_price"].to_numpy(dtype=np.float32).reshape(-1, 1)

        self._scaler_features = MinMaxScaler()
        self._scaler_target = MinMaxScaler()
        x_scaled = self._scaler_features.fit_transform(x_raw)
        y_scaled = self._scaler_target.fit_transform(y_raw)

        x_seq, y_seq = self._to_sequences(x_scaled, y_scaled)
        if self.model is None:
            self.build(n_features=x_seq.shape[2])

        assert self.model is not None  # noqa: S101 - narrow type for mypy
        history = self.model.fit(
            x_seq,
            y_seq,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="loss", patience=10, restore_best_weights=True
                ),
            ],
        )
        final_loss = float(history.history["loss"][-1])
        return {"loss": final_loss, "epochs_run": len(history.history["loss"])}

    def predict(self, df: pd.DataFrame, days: int = 5) -> list[float]:
        """Produce ``days`` next-step predictions via recursive rollout."""
        if self.model is None or self._scaler_features is None or self._scaler_target is None:
            raise RuntimeError("Model is not trained or loaded")
        if self._feature_columns is None:
            raise RuntimeError("Feature columns missing - train() or load() first")

        working = build_features(df, self.feature_config)
        preds: list[float] = []
        for _ in range(max(1, int(days))):
            x = working[self._feature_columns].to_numpy(dtype=np.float32)
            x_scaled = self._scaler_features.transform(x)
            if len(x_scaled) < self.config.sequence_length:
                raise ValueError(
                    f"need at least {self.config.sequence_length} rows, got {len(x_scaled)}"
                )
            seq = x_scaled[-self.config.sequence_length :][np.newaxis, :, :]
            y_scaled = self.model.predict(seq, verbose=0)
            y_unscaled = float(self._scaler_target.inverse_transform(y_scaled)[0, 0])
            preds.append(y_unscaled)

            # append a synthetic next row so recursion can continue
            next_row = working.iloc[-1].copy()
            next_row["close"] = y_unscaled
            working = pd.concat([working, pd.DataFrame([next_row])], ignore_index=True)
            working = build_features(working, self.feature_config)
        return preds

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(str(path))

    def load(self, path: str | Path) -> None:
        from tensorflow import keras

        self.model = keras.models.load_model(str(path))

    # --- helpers -------------------------------------------------------

    def _to_sequences(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = self.config.sequence_length
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for i in range(seq_len, len(x)):
            xs.append(x[i - seq_len : i])
            ys.append(y[i])
        return np.asarray(xs), np.asarray(ys)
