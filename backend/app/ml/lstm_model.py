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

    # Alvo do modelo: **log-retorno** em vez de preco absoluto (ADR 0011).
    # Preco bruto satura o MinMaxScaler quando o mercado supera o maximo
    # historico do treino (ex.: PETR4 fez topo e scaler_target.max=49.67
    # enquanto last_close=49.97 devolvia rollout em R$40). Log-retorno e
    # invariante a escala, estacionario e tem distribuicao leve bem
    # comportada em equities diarios (~N(0, 0.02) para blue-chips B3).
    _LOG_RETURN_CLIP: float = 0.30  # +/- 30% por passo como guard-rail

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Fit the model em log-retornos 1-step ahead."""
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow import keras

        features_df = build_features(df, self.feature_config)
        # target = log( close[t+1] / close[t] ). fillna(0) em bordas.
        close = features_df["close"].astype(float)
        log_ret = np.log(close.shift(-1) / close).fillna(0.0)
        features_df["target_return"] = log_ret
        features_df = features_df.iloc[:-1].copy()  # drop ultima linha (sem target)

        self._feature_columns = select_feature_columns(features_df, self.feature_config)
        x_raw = features_df[self._feature_columns].to_numpy(dtype=np.float32)
        y_raw = features_df["target_return"].to_numpy(dtype=np.float32).reshape(-1, 1)

        self._scaler_features = MinMaxScaler()
        # feature_range=(-1, 1) mantem log-retornos centrados em zero.
        self._scaler_target = MinMaxScaler(feature_range=(-1.0, 1.0))
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
                    monitor="loss", patience=15, restore_best_weights=True
                ),
            ],
        )
        final_loss = float(history.history["loss"][-1])
        # Metrica direcional: quantos sinais de log-retorno previstos batem
        # o sinal real no conjunto de treino (sanity pos-fit).
        y_pred_scaled = self.model.predict(x_seq, verbose=0)
        y_pred = self._scaler_target.inverse_transform(y_pred_scaled).flatten()
        y_true = self._scaler_target.inverse_transform(y_seq).flatten()
        direction_hit = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        return {
            "loss": final_loss,
            "epochs_run": len(history.history["loss"]),
            "direction_accuracy": direction_hit,
        }

    def predict(self, df: pd.DataFrame, days: int = 5) -> list[float]:
        """Rollout recursivo de ``days`` passos, cada um um log-retorno aplicado ao close anterior."""
        if self.model is None or self._scaler_features is None or self._scaler_target is None:
            raise RuntimeError("Model is not trained or loaded")
        if self._feature_columns is None:
            raise RuntimeError("Feature columns missing - train() or load() first")

        working = build_features(df, self.feature_config)
        # Garante que features externas (sentimento, macro, fx, market) que
        # o modelo viu no treino existam no DataFrame de inferencia. Quando
        # nao disponiveis (uso offline ou predict puro), entram como 0 -
        # mesma convencao usada pelo training_orchestrator quando o builder
        # falha (_safe_signal devolve 0.0).
        for col in self._feature_columns:
            if col not in working.columns:
                working[col] = 0.0
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
            log_return = float(self._scaler_target.inverse_transform(y_scaled)[0, 0])
            # Guard-rail: bloqueia previsoes absurdas antes de propagarem
            # pelo rollout recursivo. Movimentos >30% por pregao em
            # blue-chips da B3 sao fenomeno de cauda; clipar preserva
            # calibracao do modelo e evita escalada exponencial do erro.
            log_return = max(-self._LOG_RETURN_CLIP, min(self._LOG_RETURN_CLIP, log_return))
            last_close = float(working["close"].iloc[-1])
            next_close = last_close * float(np.exp(log_return))
            preds.append(next_close)

            # append a synthetic next row so recursion can continue
            next_row = working.iloc[-1].copy()
            next_row["close"] = next_close
            working = pd.concat([working, pd.DataFrame([next_row])], ignore_index=True)
            working = build_features(working, self.feature_config)
        return preds

    def save(self, path: str | Path) -> None:
        """Persist keras model + scalers + feature columns.

        Keras salva apenas a arquitetura/pesos. Scalers e a lista de
        colunas sao serializados em ``{path}.aux.joblib`` para que
        ``load()`` reconstitua o estado completo necessario para
        ``predict()`` rodar.
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        if self._scaler_features is None or self._scaler_target is None:
            raise RuntimeError("Scalers not initialized; call train() first")
        if self._feature_columns is None:
            raise RuntimeError("Feature columns not set; call train() first")

        import joblib

        path_obj = Path(path)
        self.model.save(str(path_obj))
        aux_path = path_obj.with_name(path_obj.name + ".aux.joblib")
        joblib.dump(
            {
                "feature_columns": self._feature_columns,
                "scaler_features": self._scaler_features,
                "scaler_target": self._scaler_target,
                "config": self.config,
                "feature_config": self.feature_config,
            },
            aux_path,
        )

    def load(self, path: str | Path) -> None:
        """Restore keras model and, when available, the aux state (scalers + columns)."""
        from tensorflow import keras

        path_obj = Path(path)
        self.model = keras.models.load_model(str(path_obj))
        aux_path = path_obj.with_name(path_obj.name + ".aux.joblib")
        if aux_path.exists():
            import joblib

            aux = joblib.load(aux_path)
            self._feature_columns = aux.get("feature_columns")
            self._scaler_features = aux.get("scaler_features")
            self._scaler_target = aux.get("scaler_target")
            cfg = aux.get("config")
            if cfg is not None:
                self.config = cfg
            feature_cfg = aux.get("feature_config")
            if feature_cfg is not None:
                self.feature_config = feature_cfg

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
