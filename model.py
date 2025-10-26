import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings do TensorFlow
tf.get_logger().setLevel('ERROR')


class PredictionModel:
    """Modelo LSTM para predição de preços de ações"""
    
    def __init__(self):
        self.model = None
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.sequence_length = config.SEQUENCE_LENGTH
    
    def prepare_features(self, df: pd.DataFrame) -> List[str]:
        """Seleciona colunas de features para o modelo"""
        exclude_cols = ["date", "text", "clean_text", "target_price", "target_direction"]
        
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]
        
        self.feature_columns = feature_cols
        logger.info(f"Features selecionadas: {len(feature_cols)}")
        
        return feature_cols
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cria sequências temporais para LSTM
        
        Args:
            data: Array de features
            target: Array de targets
            
        Returns:
            X, y arrays com sequências
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara dados para treinamento"""
        # Selecionar features
        if self.feature_columns is None:
            self.prepare_features(df)
        
        # Extrair features e target
        X = df[self.feature_columns].values
        y = df["target_price"].values.reshape(-1, 1)
        
        # Normalizar
        X_scaled = self.scaler_features.fit_transform(X)
        y_scaled = self.scaler_target.fit_transform(y)
        
        # Criar sequências
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        logger.info(f"Dados preparados: X shape {X_seq.shape}, y shape {y_seq.shape}")
        
        return X_seq, y_seq
    
    def build_model(self, input_shape: Tuple) -> keras.Model:
        """Constrói arquitetura do modelo LSTM"""
        model = keras.Sequential([
            # Primeira camada LSTM
            layers.LSTM(
                units=128,
                return_sequences=True,
                input_shape=input_shape,
                dropout=0.2
            ),
            
            # Segunda camada LSTM
            layers.LSTM(
                units=64,
                return_sequences=True,
                dropout=0.2
            ),
            
            # Terceira camada LSTM
            layers.LSTM(
                units=32,
                dropout=0.2
            ),
            
            # Camadas Dense
            layers.Dense(units=16, activation="relu"),
            layers.Dropout(0.2),
            
            # Saída
            layers.Dense(units=1)
        ])
        
        # Compilar
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info("✅ Modelo LSTM construído")
        model.summary(print_fn=logger.info)
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> keras.callbacks.History:
        """Treina o modelo"""
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        # Validação
        validation_data = (X_val, y_val) if X_val is not None else None
        
        logger.info("Iniciando treinamento...")
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Treinamento concluído")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        # Predizer valores normalizados
        y_pred_scaled = self.model.predict(X, verbose=0)
        
        # Desnormalizar
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Avalia performance do modelo"""
        if len(X_test) == 0:
            logger.warning("⚠️  Dataset de teste vazio")
            return {
                "MSE": 0,
                "RMSE": 0,
                "MAE": 0,
                "MAPE": 0,
                "Direction_Accuracy": 0
            }
        
        # Predições com batch_size apropriado
        batch_size = min(32, len(X_test))  # Ajustar batch size ao tamanho dos dados
        y_pred_scaled = self.model.predict(X_test, verbose=0, batch_size=batch_size)
        
        # Desnormalizar
        y_true = self.scaler_target.inverse_transform(y_test)
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled)
        
        # Métricas
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE com proteção contra divisão por zero
        mape_values = np.abs((y_true - y_pred) / (y_true + 1e-10)) * 100
        mape = np.mean(mape_values)
        
        # Acurácia direcional (precisa de pelo menos 2 pontos)
        if len(y_true) > 1:
            direction_true = np.diff(y_true.flatten()) > 0
            direction_pred = np.diff(y_pred.flatten()) > 0
            direction_accuracy = np.mean(direction_true == direction_pred) * 100
        else:
            direction_accuracy = 0
            logger.warning("⚠️  Poucos dados para calcular acurácia direcional")
        
        metrics = {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "MAPE": float(mape),
            "Direction_Accuracy": float(direction_accuracy)
        }
        
        logger.info("📊 Métricas de Avaliação:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str = None) -> None:
        """Salva modelo treinado"""
        if filepath is None:
            filepath = config.MODEL_FILE
        
        if self.model is None:
            raise ValueError("Nenhum modelo para salvar")
        
        self.model.save(filepath)
        logger.info(f"✅ Modelo salvo: {filepath}")
    
    def load_model(self, filepath: str = None) -> None:
        """Carrega modelo salvo"""
        if filepath is None:
            filepath = config.MODEL_FILE
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"✅ Modelo carregado: {filepath}")