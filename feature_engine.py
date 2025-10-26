import pandas as pd
import numpy as np
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngine:
    """Cria features técnicas para análise de ações"""
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona médias móveis simples"""
        for period in config.MA_PERIODS:
            df[f"ma_{period}"] = df["close"].rolling(window=period, min_periods=1).mean()
        
        logger.info(f"✅ Médias móveis adicionadas: {config.MA_PERIODS}")
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Adiciona Relative Strength Index (RSI)"""
        if period is None:
            period = config.RSI_PERIOD
        
        # Calcular variações
        delta = df["close"].diff()
        
        # Separar ganhos e perdas
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Calcular médias móveis exponenciais
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calcular RS e RSI
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        df["rsi"] = rsi.fillna(50)
        
        logger.info(f"✅ RSI adicionado (período: {period})")
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona MACD (Moving Average Convergence Divergence)"""
        # Calcular EMAs
        ema_fast = df["close"].ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = df["close"].ewm(span=config.MACD_SLOW, adjust=False).mean()
        
        # MACD line
        df["macd"] = ema_fast - ema_slow
        
        # Signal line
        df["macd_signal"] = df["macd"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
        
        # Histogram
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        logger.info("✅ MACD adicionado")
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona Bandas de Bollinger"""
        period = config.BOLLINGER_PERIOD
        std_dev = config.BOLLINGER_STD
        
        # Média móvel simples
        sma = df["close"].rolling(window=period, min_periods=1).mean()
        
        # Desvio padrão
        std = df["close"].rolling(window=period, min_periods=1).std()
        
        # Bandas
        df["bb_upper"] = sma + (std * std_dev)
        df["bb_middle"] = sma
        df["bb_lower"] = sma - (std * std_dev)
        
        # Bandwidth e %B
        df["bb_bandwidth"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        logger.info("✅ Bandas de Bollinger adicionadas")
        return df
    
    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Adiciona volatilidade histórica"""
        # Retornos logarítmicos
        log_returns = np.log(df["close"] / df["close"].shift(1))
        
        # Volatilidade (desvio padrão dos retornos)
        df["volatility"] = log_returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
        
        logger.info("✅ Volatilidade adicionada")
        return df
    
    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona retornos percentuais diários"""
        df["daily_return"] = df["close"].pct_change() * 100
        df["daily_return"] = df["daily_return"].fillna(0)
        
        logger.info("✅ Retornos diários adicionados")
        return df
    
    @staticmethod
    def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Adiciona indicador de momentum"""
        df["momentum"] = df["close"].diff(period)
        df["momentum"] = df["momentum"].fillna(0)
        
        logger.info("✅ Momentum adicionado")
        return df
    
    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features relacionadas ao volume"""
        # Volume médio
        df["volume_ma_5"] = df["volume"].rolling(window=5, min_periods=1).mean()
        
        # Razão volume atual / média
        df["volume_ratio"] = df["volume"] / df["volume_ma_5"].replace(0, 1)
        
        logger.info("✅ Features de volume adicionadas")
        return df
    
    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Cria todas as features técnicas"""
        logger.info("Criando features técnicas...")
        
        df = df.copy()
        
        # Adicionar todas as features
        df = cls.add_moving_averages(df)
        df = cls.add_rsi(df)
        df = cls.add_macd(df)
        df = cls.add_bollinger_bands(df)
        df = cls.add_volatility(df)
        df = cls.add_returns(df)
        df = cls.add_momentum(df)
        df = cls.add_volume_features(df)
        
        # Preencher NaN com 0 ou método forward fill
        df = df.ffill().fillna(0)
        
        logger.info(f"✅ Total de features: {len(df.columns)}")
        
        return df
    
    @staticmethod
    def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona variável target (preço futuro)"""
        # Target: preço de fechamento do próximo dia
        df["target_price"] = df["close"].shift(-1)
        
        # Target binário: 1 se sobe, 0 se cai
        df["target_direction"] = (df["target_price"] > df["close"]).astype(int)
        
        # Remover última linha (sem target)
        df = df[:-1].copy()
        
        logger.info("✅ Target variable adicionada")
        
        return df
