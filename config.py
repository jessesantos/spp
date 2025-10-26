import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configurações do sistema de predição"""
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    
    # URLs
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    # Arquivos de dados
    STOCK_DATA_FILE: str = "PETR4_2025_10_01_to_2025_10_25.csv"
    NEWS_DATA_FILE: str = "noticias_PETR4_setembro_2025.csv"
    OUTPUT_FILE: str = "dataset_PETR4_normalizado_completo.csv"
    MODEL_FILE: str = "petr4_lstm_model.keras"
    
    # Parâmetros do modelo
    SEQUENCE_LENGTH: int = 5       # Reduzido de 10 para 5 (dataset pequeno)
    TRAIN_TEST_SPLIT: float = 0.7  # Reduzido de 0.8 para 0.7
    EPOCHS: int = 50
    BATCH_SIZE: int = 4            # Reduzido de 16 para 4
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2  # 20% do treino para validação
    
    # Configurações de GPU
    GPU_ID: int = None  # None=todas, -1=CPU, 0,1,2...=GPU específica
    GPU_MEMORY_GROWTH: bool = True  # Crescimento dinâmico de memória
    GPU_MEMORY_LIMIT_MB: int = None  # Limite de memória em MB (None=sem limite)
    
    # Features técnicas
    MA_PERIODS: list = None
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: int = 2
    
    def __post_init__(self):
        if self.MA_PERIODS is None:
            self.MA_PERIODS = [5, 10, 20]
        
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY não encontrada no .env")
    
    def validate(self) -> None:
        """Valida configurações críticas"""
        if not os.path.exists(self.STOCK_DATA_FILE):
            raise FileNotFoundError(f"Arquivo de ações não encontrado: {self.STOCK_DATA_FILE}")
        
        if not os.path.exists(self.NEWS_DATA_FILE):
            raise FileNotFoundError(f"Arquivo de notícias não encontrado: {self.NEWS_DATA_FILE}")


config = Config()
