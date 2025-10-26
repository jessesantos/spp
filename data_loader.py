import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Tuple
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Carrega e pré-processa dados de ações e notícias"""
    
    def __init__(self):
        self.stock_df = None
        self.news_df = None
        self.merged_df = None
    
    def load_stock_data(self) -> pd.DataFrame:
        """Carrega dados históricos de preços da ação"""
        logger.info(f"Carregando dados de ações: {config.STOCK_DATA_FILE}")
        
        # Ler CSV
        df = pd.read_csv(config.STOCK_DATA_FILE)
        
        # Normalizar nomes de colunas
        df.columns = [col.strip() for col in df.columns]
        
        # Mapear colunas
        column_mapping = {
            "Data": "date",
            "Último": "close",
            "Abertura": "open",
            "Máxima": "high",
            "Mínima": "low",
            "Vol.": "volume",
            "Var%": "change"
        }
        
        df = df.rename(columns=column_mapping)
        
        # Converter data para formato padrão
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
        
        # Converter valores numéricos (remover vírgulas e M/K)
        numeric_columns = ["close", "open", "high", "low"]
        for col in numeric_columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
        
        # Processar volume
        df["volume"] = df["volume"].astype(str).str.replace("M", "").str.replace(",", ".").astype(float) * 1_000_000
        
        # Processar variação percentual
        df["change"] = df["change"].astype(str).str.replace("%", "").str.replace(",", ".").astype(float)
        
        # Ordenar por data crescente
        df = df.sort_values("date").reset_index(drop=True)
        
        # Remover valores nulos
        df = df.dropna(subset=["date", "close"])
        
        logger.info(f"✅ {len(df)} dias de dados carregados")
        
        self.stock_df = df
        return df
    
    def load_news_data(self) -> pd.DataFrame:
        """Carrega dados de notícias"""
        logger.info(f"Carregando notícias: {config.NEWS_DATA_FILE}")
        
        # Ler CSV
        df = pd.read_csv(config.NEWS_DATA_FILE)
        
        # Normalizar colunas
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Converter data
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        
        # Criar texto combinado (título + descrição)
        df["text"] = (
            df["title"].fillna("") + " " + df["description"].fillna("")
        ).str.strip()
        
        # Substituir textos vazios
        df.loc[df["text"] == "", "text"] = "sem notícia relevante"
        
        # Remover duplicatas
        df = df.drop_duplicates(subset=["date"]).reset_index(drop=True)
        
        logger.info(f"✅ {len(df)} notícias carregadas")
        
        self.news_df = df
        return df
    
    def merge_datasets(self) -> pd.DataFrame:
        """Combina dados de ações e notícias"""
        if self.stock_df is None:
            self.load_stock_data()
        
        if self.news_df is None:
            self.load_news_data()
        
        logger.info("Mesclando datasets...")
        
        # Garantir formato de data consistente
        self.stock_df["date"] = pd.to_datetime(self.stock_df["date"])
        self.news_df["date"] = pd.to_datetime(self.news_df["date"])
        
        # Merge left para manter todos os dias de negociação
        merged = pd.merge(
            self.stock_df,
            self.news_df[["date", "text"]],
            on="date",
            how="left"
        )
        
        # Preencher notícias ausentes
        merged["text"] = merged["text"].fillna("sem notícia relevante")
        
        # Ordenar por data
        merged = merged.sort_values("date").reset_index(drop=True)
        
        logger.info(f"✅ Dataset mesclado: {len(merged)} registros")
        
        self.merged_df = merged
        return merged
    
    def get_train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Divide dataset em treino e teste mantendo ordem temporal"""
        if self.merged_df is None:
            self.merge_datasets()
        
        split_idx = int(len(self.merged_df) * config.TRAIN_TEST_SPLIT)
        
        train_df = self.merged_df.iloc[:split_idx].copy()
        test_df = self.merged_df.iloc[split_idx:].copy()
        
        logger.info(f"Split: {len(train_df)} treino / {len(test_df)} teste")
        
        return train_df, test_df
