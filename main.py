import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys

from config import config
from data_loader import DataLoader
from sentiment_analyzer import SentimentAnalyzer
from feature_engine import FeatureEngine
from model import PredictionModel
from gpu_manager import setup_gpu

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InvestmentPredictionSystem:
    """Sistema completo de predição de investimentos"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_engine = FeatureEngine()
        self.model = PredictionModel()
        self.final_df = None
    
    def run_pipeline(self, skip_sentiment: bool = False) -> pd.DataFrame:
        """
        Executa pipeline completo do sistema
        
        Args:
            skip_sentiment: Se True, pula análise de sentimento (útil para testes rápidos)
        
        Returns:
            DataFrame final com todas as features
        """
        logger.info("=" * 60)
        logger.info("🚀 SISTEMA DE PREDIÇÃO DE INVESTIMENTOS - PETR4")
        logger.info("=" * 60)
        
        # Configurar GPU
        logger.info("\n🎮 CONFIGURAÇÃO DE GPU")
        logger.info("-" * 60)
        gpu_manager = setup_gpu(
            gpu_id=config.GPU_ID,
            memory_growth=config.GPU_MEMORY_GROWTH,
            memory_limit_mb=config.GPU_MEMORY_LIMIT_MB
        )
        gpu_manager.print_gpu_info()
        logger.info(f"Dispositivo selecionado: {gpu_manager.get_current_device()}")
        
        # Validar configuração
        try:
            config.validate()
        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}")
            sys.exit(1)
        
        # 1. Carregar dados
        logger.info("\n📂 ETAPA 1: Carregamento de Dados")
        logger.info("-" * 60)
        df = self.data_loader.merge_datasets()
        
        # 2. Análise de sentimento
        if not skip_sentiment:
            logger.info("\n🧠 ETAPA 2: Análise de Sentimento (Gemini API)")
            logger.info("-" * 60)
            df = self._add_sentiment_analysis(df)
        else:
            logger.info("\n⚠️  Pulando análise de sentimento")
            df["sentiment"] = 0
            df["sentiment_confidence"] = 0.5
        
        # 3. Feature engineering
        logger.info("\n⚙️  ETAPA 3: Feature Engineering")
        logger.info("-" * 60)
        df = self.feature_engine.create_all_features(df)
        
        # 4. Adicionar target
        df = self.feature_engine.add_target_variable(df)
        
        # 5. Salvar dataset processado
        logger.info("\n💾 ETAPA 4: Salvamento de Dados")
        logger.info("-" * 60)
        df.to_csv(config.OUTPUT_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"✅ Dataset salvo: {config.OUTPUT_FILE}")
        
        self.final_df = df
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ PIPELINE CONCLUÍDO COM SUCESSO")
        logger.info("=" * 60)
        logger.info(f"📊 Total de registros: {len(df)}")
        logger.info(f"📊 Total de features: {len(df.columns)}")
        logger.info("=" * 60)
        
        return df
    
    def _add_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona análise de sentimento ao DataFrame"""
        texts = df["text"].tolist()
        
        # Analisar sentimento em batch
        results = self.sentiment_analyzer.batch_analyze(texts)
        
        # Adicionar ao DataFrame
        sentiments = []
        confidences = []
        
        for i in range(len(df)):
            if i in results:
                sent, conf = results[i]
                sentiments.append(sent)
                confidences.append(conf)
            else:
                sentiments.append(0)
                confidences.append(0.5)
        
        df["sentiment"] = sentiments
        df["sentiment_confidence"] = confidences
        
        logger.info(f"✅ Sentimentos analisados: {len(results)}")
        logger.info(f"   Positivos: {sum(1 for s in sentiments if s == 1)}")
        logger.info(f"   Neutros: {sum(1 for s in sentiments if s == 0)}")
        logger.info(f"   Negativos: {sum(1 for s in sentiments if s == -1)}")
        
        return df
    
    def train_model(self, df: pd.DataFrame = None) -> dict:
        """
        Treina modelo de predição
        
        Args:
            df: DataFrame processado (usa self.final_df se None)
            
        Returns:
            Dicionário com métricas de avaliação
        """
        logger.info("\n" + "=" * 60)
        logger.info("🎯 TREINAMENTO DO MODELO")
        logger.info("=" * 60)
        
        if df is None:
            if self.final_df is None:
                raise ValueError("Execute run_pipeline() primeiro")
            df = self.final_df
        
        # Validar dados mínimos
        min_samples = config.SEQUENCE_LENGTH * 3  # Mínimo 3x o sequence length
        if len(df) < min_samples:
            logger.error(f"❌ Dataset muito pequeno: {len(df)} amostras")
            logger.error(f"   Mínimo necessário: {min_samples} amostras")
            logger.error(f"   Adicione mais dados históricos ou reduza SEQUENCE_LENGTH em config.py")
            raise ValueError(f"Dataset insuficiente: {len(df)} < {min_samples}")
        
        # Split temporal
        split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
        
        # Garantir mínimo de amostras para teste
        min_test = config.SEQUENCE_LENGTH + 2
        if len(df) - split_idx < min_test:
            split_idx = len(df) - min_test
            logger.warning(f"⚠️  Ajustado split para garantir teste mínimo")
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"📊 Split: {len(train_df)} treino / {len(test_df)} teste")
        
        # Validar tamanho após split
        if len(test_df) <= config.SEQUENCE_LENGTH:
            logger.error(f"❌ Dados de teste insuficientes: {len(test_df)}")
            raise ValueError("Dataset de teste muito pequeno após split")
        
        # Preparar dados
        logger.info("\n⚙️  Preparando dados...")
        X_train, y_train = self.model.prepare_data(train_df)
        
        if len(X_train) == 0:
            logger.error("❌ Nenhuma sequência de treino criada")
            raise ValueError("Dataset de treino insuficiente para criar sequências")
        
        # Preparar teste (usar mesmo scaler)
        X_test = test_df[self.model.feature_columns].values
        y_test = test_df["target_price"].values.reshape(-1, 1)
        
        X_test_scaled = self.model.scaler_features.transform(X_test)
        y_test_scaled = self.model.scaler_target.transform(y_test)
        
        X_test_seq, y_test_seq = self.model.create_sequences(X_test_scaled, y_test_scaled)
        
        if len(X_test_seq) == 0:
            logger.error("❌ Nenhuma sequência de teste criada")
            raise ValueError("Dataset de teste insuficiente para criar sequências")
        
        # Criar validação (últimos X% do treino)
        val_split_ratio = config.VALIDATION_SPLIT
        val_split = int(len(X_train) * (1 - val_split_ratio))
        
        if val_split < 1:
            # Dataset muito pequeno, treinar sem validação
            logger.warning("⚠️  Dataset pequeno: treinando sem validação")
            X_val = None
            y_val = None
        else:
            X_val = X_train[val_split:]
            y_val = y_train[val_split:]
            X_train = X_train[:val_split]
            y_train = y_train[:val_split]
        
        logger.info(f"✅ Treino: {len(X_train)} | Validação: {len(X_val) if X_val is not None else 0} | Teste: {len(X_test_seq)}")
        
        # Treinar
        logger.info("\n🚂 Iniciando treinamento...\n")
        history = self.model.train(X_train, y_train, X_val, y_val)
        
        # Avaliar
        logger.info("\n📊 Avaliando modelo...")
        metrics = self.model.evaluate(X_test_seq, y_test_seq)
        
        # Salvar modelo
        self.model.save_model()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ TREINAMENTO CONCLUÍDO")
        logger.info("=" * 60)
        
        return metrics
    
    def predict_next_day(self, df: pd.DataFrame = None) -> dict:
        """
        Faz predição para o próximo dia de negociação
        
        Args:
            df: DataFrame processado (usa self.final_df se None)
            
        Returns:
            Dicionário com predição e informações
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔮 PREDIÇÃO PARA PRÓXIMO DIA")
        logger.info("=" * 60)
        
        if df is None:
            if self.final_df is None:
                raise ValueError("Execute run_pipeline() primeiro")
            df = self.final_df
        
        # Preparar últimos N dias
        last_n_days = df.tail(config.SEQUENCE_LENGTH).copy()
        
        # Extrair features
        X = last_n_days[self.model.feature_columns].values
        X_scaled = self.model.scaler_features.transform(X)
        X_seq = X_scaled.reshape(1, config.SEQUENCE_LENGTH, -1)
        
        # Predizer
        prediction_scaled = self.model.model.predict(X_seq, verbose=0)
        prediction = self.model.scaler_target.inverse_transform(prediction_scaled)[0][0]
        
        # Informações contextuais
        last_price = df["close"].iloc[-1]
        last_date = df["date"].iloc[-1]
        next_date = last_date + timedelta(days=1)
        
        change = prediction - last_price
        change_percent = (change / last_price) * 100
        direction = "ALTA ⬆️" if change > 0 else "BAIXA ⬇️"
        
        result = {
            "last_date": last_date.strftime("%Y-%m-%d"),
            "next_date": next_date.strftime("%Y-%m-%d"),
            "last_price": last_price,
            "predicted_price": prediction,
            "change": change,
            "change_percent": change_percent,
            "direction": direction
        }
        
        # Exibir resultado
        logger.info(f"\n📅 Última data: {result['last_date']}")
        logger.info(f"💰 Último preço: R$ {result['last_price']:.2f}")
        logger.info(f"\n🔮 Predição para {result['next_date']}:")
        logger.info(f"   Preço: R$ {result['predicted_price']:.2f}")
        logger.info(f"   Variação: R$ {result['change']:.2f} ({result['change_percent']:+.2f}%)")
        logger.info(f"   Tendência: {result['direction']}")
        logger.info("=" * 60)
        
        return result


def main():
    """Função principal"""
    try:
        # Criar sistema
        system = InvestmentPredictionSystem()
        
        # Executar pipeline completo
        df = system.run_pipeline(skip_sentiment=False)
        
        # Treinar modelo
        metrics = system.train_model(df)
        
        # Fazer predição
        prediction = system.predict_next_day(df)
        
        logger.info("\n✅ Sistema executado com sucesso!")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Execução interrompida pelo usuário")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"\n❌ Erro fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
