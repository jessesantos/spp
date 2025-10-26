"""
Script para fazer predições usando modelo já treinado
"""
import pandas as pd
import numpy as np
import logging
from datetime import timedelta
import sys

from config import config
from data_loader import DataLoader
from feature_engine import FeatureEngine
from model import PredictionModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_next_days(model_path: str = None, n_days: int = 5) -> pd.DataFrame:
    """
    Faz predições para os próximos N dias
    
    Args:
        model_path: Caminho do modelo salvo
        n_days: Número de dias para prever
        
    Returns:
        DataFrame com predições
    """
    logger.info("=" * 60)
    logger.info("🔮 PREDIÇÃO MULTI-DIA")
    logger.info("=" * 60)
    
    # Carregar dados
    logger.info("\n📂 Carregando dados...")
    data_loader = DataLoader()
    df = data_loader.merge_datasets()
    
    # Processar features (sem sentimento para predição rápida)
    logger.info("⚙️  Processando features...")
    df["sentiment"] = 0
    df["sentiment_confidence"] = 0.5
    feature_engine = FeatureEngine()
    df = feature_engine.create_all_features(df)
    df = feature_engine.add_target_variable(df)
    
    # Carregar modelo
    logger.info("📦 Carregando modelo...")
    model = PredictionModel()
    model.load_model(model_path or config.MODEL_FILE)
    
    # Preparar dados (para treinar scaler)
    model.prepare_features(df)
    X_all = df[model.feature_columns].values
    y_all = df["target_price"].values.reshape(-1, 1)
    model.scaler_features.fit(X_all)
    model.scaler_target.fit(y_all)
    
    # Fazer predições iterativas
    predictions = []
    current_df = df.copy()
    
    for day in range(n_days):
        # Últimos N dias
        last_n = current_df.tail(config.SEQUENCE_LENGTH)
        
        # Preparar entrada
        X = last_n[model.feature_columns].values
        X_scaled = model.scaler_features.transform(X)
        X_seq = X_scaled.reshape(1, config.SEQUENCE_LENGTH, -1)
        
        # Predizer
        pred_scaled = model.model.predict(X_seq, verbose=0)
        pred_price = model.scaler_target.inverse_transform(pred_scaled)[0][0]
        
        # Calcular variação
        last_price = current_df["close"].iloc[-1]
        change = pred_price - last_price
        change_pct = (change / last_price) * 100
        
        # Data da predição
        last_date = current_df["date"].iloc[-1]
        pred_date = last_date + timedelta(days=1)
        
        predictions.append({
            "date": pred_date,
            "predicted_price": pred_price,
            "previous_price": last_price,
            "change": change,
            "change_percent": change_pct,
            "direction": "ALTA" if change > 0 else "BAIXA"
        })
        
        # Adicionar predição ao df para próxima iteração
        # (simulando que a predição se torna realidade)
        new_row = current_df.iloc[-1].copy()
        new_row["date"] = pred_date
        new_row["close"] = pred_price
        new_row["open"] = last_price
        new_row["high"] = max(last_price, pred_price)
        new_row["low"] = min(last_price, pred_price)
        
        # Recalcular features
        temp_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        temp_df = feature_engine.create_all_features(temp_df)
        current_df = temp_df
    
    # Criar DataFrame de resultados
    results_df = pd.DataFrame(predictions)
    
    # Exibir resultados
    logger.info("\n📊 PREDIÇÕES:")
    logger.info("-" * 60)
    for idx, row in results_df.iterrows():
        logger.info(
            f"{row['date'].strftime('%Y-%m-%d')} | "
            f"R$ {row['predicted_price']:.2f} | "
            f"{row['change_percent']:+.2f}% | "
            f"{row['direction']}"
        )
    
    logger.info("=" * 60)
    
    return results_df


def analyze_model_performance(model_path: str = None) -> dict:
    """
    Analisa performance do modelo no conjunto de teste
    
    Args:
        model_path: Caminho do modelo salvo
        
    Returns:
        Dicionário com métricas
    """
    logger.info("=" * 60)
    logger.info("📊 ANÁLISE DE PERFORMANCE")
    logger.info("=" * 60)
    
    # Carregar dados
    logger.info("\n📂 Carregando dados...")
    data_loader = DataLoader()
    df = data_loader.merge_datasets()
    
    # Processar features
    df["sentiment"] = 0
    df["sentiment_confidence"] = 0.5
    feature_engine = FeatureEngine()
    df = feature_engine.create_all_features(df)
    df = feature_engine.add_target_variable(df)
    
    # Split
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Carregar modelo
    logger.info("📦 Carregando modelo...")
    model = PredictionModel()
    model.load_model(model_path or config.MODEL_FILE)
    
    # Preparar dados
    model.prepare_features(df)
    
    # Treinar scalers com dados de treino
    X_train = train_df[model.feature_columns].values
    y_train = train_df["target_price"].values.reshape(-1, 1)
    model.scaler_features.fit(X_train)
    model.scaler_target.fit(y_train)
    
    # Preparar teste
    X_test = test_df[model.feature_columns].values
    y_test = test_df["target_price"].values.reshape(-1, 1)
    
    X_test_scaled = model.scaler_features.transform(X_test)
    y_test_scaled = model.scaler_target.transform(y_test)
    
    X_test_seq, y_test_seq = model.create_sequences(X_test_scaled, y_test_scaled)
    
    # Avaliar
    logger.info("\n📈 Métricas no conjunto de teste:")
    metrics = model.evaluate(X_test_seq, y_test_seq)
    
    logger.info("=" * 60)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preditor PETR4")
    parser.add_argument(
        "--mode",
        choices=["predict", "analyze"],
        default="predict",
        help="Modo de operação"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Número de dias para prever"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Caminho do modelo"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "predict":
            results = predict_next_days(args.model, args.days)
            results.to_csv("predictions.csv", index=False)
            logger.info("\n✅ Predições salvas em: predictions.csv")
        
        else:
            metrics = analyze_model_performance(args.model)
    
    except Exception as e:
        logger.error(f"❌ Erro: {e}", exc_info=True)
        sys.exit(1)
