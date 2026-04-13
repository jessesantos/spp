"""Machine-learning core: feature engineering, LSTM model, Claude sentiment."""

from app.ml.features import FeatureConfig, build_features
from app.ml.aggregator import aggregate_sentiment

__all__ = ["FeatureConfig", "build_features", "aggregate_sentiment"]
