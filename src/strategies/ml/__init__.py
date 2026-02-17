"""
Machine learning based trading strategies.
"""

from .knn import KNNStrategy
from .random_forest import RandomForestStrategy
from .xgboost_strategy import XGBoostStrategy
from .clustering import ClusteringStrategy

__all__ = [
    'KNNStrategy',
    'RandomForestStrategy',
    'XGBoostStrategy',
    'ClusteringStrategy'
]
