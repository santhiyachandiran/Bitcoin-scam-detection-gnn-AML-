"""
Models Package for Bitcoin Scam Detection.
Provides Logistic Regression baseline, Random Forest baseline, and PyG GCN model/trainer.
"""

from src.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from src.models.gcn import GCNClassifier, GCNTrainer

__all__ = [
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "GCNClassifier",
    "GCNTrainer",
]
