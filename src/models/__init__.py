"""
Models Package for Bitcoin Scam Detection.
Provides Logistic Regression baseline, Random Forest baseline, PyG GCN model, and Causal Recurrent GCN model.
"""

from src.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from src.models.gcn import GCNClassifier, GCNTrainer
from src.models.temporal_gcn import RecurrentGCNClassifier, RecurrentGCNTrainer

__all__ = [
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "GCNClassifier",
    "GCNTrainer",
    "RecurrentGCNClassifier",
    "RecurrentGCNTrainer",
]
