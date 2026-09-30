"""
Models Package for Bitcoin Scam Detection.
Provides Logistic Regression baseline, Random Forest baseline, PyG GCN model, Causal Recurrent GCN,
and Triplet-Style Dynamic GNN with Transformer Encoder.
"""

from src.models.baselines import LogisticRegressionBaseline, RandomForestBaseline
from src.models.gcn import GCNClassifier, GCNTrainer
from src.models.temporal_gcn import RecurrentGCNClassifier, RecurrentGCNTrainer
from src.models.triplet_transformer_gcn import (
    TripletTransformerGCNClassifier,
    TripletTransformerGCNTrainer,
    compute_online_triplet_loss,
)

__all__ = [
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "GCNClassifier",
    "GCNTrainer",
    "RecurrentGCNClassifier",
    "RecurrentGCNTrainer",
    "TripletTransformerGCNClassifier",
    "TripletTransformerGCNTrainer",
    "compute_online_triplet_loss",
]
