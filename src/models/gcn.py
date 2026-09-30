"""
Graph Convolutional Network (GCN) Model and Trainer for Elliptic Bitcoin Scam Detection.
Uses PyTorch Geometric GCNConv with weighted loss for handling severe class imbalance.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("gcn_model")


class GCNClassifier(nn.Module):
    """
    PyTorch Geometric Graph Convolutional Network (GCN) for node classification.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize GCN model architecture.

        Args:
            in_channels: Input feature dimension per node.
            hidden_channels: Hidden representation dimension.
            out_channels: Output dimension (1 for binary logits).
            num_layers: Number of GCNConv layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN model.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Graph edge indices of shape (2, E).

        Returns:
            Logits tensor of shape (N,).
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.convs[-1](x, edge_index)
        return logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits


class GCNTrainer:
    """
    Trainer for GCN model on PyTorch Geometric graph data.
    Handles weighted BCE loss, early stopping, and evaluation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: Optional[str] = None,
    ):
        """
        Initialize GCN Trainer.

        Args:
            in_channels: Input feature dimension per node.
            hidden_channels: Hidden representation dimension.
            num_layers: Number of GCN layer steps.
            dropout: Dropout probability.
            device: Computing device ('cuda' or 'cpu'). If None, auto-selects.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GCNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.is_fitted = False
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.training_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": []
        }

    def fit(
        self,
        data: Data,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        patience: int = 15,
    ) -> "GCNTrainer":
        """
        Train the GCN model using PyG Data object and temporal train/val masks.

        Args:
            data: PyG Data object containing x, edge_index, y, train_mask, val_mask.
            epochs: Maximum training epochs.
            lr: Learning rate.
            weight_decay: L2 regularization strength.
            patience: Early stopping patience epochs.

        Returns:
            Self instance.
        """
        logger.info(f"Training GCN classifier on device: {self.device} for up to {epochs} epochs...")
        data = data.to(self.device)

        train_mask = data.train_mask
        val_mask = data.val_mask

        # Compute pos_weight for class imbalance handling strictly on train mask
        train_y = data.y[train_mask]
        num_licit = (train_y == 0).sum().item()
        num_illicit = (train_y == 1).sum().item()

        if num_illicit > 0:
            pos_weight = torch.tensor([num_licit / num_illicit], dtype=torch.float32, device=self.device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float32, device=self.device)

        logger.info(f"Class imbalance handling: train_licit={num_licit}, train_illicit={num_illicit} -> pos_weight={pos_weight.item():.4f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            logits = self.model(data.x, data.edge_index)
            train_loss = criterion(logits[train_mask], data.y[train_mask].float())

            train_loss.backward()
            optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(data.x, data.edge_index)
                val_loss = criterion(val_logits[val_mask], data.y[val_mask].float()).item()
                val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                val_preds = (val_probs >= 0.5).astype(int)
                
                val_y_true = data.y[val_mask].cpu().numpy()
                
                # Compute Val F1 score
                tp = np.logical_and(val_preds == 1, val_y_true == 1).sum()
                fp = np.logical_and(val_preds == 1, val_y_true == 0).sum()
                fn = np.logical_and(val_preds == 0, val_y_true == 1).sum()
                val_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            self.training_history["train_loss"].append(train_loss.item())
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_f1"].append(val_f1)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

            # Early stopping check based on val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f}).")
                    break

        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})

        self.is_fitted = True
        logger.info("GCN model training completed.")
        return self

    def predict_proba(self, data: Data, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Predict class probabilities for positive class (class 1: Illicit).

        Args:
            data: PyG Data object.
            mask: Optional boolean tensor mask. If provided, returns probabilities for masked nodes.

        Returns:
            Probability of class 1 array of shape (N_mask,).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predict_proba().")

        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            if mask is not None:
                logits = logits[mask]
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, data: Data, mask: Optional[torch.Tensor] = None, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            data: PyG Data object.
            mask: Optional boolean tensor mask.
            threshold: Probability decision threshold (default: 0.5).

        Returns:
            Predicted binary labels array of shape (N_mask,).
        """
        probs = self.predict_proba(data, mask=mask)
        return (probs >= threshold).astype(np.int64)

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save GCN model checkpoint to disk.

        Args:
            filepath: Destination path.

        Returns:
            Saved Path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.model.in_channels,
            "hidden_channels": self.model.hidden_channels,
            "num_layers": self.model.num_layers,
            "dropout": self.model.dropout,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved GCN model checkpoint to {filepath}")
        return filepath

    def load(self, filepath: Union[str, Path]) -> "GCNTrainer":
        """
        Load GCN model checkpoint from disk.

        Args:
            filepath: Source file path.

        Returns:
            Loaded self instance.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GCN checkpoint not found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model = GCNClassifier(
            in_channels=checkpoint["in_channels"],
            hidden_channels=checkpoint["hidden_channels"],
            out_channels=1,
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict({k: v.to(self.device) for k, v in checkpoint["model_state_dict"].items()})
        self.training_history = checkpoint.get("training_history", {})
        self.is_fitted = True
        logger.info(f"Loaded GCN model checkpoint from {filepath}")
        return self
