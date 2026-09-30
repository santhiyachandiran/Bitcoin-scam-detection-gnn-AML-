"""
Strictly Causal Dynamic Temporal Graph Neural Network (Recurrent GCN).
Combines Spatial GCN convolutions with a Node-Level Temporal GRU Memory mechanism.

Ensures 100% causal integrity:
- Processes 49 temporal snapshots sequentially (t = 1..49).
- Spatial message passing operates exclusively on intra-snapshot edges E_t and features X_t.
- Temporal information flows strictly forward in time (t-1 -> t) via GRU node memory states.
- Zero future temporal data leakage.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("temporal_gcn")


class RecurrentGCNClassifier(nn.Module):
    """
    Causal Recurrent Graph Convolutional Network.
    Applies spatial GCN convolution per snapshot and updates node temporal memory via GRU.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_gcn_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize Recurrent GCN Model.

        Args:
            in_channels: Feature vector dimension per node.
            hidden_channels: Hidden representation dimension.
            num_gcn_layers: Number of spatial GCNConv layers per snapshot.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_gcn_layers = num_gcn_layers
        self.dropout = dropout

        # 1. Spatial GCN Encoder
        self.convs = nn.ModuleList()
        if num_gcn_layers == 1:
            self.convs.append(GCNConv(in_channels, hidden_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_gcn_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # 2. Temporal Memory Mechanism (GRU Cell)
        self.gru_cell = nn.GRUCell(input_size=hidden_channels, hidden_size=hidden_channels)

        # 3. Classifier Output Head
        self.classifier = nn.Linear(hidden_channels, 1)

    def encode_snapshot(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial GCN embeddings for a single temporal snapshot.

        Args:
            x: Node features of snapshot t, shape (N_t, in_channels).
            edge_index: Local intra-snapshot edge indices of shape (2, E_t).

        Returns:
            Spatial node embeddings of shape (N_t, hidden_channels).
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        prev_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single temporal snapshot G_t.

        Args:
            x: Node feature matrix of snapshot t (N_t, in_channels).
            edge_index: Intra-snapshot edge indices (2, E_t).
            prev_states: Temporal memory states from previous timesteps (N_t, hidden_channels).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Logits of shape (N_t,)
                - Updated node temporal memory states of shape (N_t, hidden_channels)
        """
        # Step 1: Spatial GCN message passing on snapshot G_t
        spatial_embeds = self.encode_snapshot(x, edge_index)

        # Step 2: Temporal GRU memory update (forward in time)
        updated_states = self.gru_cell(spatial_embeds, prev_states)

        # Step 3: Binary classification logits
        logits = self.classifier(updated_states).squeeze(-1)

        return logits, updated_states


class RecurrentGCNTrainer:
    """
    Trainer for strictly causal Recurrent GCN on temporal graph snapshots.
    Handles sequential snapshot processing, weighted loss, early stopping, and checkpointing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_gcn_layers: int = 2,
        dropout: float = 0.2,
        device: Optional[str] = None,
    ):
        """
        Initialize Recurrent GCN Trainer.

        Args:
            in_channels: Node feature dimension.
            hidden_channels: Hidden representation dimension.
            num_gcn_layers: Number of spatial GCN layers per snapshot.
            dropout: Dropout probability.
            device: Computing device ('cuda' or 'cpu'). Auto-detects if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = RecurrentGCNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gcn_layers=num_gcn_layers,
            dropout=dropout,
        ).to(self.device)

        self.is_fitted = False
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.training_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": []
        }

    def _compute_train_pos_weight(self, snapshots: List[Data], train_timesteps: Tuple[int, int] = (1, 34)) -> torch.Tensor:
        """
        Compute positive class weight pos_weight = num_licit / num_illicit
        strictly from training timesteps.
        """
        num_licit = 0
        num_illicit = 0

        for t in range(train_timesteps[0], train_timesteps[1] + 1):
            snap = snapshots[t - 1]
            train_mask = snap.train_mask
            y_train = snap.y[train_mask]

            num_licit += int((y_train == 0).sum().item())
            num_illicit += int((y_train == 1).sum().item())

        if num_illicit > 0:
            pos_weight_val = num_licit / num_illicit
        else:
            pos_weight_val = 1.0

        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)
        logger.info(f"Class imbalance handling (Train Timesteps {train_timesteps}): train_licit={num_licit}, train_illicit={num_illicit} -> pos_weight={pos_weight_val:.4f}")
        return pos_weight

    def fit(
        self,
        snapshots: List[Data],
        total_nodes: int,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        patience: int = 15,
        train_timesteps: Tuple[int, int] = (1, 34),
        val_timesteps: Tuple[int, int] = (35, 39),
    ) -> "RecurrentGCNTrainer":
        """
        Train Recurrent GCN strictly sequentially across temporal snapshots.

        Args:
            snapshots: List of 49 PyG Data snapshot objects (t=1..49).
            total_nodes: Total number of unique nodes across full dataset.
            epochs: Maximum training epochs.
            lr: Learning rate.
            weight_decay: L2 regularization strength.
            patience: Early stopping patience epochs.
            train_timesteps: Inclusive start and end timesteps for training (default: 1-34).
            val_timesteps: Inclusive start and end timesteps for validation (default: 35-39).

        Returns:
            Self instance.
        """
        logger.info(f"Training Causal Recurrent GCN on device: {self.device} across {len(snapshots)} temporal snapshots...")
        pos_weight = self._compute_train_pos_weight(snapshots, train_timesteps=train_timesteps)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            # Global node temporal state memory initialized to zeros for each training pass
            global_node_states = torch.zeros((total_nodes, self.model.hidden_channels), device=self.device)

            train_loss_accum = torch.tensor(0.0, device=self.device)
            num_train_nodes_total = 0

            # -----------------------------------------------------------------
            # STEP 1: Process Training Snapshots Sequentially (t = 1 .. train_end)
            # -----------------------------------------------------------------
            for ts in range(1, train_timesteps[1] + 1):
                snap = snapshots[ts - 1].to(self.device)
                if snap.num_nodes == 0:
                    continue

                global_idx = snap.global_node_idx
                prev_states = global_node_states[global_idx]

                logits_t, updated_states = self.model.forward_snapshot(snap.x, snap.edge_index, prev_states)
                
                # Update global node memory for persistence into subsequent timesteps
                global_node_states[global_idx] = updated_states

                # Accumulate loss ONLY on labelled training mask nodes
                train_mask = snap.train_mask
                if train_mask.sum() > 0:
                    loss_t = criterion(logits_t[train_mask], snap.y[train_mask].float())
                    train_loss_accum = train_loss_accum + loss_t * train_mask.sum()
                    num_train_nodes_total += int(train_mask.sum().item())

            if num_train_nodes_total > 0:
                epoch_train_loss = train_loss_accum / num_train_nodes_total
                epoch_train_loss.backward()
                optimizer.step()
            else:
                epoch_train_loss = torch.tensor(0.0)

            # -----------------------------------------------------------------
            # STEP 2: Validation Evaluation (t = val_start .. val_end)
            # -----------------------------------------------------------------
            self.model.eval()
            val_loss_accum = 0.0
            num_val_nodes_total = 0

            val_preds_list = []
            val_trues_list = []

            with torch.no_grad():
                # Re-run forward propagation up through validation range sequentially
                eval_global_states = torch.zeros((total_nodes, self.model.hidden_channels), device=self.device)

                for ts in range(1, val_timesteps[1] + 1):
                    snap = snapshots[ts - 1].to(self.device)
                    if snap.num_nodes == 0:
                        continue

                    global_idx = snap.global_node_idx
                    prev_states = eval_global_states[global_idx]

                    logits_t, updated_states = self.model.forward_snapshot(snap.x, snap.edge_index, prev_states)
                    eval_global_states[global_idx] = updated_states

                    # Compute validation loss ONLY for timesteps in validation range
                    if val_timesteps[0] <= ts <= val_timesteps[1]:
                        val_mask = snap.val_mask
                        if val_mask.sum() > 0:
                            loss_t = criterion(logits_t[val_mask], snap.y[val_mask].float()).item()
                            val_loss_accum += loss_t * int(val_mask.sum().item())
                            num_val_nodes_total += int(val_mask.sum().item())

                            probs_t = torch.sigmoid(logits_t[val_mask]).cpu().numpy()
                            preds_t = (probs_t >= 0.5).astype(int)
                            trues_t = snap.y[val_mask].cpu().numpy()

                            val_preds_list.extend(preds_t)
                            val_trues_list.extend(trues_t)

            val_loss = (val_loss_accum / num_val_nodes_total) if num_val_nodes_total > 0 else 0.0

            # Calculate Val F1 score
            val_preds_arr = np.array(val_preds_list)
            val_trues_arr = np.array(val_trues_list)
            tp = np.logical_and(val_preds_arr == 1, val_trues_arr == 1).sum()
            fp = np.logical_and(val_preds_arr == 1, val_trues_arr == 0).sum()
            fn = np.logical_and(val_preds_arr == 0, val_trues_arr == 1).sum()
            val_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            self.training_history["train_loss"].append(epoch_train_loss.item())
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_f1"].append(val_f1)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {epoch_train_loss.item():.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

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

        # Restore best model state dict
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})

        self.is_fitted = True
        logger.info("Recurrent GCN model training complete.")
        return self

    def predict_snapshot_range(
        self,
        snapshots: List[Data],
        total_nodes: int,
        target_timesteps: Tuple[int, int],
        mask_attr: str = "test_mask",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run causal sequential forward pass and extract predictions for target timesteps.

        Args:
            snapshots: List of 49 PyG Data snapshot objects.
            total_nodes: Total number of unique nodes across full dataset.
            target_timesteps: Range of timesteps (start, end) to extract predictions for.
            mask_attr: Name of mask attribute ('train_mask', 'val_mask', 'test_mask').

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Ground truth labels array y_true
                - Predicted binary labels array y_pred (threshold 0.5)
                - Predicted positive class probabilities y_prob
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predicting.")

        self.model.eval()
        all_trues = []
        all_probs = []

        eval_global_states = torch.zeros((total_nodes, self.model.hidden_channels), device=self.device)

        with torch.no_grad():
            for ts in range(1, target_timesteps[1] + 1):
                snap = snapshots[ts - 1].to(self.device)
                if snap.num_nodes == 0:
                    continue

                global_idx = snap.global_node_idx
                prev_states = eval_global_states[global_idx]

                logits_t, updated_states = self.model.forward_snapshot(snap.x, snap.edge_index, prev_states)
                eval_global_states[global_idx] = updated_states

                if target_timesteps[0] <= ts <= target_timesteps[1]:
                    mask = getattr(snap, mask_attr)
                    if mask.sum() > 0:
                        probs_t = torch.sigmoid(logits_t[mask]).cpu().numpy()
                        trues_t = snap.y[mask].cpu().numpy()

                        all_probs.extend(probs_t)
                        all_trues.extend(trues_t)

        y_prob = np.array(all_probs, dtype=np.float32)
        y_true = np.array(all_trues, dtype=np.int64)
        y_pred = (y_prob >= 0.5).astype(np.int64)

        return y_true, y_pred, y_prob

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save Recurrent GCN model checkpoint to disk.

        Args:
            filepath: Destination file path.

        Returns:
            Saved Path.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.model.in_channels,
            "hidden_channels": self.model.hidden_channels,
            "num_gcn_layers": self.model.num_gcn_layers,
            "dropout": self.model.dropout,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved Recurrent GCN model checkpoint to {filepath}")
        return filepath

    def load(self, filepath: Union[str, Path]) -> "RecurrentGCNTrainer":
        """
        Load Recurrent GCN model checkpoint from disk.

        Args:
            filepath: Source file path.

        Returns:
            Loaded self instance.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Recurrent GCN checkpoint not found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model = RecurrentGCNClassifier(
            in_channels=checkpoint["in_channels"],
            hidden_channels=checkpoint["hidden_channels"],
            num_gcn_layers=checkpoint["num_gcn_layers"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict({k: v.to(self.device) for k, v in checkpoint["model_state_dict"].items()})
        self.training_history = checkpoint.get("training_history", {})
        self.is_fitted = True
        logger.info(f"Loaded Recurrent GCN model checkpoint from {filepath}")
        return self
