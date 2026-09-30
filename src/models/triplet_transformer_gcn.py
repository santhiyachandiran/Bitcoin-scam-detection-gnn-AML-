"""
Triplet-Style Dynamic Graph Network with Transformer Encoder for Bitcoin Scam Detection.
Inspired by "Triplet-Style Dynamic Graph Network With Transformer Encoder for Scam Detection in Cryptocurrency Transactions".

Combines:
1. Spatial GCN Convolution (GCNConv) for structural neighborhood encoding per snapshot G_t.
2. Temporal Transformer Encoder (TransformerEncoderLayer) for temporal sequence dynamics.
3. Online Triplet Margin Loss for contrastive representation learning (Illicit vs Licit separation).
4. Strictly causal temporal memory propagation (zero future leakage).
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

logger = get_logger("triplet_transformer_gcn")


class TripletTransformerGCNClassifier(nn.Module):
    """
    Triplet-Style Dynamic GNN with Spatial GCN, Transformer Encoder, and Embedding Projection Head.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_gcn_layers: int = 2,
        nhead: int = 4,
        num_transformer_layers: int = 1,
        dropout: float = 0.2,
    ):
        """
        Initialize Triplet Transformer GCN Classifier.

        Args:
            in_channels: Feature vector dimension per node.
            hidden_channels: Hidden representation dimension.
            num_gcn_layers: Number of spatial GCNConv layers per snapshot.
            nhead: Number of attention heads in Transformer Encoder.
            num_transformer_layers: Number of Transformer Encoder layers.
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

        # 2. Temporal Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=hidden_channels * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers,
        )

        # 3. Embedding Projection Head (for Triplet Loss contrastive representation)
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # 4. Classification Output Head
        self.classifier = nn.Linear(hidden_channels, 1)

    def encode_spatial(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial GCN embeddings for a single snapshot G_t.

        Args:
            x: Node features of snapshot t, shape (N_t, in_channels).
            edge_index: Intra-snapshot edge indices of shape (2, E_t).

        Returns:
            Spatial embeddings of shape (N_t, hidden_channels).
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        prev_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single temporal snapshot G_t.

        Args:
            x: Node feature matrix (N_t, in_channels).
            edge_index: Intra-snapshot edge indices (2, E_t).
            prev_states: Temporal memory states from previous timesteps (N_t, hidden_channels).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Logits tensor of shape (N_t,)
                - L2-normalized embedding projections of shape (N_t, hidden_channels) for Triplet Loss
                - Updated node temporal representations of shape (N_t, hidden_channels)
        """
        # Step 1: Spatial GCN message passing on snapshot G_t
        spatial_embeds = self.encode_spatial(x, edge_index)

        # Step 2: Combine historical state and spatial embedding into sequence tensor (N_t, 2, hidden_dim)
        seq_input = torch.stack([prev_states, spatial_embeds], dim=1)

        # Step 3: Pass through Transformer Encoder for temporal self-attention
        tf_output = self.transformer_encoder(seq_input)
        updated_states = tf_output[:, -1, :]  # Extract refined temporal-spatial representation

        # Step 4: Compute L2-normalized embedding projection for Triplet Loss
        raw_embeds = self.embedding_head(updated_states)
        norm_embeds = F.normalize(raw_embeds, p=2, dim=-1)

        # Step 5: Compute binary classification logits
        logits = self.classifier(updated_states).squeeze(-1)

        return logits, norm_embeds, updated_states


def compute_online_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    max_triplets: int = 500,
) -> torch.Tensor:
    """
    Mine online triplets (Anchor=Illicit, Positive=Illicit, Negative=Licit) and compute Triplet Margin Loss.

    Args:
        embeddings: Normalized node embedding matrix of shape (N, hidden_dim).
        labels: Binary labels of shape (N,), where 1=Illicit, 0=Licit.
        margin: Triplet loss margin distance.
        max_triplets: Maximum number of mined triplets per snapshot batch to bound computation.

    Returns:
        Scalar torch.Tensor containing average Triplet Loss.
    """
    illicit_indices = torch.where(labels == 1)[0]
    licit_indices = torch.where(labels == 0)[0]

    num_illicit = len(illicit_indices)
    num_licit = len(licit_indices)

    # Requires at least 2 illicit nodes and 1 licit node to form valid triplets
    if num_illicit < 2 or num_licit < 1:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    anchors = []
    positives = []
    negatives = []

    count = 0
    # Mine valid triplets
    for i in range(num_illicit):
        anchor_idx = illicit_indices[i]
        for j in range(i + 1, num_illicit):
            positive_idx = illicit_indices[j]
            # Randomly pick negative licit node
            neg_choice = torch.randint(0, num_licit, (1,)).item()
            negative_idx = licit_indices[neg_choice]

            anchors.append(embeddings[anchor_idx])
            positives.append(embeddings[positive_idx])
            negatives.append(embeddings[negative_idx])

            count += 1
            if count >= max_triplets:
                break
        if count >= max_triplets:
            break

    if len(anchors) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    anchor_tensor = torch.stack(anchors)
    positive_tensor = torch.stack(positives)
    negative_tensor = torch.stack(negatives)

    triplet_criterion = nn.TripletMarginLoss(margin=margin, p=2)
    loss = triplet_criterion(anchor_tensor, positive_tensor, negative_tensor)
    return loss


class TripletTransformerGCNTrainer:
    """
    Trainer for Causal Triplet-Style Dynamic GNN with Transformer Encoder.
    Handles sequential snapshot processing, combined BCE + Triplet loss, early stopping, and checkpointing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_gcn_layers: int = 2,
        nhead: int = 4,
        num_transformer_layers: int = 1,
        dropout: float = 0.2,
        triplet_weight: float = 0.5,
        triplet_margin: float = 1.0,
        device: Optional[str] = None,
    ):
        """
        Initialize Triplet Transformer GCN Trainer.

        Args:
            in_channels: Node feature dimension.
            hidden_channels: Hidden representation dimension.
            num_gcn_layers: Spatial GCN layers per snapshot.
            nhead: Number of Transformer attention heads.
            num_transformer_layers: Number of Transformer encoder layers.
            dropout: Dropout rate.
            triplet_weight: Alpha scaling factor for Triplet loss in combined objective.
            triplet_margin: Margin parameter for TripletMarginLoss.
            device: Device ('cuda' or 'cpu').
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin

        self.model = TripletTransformerGCNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_gcn_layers=num_gcn_layers,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
        ).to(self.device)

        self.is_fitted = False
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.training_history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "triplet_loss": []
        }

    def _compute_train_pos_weight(self, snapshots: List[Data], train_timesteps: Tuple[int, int] = (1, 34)) -> torch.Tensor:
        """Compute positive class weight strictly from training timesteps."""
        num_licit = 0
        num_illicit = 0

        for t in range(train_timesteps[0], train_timesteps[1] + 1):
            snap = snapshots[t - 1]
            train_mask = snap.train_mask
            y_train = snap.y[train_mask]

            num_licit += int((y_train == 0).sum().item())
            num_illicit += int((y_train == 1).sum().item())

        pos_weight_val = (num_licit / num_illicit) if num_illicit > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=self.device)
        logger.info(f"Stage 5 Class imbalance weight: train_licit={num_licit}, train_illicit={num_illicit} -> pos_weight={pos_weight_val:.4f}")
        return pos_weight

    def fit(
        self,
        snapshots: List[Data],
        total_nodes: int,
        epochs: int = 100,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        patience: int = 15,
        train_timesteps: Tuple[int, int] = (1, 34),
        val_timesteps: Tuple[int, int] = (35, 39),
    ) -> "TripletTransformerGCNTrainer":
        """
        Train Triplet Transformer GCN strictly sequentially across temporal snapshots.

        Args:
            snapshots: List of 49 PyG Data snapshot objects.
            total_nodes: Total number of unique nodes across full dataset.
            epochs: Maximum training epochs.
            lr: Learning rate.
            weight_decay: Weight decay factor.
            patience: Early stopping patience epochs.
            train_timesteps: Training timesteps tuple (1, 34).
            val_timesteps: Validation timesteps tuple (35, 39).

        Returns:
            Self instance.
        """
        logger.info(f"Training Stage 5 Triplet Transformer GCN on {self.device} across {len(snapshots)} temporal snapshots...")
        pos_weight = self._compute_train_pos_weight(snapshots, train_timesteps=train_timesteps)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            global_node_states = torch.zeros((total_nodes, self.model.hidden_channels), device=self.device)

            train_loss_accum = torch.tensor(0.0, device=self.device)
            triplet_loss_accum = torch.tensor(0.0, device=self.device)
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

                logits_t, norm_embeds_t, updated_states = self.model.forward_snapshot(snap.x, snap.edge_index, prev_states)
                global_node_states[global_idx] = updated_states

                train_mask = snap.train_mask
                if train_mask.sum() > 0:
                    # 1. Weighted BCE Loss
                    bce_loss_t = bce_criterion(logits_t[train_mask], snap.y[train_mask].float())
                    
                    # 2. Triplet Contrastive Loss on labelled training embeddings
                    train_embeds = norm_embeds_t[train_mask]
                    train_labels = snap.y[train_mask]
                    t_loss_t = compute_online_triplet_loss(train_embeds, train_labels, margin=self.triplet_margin)

                    combined_snap_loss = bce_loss_t + self.triplet_weight * t_loss_t

                    train_loss_accum = train_loss_accum + combined_snap_loss * train_mask.sum()
                    triplet_loss_accum = triplet_loss_accum + t_loss_t * train_mask.sum()
                    num_train_nodes_total += int(train_mask.sum().item())

            if num_train_nodes_total > 0:
                epoch_train_loss = train_loss_accum / num_train_nodes_total
                epoch_triplet_loss = triplet_loss_accum / num_train_nodes_total
                epoch_train_loss.backward()
                optimizer.step()
            else:
                epoch_train_loss = torch.tensor(0.0)
                epoch_triplet_loss = torch.tensor(0.0)

            # -----------------------------------------------------------------
            # STEP 2: Validation Evaluation (t = val_start .. val_end)
            # -----------------------------------------------------------------
            self.model.eval()
            val_loss_accum = 0.0
            num_val_nodes_total = 0

            val_preds_list = []
            val_trues_list = []

            with torch.no_grad():
                eval_global_states = torch.zeros((total_nodes, self.model.hidden_channels), device=self.device)

                for ts in range(1, val_timesteps[1] + 1):
                    snap = snapshots[ts - 1].to(self.device)
                    if snap.num_nodes == 0:
                        continue

                    global_idx = snap.global_node_idx
                    prev_states = eval_global_states[global_idx]

                    logits_t, norm_embeds_t, updated_states = self.model.forward_snapshot(snap.x, snap.edge_index, prev_states)
                    eval_global_states[global_idx] = updated_states

                    if val_timesteps[0] <= ts <= val_timesteps[1]:
                        val_mask = snap.val_mask
                        if val_mask.sum() > 0:
                            bce_val = bce_criterion(logits_t[val_mask], snap.y[val_mask].float()).item()
                            val_loss_accum += bce_val * int(val_mask.sum().item())
                            num_val_nodes_total += int(val_mask.sum().item())

                            probs_t = torch.sigmoid(logits_t[val_mask]).cpu().numpy()
                            preds_t = (probs_t >= 0.5).astype(int)
                            trues_t = snap.y[val_mask].cpu().numpy()

                            val_preds_list.extend(preds_t)
                            val_trues_list.extend(trues_t)

            val_loss = (val_loss_accum / num_val_nodes_total) if num_val_nodes_total > 0 else 0.0

            val_preds_arr = np.array(val_preds_list)
            val_trues_arr = np.array(val_trues_list)
            tp = np.logical_and(val_preds_arr == 1, val_trues_arr == 1).sum()
            fp = np.logical_and(val_preds_arr == 1, val_trues_arr == 0).sum()
            fn = np.logical_and(val_preds_arr == 0, val_trues_arr == 1).sum()
            val_f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

            self.training_history["train_loss"].append(epoch_train_loss.item())
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_f1"].append(val_f1)
            self.training_history["triplet_loss"].append(epoch_triplet_loss.item())

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {epoch_train_loss.item():.4f} (Triplet: {epoch_triplet_loss.item():.4f}) | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

            # Early stopping check based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.4f}).")
                    break

        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})

        self.is_fitted = True
        logger.info("Triplet Transformer GCN model training complete.")
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
            target_timesteps: Range of timesteps (start, end) to predict.
            mask_attr: Mask attribute name ('train_mask', 'val_mask', 'test_mask').

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (y_true, y_pred, y_prob)
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

                logits_t, norm_embeds_t, updated_states = self.model.forward_snapshot(snap.x, snap.edge_index, prev_states)
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
        """Save model checkpoint to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "in_channels": self.model.in_channels,
            "hidden_channels": self.model.hidden_channels,
            "num_gcn_layers": self.model.num_gcn_layers,
            "nhead": getattr(self.model.transformer_encoder.layers[0].self_attn, "num_heads", 4),
            "num_transformer_layers": len(self.model.transformer_encoder.layers),
            "dropout": self.model.dropout,
            "triplet_weight": self.triplet_weight,
            "triplet_margin": self.triplet_margin,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved Triplet Transformer GCN model checkpoint to {filepath}")
        return filepath

    def load(self, filepath: Union[str, Path]) -> "TripletTransformerGCNTrainer":
        """Load model checkpoint from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.triplet_weight = checkpoint.get("triplet_weight", 0.5)
        self.triplet_margin = checkpoint.get("triplet_margin", 1.0)

        self.model = TripletTransformerGCNClassifier(
            in_channels=checkpoint["in_channels"],
            hidden_channels=checkpoint["hidden_channels"],
            num_gcn_layers=checkpoint["num_gcn_layers"],
            nhead=checkpoint.get("nhead", 4),
            num_transformer_layers=checkpoint.get("num_transformer_layers", 1),
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict({k: v.to(self.device) for k, v in checkpoint["model_state_dict"].items()})
        self.training_history = checkpoint.get("training_history", {})
        self.is_fitted = True
        logger.info(f"Loaded Triplet Transformer GCN model checkpoint from {filepath}")
        return self

