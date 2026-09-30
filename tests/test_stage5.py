"""
Unit tests for Stage 5: Triplet Transformer Dynamic GNN, Triplet Loss Mining, and Temporal Causality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

from src.models import (
    TripletTransformerGCNClassifier,
    TripletTransformerGCNTrainer,
    compute_online_triplet_loss,
)


@pytest.fixture(scope="module")
def synthetic_snapshots():
    """Generate synthetic temporal snapshots (t=1..10) for Stage 5 testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    num_timesteps = 10
    nodes_per_step = 20
    total_nodes = num_timesteps * nodes_per_step

    snapshots = []
    for ts in range(1, num_timesteps + 1):
        global_indices = np.arange((ts - 1) * nodes_per_step, ts * nodes_per_step)
        x_t = torch.randn((nodes_per_step, 10), dtype=torch.float32)
        y_t = torch.tensor(np.random.choice([0, 1], size=nodes_per_step, p=[0.8, 0.2]), dtype=torch.long)

        src = np.arange(nodes_per_step)
        dst = (src + 1) % nodes_per_step
        edge_index_t = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        train_mask_t = torch.tensor([ts <= 6] * nodes_per_step, dtype=torch.bool)
        val_mask_t = torch.tensor([ts in (7, 8)] * nodes_per_step, dtype=torch.bool)
        test_mask_t = torch.tensor([ts >= 9] * nodes_per_step, dtype=torch.bool)

        snap = Data(
            x=x_t,
            edge_index=edge_index_t,
            y=y_t,
            time_step=torch.full((nodes_per_step,), ts, dtype=torch.long),
            train_mask=train_mask_t,
            val_mask=val_mask_t,
            test_mask=test_mask_t,
            labelled_mask=torch.ones(nodes_per_step, dtype=torch.bool),
            global_node_idx=torch.tensor(global_indices, dtype=torch.long),
            num_nodes=nodes_per_step,
        )
        snapshots.append(snap)

    return snapshots, total_nodes


def test_online_triplet_loss_mining():
    """Test online triplet loss mining and margin computation."""
    torch.manual_seed(42)
    embeddings = torch.randn((10, 16), dtype=torch.float32)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    
    # 3 illicit nodes (1), 7 licit nodes (0)
    labels = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)

    loss = compute_online_triplet_loss(embeddings, labels, margin=1.0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    # Edge case: No illicit nodes
    no_illicit_labels = torch.zeros(10, dtype=torch.long)
    loss_empty = compute_online_triplet_loss(embeddings, no_illicit_labels, margin=1.0)
    assert loss_empty.item() == 0.0


def test_triplet_transformer_gcn_classifier_forward(synthetic_snapshots):
    """Test TripletTransformerGCNClassifier forward pass shapes."""
    snapshots, _ = synthetic_snapshots
    snap0 = snapshots[0]

    model = TripletTransformerGCNClassifier(
        in_channels=10,
        hidden_channels=16,
        num_gcn_layers=2,
        nhead=2,
        num_transformer_layers=1,
    )
    prev_states = torch.zeros((snap0.num_nodes, 16), dtype=torch.float32)

    logits, norm_embeds, updated_states = model.forward_snapshot(snap0.x, snap0.edge_index, prev_states)

    assert logits.dim() == 1
    assert logits.size(0) == snap0.num_nodes
    assert norm_embeds.shape == (snap0.num_nodes, 16)
    assert updated_states.shape == (snap0.num_nodes, 16)

    # Check L2 normalization of embedding projections
    norm_values = torch.norm(norm_embeds, p=2, dim=-1)
    np.testing.assert_allclose(norm_values.detach().numpy(), np.ones(snap0.num_nodes), rtol=1e-4)


def test_triplet_transformer_gcn_trainer_fit_and_predict(synthetic_snapshots, tmp_path):
    """Test TripletTransformerGCNTrainer training loop, prediction, and checkpoint persistence."""
    snapshots, total_nodes = synthetic_snapshots

    trainer = TripletTransformerGCNTrainer(
        in_channels=10,
        hidden_channels=16,
        num_gcn_layers=2,
        nhead=2,
        num_transformer_layers=1,
        triplet_weight=0.5,
    )
    assert not trainer.is_fitted

    trainer.fit(
        snapshots=snapshots,
        total_nodes=total_nodes,
        epochs=3,
        lr=0.01,
        patience=3,
        train_timesteps=(1, 6),
        val_timesteps=(7, 8),
    )
    assert trainer.is_fitted
    assert len(trainer.training_history["triplet_loss"]) == 3

    # Predict test snapshot range (9..10)
    y_true, y_pred, y_prob = trainer.predict_snapshot_range(
        snapshots=snapshots,
        total_nodes=total_nodes,
        target_timesteps=(9, 10),
        mask_attr="test_mask",
    )

    assert len(y_true) > 0
    assert len(y_true) == len(y_pred) == len(y_prob)
    assert set(y_pred).issubset({0, 1})
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))

    # Test Save & Load Roundtrip
    save_path = tmp_path / "triplet_transformer_gcn.pth"
    trainer.save(save_path)
    assert save_path.exists()

    reloaded_trainer = TripletTransformerGCNTrainer(in_channels=10).load(save_path)
    assert reloaded_trainer.is_fitted

    y_true_r, y_pred_r, y_prob_r = reloaded_trainer.predict_snapshot_range(
        snapshots=snapshots,
        total_nodes=total_nodes,
        target_timesteps=(9, 10),
        mask_attr="test_mask",
    )
    np.testing.assert_allclose(y_prob, y_prob_r, rtol=1e-4)
