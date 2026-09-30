"""
Unit tests for Stage 4: Strictly Causal Dynamic Temporal GNN, Causality Rules, and Temporal Leakage Prevention.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

from src.models import RecurrentGCNClassifier, RecurrentGCNTrainer


@pytest.fixture(scope="module")
def synthetic_snapshots():
    """Generate 10 synthetic temporal snapshots (t=1..10) for testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    num_timesteps = 10
    nodes_per_step = 20
    total_nodes = num_timesteps * nodes_per_step

    snapshots = []
    for ts in range(1, num_timesteps + 1):
        global_indices = np.arange((ts - 1) * nodes_per_step, ts * nodes_per_step)
        x_t = torch.randn((nodes_per_step, 10), dtype=torch.float32)
        y_t = torch.tensor(np.random.choice([0, 1], size=nodes_per_step, p=[0.85, 0.15]), dtype=torch.long)

        # Ring edge index for local snapshot
        src = np.arange(nodes_per_step)
        dst = (src + 1) % nodes_per_step
        edge_index_t = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # Assign masks according to timestep
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


def test_temporal_snapshot_causality_and_disjoint_masks(synthetic_snapshots):
    """Verify snapshot causality and disjoint split masks."""
    snapshots, total_nodes = synthetic_snapshots

    all_train_nodes = []
    all_val_nodes = []
    all_test_nodes = []

    for t, snap in enumerate(snapshots, start=1):
        # 1. Verify snapshot contains only node indices belonging to timestep t
        assert torch.all(snap.time_step == t)

        # 2. Verify edge indices stay within local node count 0..N_t-1
        if snap.edge_index.numel() > 0:
            assert snap.edge_index.max().item() < snap.num_nodes
            assert snap.edge_index.min().item() >= 0

        g_idx = snap.global_node_idx.numpy()
        if torch.any(snap.train_mask):
            all_train_nodes.extend(g_idx[snap.train_mask.numpy()])
        if torch.any(snap.val_mask):
            all_val_nodes.extend(g_idx[snap.val_mask.numpy()])
        if torch.any(snap.test_mask):
            all_test_nodes.extend(g_idx[snap.test_mask.numpy()])

    set_train = set(all_train_nodes)
    set_val = set(all_val_nodes)
    set_test = set(all_test_nodes)

    # 3. Disjointness check
    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0


def test_recurrent_gcn_classifier_forward(synthetic_snapshots):
    """Test RecurrentGCNClassifier single snapshot forward pass."""
    snapshots, total_nodes = synthetic_snapshots
    snap0 = snapshots[0]

    model = RecurrentGCNClassifier(in_channels=10, hidden_channels=16, num_gcn_layers=2)
    prev_states = torch.zeros((snap0.num_nodes, 16), dtype=torch.float32)

    logits, new_states = model.forward_snapshot(snap0.x, snap0.edge_index, prev_states)

    assert logits.dim() == 1
    assert logits.size(0) == snap0.num_nodes
    assert new_states.shape == (snap0.num_nodes, 16)


def test_recurrent_gcn_trainer_fit_and_predict(synthetic_snapshots, tmp_path):
    """Test RecurrentGCNTrainer training, prediction, and checkpoint persistence."""
    snapshots, total_nodes = synthetic_snapshots

    trainer = RecurrentGCNTrainer(in_channels=10, hidden_channels=16, num_gcn_layers=2)
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

    # Predict test range (9..10)
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
    save_path = tmp_path / "recurrent_gcn.pth"
    trainer.save(save_path)
    assert save_path.exists()

    reloaded_trainer = RecurrentGCNTrainer(in_channels=10).load(save_path)
    assert reloaded_trainer.is_fitted

    y_true_r, y_pred_r, y_prob_r = reloaded_trainer.predict_snapshot_range(
        snapshots=snapshots,
        total_nodes=total_nodes,
        target_timesteps=(9, 10),
        mask_attr="test_mask",
    )
    np.testing.assert_allclose(y_prob, y_prob_r, rtol=1e-4)


def test_class_weights_train_only(synthetic_snapshots):
    """Verify class weight calculation is fitted strictly on training snapshots."""
    snapshots, _ = synthetic_snapshots
    trainer = RecurrentGCNTrainer(in_channels=10, hidden_channels=16)

    pos_weight = trainer._compute_train_pos_weight(snapshots, train_timesteps=(1, 6))

    # Calculate manually
    num_licit = 0
    num_illicit = 0
    for t in range(1, 7):
        snap = snapshots[t - 1]
        y_train = snap.y[snap.train_mask]
        num_licit += int((y_train == 0).sum().item())
        num_illicit += int((y_train == 1).sum().item())

    expected_pos_weight = num_licit / num_illicit if num_illicit > 0 else 1.0
    assert abs(pos_weight.item() - expected_pos_weight) < 1e-4
