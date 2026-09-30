"""
Unit tests for Stage 2: Data Preprocessing, PyTorch Geometric Graph Construction, and Persistence.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.data.download import generate_sample_dataset
from src.data.loader import EllipticDataLoader
from src.graph.preprocessor import EllipticPreprocessor
from src.graph.builder import EllipticGraphBuilder
from src.graph.saver import save_processed_graph, load_processed_data


@pytest.fixture(scope="module")
def sample_data(tmp_path_factory):
    """Fixture generating synthetic dataset for Stage 2 testing."""
    tmp_dir = tmp_path_factory.mktemp("stage2_sample")
    generate_sample_dataset(tmp_dir, num_nodes=200, num_edges=300)
    loader = EllipticDataLoader(data_dir=tmp_dir)
    return loader.load_all(), tmp_dir


def test_preprocessor_normalization_and_masks(sample_data):
    """Test feature preprocessing, normalization on train nodes only, and split masks."""
    data_dict, _ = sample_data
    
    preprocessor = EllipticPreprocessor(
        data_dict=data_dict,
        train_timesteps=(1, 30),
        val_timesteps=(31, 40),
        test_timesteps=(41, 49)
    )
    processed = preprocessor.preprocess()

    assert "x" in processed
    assert "y" in processed
    assert "train_mask" in processed
    assert "val_mask" in processed
    assert "test_mask" in processed

    # Check shapes
    num_nodes = len(data_dict["features"])
    assert processed["x"].shape[0] == num_nodes
    assert processed["x"].shape[1] == 166
    assert processed["y"].shape[0] == num_nodes

    # Check mask disjointness (no data leakage)
    train_m = processed["train_mask"]
    val_m = processed["val_mask"]
    test_m = processed["test_mask"]

    assert np.logical_and(train_m, val_m).sum() == 0
    assert np.logical_and(train_m, test_m).sum() == 0
    assert np.logical_and(val_m, test_m).sum() == 0

    # Verify normalization parameters (scaler fit ONLY on train timesteps)
    train_time_mask = (processed["time_step"] >= 1) & (processed["time_step"] <= 30)
    raw_feature_cols = [c for c in data_dict["features"].columns if c.startswith("feat_")]
    
    # Sort data_dict features identically to preprocessor (time_step, txId)
    df_sorted = data_dict["features"].sort_values(by=["time_step", "txId"]).reset_index(drop=True)
    raw_train_x = df_sorted.loc[train_time_mask, raw_feature_cols].values

    expected_scaler = StandardScaler().fit(raw_train_x)
    np.testing.assert_allclose(preprocessor.scaler.mean_, expected_scaler.mean_, rtol=1e-3, atol=1e-5)


def test_graph_builder_pyg_data(sample_data):
    """Test PyTorch Geometric graph construction and PyG validation."""
    data_dict, _ = sample_data
    
    preprocessor = EllipticPreprocessor(data_dict=data_dict)
    processed = preprocessor.preprocess()

    builder = EllipticGraphBuilder(preprocessed_data=processed, edges_df=data_dict["edges"])
    full_data, stats = builder.build_full_graph()

    # PyG Schema assertions
    assert isinstance(full_data.x, torch.Tensor)
    assert isinstance(full_data.edge_index, torch.Tensor)
    assert full_data.x.dtype == torch.float32
    assert full_data.edge_index.dtype == torch.long
    assert full_data.edge_index.dim() == 2
    assert full_data.edge_index.size(0) == 2

    # Verify edge index boundaries
    assert full_data.edge_index.max().item() < full_data.num_nodes
    assert full_data.edge_index.min().item() >= 0

    # Validate method call
    full_data.validate(raise_on_error=True)

    # Check statistics output
    assert stats["num_nodes"] == full_data.num_nodes
    assert stats["num_edges"] == full_data.edge_index.size(1)
    assert "in_degree" in stats
    assert "out_degree" in stats


def test_temporal_snapshots(sample_data):
    """Test creation of temporal graph snapshots."""
    data_dict, _ = sample_data
    
    preprocessor = EllipticPreprocessor(data_dict=data_dict)
    processed = preprocessor.preprocess()

    builder = EllipticGraphBuilder(preprocessed_data=processed, edges_df=data_dict["edges"])
    snapshots = builder.build_temporal_snapshots()

    assert len(snapshots) == 49
    for snap in snapshots:
        assert isinstance(snap.x, torch.Tensor)
        assert isinstance(snap.edge_index, torch.Tensor)
        if snap.edge_index.numel() > 0:
            assert snap.edge_index.max().item() < snap.num_nodes
            assert snap.edge_index.min().item() >= 0
        snap.validate(raise_on_error=True)


def test_save_and_load_roundtrip(sample_data, tmp_path):
    """Test saving and reloading processed PyG graph artifacts."""
    data_dict, _ = sample_data
    
    preprocessor = EllipticPreprocessor(data_dict=data_dict)
    processed = preprocessor.preprocess()

    builder = EllipticGraphBuilder(preprocessed_data=processed, edges_df=data_dict["edges"])
    full_data, stats = builder.build_full_graph()
    snapshots = builder.build_temporal_snapshots()

    metadata = {"test_run": True, "num_nodes": full_data.num_nodes}

    saved_paths = save_processed_graph(
        data=full_data,
        snapshots=snapshots,
        metadata=metadata,
        scaler=preprocessor.scaler,
        output_dir=tmp_path
    )

    for p in saved_paths.values():
        assert Path(p).exists()

    reloaded_data, reloaded_snapshots, reloaded_meta, scaler_state = load_processed_data(processed_dir=tmp_path)
    
    assert reloaded_data.num_nodes == full_data.num_nodes
    assert reloaded_data.edge_index.size(1) == full_data.edge_index.size(1)
    assert len(reloaded_snapshots) == len(snapshots)
    assert reloaded_meta["num_nodes"] == full_data.num_nodes
    assert "mean" in scaler_state
