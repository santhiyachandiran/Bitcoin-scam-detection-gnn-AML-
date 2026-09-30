"""
Persistence and Data I/O utilities for Stage 2 Processed Elliptic Graph Data.
Saves and loads PyTorch Geometric Data objects, temporal snapshots, scaler, and metadata.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from torch_geometric.data import Data

from src.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger("graph_saver")


def save_processed_graph(
    data: Data,
    snapshots: List[Data],
    metadata: Dict[str, Any],
    scaler: Any,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Dict[str, Path]:
    """
    Save PyG Data objects, temporal snapshots, metadata, and scaler to output directory.
    
    Args:
        data: Unified PyG Data object for full transaction graph.
        snapshots: List of 49 PyG Data objects for temporal snapshots.
        metadata: Summary statistics and pipeline metadata.
        scaler: Fitted StandardScaler instance.
        output_dir: Directory path where processed files will be saved.
        
    Returns:
        Dict[str, Path]: Dictionary mapping item names to saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pyg_path = output_dir / "elliptic_pyg_data.pt"
    snapshots_path = output_dir / "elliptic_temporal_snapshots.pt"
    metadata_path = output_dir / "preprocessing_metadata.json"
    scaler_path = output_dir / "scaler.pt"

    logger.info(f"Saving processed PyG full graph to {pyg_path}...")
    torch.save(data, pyg_path)

    logger.info(f"Saving {len(snapshots)} temporal snapshots to {snapshots_path}...")
    torch.save(snapshots, snapshots_path)

    logger.info(f"Saving scaler state to {scaler_path}...")
    scaler_state = {
        "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
        "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
        "var": scaler.var_.tolist() if hasattr(scaler, "var_") else None,
        "n_samples_seen": int(scaler.n_samples_seen_) if hasattr(scaler, "n_samples_seen_") else None
    }
    torch.save(scaler_state, scaler_path)

    logger.info(f"Saving preprocessing metadata to {metadata_path}...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    saved_files = {
        "pyg_data": pyg_path,
        "snapshots": snapshots_path,
        "metadata": metadata_path,
        "scaler": scaler_path
    }
    logger.info("Processed dataset files saved successfully.")
    return saved_files


def load_processed_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> Tuple[Data, List[Data], Dict[str, Any], Dict[str, Any]]:
    """
    Load saved PyG full graph, temporal snapshots, metadata, and scaler state.
    
    Args:
        processed_dir: Path to directory containing processed .pt and .json files.
        
    Returns:
        Tuple[Data, List[Data], Dict[str, Any], Dict[str, Any]]:
            - Full graph Data object
            - List of temporal snapshot Data objects
            - Metadata dictionary
            - Scaler state dictionary
    """
    processed_dir = Path(processed_dir)
    pyg_path = processed_dir / "elliptic_pyg_data.pt"
    snapshots_path = processed_dir / "elliptic_temporal_snapshots.pt"
    metadata_path = processed_dir / "preprocessing_metadata.json"
    scaler_path = processed_dir / "scaler.pt"

    if not pyg_path.exists():
        raise FileNotFoundError(f"Processed PyG graph not found at {pyg_path}")
    if not snapshots_path.exists():
        raise FileNotFoundError(f"Processed temporal snapshots not found at {snapshots_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    logger.info(f"Loading PyG graph from {pyg_path}...")
    data = torch.load(pyg_path, weights_only=False)

    logger.info(f"Loading temporal snapshots from {snapshots_path}...")
    snapshots = torch.load(snapshots_path, weights_only=False)

    logger.info(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scaler_state = {}
    if scaler_path.exists():
        scaler_state = torch.load(scaler_path, weights_only=False)

    return data, snapshots, metadata, scaler_state
