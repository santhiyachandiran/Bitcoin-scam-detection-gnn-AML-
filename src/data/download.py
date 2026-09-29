"""
Automated downloader and dataset manager for Elliptic Bitcoin Transaction Dataset.
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
    FEATURES_FILE,
    CLASSES_FILE,
    EDGES_FILE,
    KAGGLE_DATASET_SLUG,
    TOTAL_FEATURES,
)
from src.utils.logger import get_logger

logger = get_logger("data_downloader")


def is_dataset_present(target_dir: Path) -> bool:
    """Check if all three required CSV files are present in target_dir."""
    required_files = [
        target_dir / FEATURES_FILE,
        target_dir / CLASSES_FILE,
        target_dir / EDGES_FILE,
    ]
    return all(f.exists() and f.stat().st_size > 0 for f in required_files)


def download_dataset(target_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Downloads the Elliptic dataset via kagglehub if not present in target_dir.
    
    Args:
        target_dir: Path to directory where raw files should reside.
        
    Returns:
        Path: Target directory containing dataset files.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    if is_dataset_present(target_dir):
        logger.info(f"Elliptic dataset already exists in {target_dir}")
        return target_dir

    logger.info("Dataset not found in target_dir. Attempting automated download via kagglehub...")
    try:
        import kagglehub
        cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
        logger.info(f"Dataset downloaded to kagglehub cache: {cache_path}")

        # Move or copy files to target_dir
        for file_name in [FEATURES_FILE, CLASSES_FILE, EDGES_FILE]:
            src_file = cache_path / file_name
            if src_file.exists():
                shutil.copy(src_file, target_dir / file_name)
                logger.info(f"Copied {file_name} -> {target_dir / file_name}")
            else:
                # Search recursively in case kaggle hub uncompressed into subfolder
                matches = list(cache_path.glob(f"**/{file_name}"))
                if matches:
                    shutil.copy(matches[0], target_dir / file_name)
                    logger.info(f"Copied {file_name} from subfolder -> {target_dir / file_name}")
                else:
                    raise FileNotFoundError(f"File {file_name} not found in kagglehub download.")

        if is_dataset_present(target_dir):
            logger.info("Dataset successfully downloaded and installed into raw data directory.")
            return target_dir

    except Exception as e:
        logger.warning(f"Kagglehub download failed or unavailable: {e}")
        logger.info("Fallback: Checking system cache or generating sample synthetic dataset for testing...")

    # Check common cache locations
    home_dir = Path.home()
    possible_caches = list(home_dir.glob(".cache/kagglehub/datasets/ellipticco/elliptic-data-set/**/elliptic_txs_features.csv"))
    if possible_caches:
        cache_dir = possible_caches[0].parent
        logger.info(f"Found existing cached files at {cache_dir}, copying...")
        for file_name in [FEATURES_FILE, CLASSES_FILE, EDGES_FILE]:
            src_file = cache_dir / file_name
            if src_file.exists():
                shutil.copy(src_file, target_dir / file_name)
        if is_dataset_present(target_dir):
            logger.info("Dataset populated from local cache.")
            return target_dir

    raise RuntimeError("Dataset could not be loaded or downloaded. Please check Kaggle credentials or network.")


def generate_sample_dataset(output_dir: Path = SAMPLE_DATA_DIR, num_nodes: int = 500, num_edges: int = 800) -> Path:
    """
    Generates a realistic mock Elliptic dataset for offline testing/verification.
    
    Args:
        output_dir: Destination directory.
        num_nodes: Number of transaction nodes.
        num_edges: Number of edges.
        
    Returns:
        Path to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating synthetic sample dataset with {num_nodes} nodes and {num_edges} edges...")

    node_ids = np.arange(100000, 100000 + num_nodes)
    time_steps = np.random.randint(1, 50, size=num_nodes)
    
    # 1. Features
    features_data = []
    for i, nid in enumerate(node_ids):
        t = time_steps[i]
        feats = np.random.randn(TOTAL_FEATURES)
        features_data.append([nid, t] + feats.tolist())

    feat_cols = [0, 1] + [f"feat_{i}" for i in range(1, TOTAL_FEATURES + 1)]
    df_feat = pd.DataFrame(features_data, columns=feat_cols)
    df_feat.to_csv(output_dir / FEATURES_FILE, header=False, index=False)

    # 2. Classes
    classes = np.random.choice(["1", "2", "unknown"], size=num_nodes, p=[0.1, 0.4, 0.5])
    df_class = pd.DataFrame({"txId": node_ids, "class": classes})
    df_class.to_csv(output_dir / CLASSES_FILE, index=False)

    # 3. Edges
    src_nodes = np.random.choice(node_ids, size=num_edges)
    dst_nodes = np.random.choice(node_ids, size=num_edges)
    # filter self loops
    mask = src_nodes != dst_nodes
    df_edges = pd.DataFrame({"txId1": src_nodes[mask], "txId2": dst_nodes[mask]})
    df_edges.to_csv(output_dir / EDGES_FILE, index=False)

    logger.info(f"Sample dataset successfully generated in {output_dir}")
    return output_dir
