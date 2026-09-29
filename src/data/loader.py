"""
Dataset loader module for Elliptic Bitcoin Transaction Dataset.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    FEATURES_FILE,
    CLASSES_FILE,
    EDGES_FILE,
    CLASS_MAP,
    NUM_LOCAL_FEATURES,
    NUM_AGG_FEATURES,
)
from src.utils.logger import get_logger

logger = get_logger("data_loader")


class EllipticDataLoader:
    """
    Modular loader for Elliptic Bitcoin Transaction dataset files.
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        """
        Initialize the loader with the target directory.
        
        Args:
            data_dir: Path to directory containing raw/sample CSV files.
        """
        self.data_dir = Path(data_dir)
        self.features_path = self.data_dir / FEATURES_FILE
        self.classes_path = self.data_dir / CLASSES_FILE
        self.edges_path = self.data_dir / EDGES_FILE

    def load_features(self) -> pd.DataFrame:
        """
        Load transaction features dataset.
        
        Returns:
            pd.DataFrame: Features DataFrame with column names (txId, time_step, feat_1..feat_166).
        """
        logger.info(f"Loading features from {self.features_path}...")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found at {self.features_path}")

        # Check if the file has headers or raw index
        df_sample = pd.read_csv(self.features_path, nrows=5, header=None)
        
        # If first column is header string like 'txId' or '0'
        if str(df_sample.iloc[0, 0]).lower() == "txid":
            df_features = pd.read_csv(self.features_path)
        else:
            # Assign standardized column names
            col_names = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, NUM_LOCAL_FEATURES + NUM_AGG_FEATURES + 1)]
            df_features = pd.read_csv(self.features_path, header=None, names=col_names)

        # Ensure txId and time_step are integers
        df_features["txId"] = df_features["txId"].astype(int)
        df_features["time_step"] = df_features["time_step"].astype(int)
        logger.info(f"Loaded features shape: {df_features.shape}")
        return df_features

    def load_classes(self) -> pd.DataFrame:
        """
        Load transaction classes dataset.
        
        Returns:
            pd.DataFrame: Classes DataFrame with txId, raw class, label name, and binary label.
        """
        logger.info(f"Loading classes from {self.classes_path}...")
        if not self.classes_path.exists():
            raise FileNotFoundError(f"Classes file not found at {self.classes_path}")

        df_classes = pd.read_csv(self.classes_path)
        df_classes["txId"] = df_classes["txId"].astype(int)
        df_classes["class"] = df_classes["class"].astype(str)

        # Add human readable label
        df_classes["class_name"] = df_classes["class"].map(CLASS_MAP).fillna("Unknown")
        
        # Binary target: 1 for Illicit, 0 for Licit, -1 for Unknown
        binary_map = {"1": 1, "2": 0, "unknown": -1}
        df_classes["binary_label"] = df_classes["class"].map(binary_map).fillna(-1)

        logger.info(f"Loaded classes shape: {df_classes.shape}")
        return df_classes

    def load_edges(self) -> pd.DataFrame:
        """
        Load transaction edges (graph directed connections).
        
        Returns:
            pd.DataFrame: Edges DataFrame with txId1 (source) and txId2 (destination).
        """
        logger.info(f"Loading transaction edges from {self.edges_path}...")
        if not self.edges_path.exists():
            raise FileNotFoundError(f"Edges file not found at {self.edges_path}")

        df_edges = pd.read_csv(self.edges_path)
        df_edges["txId1"] = df_edges["txId1"].astype(int)
        df_edges["txId2"] = df_edges["txId2"].astype(int)
        logger.info(f"Loaded edges shape: {df_edges.shape}")
        return df_edges

    def load_merged_data(self) -> pd.DataFrame:
        """
        Load and merge features and classes into a single DataFrame.
        
        Returns:
            pd.DataFrame: Merged features and target labels.
        """
        df_features = self.load_features()
        df_classes = self.load_classes()

        df_merged = pd.merge(df_features, df_classes, on="txId", how="inner")
        logger.info(f"Merged features & classes shape: {df_merged.shape}")
        return df_merged

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all files into a dictionary.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'features', 'classes', 'edges', 'merged'.
        """
        df_features = self.load_features()
        df_classes = self.load_classes()
        df_edges = self.load_edges()
        df_merged = pd.merge(df_features, df_classes, on="txId", how="inner")

        return {
            "features": df_features,
            "classes": df_classes,
            "edges": df_edges,
            "merged": df_merged,
        }
