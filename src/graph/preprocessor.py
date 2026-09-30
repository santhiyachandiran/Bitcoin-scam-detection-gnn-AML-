"""
Data Preprocessing Module for Elliptic Bitcoin Transaction Dataset.
Handles feature cleaning, normalization (without data leakage), 
continuous ID mapping, and temporal train/val/test splits.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import TOTAL_TIMESTEPS, NUM_LOCAL_FEATURES, NUM_AGG_FEATURES
from src.data.loader import EllipticDataLoader
from src.data.validator import DatasetValidator
from src.utils.logger import get_logger

logger = get_logger("graph_preprocessor")


class EllipticPreprocessor:
    """
    Modular preprocessor for Elliptic Bitcoin transaction features and labels.
    Ensures feature normalization parameters are fitted ONLY on training timesteps.
    """

    def __init__(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        train_timesteps: Tuple[int, int] = (1, 34),
        val_timesteps: Tuple[int, int] = (35, 39),
        test_timesteps: Tuple[int, int] = (40, 49),
    ):
        """
        Initialize the preprocessor.
        
        Args:
            data_dict: Dictionary containing DataFrames loaded by EllipticDataLoader.
            train_timesteps: Inclusive range of time-steps for training (default: 1-34).
            val_timesteps: Inclusive range of time-steps for validation (default: 35-39).
            test_timesteps: Inclusive range of time-steps for testing (default: 40-49).
        """
        self.data_dict = data_dict
        self.train_timesteps = train_timesteps
        self.val_timesteps = val_timesteps
        self.test_timesteps = test_timesteps
        
        self.scaler = StandardScaler()
        self.txId_to_idx: Dict[int, int] = {}
        self.idx_to_txId: Optional[np.ndarray] = None
        self.feature_names = [f"feat_{i}" for i in range(1, NUM_LOCAL_FEATURES + NUM_AGG_FEATURES + 1)]

    def preprocess(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Any]:
        """
        Clean, normalize features, create node mappings, and compute split masks.
        
        Args:
            data_dict: Optional DataFrames dictionary. If not provided, uses self.data_dict.
            
        Returns:
            Dict[str, Any]: Dictionary containing processed numpy arrays and metadata.
        """
        if data_dict is not None:
            self.data_dict = data_dict

        if self.data_dict is None:
            raise ValueError("No data_dict provided to EllipticPreprocessor.")

        # Step 1: Validate dataset integrity
        logger.info("Validating dataset integrity prior to preprocessing...")
        validator = DatasetValidator(self.data_dict)
        val_report = validator.validate_all()
        if not val_report["is_valid"]:
            logger.warning(f"Dataset validation produced issues: {val_report['issues']}")

        df_features = self.data_dict["features"].copy()
        df_classes = self.data_dict["classes"].copy()

        # Step 2: Merge features and classes on txId to ensure perfect alignment
        logger.info("Aligning features and target labels...")
        df_merged = pd.merge(df_features, df_classes[["txId", "class", "binary_label"]], on="txId", how="inner")
        
        # Sort nodes primarily by time_step and secondarily by txId for deterministic indexing
        df_merged.sort_values(by=["time_step", "txId"], inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        num_nodes = len(df_merged)
        logger.info(f"Total merged transaction nodes to preprocess: {num_nodes}")

        # Step 3: Map continuous 0..N-1 node indices
        raw_tx_ids = df_merged["txId"].values.astype(np.int64)
        self.idx_to_txId = raw_tx_ids
        self.txId_to_idx = {tx_id: idx for idx, tx_id in enumerate(raw_tx_ids)}

        # Step 4: Extract features and check for missing/invalid values
        feature_cols = [c for c in df_merged.columns if c.startswith("feat_")]
        if len(feature_cols) != NUM_LOCAL_FEATURES + NUM_AGG_FEATURES:
            logger.warning(f"Found {len(feature_cols)} feature columns instead of expected {NUM_LOCAL_FEATURES + NUM_AGG_FEATURES}.")

        X_raw = df_merged[feature_cols].values.astype(np.float32)
        
        # Handle inf or NaN values if present
        num_nans = np.isnan(X_raw).sum()
        num_infs = np.isinf(X_raw).sum()
        if num_nans > 0 or num_infs > 0:
            logger.warning(f"Detected {num_nans} NaNs and {num_infs} Infs in feature matrix. Cleaning...")
            X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 5: Identify Train/Val/Test node indices by timestep
        timesteps = df_merged["time_step"].values.astype(np.int64)
        
        train_time_mask = (timesteps >= self.train_timesteps[0]) & (timesteps <= self.train_timesteps[1])
        val_time_mask = (timesteps >= self.val_timesteps[0]) & (timesteps <= self.val_timesteps[1])
        test_time_mask = (timesteps >= self.test_timesteps[0]) & (timesteps <= self.test_timesteps[1])

        # Step 6: Normalize features WITHOUT data leakage (fit scaler only on training timesteps)
        logger.info(f"Fitting StandardScaler on train nodes (timesteps {self.train_timesteps[0]}-{self.train_timesteps[1]})...")
        train_indices = np.where(train_time_mask)[0]
        
        if len(train_indices) == 0:
            raise ValueError(f"No training nodes found in timesteps range {self.train_timesteps}")

        self.scaler.fit(X_raw[train_indices])
        X_scaled = self.scaler.transform(X_raw)

        # Step 7: Process labels & construct split masks
        y = df_merged["binary_label"].values.astype(np.int64) # 1=Illicit, 0=Licit, -1=Unknown
        
        labelled_mask = (y != -1)
        train_mask = train_time_mask & labelled_mask
        val_mask = val_time_mask & labelled_mask
        test_mask = test_time_mask & labelled_mask

        # Integrity Check: Ensure split masks are mutually exclusive
        overlap_train_val = np.logical_and(train_mask, val_mask).sum()
        overlap_train_test = np.logical_and(train_mask, test_mask).sum()
        overlap_val_test = np.logical_and(val_mask, test_mask).sum()

        if overlap_train_val + overlap_train_test + overlap_val_test > 0:
            raise ValueError(f"Data leakage detected! Split masks overlap. Train/Val: {overlap_train_val}, Train/Test: {overlap_train_test}, Val/Test: {overlap_val_test}")

        logger.info(f"Feature normalization complete.")
        logger.info(f"Split breakdown: Train labelled={train_mask.sum()}, Val labelled={val_mask.sum()}, Test labelled={test_mask.sum()}, Total labelled={labelled_mask.sum()}, Unknown={num_nodes - labelled_mask.sum()}")

        processed_data = {
            "x": X_scaled,
            "y": y,
            "time_step": timesteps,
            "tx_id": raw_tx_ids,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "labelled_mask": labelled_mask,
            "txId_to_idx": self.txId_to_idx,
            "idx_to_txId": self.idx_to_txId,
            "scaler": self.scaler,
            "stats": {
                "num_nodes": num_nodes,
                "num_features": X_scaled.shape[1],
                "train_nodes_total": int(train_time_mask.sum()),
                "train_nodes_labelled": int(train_mask.sum()),
                "train_illicit": int((train_mask & (y == 1)).sum()),
                "train_licit": int((train_mask & (y == 0)).sum()),
                "val_nodes_total": int(val_time_mask.sum()),
                "val_nodes_labelled": int(val_mask.sum()),
                "val_illicit": int((val_mask & (y == 1)).sum()),
                "val_licit": int((val_mask & (y == 0)).sum()),
                "test_nodes_total": int(test_time_mask.sum()),
                "test_nodes_labelled": int(test_mask.sum()),
                "test_illicit": int((test_mask & (y == 1)).sum()),
                "test_licit": int((test_mask & (y == 0)).sum()),
                "unknown_nodes": int((y == -1).sum()),
            }
        }

        return processed_data
