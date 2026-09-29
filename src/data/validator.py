"""
Data integrity and schema validator for Elliptic dataset.
"""

from typing import Dict, Any, List
import pandas as pd

from src.config import TOTAL_TIMESTEPS, NUM_LOCAL_FEATURES, NUM_AGG_FEATURES
from src.utils.logger import get_logger

logger = get_logger("dataset_validator")


class DatasetValidator:
    """
    Validates shapes, schemas, and integrity across features, classes, and edge tables.
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Initialize validator with dictionary of loaded DataFrames.
        
        Args:
            data_dict: Dictionary containing 'features', 'classes', 'edges', 'merged'.
        """
        self.df_features = data_dict["features"]
        self.df_classes = data_dict["classes"]
        self.df_edges = data_dict["edges"]
        self.df_merged = data_dict.get("merged")

    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.
        
        Returns:
            Dict[str, Any]: Validation summary report containing passes, warnings, and errors.
        """
        logger.info("Running dataset integrity validation checks...")
        
        checks = {}
        issues: List[str] = []

        # 1. Feature Shape & Column Check
        num_feature_cols = self.df_features.shape[1] - 2 # excluding txId, time_step
        expected_features = NUM_LOCAL_FEATURES + NUM_AGG_FEATURES
        feat_shape_valid = num_feature_cols == expected_features
        checks["feature_columns_count"] = {
            "valid": feat_shape_valid,
            "actual": num_feature_cols,
            "expected": expected_features
        }
        if not feat_shape_valid:
            issues.append(f"Features column count mismatch: got {num_feature_cols}, expected {expected_features}")

        # 2. Classes Columns Check
        required_class_cols = {"txId", "class"}
        classes_valid = required_class_cols.issubset(set(self.df_classes.columns))
        checks["classes_schema"] = {
            "valid": classes_valid,
            "actual": list(self.df_classes.columns),
            "expected": list(required_class_cols)
        }
        if not classes_valid:
            issues.append("Classes schema missing required columns 'txId' or 'class'.")

        # 3. Edges Schema Check
        required_edge_cols = {"txId1", "txId2"}
        edges_valid = required_edge_cols.issubset(set(self.df_edges.columns))
        checks["edges_schema"] = {
            "valid": edges_valid,
            "actual": list(self.df_edges.columns),
            "expected": list(required_edge_cols)
        }
        if not edges_valid:
            issues.append("Edges schema missing required columns 'txId1' or 'txId2'.")

        # 4. ID Alignment between Features and Classes
        feat_tx_ids = set(self.df_features["txId"])
        class_tx_ids = set(self.df_classes["txId"])
        ids_match = feat_tx_ids == class_tx_ids
        checks["id_alignment"] = {
            "valid": ids_match,
            "features_node_count": len(feat_tx_ids),
            "classes_node_count": len(class_tx_ids),
            "missing_in_classes": len(feat_tx_ids - class_tx_ids),
            "missing_in_features": len(class_tx_ids - feat_tx_ids)
        }
        if not ids_match:
            issues.append(f"Node ID mismatch between features and classes tables!")

        # 5. Edge Nodes existence in Features
        edge_nodes = set(self.df_edges["txId1"]).union(set(self.df_edges["txId2"]))
        orphaned_edge_nodes = edge_nodes - feat_tx_ids
        edges_node_exist = len(orphaned_edge_nodes) == 0
        checks["edges_node_coverage"] = {
            "valid": edges_node_exist,
            "total_unique_edge_nodes": len(edge_nodes),
            "orphaned_edge_nodes": len(orphaned_edge_nodes)
        }
        if not edges_node_exist:
            issues.append(f"{len(orphaned_edge_nodes)} edge endpoints do not exist in features table!")

        # 6. Time-step range check
        min_ts = self.df_features["time_step"].min()
        max_ts = self.df_features["time_step"].max()
        ts_valid = (min_ts >= 1) and (max_ts <= TOTAL_TIMESTEPS)
        checks["timestep_range"] = {
            "valid": ts_valid,
            "min_timestep": int(min_ts),
            "max_timestep": int(max_ts),
            "expected_range": [1, TOTAL_TIMESTEPS]
        }
        if not ts_valid:
            issues.append(f"Time-step range out of expected [1, {TOTAL_TIMESTEPS}] bounds: [{min_ts}, {max_ts}]")

        is_overall_valid = len(issues) == 0
        logger.info(f"Dataset validation completed. Overall valid: {is_overall_valid}")
        
        return {
            "is_valid": is_overall_valid,
            "checks": checks,
            "issues": issues,
            "shapes": {
                "features": self.df_features.shape,
                "classes": self.df_classes.shape,
                "edges": self.df_edges.shape
            }
        }
