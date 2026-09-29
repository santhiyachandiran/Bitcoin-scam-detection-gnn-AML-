"""
Exploratory Data Analysis (EDA) Analyzer for Elliptic Bitcoin Transaction Dataset.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import networkx as nx

from src.config import CLASS_MAP, NUM_LOCAL_FEATURES, NUM_AGG_FEATURES
from src.utils.logger import get_logger

logger = get_logger("eda_analyzer")


class EllipticEDAAnalyzer:
    """
    Comprehensive analyzer for missing values, duplicates, class distribution,
    time-step dynamics, feature statistics, and graph topological metrics.
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Initialize EDA analyzer.
        
        Args:
            data_dict: Dictionary containing 'features', 'classes', 'edges', 'merged'.
        """
        self.df_features = data_dict["features"]
        self.df_classes = data_dict["classes"]
        self.df_edges = data_dict["edges"]
        self.df_merged = data_dict.get("merged")

    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values across all tables.
        
        Returns:
            Dict containing missing count and percentage for each table.
        """
        logger.info("Analyzing missing values...")
        res = {}
        for name, df in [("features", self.df_features), ("classes", self.df_classes), ("edges", self.df_edges)]:
            missing_series = df.isnull().sum()
            total_missing = missing_series.sum()
            res[name] = {
                "total_missing_cells": int(total_missing),
                "columns_with_missing": missing_series[missing_series > 0].to_dict(),
                "has_missing": total_missing > 0,
            }
        return res

    def analyze_duplicates(self) -> Dict[str, Any]:
        """
        Analyze duplicate rows and duplicate node IDs across tables.
        
        Returns:
            Dict containing duplicate statistics.
        """
        logger.info("Analyzing duplicate values...")
        return {
            "features_duplicate_rows": int(self.df_features.duplicated().sum()),
            "features_duplicate_txIds": int(self.df_features["txId"].duplicated().sum()),
            "classes_duplicate_rows": int(self.df_classes.duplicated().sum()),
            "classes_duplicate_txIds": int(self.df_classes["txId"].duplicated().sum()),
            "edges_duplicate_pairs": int(self.df_edges.duplicated().sum()),
        }

    def analyze_class_distribution(self) -> Dict[str, Any]:
        """
        Analyze class labels distribution (Illicit, Licit, Unknown).
        
        Returns:
            Dict containing counts, percentages, and labelled vs unknown breakdown.
        """
        logger.info("Analyzing class label distribution...")
        counts = self.df_classes["class"].value_counts()
        total_txs = len(self.df_classes)

        illicit_count = int(counts.get("1", 0))
        licit_count = int(counts.get("2", 0))
        unknown_count = int(counts.get("unknown", 0))
        labelled_count = illicit_count + licit_count

        return {
            "total_transactions": total_txs,
            "raw_counts": {
                "Illicit (1)": illicit_count,
                "Licit (2)": licit_count,
                "Unknown": unknown_count,
            },
            "percentages": {
                "Illicit (1)": round((illicit_count / total_txs) * 100, 2),
                "Licit (2)": round((licit_count / total_txs) * 100, 2),
                "Unknown": round((unknown_count / total_txs) * 100, 2),
            },
            "labelled_vs_unknown": {
                "labelled_count": labelled_count,
                "labelled_percentage": round((labelled_count / total_txs) * 100, 2),
                "unknown_count": unknown_count,
                "unknown_percentage": round((unknown_count / total_txs) * 100, 2),
            },
            "illicit_ratio_in_labelled": round((illicit_count / max(1, labelled_count)) * 100, 2)
        }

    def analyze_timestep_distribution(self) -> Dict[str, Any]:
        """
        Analyze distribution of transactions across the 49 time-steps.
        
        Returns:
            Dict containing time-step summary statistics and breakdown per class.
        """
        logger.info("Analyzing time-step distribution...")
        
        # Transactions count per time step
        ts_counts = self.df_features["time_step"].value_counts().sort_index()

        # Breakdown by class if merged is available
        ts_class_df = self.df_merged.groupby(["time_step", "class_name"]).size().unstack(fill_value=0)
        
        # Calculate illicit ratio per time step
        if "Illicit" in ts_class_df.columns and "Licit" in ts_class_df.columns:
            labelled_total = ts_class_df["Illicit"] + ts_class_df["Licit"]
            illicit_ratio_ts = (ts_class_df["Illicit"] / labelled_total.replace(0, np.nan) * 100).fillna(0)
        else:
            illicit_ratio_ts = pd.Series(0, index=ts_counts.index)

        return {
            "total_timesteps": int(self.df_features["time_step"].nunique()),
            "min_transactions_per_step": int(ts_counts.min()),
            "max_transactions_per_step": int(ts_counts.max()),
            "mean_transactions_per_step": round(float(ts_counts.mean()), 2),
            "median_transactions_per_step": float(ts_counts.median()),
            "counts_per_timestep": ts_counts.to_dict(),
            "class_breakdown_per_timestep": ts_class_df.to_dict(orient="index"),
            "illicit_ratio_per_timestep": illicit_ratio_ts.to_dict()
        }

    def analyze_feature_statistics(self) -> Dict[str, Any]:
        """
        Analyze summary statistics for local features (1-94) and aggregate features (95-166).
        
        Returns:
            Dict containing feature statistics summaries.
        """
        logger.info("Analyzing feature numerical statistics...")
        feat_cols = [c for c in self.df_features.columns if c.startswith("feat_")]
        local_cols = feat_cols[:NUM_LOCAL_FEATURES]
        agg_cols = feat_cols[NUM_LOCAL_FEATURES:NUM_LOCAL_FEATURES + NUM_AGG_FEATURES]

        df_local = self.df_features[local_cols]
        df_agg = self.df_features[agg_cols]

        def get_summary(df_sub):
            means = df_sub.mean()
            stds = df_sub.std()
            mins = df_sub.min()
            maxs = df_sub.max()
            zeros_ratio = (df_sub == 0).mean()
            skews = df_sub.skew()
            return {
                "mean_of_means": round(float(means.mean()), 4),
                "mean_of_stds": round(float(stds.mean()), 4),
                "min_overall": round(float(mins.min()), 4),
                "max_overall": round(float(maxs.max()), 4),
                "avg_zero_ratio_pct": round(float(zeros_ratio.mean()) * 100, 2),
                "avg_skewness": round(float(skews.mean()), 4)
            }

        return {
            "total_features_count": len(feat_cols),
            "local_features_count": len(local_cols),
            "aggregate_features_count": len(agg_cols),
            "local_features_summary": get_summary(df_local),
            "aggregate_features_summary": get_summary(df_agg)
        }

    def analyze_graph_statistics(self) -> Dict[str, Any]:
        """
        Analyze graph, node, edge, and network topology statistics.
        
        Returns:
            Dict containing node counts, edge counts, degree distribution, density, etc.
        """
        logger.info("Analyzing graph node/edge topology statistics...")

        total_nodes = len(self.df_features)
        total_edges = len(self.df_edges)

        # In-degree and Out-degree calculation via Pandas
        in_degrees = self.df_edges["txId2"].value_counts()
        out_degrees = self.df_edges["txId1"].value_counts()

        # Combine degrees per node
        degree_df = pd.DataFrame({"txId": self.df_features["txId"]})
        degree_df["in_degree"] = degree_df["txId"].map(in_degrees).fillna(0).astype(int)
        degree_df["out_degree"] = degree_df["txId"].map(out_degrees).fillna(0).astype(int)
        degree_df["total_degree"] = degree_df["in_degree"] + degree_df["out_degree"]

        isolated_nodes = int((degree_df["total_degree"] == 0).sum())

        # Graph Density for directed graph: E / (V * (V - 1))
        density = total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0.0

        return {
            "node_count": total_nodes,
            "edge_count": total_edges,
            "directed": True,
            "graph_density": float(density),
            "isolated_nodes_count": isolated_nodes,
            "isolated_nodes_percentage": round((isolated_nodes / total_nodes) * 100, 2),
            "degree_stats": {
                "in_degree": {
                    "max": int(degree_df["in_degree"].max()),
                    "mean": round(float(degree_df["in_degree"].mean()), 4),
                    "median": float(degree_df["in_degree"].median()),
                    "std": round(float(degree_df["in_degree"].std()), 4)
                },
                "out_degree": {
                    "max": int(degree_df["out_degree"].max()),
                    "mean": round(float(degree_df["out_degree"].mean()), 4),
                    "median": float(degree_df["out_degree"].median()),
                    "std": round(float(degree_df["out_degree"].std()), 4)
                },
                "total_degree": {
                    "max": int(degree_df["total_degree"].max()),
                    "mean": round(float(degree_df["total_degree"].mean()), 4),
                    "median": float(degree_df["total_degree"].median()),
                    "std": round(float(degree_df["total_degree"].std()), 4)
                }
            }
        }

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Execute full analysis pipeline and aggregate all statistics.
        
        Returns:
            Dict containing complete EDA results.
        """
        logger.info("Running full EDA analysis pipeline...")
        return {
            "missing_values": self.analyze_missing_values(),
            "duplicates": self.analyze_duplicates(),
            "class_distribution": self.analyze_class_distribution(),
            "timestep_distribution": self.analyze_timestep_distribution(),
            "feature_statistics": self.analyze_feature_statistics(),
            "graph_statistics": self.analyze_graph_statistics(),
        }
