"""
Visualizer module for generating EDA plots for Elliptic Bitcoin Transaction Dataset.
"""

from pathlib import Path
from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg") # Non-interactive backend for headless file rendering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx

from src.config import FIGURES_DIR, CLASS_MAP
from src.utils.logger import get_logger

logger = get_logger("eda_visualizer")

# Custom aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
COLOR_PALETTE = {"Illicit": "#E63946", "Licit": "#2A9D8F", "Unknown": "#A8DADC"}


class EDAVisualizer:
    """
    Generates and saves publication-quality figures for the EDA report.
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame], output_dir: Path = FIGURES_DIR):
        """
        Initialize visualizer with DataFrames and target figures directory.
        
        Args:
            data_dict: Dictionary containing 'features', 'classes', 'edges', 'merged'.
            output_dir: Output directory for plots.
        """
        self.df_features = data_dict["features"]
        self.df_classes = data_dict["classes"]
        self.df_edges = data_dict["edges"]
        self.df_merged = data_dict.get("merged")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_class_distribution(self, save_name: str = "class_distribution.png") -> Path:
        """Plot class distribution bar chart and donut chart."""
        logger.info(f"Generating class distribution plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        counts = self.df_classes["class_name"].value_counts()
        colors = [COLOR_PALETTE.get(k, "#457B9D") for k in counts.index]

        # 1. Bar Chart
        sns.barplot(x=counts.index, y=counts.values, ax=axes[0], palette=colors)
        axes[0].set_title("Transaction Count by Class Label", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Class Label", fontsize=12)
        axes[0].set_ylabel("Transaction Count", fontsize=12)
        for p in axes[0].patches:
            axes[0].annotate(f"{int(p.get_height()):,}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="bottom", fontsize=10, xytext=(0, 5), textcoords="offset points")

        # 2. Donut Chart
        axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=140,
                    wedgeprops=dict(width=0.4, edgecolor="w"))
        axes[1].set_title("Proportional Class Breakdown", fontsize=14, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {save_path}")
        return save_path

    def plot_unknown_vs_labelled(self, save_name: str = "unknown_vs_labelled.png") -> Path:
        """Plot Labelled vs Unknown transaction distribution."""
        logger.info(f"Generating labelled vs unknown plot...")
        fig, ax = plt.subplots(figsize=(8, 6))

        is_unknown = self.df_classes["class_name"] == "Unknown"
        counts = pd.Series({"Labelled (Licit/Illicit)": (~is_unknown).sum(), "Unknown (Unlabelled)": is_unknown.sum()})
        
        colors = ["#1D3557", "#A8DADC"]
        bars = ax.bar(counts.index, counts.values, color=colors, width=0.5)
        ax.set_title("Labelled vs Unknown Bitcoin Transactions", fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Transactions", fontsize=12)

        for bar in bars:
            height = bar.get_height()
            pct = (height / len(self.df_classes)) * 100
            ax.annotate(f"{int(height):,}\n({pct:.1f}%)",
                        (bar.get_x() + bar.get_width() / 2., height / 2),
                        ha="center", va="center", color="white" if bar.get_x() == 0 else "black",
                        fontsize=12, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {save_path}")
        return save_path

    def plot_timestep_dynamics(self, save_name: str = "timestep_distribution.png") -> Path:
        """Plot transactions timeline over 49 time-steps."""
        logger.info(f"Generating time-step distribution dynamics plot...")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # 1. Total volume per time step
        ts_counts = self.df_features["time_step"].value_counts().sort_index()
        axes[0].plot(ts_counts.index, ts_counts.values, marker="o", linewidth=2, color="#457B9D")
        axes[0].fill_between(ts_counts.index, ts_counts.values, color="#457B9D", alpha=0.2)
        axes[0].set_title("Total Transaction Volume Across 49 Time-Steps", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Transaction Count", fontsize=12)

        # 2. Stacked Area Chart per Class
        ts_class = self.df_merged.groupby(["time_step", "class_name"]).size().unstack(fill_value=0)
        cols = [c for c in ["Illicit", "Licit", "Unknown"] if c in ts_class.columns]
        colors = [COLOR_PALETTE[c] for c in cols]

        axes[1].stackplot(ts_class.index, [ts_class[c] for c in cols], labels=cols, colors=colors, alpha=0.85)
        axes[1].set_title("Class Composition Breakdown Across Time-Steps", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Time-Step (Discrete Interval ~ 2 Weeks)", fontsize=12)
        axes[1].set_ylabel("Transaction Count", fontsize=12)
        axes[1].legend(loc="upper right", fontsize=11)
        axes[1].set_xticks(range(1, 50, 2))

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {save_path}")
        return save_path

    def plot_illicit_ratio_over_time(self, save_name: str = "illicit_ratio_over_time.png") -> Path:
        """Plot ratio of illicit transactions over labelled set per time-step."""
        logger.info(f"Generating illicit ratio plot...")
        fig, ax = plt.subplots(figsize=(12, 5))

        ts_class = self.df_merged.groupby(["time_step", "class_name"]).size().unstack(fill_value=0)
        if "Illicit" in ts_class.columns and "Licit" in ts_class.columns:
            labelled = ts_class["Illicit"] + ts_class["Licit"]
            ratio = (ts_class["Illicit"] / labelled.replace(0, np.nan)) * 100
        else:
            ratio = pd.Series(0, index=range(1, 50))

        ax.plot(ratio.index, ratio.values, marker="s", color="#E63946", linewidth=2.5, label="Illicit Ratio (%)")
        ax.set_title("Illicit Transaction Ratio Among Labelled Data Over Time-Steps", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time-Step", fontsize=12)
        ax.set_ylabel("Illicit % of Labelled Txs", fontsize=12)
        ax.set_xticks(range(1, 50, 2))
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {save_path}")
        return save_path

    def plot_node_degree_distribution(self, save_name: str = "node_degree_distribution.png") -> Path:
        """Plot In-degree and Out-degree distributions (Histogram & Log-Log scale)."""
        logger.info(f"Generating node degree distribution plot...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        in_degrees = self.df_edges["txId2"].value_counts().values
        out_degrees = self.df_edges["txId1"].value_counts().values

        # 1. Degree Distribution Histograms
        axes[0].hist(in_degrees, bins=50, alpha=0.7, color="#2A9D8F", label="In-degree", log=True)
        axes[0].hist(out_degrees, bins=50, alpha=0.7, color="#E76F51", label="Out-degree", log=True)
        axes[0].set_title("In-Degree vs Out-Degree Histogram (Log Scale)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Degree Count", fontsize=12)
        axes[0].set_ylabel("Frequency (Log Scale)", fontsize=12)
        axes[0].legend(fontsize=11)

        # 2. Log-Log Degree Rank Plot
        in_sorted = np.sort(in_degrees)[::-1]
        out_sorted = np.sort(out_degrees)[::-1]
        axes[1].loglog(range(1, len(in_sorted) + 1), in_sorted, color="#2A9D8F", label="In-Degree", linewidth=2)
        axes[1].loglog(range(1, len(out_sorted) + 1), out_sorted, color="#E76F51", label="Out-Degree", linewidth=2)
        axes[1].set_title("Power-Law Degree Rank Distribution (Log-Log)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Node Rank", fontsize=12)
        axes[1].set_ylabel("Degree", fontsize=12)
        axes[1].legend(fontsize=11)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {save_path}")
        return save_path

    def plot_feature_correlation(self, save_name: str = "feature_correlation_sample.png") -> Path:
        """Plot correlation heatmap for a subset of local features."""
        logger.info(f"Generating feature correlation heatmap...")
        fig, ax = plt.subplots(figsize=(10, 8))

        feat_cols = [f"feat_{i}" for i in range(1, 16)] # First 15 features
        corr = self.df_features[feat_cols].corr()

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar=True)
        ax.set_title("Correlation Heatmap (Local Features 1-15 Sample)", fontsize=14, fontweight="bold")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {save_path}")
        return save_path

    def plot_subgraph_sample(self, timestep: int = 1, max_nodes: int = 80, save_name: str = "subgraph_sample.png") -> Path:
        """Plot a sample network graph visualization for a single time-step."""
        logger.info(f"Generating sample temporal graph visualization for time-step {timestep}...")
        fig, ax = plt.subplots(figsize=(10, 8))

        ts_nodes = set(self.df_merged[self.df_merged["time_step"] == timestep]["txId"])
        edges_sub = self.df_edges[
            self.df_edges["txId1"].isin(ts_nodes) & self.df_edges["txId2"].isin(ts_nodes)
        ]

        if len(edges_sub) > max_nodes * 2:
            edges_sub = edges_sub.sample(n=max_nodes * 2, random_state=42)

        G = nx.from_pandas_edgelist(edges_sub, source="txId1", target="txId2", create_using=nx.DiGraph())
        
        # Color mapping for nodes
        node_class_map = self.df_classes.set_index("txId")["class_name"].to_dict()
        node_colors = [COLOR_PALETTE.get(node_class_map.get(n, "Unknown"), "#A8DADC") for n in G.nodes()]

        pos = nx.spring_layout(G, k=0.3, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=60, node_color=node_colors, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray", arrows=True, arrowsize=8, ax=ax)

        # Legend
        for class_name, color in COLOR_PALETTE.items():
            ax.scatter([], [], color=color, label=class_name, s=100)
        ax.legend(loc="upper right", title="Class Label")
        ax.set_title(f"Sample Transaction Graph Topology (Time-Step {timestep})", fontsize=14, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {save_path}")
        return save_path

    def generate_all_plots(self) -> Dict[str, Path]:
        """Generate all EDA plots."""
        logger.info("Generating full suite of EDA visualizations...")
        return {
            "class_distribution": self.plot_class_distribution(),
            "unknown_vs_labelled": self.plot_unknown_vs_labelled(),
            "timestep_dynamics": self.plot_timestep_dynamics(),
            "illicit_ratio": self.plot_illicit_ratio_over_time(),
            "node_degree": self.plot_node_degree_distribution(),
            "feature_correlation": self.plot_feature_correlation(),
            "subgraph_sample": self.plot_subgraph_sample(),
        }
