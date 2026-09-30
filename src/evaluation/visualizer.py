"""
Visualization Engine for Stage 3 Model Evaluation.
Generates publication-quality plots: ROC curves, Precision-Recall curves,
Confusion Matrices, and comparative performance bar charts.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.utils.logger import get_logger

logger = get_logger("evaluation_visualizer")

# Set publication style aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def plot_roc_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_trues: Dict[str, np.ndarray],
    y_probs: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "ROC Curves Comparison (Stage 3)",
) -> Path:
    """
    Plot ROC curves for multiple models on the same figure.

    Args:
        eval_results: Dictionary mapping model_name to its evaluation metrics dict.
        y_trues: Dictionary mapping model_name to true target binary array.
        y_probs: Dictionary mapping model_name to predicted probabilities array.
        output_path: Path to save generated figure.
        title: Plot title.

    Returns:
        Path of saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (model_name, metrics) in enumerate(eval_results.items()):
        y_true = y_trues[model_name]
        y_prob = y_probs[model_name]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = metrics.get("roc_auc", auc(fpr, tpr))
        
        color = colors[idx % len(colors)]
        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.5,
            label=f"{model_name} (AUC = {roc_auc:.4f})" if roc_auc is not None else f"{model_name}",
        )

    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved ROC curves figure to {output_path}")
    return output_path


def plot_precision_recall_curves(
    eval_results: Dict[str, Dict[str, Any]],
    y_trues: Dict[str, np.ndarray],
    y_probs: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Precision-Recall Curves Comparison (Stage 3)",
) -> Path:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        eval_results: Dictionary mapping model_name to evaluation metrics dict.
        y_trues: Dictionary mapping model_name to true binary array.
        y_probs: Dictionary mapping model_name to predicted probabilities array.
        output_path: Path to save generated figure.
        title: Plot title.

    Returns:
        Path of saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (model_name, metrics) in enumerate(eval_results.items()):
        y_true = y_trues[model_name]
        y_prob = y_probs[model_name]

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        color = colors[idx % len(colors)]
        ax.plot(
            recall,
            precision,
            color=color,
            lw=2.5,
            label=f"{model_name} (PR-AUC = {pr_auc:.4f})",
        )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Illicit)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision (Illicit)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved Precision-Recall curves figure to {output_path}")
    return output_path


def plot_confusion_matrices(
    eval_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Confusion Matrices Comparison (Stage 3)",
) -> Path:
    """
    Plot side-by-side heatmaps of confusion matrices for all evaluated models.

    Args:
        eval_results: Dictionary mapping model_name to evaluation metrics dict.
        output_path: Path to save figure.
        title: Plot title.

    Returns:
        Path of saved figure.
    """
    num_models = len(eval_results)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5), dpi=300)

    if num_models == 1:
        axes = [axes]

    class_names = ["Licit (0)", "Illicit (1)"]

    for idx, (model_name, metrics) in enumerate(eval_results.items()):
        ax = axes[idx]
        cm = np.array(metrics["confusion_matrix"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=class_names,
            yticklabels=class_names,
        )

        ax.set_title(f"{model_name}\nF1 (Illicit): {metrics['f1_score']:.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Actual Class", fontsize=11, fontweight="bold")

    plt.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrices figure to {output_path}")
    return output_path


def plot_metrics_comparison(
    eval_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    title: str = "Model Performance Benchmark Comparison (Stage 3)",
) -> Path:
    """
    Plot grouped bar chart comparing key evaluation metrics across models.

    Args:
        eval_results: Dictionary mapping model_name to evaluation metrics dict.
        output_path: Path to save figure.
        title: Plot title.

    Returns:
        Path of saved figure.
    """
    metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "macro_f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 (Illicit)", "Macro-F1", "ROC-AUC"]

    models = list(eval_results.keys())
    x = np.arange(len(metric_labels))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, model_name in enumerate(models):
        m = eval_results[model_name]
        values = []
        for metric_key in metrics_to_compare:
            val = m.get(metric_key, 0.0)
            values.append(val if val is not None else 0.0)

        offset = x + (i - len(models) / 2 + 0.5) * width
        rects = ax.bar(offset, values, width, label=model_name, color=colors[i % len(colors)], edgecolor="black", linewidth=0.8)
        
        # Value labels on bars
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if len(models) > 2 else 0,
                )

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_ylim([0.0, 1.15])
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved metrics comparison figure to {output_path}")
    return output_path
