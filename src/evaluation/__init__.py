"""
Evaluation Package for Bitcoin Scam Detection Project.
Provides evaluation metrics calculation and visualization utilities.
"""

from src.evaluation.metrics import (
    compute_evaluation_metrics,
    generate_metrics_summary_table,
    dataframe_to_markdown,
)
from src.evaluation.visualizer import (
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_metrics_comparison,
)

__all__ = [
    "compute_evaluation_metrics",
    "generate_metrics_summary_table",
    "dataframe_to_markdown",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_confusion_matrices",
    "plot_metrics_comparison",
]
