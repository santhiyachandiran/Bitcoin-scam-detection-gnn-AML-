"""
Evaluation Metrics Module for Bitcoin Scam Detection Project.
Computes Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, and Confusion Matrix.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.utils.logger import get_logger

logger = get_logger("evaluation_metrics")


def compute_evaluation_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[Union[np.ndarray, List[float]]] = None,
    model_name: str = "Model",
    split_name: str = "Test",
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for binary classification.

    Args:
        y_true: Ground truth binary labels (0=Licit, 1=Illicit).
        y_pred: Predicted binary class labels (0 or 1).
        y_prob: Predicted probabilities for positive class (class 1: Illicit). Optional for ROC-AUC.
        model_name: Identifier name of the model.
        split_name: Identifier for the dataset split (e.g., 'Validation', 'Test').

    Returns:
        Dict[str, Any]: Dictionary containing all metric scores and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    roc_auc: Optional[float] = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        # Handle case where test set has only 1 unique class
        if len(np.unique(y_true)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception as e:
                logger.warning(f"Failed to compute ROC-AUC for {model_name} on {split_name}: {e}")
                roc_auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist() # [[TN, FP], [FN, TP]]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model_name": model_name,
        "split_name": split_name,
        "num_samples": int(len(y_true)),
        "num_illicit": int((y_true == 1).sum()),
        "num_licit": int((y_true == 0).sum()),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    logger.info(
        f"Metrics [{model_name} | {split_name}]: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Macro-F1={macro_f1:.4f}"
        + (f", ROC-AUC={roc_auc:.4f}" if roc_auc is not None else "")
    )
    return metrics


def generate_metrics_summary_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generate a summary pandas DataFrame from a list of metrics dictionaries.

    Args:
        metrics_list: List of metric result dictionaries from compute_evaluation_metrics.

    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model_name"],
            "Split": m["split_name"],
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision (Illicit)": f"{m['precision']:.4f}",
            "Recall (Illicit)": f"{m['recall']:.4f}",
            "F1-Score (Illicit)": f"{m['f1_score']:.4f}",
            "Macro-F1": f"{m['macro_f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "N/A",
            "TN": m["tn"],
            "FP": m["fp"],
            "FN": m["fn"],
            "TP": m["tp"],
        })
    df = pd.DataFrame(rows)
    return df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert pandas DataFrame to Markdown table string without requiring tabulate.

    Args:
        df: Input pandas DataFrame.

    Returns:
        str: Markdown table string.
    """
    headers = [str(col) for col in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_str = " | ".join([str(val) for val in row])
        lines.append("| " + row_str + " |")
    return "\n".join(lines)

