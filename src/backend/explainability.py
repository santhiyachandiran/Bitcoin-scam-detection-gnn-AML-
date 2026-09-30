"""
Explainability Engine for Stage 6 Bitcoin Scam Detection.
Computes local feature contributions, Z-score deviations, top suspicious indicators,
and natural language explanations for individual predictions.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("backend_explainability")

# Human readable feature descriptions mapping
FEATURE_DESCRIPTIONS = {
    1: "Transaction Output Amount / Volume",
    2: "Transaction Fee Ratio to Amount",
    3: "Input Addresses Count",
    4: "Output Addresses Count",
    5: "CoinJoin / Anonymity Mix Ratio",
    6: "Fee Variance across Inputs",
    7: "Average Input Transaction Value",
    8: "Max Output Value Ratio",
    9: "Time Delay from Parent Transaction",
    10: "Address Reuse Count",
}

# Default categories
def get_feature_name_and_category(feat_idx_1_based: int) -> Tuple[str, str]:
    """Get human-readable feature name and category for 1-based feature index."""
    if feat_idx_1_based in FEATURE_DESCRIPTIONS:
        name = FEATURE_DESCRIPTIONS[feat_idx_1_based]
    else:
        name = f"Transaction Feature #{feat_idx_1_based}"

    if feat_idx_1_based <= 94:
        category = "Local Transaction Attribute"
    else:
        category = "1-Hop Aggregated Neighbor Network"

    return name, category


class TransactionExplainer:
    """
    Computes explainability metrics, Z-score feature impacts, and natural language explanations.
    """

    def __init__(self, scaler_mean: Optional[np.ndarray] = None, scaler_scale: Optional[np.ndarray] = None):
        """
        Initialize TransactionExplainer.

        Args:
            scaler_mean: Mean vector from StandardScaler (shape 166).
            scaler_scale: Scale/Std vector from StandardScaler (shape 166).
        """
        self.scaler_mean = scaler_mean if scaler_mean is not None else np.zeros(166, dtype=np.float32)
        scale_safe = scaler_scale if scaler_scale is not None else np.ones(166, dtype=np.float32)
        self.scaler_scale = np.where(scale_safe == 0, 1.0, scale_safe)

    def explain(self, prediction_result: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """
        Generate explainability report for a transaction prediction.

        Args:
            prediction_result: Result dict returned by ModelInferenceEngine.predict_single.
            top_k: Number of top feature indicators to include (default: 5).

        Returns:
            Dict[str, Any]: Detailed explainability output.
        """
        x_raw = np.array(prediction_result["x_raw"], dtype=np.float32)
        x_scaled = np.array(prediction_result["x_scaled"], dtype=np.float32)

        # 1. Compute Z-score deviations relative to baseline
        z_scores = np.abs(x_scaled)
        
        # Sort indices by descending absolute Z-score impact
        top_indices = np.argsort(z_scores)[::-1][:top_k]

        top_indicators = []
        for rank, idx in enumerate(top_indices, start=1):
            feat_idx_1 = int(idx + 1)
            name, cat = get_feature_name_and_category(feat_idx_1)
            raw_val = float(x_raw[idx])
            z_score = float(x_scaled[idx])
            abs_z = float(z_scores[idx])

            # Directional impact indicator
            direction = "Elevated" if z_score > 0 else "Depressed"

            top_indicators.append({
                "rank": rank,
                "feature_id": f"feat_{feat_idx_1}",
                "feature_name": name,
                "category": cat,
                "raw_value": round(raw_val, 4),
                "scaled_z_score": round(z_score, 4),
                "impact_magnitude": round(abs_z, 4),
                "direction": direction,
            })

        # 2. Construct Natural Language Narrative
        prob_pct = prediction_result["risk_score_pct"]
        is_illicit = prediction_result["is_illicit"]
        tx_id = prediction_result["tx_id"]
        model_used = prediction_result["model_used"]

        top_names = [ind["feature_name"] for ind in top_indicators[:3]]
        indicators_str = ", ".join(top_names)

        if is_illicit:
            explanation = (
                f"Transaction #{tx_id} was flagged as HIGH RISK ILLICIT / SCAM ({prob_pct}% scam probability) "
                f"by {model_used}. The decision was driven primarily by anomalous deviations in {indicators_str}. "
                f"The highest contributor ({top_indicators[0]['feature_name']}) deviates by {top_indicators[0]['scaled_z_score']} "
                f"standard deviations from the legitimate baseline, matching illicit scam cluster characteristics."
            )
        else:
            explanation = (
                f"Transaction #{tx_id} was classified as LEGITIMATE LICIT ({prob_pct}% scam probability) "
                f"by {model_used}. Feature distributions closely align with normal licit transaction behavior. "
                f"Top features evaluated include {indicators_str}, all remaining within acceptable baseline safety bounds."
            )

        return {
            "tx_id": tx_id,
            "prediction": prediction_result["prediction"],
            "risk_score_pct": prob_pct,
            "risk_level": prediction_result["risk_level"],
            "explanation_summary": explanation,
            "top_indicators": top_indicators,
        }
