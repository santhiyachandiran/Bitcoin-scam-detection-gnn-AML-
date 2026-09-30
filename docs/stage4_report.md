# Stage 4: Dynamic Temporal GNN Evaluation Report

## 📌 Executive Summary

In Stage 4, we designed, implemented, and benchmarked a **Strictly Causal Dynamic Temporal Graph Neural Network (Recurrent GCN)** to resolve the temporal data leakage limitation identified in Stage 3 static GCN models.

### Key Architectural Highlights:
1. **Strict Temporal Causality**:
   - Message passing operates exclusively per-snapshot $G_t = (V_t, E_t)$ on intra-timestep edges $E_t$ and node features $X_t$.
   - Temporal memory propagates strictly forward in time ($t-1 	o t$) via a node-level `GRUCell`.
   - **Zero future temporal data leakage**: Future nodes ($t' > t$) are completely excluded from graph convolutions and memory states during training at timestep $t$.
2. **Temporal Split & Imbalance Protocol**:
   - **Train**: Timesteps 1–34 (29,894 labelled training nodes).
   - **Validation**: Timesteps 35–39 (5,486 labelled nodes for early stopping).
   - **Test**: Timesteps 40–49 (11,184 labelled nodes for out-of-sample testing).
   - Class weight `pos_weight` (7.6349) is calculated strictly on training timesteps.

---

## 📊 Comprehensive Performance Benchmark (Test Set: Timesteps 40–49)

| Model | Split | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Test | 0.7398 | 0.1559 | 0.8097 | 0.2614 | 0.5518 | 0.8549 | 7759 | 2789 | 121 | 515 |
| Random Forest | Test | 0.9702 | 0.8359 | 0.5928 | 0.6937 | 0.8390 | 0.8858 | 10474 | 74 | 259 | 377 |
| GCN Model | Test | 0.8557 | 0.1889 | 0.4670 | 0.2690 | 0.5945 | 0.7767 | 9273 | 1275 | 339 | 297 |
| Recurrent GCN (Dynamic) | Test | 0.9193 | 0.3321 | 0.4135 | 0.3683 | 0.6626 | 0.8136 | 10019 | 529 | 373 | 263 |

---

## 🔍 Validation Set Benchmark (Timesteps 35–39)

| Model | Split | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Validation | 0.7838 | 0.2675 | 0.9508 | 0.4175 | 0.6424 | 0.9199 | 3875 | 1164 | 22 | 425 |
| Random Forest | Validation | 0.9892 | 0.9429 | 0.9239 | 0.9333 | 0.9637 | 0.9881 | 5014 | 25 | 34 | 413 |
| GCN Model | Validation | 0.8101 | 0.2674 | 0.7651 | 0.3963 | 0.6418 | 0.8627 | 4102 | 937 | 105 | 342 |
| Recurrent GCN (Dynamic) | Validation | 0.8914 | 0.4053 | 0.7136 | 0.5170 | 0.7279 | 0.8815 | 4571 | 468 | 128 | 319 |

---

## 💡 Comparative Insights: Stage 3 vs Stage 4

1. **Causal Integrity vs. Transductive Leakage**:
   - The Stage 3 static GCN processed all 49 timesteps in a single global graph matrix, allowing future node features to leak backward during spatial graph convolutions.
   - The Stage 4 **Recurrent GCN** enforces strict temporal causality, ensuring that predictions at timestep $t$ depend solely on information available at or before $t$.

2. **Temporal Dynamics**:
   - The integration of a `GRUCell` enables persistent transaction nodes to carry historical state forward in time, improving illicit transaction pattern recognition across dynamic time-series snapshots.

---

## 📁 Saved Artifacts
- **Model Checkpoint:** `models/recurrent_gcn_model.pth`
- **Metrics Exports:** `reports/stage4_metrics.json`, `reports/stage4_metrics.csv`
- **Visualizations:** `reports/figures/`
  - `stage4_roc_curves.png`
  - `stage4_pr_curves.png`
  - `stage4_confusion_matrices.png`
  - `stage4_metrics_comparison.png`
  - `stage4_training_loss.png`
