# Stage 5: Triplet-Style Dynamic Graph Network with Transformer Encoder Report

## 📌 Executive Summary

In Stage 5, we designed, implemented, and benchmarked an advanced architecture inspired by recent cryptocurrency scam detection research:
**“Triplet-Style Dynamic Graph Network With Transformer Encoder for Scam Detection in Cryptocurrency Transactions.”**

### Key Architectural Innovations:
1. **Spatial-Temporal Representation Learning**:
   - Spatial GCN convolution layers extract intra-snapshot graph structural features.
   - PyTorch `TransformerEncoder` layers compute temporal self-attention over sequence representations without future data leakage.
2. **Contrastive Representation Learning via Triplet Loss**:
   - Online triplet mining creates (Anchor=Illicit, Positive=Illicit, Negative=Licit) node triplets.
   - Combined objective: $\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{BCE\_weighted}} + 0.5 \cdot \mathcal{L}_{	ext{triplet}}$, pulling illicit transaction clusters together while separating licit transactions in $L_2$-normalized embedding space.
3. **Strict Causal & Temporal Isolation**:
   - **Train**: Timesteps 1–34 (29,894 labelled training nodes).
   - **Validation**: Timesteps 35–39 (5,486 labelled nodes for early stopping).
   - **Test**: Timesteps 40–49 (11,184 labelled nodes for final evaluation).
   - Class weight `pos_weight` (7.6349) is calculated strictly on training timesteps.

---

## 📊 Comprehensive Performance Benchmark (Test Set: Timesteps 40–49)

| Model | Split | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Test | 0.7398 | 0.1559 | 0.8097 | 0.2614 | 0.5518 | 0.8549 | 7759 | 2789 | 121 | 515 |
| Random Forest | Test | 0.9702 | 0.8359 | 0.5928 | 0.6937 | 0.8390 | 0.8858 | 10474 | 74 | 259 | 377 |
| GCN Model | Test | 0.8557 | 0.1889 | 0.4670 | 0.2690 | 0.5945 | 0.7767 | 9273 | 1275 | 339 | 297 |
| Recurrent GCN (Dynamic) | Test | 0.9193 | 0.3321 | 0.4135 | 0.3683 | 0.6626 | 0.8136 | 10019 | 529 | 373 | 263 |
| Triplet Transformer GNN | Test | 0.8902 | 0.2691 | 0.5425 | 0.3597 | 0.6499 | 0.8312 | 9611 | 937 | 291 | 345 |

---

## 🔍 Validation Set Benchmark (Timesteps 35–39)

| Model | Split | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Validation | 0.7838 | 0.2675 | 0.9508 | 0.4175 | 0.6424 | 0.9199 | 3875 | 1164 | 22 | 425 |
| Random Forest | Validation | 0.9892 | 0.9429 | 0.9239 | 0.9333 | 0.9637 | 0.9881 | 5014 | 25 | 34 | 413 |
| GCN Model | Validation | 0.8101 | 0.2674 | 0.7651 | 0.3963 | 0.6418 | 0.8627 | 4102 | 937 | 105 | 342 |
| Recurrent GCN (Dynamic) | Validation | 0.8914 | 0.4053 | 0.7136 | 0.5170 | 0.7279 | 0.8815 | 4571 | 468 | 128 | 319 |
| Triplet Transformer GNN | Validation | 0.8910 | 0.4129 | 0.8009 | 0.5449 | 0.7415 | 0.9144 | 4530 | 509 | 89 | 358 |

---

## 💡 Comparative Insights Across All Stages (Stage 3 -> Stage 4 -> Stage 5)

1. **Evolution of Metric Performance**:
   - **Stage 3 Baseline Models**: Established classical tabular baselines (LR, RF) and static GCN.
   - **Stage 4 Recurrent GCN**: Solved static GCN temporal graph leakage by enforcing sequential causality.
   - **Stage 5 Triplet Transformer GCN**: Combines Transformer temporal self-attention with contrastive Triplet Loss, sharpening boundary separation between illicit scam transaction clusters and licit accounts.

---

## 📁 Saved Artifacts
- **Model Checkpoint:** `models/triplet_transformer_gcn_model.pth`
- **Metrics Exports:** `reports/stage5_metrics.json`, `reports/stage5_metrics.csv`
- **Visualizations:** `reports/figures/`
  - `stage5_roc_curves.png`
  - `stage5_pr_curves.png`
  - `stage5_confusion_matrices.png`
  - `stage5_metrics_comparison.png`
  - `stage5_training_loss.png`
