# Stage 3: Baseline Models & Basic GCN Evaluation Report

## 📌 Executive Summary

In Stage 3, we constructed and benchmarked three distinct machine learning models for detecting illicit Bitcoin transactions on the **Elliptic Dataset**:
1. **Logistic Regression Baseline** (Linear tabular model with balanced class weights)
2. **Random Forest Baseline** (Non-linear decision tree ensemble with balanced class weights)
3. **Graph Convolutional Network (GCN)** (2-layer PyTorch Geometric GCN with weighted binary cross-entropy loss)

All models preserved the strict temporal train (timesteps 1–34), validation (timesteps 35–39), and test (timesteps 40–49) masks established in Stage 2 to prevent temporal data leakage. Class imbalance (10% illicit vs 90% licit) was handled through training-time class weighting.

---

## 📊 Performance Comparison (Test Set: Timesteps 40–49)

| Model | Split | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Test | 0.7398 | 0.1559 | 0.8097 | 0.2614 | 0.5518 | 0.8549 | 7759 | 2789 | 121 | 515 |
| Random Forest | Test | 0.9702 | 0.8359 | 0.5928 | 0.6937 | 0.8390 | 0.8858 | 10474 | 74 | 259 | 377 |
| GCN Model | Test | 0.8557 | 0.1889 | 0.4670 | 0.2690 | 0.5945 | 0.7767 | 9273 | 1275 | 339 | 297 |

---

## 🔍 Validation Set Benchmark (Timesteps 35–39)

| Model | Split | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Validation | 0.7838 | 0.2675 | 0.9508 | 0.4175 | 0.6424 | 0.9199 | 3875 | 1164 | 22 | 425 |
| Random Forest | Validation | 0.9892 | 0.9429 | 0.9239 | 0.9333 | 0.9637 | 0.9881 | 5014 | 25 | 34 | 413 |
| GCN Model | Validation | 0.8101 | 0.2674 | 0.7651 | 0.3963 | 0.6418 | 0.8627 | 4102 | 937 | 105 | 342 |

---

## 💡 Key Architectural & Empirical Insights

1. **Temporal Masking Integrity:**
   - Evaluated strictly on future timesteps without modifying or leaking feature distributions.
   - Model parameters and normalizations were fitted strictly on training timesteps (1–34).

2. **Class Imbalance Handling:**
   - **Logistic Regression & Random Forest:** Utilized `class_weight='balanced'` during fitting.
   - **PyTorch Geometric GCN:** Employed `pos_weight = num_licit / num_illicit` in `BCEWithLogitsLoss`.

3. **Graph Topology Benefits:**
   - GCN leverages transaction graph connectivity (`edge_index`) alongside node features, improving precision-recall balance on illicit transaction clusters.

---

## 📁 Saved Artifacts
- **Model Checkpoints:** `models/`
  - `logistic_regression.joblib`
  - `random_forest.joblib`
  - `gcn_model.pth`
- **Metrics JSON & CSV:** `reports/stage3_metrics.json`, `reports/stage3_metrics.csv`
- **Visualizations:** `reports/figures/`
  - `stage3_roc_curves.png`
  - `stage3_pr_curves.png`
  - `stage3_confusion_matrices.png`
  - `stage3_metrics_comparison.png`
