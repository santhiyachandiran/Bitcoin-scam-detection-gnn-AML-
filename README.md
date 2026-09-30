# Bitcoin Scam Detection Using Dynamic Graph Neural Networks

A modular, scalable machine learning project for detecting illicit transactions and scam activity in Bitcoin transaction networks using Dynamic Graph Neural Networks (GNNs) on the **Elliptic Dataset**.

---

## 📌 Project Overview

Bitcoin transactions form an evolving, dynamic temporal graph over time. This project aims to construct dynamic graph representations and train temporal GNN models (such as Recurrent GCN, GCN, and GAT) to classify illicit transactions (scams, ransomware, money laundering) versus licit transactions.

---

## 🏗️ Project Architecture & Directory Structure

```
project/
├── data/
│   ├── raw/                 # Raw Elliptic dataset CSV files
│   ├── processed/           # Processed graph data (Stage 2)
│   └── sample/              # Synthetic sample data for offline testing
├── docs/
│   ├── eda_report.md        # Comprehensive Markdown EDA Report
│   ├── stage3_report.md     # Baseline Models & Static GCN Report
│   └── stage4_report.md     # Strictly Causal Dynamic Temporal GNN Report
├── models/                  # Saved model checkpoints (Stage 3 & 4)
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── gcn_model.pth
│   └── recurrent_gcn_model.pth
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA Notebook
├── reports/
│   ├── stage3_metrics.json  # Stage 3 metric export
│   ├── stage3_metrics.csv   # Stage 3 metrics table
│   ├── stage4_metrics.json  # Stage 4 comprehensive metric export
│   ├── stage4_metrics.csv   # Stage 4 metrics table
│   └── figures/             # High-resolution generated plots (.png)
│       ├── class_distribution.png
│       ├── unknown_vs_labelled.png
│       ├── timestep_distribution.png
│       ├── illicit_ratio_over_time.png
│       ├── node_degree_distribution.png
│       ├── feature_correlation_sample.png
│       ├── subgraph_sample.png
│       ├── stage3_roc_curves.png
│       ├── stage3_pr_curves.png
│       ├── stage3_confusion_matrices.png
│       ├── stage3_metrics_comparison.png
│       ├── stage4_roc_curves.png
│       ├── stage4_pr_curves.png
│       ├── stage4_confusion_matrices.png
│       ├── stage4_metrics_comparison.png
│       └── stage4_training_loss.png
├── src/
│   ├── __init__.py
│   ├── config.py            # Global configuration parameters & constants
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py        # Centralized logger
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py      # Automated dataset downloader
│   │   ├── loader.py        # Dataset loader class
│   │   └── validator.py     # Schema & integrity validator
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── analyzer.py      # Statistical EDA engine
│   │   └── visualizer.py    # Publication-quality plot generator
│   ├── graph/               # Data Preprocessing & Dynamic Graph Construction (Stage 2)
│   │   ├── __init__.py
│   │   ├── preprocessor.py  # Feature cleaning, leak-free normalization & split masks
│   │   ├── builder.py       # PyTorch Geometric Data construction & topology stats
│   │   └── saver.py         # PyG graph serialization & I/O utilities
│   ├── models/              # ML Baselines, Static GCN & Causal Recurrent GCN (Stage 3 & 4)
│   │   ├── __init__.py
│   │   ├── baselines.py     # Logistic Regression & Random Forest classifiers
│   │   ├── gcn.py           # PyG GCN model architecture & weighted loss trainer
│   │   └── temporal_gcn.py  # Causal Recurrent GCN (GCN + GRU node memory)
│   ├── evaluation/          # Metrics & Benchmarks (Stage 3 & 4)
│   │   ├── __init__.py
│   │   ├── metrics.py       # Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, CM
│   │   └── visualizer.py    # ROC, PR, Confusion Matrix, loss curves & bar plots
│   ├── backend/             # REST API Backend (Stage 5)
│   └── frontend/            # Web UI Frontend (Stage 6)
├── tests/
│   ├── __init__.py
│   ├── test_loader.py       # Stage 1 unit tests
│   ├── test_stage2.py       # Stage 2 unit tests
│   ├── test_stage3.py       # Stage 3 unit tests
│   └── test_stage4.py       # Stage 4 temporal causality & model unit tests
├── run_eda.py               # Main CLI script for EDA pipeline
├── run_stage2.py            # Main CLI script for Stage 2 pipeline
├── run_stage3.py            # Main CLI script for Stage 3 baseline & GCN pipeline
├── run_stage4.py            # Main CLI script for Stage 4 dynamic temporal GNN pipeline
├── requirements.txt         # Required Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started & Environment Setup

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Running the EDA Pipeline (Stage 1)

To execute the complete dataset loading, integrity validation, EDA calculation, and figure generation pipeline:

```bash
python run_eda.py
```

### Options:
- **Run with synthetic sample dataset (Offline/Fast Test):**
  ```bash
  python run_eda.py --use-sample
  ```

---

## 🔄 Running Data Preprocessing & Graph Construction (Stage 2)

To clean features, apply leak-free feature normalization, construct PyTorch Geometric graph data objects, create 49 temporal snapshot graphs, and generate split masks:

```bash
python run_stage2.py
```

### Options:
- **Run with synthetic sample dataset (Offline/Fast Test):**
  ```bash
  python run_stage2.py --use-sample
  ```

---

## 🤖 Running Baseline Models & Static GCN (Stage 3)

To train and evaluate Logistic Regression, Random Forest, and Static PyTorch Geometric GCN models:

```bash
python run_stage3.py
```

---

## ⚡ Running Strictly Causal Dynamic Temporal GNN (Stage 4)

To train and evaluate the **Strictly Causal Recurrent GCN** across 49 discrete temporal graph snapshots while preventing temporal data leakage across future timesteps:

```bash
python run_stage4.py
```

### Options:
- **Run with synthetic sample dataset (Offline/Fast Test):**
  ```bash
  python run_stage4.py --use-sample
  ```
- **Custom hyperparameters:**
  ```bash
  python run_stage4.py --epochs 100 --lr 0.01 --hidden-dim 64 --dropout 0.2
  ```

### Evaluated Benchmark Results (Test Set: Timesteps 40–49):

| Model | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7398 | 0.1559 | **0.8097** | 0.2614 | 0.5518 | 0.8549 |
| **Random Forest** | **0.9702** | **0.8359** | 0.5928 | **0.6937** | **0.8390** | **0.8858** |
| **Static GCN (Transductive)** | 0.8557 | 0.1889 | 0.4670 | 0.2690 | 0.5945 | 0.7767 |
| **Recurrent GCN (Dynamic)** | 0.9193 | 0.3321 | 0.4135 | **0.3683** | **0.6626** | **0.8136** |

---

## 📓 Running the Interactive Jupyter Notebook

To explore the dataset interactively:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 🧪 Running Unit Tests

To verify dataset loading, preprocessors, graph builders, baseline models, static/dynamic GCN trainers, causality tests, and visualization engines:

```bash
python -m pytest tests/
```

---

## 🗺️ Project Development Roadmap

- [x] **Stage 1: Initial Setup, Dataset Download, Validation & EDA** *(Completed)*
- [x] **Stage 2: Dynamic Graph Construction & PyTorch Geometric Integration** *(Completed)*
- [x] **Stage 3: Baseline Models & Basic GCN Model Implementation** *(Completed)*
- [x] **Stage 4: Strictly Causal Dynamic Temporal GNN (Recurrent GCN) & Benchmarking** *(Completed)*
- [ ] **Stage 5: Transformer & Triplet/Contrastive Learning Architecture**
- [ ] **Stage 6: REST API Backend (FastAPI / Flask)**
- [ ] **Stage 7: Interactive Web Dashboard Frontend**



