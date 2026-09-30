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
│   ├── models/              # ML Baselines, Static GCN, Dynamic GCN & Triplet Transformer (Stage 3–5)
│   │   ├── __init__.py
│   │   ├── baselines.py     # Logistic Regression & Random Forest classifiers
│   │   ├── gcn.py           # PyG GCN model architecture & weighted loss trainer
│   │   ├── temporal_gcn.py  # Causal Recurrent GCN (GCN + GRU node memory)
│   │   └── triplet_transformer_gcn.py # Triplet Transformer Dynamic GNN
│   ├── evaluation/          # Metrics & Benchmarks (Stage 3–5)
│   │   ├── __init__.py
│   │   ├── metrics.py       # Accuracy, Precision, Recall, F1, Macro-F1, ROC-AUC, CM
│   │   └── visualizer.py    # ROC, PR, Confusion Matrix, loss curves & bar plots
│   ├── backend/             # FastAPI REST API & XAI Engine (Stage 6)
│   │   ├── main.py          # REST API server & static asset mount
│   │   ├── inference.py     # Model inference engine loading Stage 2 scaler & Stage 5 model
│   │   └── explainability.py# XAI Z-score contributor engine & natural language explanations
│   └── frontend/            # React Single Page Application (Stage 6)
│       └── index.html       # Responsive Dark Mode Dashboard with Lucide icons
├── tests/
│   ├── __init__.py
│   ├── test_loader.py       # Stage 1 unit tests
│   ├── test_stage2.py       # Stage 2 unit tests
│   ├── test_stage3.py       # Stage 3 unit tests
│   ├── test_stage4.py       # Stage 4 temporal causality unit tests
│   ├── test_stage5.py       # Stage 5 Triplet Transformer GNN unit tests
│   └── test_stage6.py       # Stage 6 API, inference, and explainability unit tests
├── run_eda.py               # Main CLI script for EDA pipeline
├── run_stage2.py            # Main CLI script for Stage 2 pipeline
├── run_stage3.py            # Main CLI script for Stage 3 baseline & GCN pipeline
├── run_stage4.py            # Main CLI script for Stage 4 dynamic temporal GNN pipeline
├── run_stage5.py            # Main CLI script for Stage 5 Triplet Transformer GNN pipeline
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

---

## ⚡ Running Triplet-Style Dynamic GNN with Transformer Encoder (Stage 5)

To train and evaluate the **Triplet-Style Dynamic Graph Network with Transformer Encoder** across 49 discrete temporal graph snapshots:

```bash
python run_stage5.py
```

### Evaluated Benchmark Results Across All Models (Test Set: Timesteps 40–49):

| Model | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7398 | 0.1559 | **0.8097** | 0.2614 | 0.5518 | 0.8549 |
| **Random Forest** | **0.9702** | **0.8359** | 0.5928 | **0.6937** | **0.8390** | **0.8858** |
| **Static GCN (Transductive)** | 0.8557 | 0.1889 | 0.4670 | 0.2690 | 0.5945 | 0.7767 |
| **Recurrent GCN (Dynamic)** | 0.9193 | 0.3321 | 0.4135 | 0.3683 | 0.6626 | 0.8136 |
| **Triplet Transformer GNN** | 0.8902 | 0.2691 | 0.5425 | **0.3597** | **0.6499** | **0.8312** |

---

## 🖥️ Running the Stage 6 Web Application & Explainability Engine

Stage 6 provides an interactive React single-page application and FastAPI backend to deploy the trained Stage 5 Triplet Transformer model for real-time inference and Explainable AI (XAI) risk diagnosis.

### 1. Launch Server
```bash
python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Access UI
Open your web browser and navigate to:
`http://127.0.0.1:8000`

### 3. Features Included
- **Single Transaction Inspector**: Enter custom transaction parameters or test features to receive real-time risk predictions, confidence scores, and natural language explanations.
- **Batch CSV Upload**: Upload custom Bitcoin transaction CSV batches for bulk inference.
- **Explainability Drawer**: Displays feature Z-score deviations and top contributing risk factors.
- **Monitoring Dashboard**: Live KPIs (Total, Illicit, Licit, High Risk count, Mean Risk Score).
- **Session History Log**: In-memory prediction session tracking with CSV download option.
- **Pipeline Figure Gallery**: Browse high-resolution EDA and model benchmark visual plots.

---

## 📓 Running the Interactive Jupyter Notebook

To explore the dataset interactively:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 🧪 Running Unit Tests

To verify dataset loading, preprocessors, graph builders, baseline models, static/dynamic GCN trainers, causality tests, model inference, explainability engines, and API endpoints:

```bash
python -m pytest tests/
```

---

## 🗺️ Project Development Roadmap

- [x] **Stage 1: Initial Setup, Dataset Download, Validation & EDA** *(Completed)*
- [x] **Stage 2: Dynamic Graph Construction & PyTorch Geometric Integration** *(Completed)*
- [x] **Stage 3: Baseline Models & Basic GCN Model Implementation** *(Completed)*
- [x] **Stage 4: Strictly Causal Dynamic Temporal GNN (Recurrent GCN) & Benchmarking** *(Completed)*
- [x] **Stage 5: Triplet-Style Dynamic GNN with Transformer Encoder** *(Completed)*
- [x] **Stage 6: REST API Backend (FastAPI), React Single-Page App, Explainable AI & Monitoring Dashboard** *(Completed)*





