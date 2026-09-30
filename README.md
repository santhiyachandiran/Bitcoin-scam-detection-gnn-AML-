# Bitcoin Scam Detection Using Dynamic Graph Neural Networks

A modular, scalable machine learning project for detecting illicit transactions and scam activity in Bitcoin transaction networks using Dynamic Graph Neural Networks (GNNs) on the **Elliptic Dataset**.

---

## 📌 Project Overview

Bitcoin transactions form an evolving, dynamic temporal graph over time. This project aims to construct dynamic graph representations and train temporal GNN models (such as EvolveGCN, GCN, and GAT) to classify illicit transactions (scams, ransomware, money laundering) versus licit transactions.

---

## 🏗️ Project Architecture & Directory Structure

```
project/
├── data/
│   ├── raw/                 # Raw Elliptic dataset CSV files
│   ├── processed/           # Processed graph data (Stage 2)
│   └── sample/              # Synthetic sample data for offline testing
├── docs/
│   └── eda_report.md        # Comprehensive Markdown EDA Report
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA Notebook
├── reports/
│   └── figures/             # High-resolution generated EDA plots (.png)
│       ├── class_distribution.png
│       ├── unknown_vs_labelled.png
│       ├── timestep_distribution.png
│       ├── illicit_ratio_over_time.png
│       ├── node_degree_distribution.png
│       ├── feature_correlation_sample.png
│       └── subgraph_sample.png
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
│   ├── models/              # Dynamic GNN models (Stage 3)
│   ├── evaluation/          # Evaluation & benchmarks (Stage 4)
│   ├── backend/             # REST API Backend (Stage 5)
│   └── frontend/            # Web UI Frontend (Stage 6)
├── tests/
│   ├── __init__.py
│   ├── test_loader.py       # Stage 1 unit tests
│   └── test_stage2.py       # Stage 2 unit tests
├── run_eda.py               # Main CLI script for EDA pipeline
├── run_stage2.py            # Main CLI script for Stage 2 pipeline
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
- **Custom temporal train/val/test split ranges:**
  ```bash
  python run_stage2.py --train-start 1 --train-end 34 --val-start 35 --val-end 39 --test-start 40 --test-end 49
  ```

### Saved Artifacts (`data/processed/`):
- `elliptic_pyg_data.pt`: Unified full PyTorch Geometric `Data` graph object (203,769 nodes, 234,355 edges, 166 features).
- `elliptic_temporal_snapshots.pt`: List of 49 discrete temporal `Data` graph snapshots (one per timestep).
- `scaler.pt`: Fitted `StandardScaler` state for feature normalization.
- `preprocessing_metadata.json`: Preprocessing metrics, graph topology statistics, and validation checks.

---

## 📓 Running the Interactive Jupyter Notebook

To explore the dataset interactively:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 🧪 Running Unit Tests

To verify that the dataset loader, validator, preprocessor, graph builder, PyG schemas, and sample generators function correctly:

```bash
python -m pytest tests/
```

---

## 🗺️ Project Development Roadmap

- [x] **Stage 1: Initial Setup, Dataset Download, Validation & EDA** *(Completed)*
- [x] **Stage 2: Dynamic Graph Construction & PyTorch Geometric Integration** *(Completed)*
- [ ] **Stage 3: Dynamic GNN Model Implementations (EvolveGCN / GCN / GAT)**
- [ ] **Stage 4: Model Training, Hyperparameter Tuning & Benchmark Evaluation**
- [ ] **Stage 5: REST API Backend (FastAPI / Flask)**
- [ ] **Stage 6: Interactive Web Dashboard Frontend**

