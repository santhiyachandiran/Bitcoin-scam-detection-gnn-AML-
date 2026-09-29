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
│   ├── graph/               # Dynamic graph construction (Stage 2)
│   ├── models/              # Dynamic GNN models (Stage 3)
│   ├── evaluation/          # Evaluation & benchmarks (Stage 4)
│   ├── backend/             # REST API Backend (Stage 5)
│   └── frontend/            # Web UI Frontend (Stage 6)
├── tests/
│   ├── __init__.py
│   └── test_loader.py       # Pytest unit tests
├── run_eda.py               # Main CLI script for EDA pipeline
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

## 📊 Running the EDA Pipeline

To execute the complete dataset loading, integrity validation, EDA calculation, and figure generation pipeline:

```bash
python run_eda.py
```

### Options:
- **Run with synthetic sample dataset (Offline/Fast Test):**
  ```bash
  python run_eda.py --use-sample
  ```

### Outputs Generated:
1. **Console Terminal:** Displays shapes, columns, integrity checks, and summary tables.
2. **Figures Folder:** Saves all high-res plots in `reports/figures/`.
3. **Markdown Report:** Writes detailed findings to `docs/eda_report.md`.

---

## 📓 Running the Interactive Jupyter Notebook

To explore the dataset interactively:

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

---

## 🧪 Running Unit Tests

To verify that the dataset loader, validator, and sample generators function correctly:

```bash
pytest tests/
```

---

## 🗺️ Project Development Roadmap

- [x] **Stage 1: Initial Setup, Dataset Download, Validation & EDA** *(Completed)*
- [ ] **Stage 2: Dynamic Graph Construction & PyTorch Geometric Integration**
- [ ] **Stage 3: Dynamic GNN Model Implementations (EvolveGCN / GCN / GAT)**
- [ ] **Stage 4: Model Training, Hyperparameter Tuning & Benchmark Evaluation**
- [ ] **Stage 5: REST API Backend (FastAPI / Flask)**
- [ ] **Stage 6: Interactive Web Dashboard Frontend**
