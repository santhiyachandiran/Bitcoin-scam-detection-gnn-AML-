# Bitcoin Fraud & Risk Intelligence Platform using Dynamic Graph Neural Networks

A modular, scalable machine learning system for detecting illicit transactions and scam activity in Bitcoin networks using Dynamic Graph Neural Networks (GNNs) on the **Elliptic Dataset**.

---

## 📌 Project Overview

Bitcoin transactions form an evolving, dynamic temporal graph over time. This project constructs dynamic graph representations and trains temporal GNN models (Spatial GCN + Transformer Encoder + Triplet Loss) to classify illicit scam transactions versus legitimate transactions.

The platform includes a real-time web application featuring a built-in transaction simulator, local feature explainability, interactive 2-hop neighborhood graph topology visualization, and session risk monitoring.

---

## 🏗️ Project Architecture & Directory Structure

```
project/
├── data/
│   ├── raw/                 # Raw Elliptic dataset CSV files
│   ├── processed/           # Processed PyG graph & scaler artifacts
│   └── sample/              # Synthetic sample data for testing
├── docs/
│   ├── eda_report.md        # Comprehensive Markdown EDA Report
│   ├── stage2_report.md     # Preprocessing & Dynamic Graph Report
│   ├── stage3_report.md     # Baseline Models & Static GCN Report
│   ├── stage4_report.md     # Strictly Causal Dynamic Temporal GNN Report
│   ├── stage5_report.md     # Triplet Transformer Dynamic GNN Report
│   └── stage6_report.md     # Web Application & Interactive Platform Report
├── models/                  # Offline trained model checkpoints
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── gcn_model.pth
│   ├── recurrent_gcn_model.pth
│   └── triplet_transformer_gcn_model.pth # Stage 5 Checkpoint
├── reports/
│   └── figures/             # High-resolution generated plots (.png)
├── src/
│   ├── config.py            # Global configuration parameters & constants
│   ├── utils/               # Centralized logging utilities
│   ├── data/                # Dataset loader & integrity validator
│   ├── eda/                 # Statistical EDA engine & visualizers
│   ├── graph/               # Data Preprocessing & PyG Graph Construction
│   ├── models/              # ML Baselines, Static GCN, Dynamic GCN & Triplet Transformer
│   ├── evaluation/          # Metrics & Benchmarks
│   ├── backend/             # FastAPI REST API, Inference Engine, Explainer & Simulator
│   │   ├── main.py          # REST API server
│   │   ├── inference.py     # Model inference engine loading Stage 2 scaler & Stage 5 model
│   │   ├── explainability.py# Local feature contributor & explanation engine
│   │   └── simulator.py     # Built-in Elliptic dataset simulator & subgraph provider
│   └── frontend/            # Interactive Web Application
│       └── index.html       # Light White Theme SPA (Cytoscape.js, Tab Routing)
├── tests/
│   ├── test_loader.py       # Dataset loading unit tests
│   ├── test_stage2.py       # Preprocessing unit tests
│   ├── test_stage3.py       # Baseline unit tests
│   ├── test_stage4.py       # Temporal causality unit tests
│   ├── test_stage5.py       # Triplet Transformer GNN unit tests
│   └── test_stage6.py       # Simulator, API, inference, graph & navigation unit tests
├── run_eda.py               # Main CLI script for EDA pipeline
├── run_stage2.py            # Main CLI script for Stage 2 pipeline
├── run_stage3.py            # Main CLI script for Stage 3 baseline & GCN pipeline
├── run_stage4.py            # Main CLI script for Stage 4 dynamic temporal GNN pipeline
├── run_stage5.py            # Main CLI script for Stage 5 Triplet Transformer pipeline
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

## 📊 Running Pipeline Stages (Offline Training & Validation)

- **Stage 1 EDA:** `python run_eda.py`
- **Stage 2 Preprocessing:** `python run_stage2.py`
- **Stage 3 Baselines & GCN:** `python run_stage3.py`
- **Stage 4 Recurrent GCN:** `python run_stage4.py`
- **Stage 5 Triplet Transformer GNN:** `python run_stage5.py`

### Benchmark Evaluation Summary (Test Set: Timesteps 40–49):

| Model | Accuracy | Precision (Illicit) | Recall (Illicit) | F1-Score (Illicit) | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7398 | 0.1559 | **0.8097** | 0.2614 | 0.5518 | 0.8549 |
| **Random Forest** | **0.9702** | **0.8359** | 0.5928 | **0.6937** | **0.8390** | **0.8858** |
| **Static GCN (Transductive)** | 0.8557 | 0.1889 | 0.4670 | 0.2690 | 0.5945 | 0.7767 |
| **Recurrent GCN (Dynamic)** | 0.9193 | 0.3321 | 0.4135 | 0.3683 | 0.6626 | 0.8136 |
| **Triplet Transformer Dynamic GNN** | 0.8902 | 0.2691 | 0.5425 | **0.3597** | **0.6499** | **0.8312** |

---

## 🖥️ Running the Interactive Web Application

The interactive web application uses the offline trained Stage 5 model and the local Elliptic Bitcoin Transaction Dataset.

### Key Features:
1. **No CSV Upload Required**: File upload UI is completely removed.
2. **Built-in Transaction Simulator**: Select transactions from the local Elliptic dataset, generate random transactions, or pick timesteps (1–49).
3. **Understandable Property Inputs**: Adjust BTC volume, fee ratios, input/output counts, and anonymity mix ratios without exposing raw 166-feature vectors.
4. **Dedicated Interactive Graph Page**: Visualize 2-hop neighborhood transaction flow graphs with Cytoscape.js, zoom/pan controls, risk color highlighting, and node click details drawer.
5. **Light White Theme**: Professional academic/startup aesthetic in white, burgundy (`#7a0c2e`), and light red. No dark mode or glassmorphism.
6. **No Stage Numbers in UI**: Development stage numbers are excluded from the web application interface.

### 1. Launch Server
```bash
python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Access Web Application
Open your web browser and navigate to:
`http://127.0.0.1:8000`

### 3. Application Navigation Pages
- **Home**: Overview, key capability highlights, system metrics, and quick start guide.
- **Transaction Analysis**: Built-in simulator, custom property tweaks, GNN risk scoring, top risk indicators table, and natural language narrative.
- **Transaction Graph**: Dedicated interactive network explorer with node search, timestep snapshot selection, and node detail drawer.
- **Risk Monitoring**: Real-time session prediction stream, risk distribution breakdown, and audit log.
- **Analytics**: Dataset metrics (203k nodes, 234k edges) and model performance evaluation specs.
- **About**: System architecture overview and methodology.

---

## 🧪 Running Unit Tests

To run the complete unit test suite across all pipeline components and API routes:

```bash
python -m pytest tests/
```
