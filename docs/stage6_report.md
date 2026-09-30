# Stage 6 Report: Production Web Application & Explainable AI Engine

## 1. Overview & Objectives

Stage 6 completes the **Bitcoin Scam Detection Using Dynamic Graph Neural Networks** project by deploying the trained Stage 5 **Triplet Transformer Dynamic GNN** into an interactive web application.

### Key Objectives
1. **Interactive Demo Platform**: Deliver a responsive UI for anti-money laundering (AML) analysts and academic reviewers.
2. **Dual-Input Mode**: Support both single transaction inspection (manual input) and batch CSV transaction processing.
3. **Stage 2 Preprocessing Pipeline Reuse**: Ensure strict feature standardization using the frozen `scaler.pt` from Stage 2.
4. **Stage 5 Model Inference**: Perform real-time inference using the trained `triplet_transformer_gcn_model.pth` checkpoint without retraining or modifying weights.
5. **Explainable AI (XAI)**: Generate feature-level risk contributions (Z-scores) and clear natural language explanations for flagged transactions.
6. **Monitoring Dashboard**: Provide live session metrics (KPI cards, risk distribution, trends) and integrate EDA visualization galleries from Stages 1–5.
7. **Session History**: Maintain in-memory prediction audit logs for real-time tracking during analyst sessions.

---

## 2. Web Application Architecture

```
[ Web Browser Frontend ]
         │
         │ REST API (JSON / Multipart CSV)
         ▼
[ Python FastAPI Server (src/backend/main.py) ]
    ├── Model Inference Engine (src/backend/inference.py)
    │     ├── Scaler Loader (data/processed/scaler.pt)
    │     └── Triplet Transformer Model (models/triplet_transformer_gcn_model.pth)
    ├── Explainability Engine (src/backend/explainability.py)
    │     ├── Baseline Stats (data/processed/elliptic_features.pt)
    │     └── Feature Z-Score Contributor Analysis
    └── Asset Server
          ├── Stage 1–5 Figures (reports/figures/)
          └── Web Application Assets (src/frontend/)
```

### Components

#### A. Backend API (`src/backend/main.py`)
- Framework: FastAPI + Uvicorn.
- Features: CORS enabled, auto-mounts static figure endpoints, exposes REST API endpoints, and serves SPA static files.

#### B. Preprocessing & Inference Engine (`src/backend/inference.py`)
- Loads `scaler.pt` saved during Stage 2 preprocessing.
- Reconstructs `TripletTransformerGCNClassifier` architecture (GCNConv + TransformerEncoder + Triplet Loss Margin Head).
- Loads state dict from `models/triplet_transformer_gcn_model.pth`.
- Implements single transaction and batch array preprocessing (166 features).
- Outputs sigmoid probability score $P(\text{Illicit})$, class label (`Illicit` if $P \ge 0.50$, else `Licit`), risk level (`CRITICAL`, `HIGH`, `MODERATE`, `LOW`), and inference latency (ms).

#### C. Explainability Engine (`src/backend/explainability.py`)
- Calculates population baseline statistics (mean and standard deviation per feature) from `data/processed/elliptic_features.pt`.
- Computes feature Z-scores for incoming test transactions to measure deviation from normal transaction distributions.
- Ranks top 5 contributing features for risk score calculation.
- Generates natural language summaries explaining why a transaction was flagged as suspicious.

#### D. React Single-Page Application (`src/frontend/index.html`)
- Built using React 18, Babel standalone, and Lucide icons via CDN.
- Styled with CSS3 modern dark theme (glassmorphism, vibrant neon accents, high contrast readability).
- Sections:
  1. **Monitoring Dashboard**: Live KPIs (Total, Licit, Illicit, High Risk counts, Average Risk Score).
  2. **Single Transaction Inspector**: Form inputs for transaction ID, timestep, and feature values with automated risk scoring & explanation drawer.
  3. **Batch CSV Uploader**: Drag-and-drop CSV file uploader with preview table and risk sorting.
  4. **Live History & Analytics**: Real-time tabular audit log of session predictions with CSV export functionality.
  5. **EDA & Pipeline Gallery**: Interactive modal/image gallery featuring visualizations from Stages 1–5.

---

## 3. API Endpoints Reference

| Endpoint | Method | Input | Output Description |
|---|---|---|---|
| `/api/predict` | `POST` | JSON (`tx_id`, `time_step`, `features`) | Single transaction prediction, risk score, top indicators, and explanation. |
| `/api/predict/csv` | `POST` | CSV file upload | Batch prediction summary table with individual transaction scores and indicators. |
| `/api/dashboard/stats` | `GET` | None | Session aggregate metrics (totals, counts, average risk score, risk distribution). |
| `/api/history` | `GET` | None | Complete list of predictions executed in the current session. |
| `/api/history/clear` | `DELETE` | None | Resets session prediction history and stats. |
| `/api/figures` | `GET` | None | List of available figure URLs from Stages 1–5 reports. |

---

## 4. Verification & Testing

The Stage 6 web application and underlying modules were validated using `pytest`:

```bash
python -m pytest tests/
```

### Test Results
- Total Tests: 28
- Status: **28 Passed, 0 Failed**
- Stage 6 Specific Tests (`tests/test_stage6.py`):
  1. `test_inference_engine_single`: Validates single vector scaling and forward pass probability.
  2. `test_inference_engine_batch`: Validates multi-row matrix inference.
  3. `test_explainer_contributions`: Validates Z-score ranking and top feature extraction.
  4. `test_api_health`: Validates `/api/health` status route.
  5. `test_api_predict`: Validates POST `/api/predict` endpoint response structure.
  6. `test_api_predict_csv`: Validates POST `/api/predict/csv` file upload parsing and predictions.
  7. `test_api_dashboard_stats`: Validates GET `/api/dashboard/stats` metrics calculation.

---

## 5. Execution Instructions

To launch the web application locally:

```bash
python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open a browser and navigate to:
`http://127.0.0.1:8000`

---

## 6. Project Completion Summary

All 6 project stages are complete, fully verified, and tested:

- **Stage 1**: Exploratory Data Analysis & Graph Structural Analytics.
- **Stage 2**: Temporal Graph Construction, Preprocessing & Scaling (`scaler.pt`).
- **Stage 3**: Non-Temporal Baselines (Logistic Regression, Random Forest, Transductive GCN).
- **Stage 4**: Causal Dynamic Temporal GNN (Snapshot GRU + GCN).
- **Stage 5**: Triplet Transformer Dynamic GNN (Margin Triplet Loss + Transformer Attention).
- **Stage 6**: Production Web Application, Explainability Engine, and Live Monitoring Dashboard.
