# Stage 6 Report: Production Web Application, Interactive Graph Explorer & Explainable AI Engine

## 1. Overview & Objectives

Stage 6 deploys the trained **Triplet Transformer Dynamic GNN** model into a production-grade, interactive web application.

### Key Objectives & Design Principles:
1. **Interactive Web Application**: Deliver a responsive single-page web application featuring tabbed section navigation (Home, Transaction Analysis, Transaction Graph, Risk Monitoring, Analytics, About).
2. **No CSV Upload Requirement**: Completely removed CSV upload workflow in favor of an interactive transaction analysis platform.
3. **Elliptic Dataset Integration**: Uses the locally stored Elliptic Bitcoin Transaction Dataset (203,769 transactions, 234,355 directed edges, 49 timesteps) as the sole research dataset.
4. **Offline Model Inference Engine**: Loads the trained Stage 5 `triplet_transformer_gcn_model.pth` checkpoint and Stage 2 `scaler.pt`. Performs inference without retraining.
5. **Built-in Transaction Simulator**:
   - Allows users to select dataset transactions by ID or filter by timestep (1–49).
   - Provides random transaction generation.
   - Offers a custom-input form for understandable transaction properties (BTC volume, fee ratio, input/output address counts, CoinJoin mix ratio, fee variance, average input value, time delay, address reuse count).
   - Hides raw 166-feature vectors from the end user.
6. **Dedicated Interactive Graph Page**:
   - Visualizes 2-hop local subgraphs and snapshot graph topologies around target transaction nodes.
   - Powered by Cytoscape.js with zooming, panning, layout toggles, node selection, and highlighting of high-risk nodes.
   - Interactive node details drawer displaying node risk probability and direct option to analyze in predictor.
7. **UI/UX Aesthetics**:
   - Clean **LIGHT WHITE THEME** with white background, deep burgundy (`#7a0c2e`), and light red (`#dc2626`) accents.
   - Professional academic / startup aesthetic.
   - No dark mode, no glassmorphism.
   - Stage numbers omitted from UI strings.

---

## 2. Web Application System Architecture

```
[ Interactive Light-Theme SPA (src/frontend/index.html) ]
    ├── Tab Navigation (Home, Analysis, Graph, Monitoring, Analytics, About)
    ├── Understandable Parameter Controls Form
    └── Cytoscape.js Network Graph Renderer
         │
         │ REST API (JSON)
         ▼
[ FastAPI Backend Server (src/backend/main.py) ]
    ├── Model Inference Engine (src/backend/inference.py)
    │     ├── Scaler Loader (data/processed/scaler.pt)
    │     └── Triplet Transformer Model (models/triplet_transformer_gcn_model.pth)
    ├── Explainability Engine (src/backend/explainability.py)
    │     └── Z-Score Impact Contributor Analysis
    └── Dataset Simulator & Graph Provider (src/backend/simulator.py)
          ├── PyG Data Loader (data/processed/elliptic_pyg_data.pt)
          ├── Transaction Index & Property Mapper
          └── Subgraph Extraction Engine
```

---

## 3. API Endpoints Reference

| Endpoint | Method | Parameters | Description |
|---|---|---|---|
| `/api/health` | `GET` | None | API health status, loaded model info, and simulator status. |
| `/api/simulator/transactions` | `GET` | `time_step`, `filter_class`, `search_query`, `limit` | List dataset transactions matching filters. |
| `/api/simulator/transaction/{tx_id}` | `GET` | `tx_id` | Fetch details and understandable properties for a transaction. |
| `/api/simulator/random` | `GET` | `time_step`, `filter_class` | Pick a random transaction from the local dataset. |
| `/api/simulator/properties-meta` | `GET` | None | Metadata schema for understandable transaction properties. |
| `/api/predict/single` | `POST` | JSON (`tx_id`, `time_step`, `custom_properties`, `features`) | Run GNN prediction, risk scoring, top risk indicators, and explanation. |
| `/api/graph/subgraph/{tx_id}` | `GET` | `tx_id`, `max_nodes` | Extract 2-hop local subgraph topology with node risk predictions. |
| `/api/graph/timestep/{time_step}` | `GET` | `time_step`, `max_nodes` | Fetch timestep network snapshot topology with risk scores. |
| `/api/dashboard/stats` | `GET` | None | Session aggregate metrics and risk distribution counts. |
| `/api/history` | `GET` | `limit` | Retrieve session prediction history log. |
| `/api/history` | `DELETE` | None | Clear session prediction history. |
| `/api/analytics/summary` | `GET` | None | Dataset overview and model evaluation performance metrics. |

---

## 4. Verification & Testing

All backend components, simulator functions, graph routes, explainability calculations, and UI serve endpoints were validated using `pytest`:

```bash
python -m pytest tests/test_stage6.py
```

### Test Results:
- Total Tests: 9 Passed / 0 Failed
- Validated components:
  1. `test_inference_engine_prediction`: Model loading, scaling, single/batch prediction.
  2. `test_transaction_explainer`: Z-score deviation ranking, natural language summary construction.
  3. `test_dataset_simulator_and_subgraph`: Dataset transaction queries, random generation, property mapping, and subgraph topology extraction.
  4. `test_fastapi_health_route`: Health check API response.
  5. `test_fastapi_simulator_routes`: Simulator transactions list, metadata, and random tx routes.
  6. `test_fastapi_predict_single_route`: Prediction analysis with custom parameter form payload.
  7. `test_fastapi_graph_routes`: Subgraph and timestep network API endpoints.
  8. `test_fastapi_dashboard_stats_and_analytics`: Session monitoring stats, history logging, clear history, and analytics summary.
  9. `test_serve_frontend_index`: Single-page app HTML rendering without CSV upload UI.

---

## 5. Execution Instructions

To start the web application locally:

```bash
python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open a browser and navigate to: `http://127.0.0.1:8000`
