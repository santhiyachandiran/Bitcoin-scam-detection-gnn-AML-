"""
Unit tests for FastAPI Backend, Model Inference Engine, Local Explainability Engine,
Dataset Simulator, Graph Subgraph Provider, and Navigation API Routes.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.backend.inference import ModelInferenceEngine
from src.backend.explainability import TransactionExplainer
from src.backend.simulator import EllipticDatasetSimulator
from src.backend.main import app

client = TestClient(app)


def test_inference_engine_prediction():
    """Test ModelInferenceEngine single and batch predictions."""
    engine = ModelInferenceEngine()
    assert engine.is_loaded

    x_raw = np.random.randn(166).astype(np.float32)
    result = engine.predict_single(x_raw, tx_id=123456, time_step=42)

    assert result["tx_id"] == 123456
    assert result["time_step"] == 42
    assert result["prediction"] in ["Illicit", "Licit"]
    assert 0.0 <= result["probability"] <= 1.0
    assert 0.0 <= result["risk_score_pct"] <= 100.0
    assert "risk_level" in result

    # Batch test
    x_batch = np.random.randn(5, 166).astype(np.float32)
    batch_results = engine.predict_batch(x_batch)
    assert len(batch_results) == 5


def test_transaction_explainer():
    """Test TransactionExplainer feature deviation and narrative generation."""
    scaler_mean = np.zeros(166, dtype=np.float32)
    scaler_scale = np.ones(166, dtype=np.float32)
    explainer = TransactionExplainer(scaler_mean=scaler_mean, scaler_scale=scaler_scale)

    pred_res = {
        "tx_id": 999999,
        "prediction": "Illicit",
        "is_illicit": True,
        "risk_score_pct": 92.5,
        "risk_level": "CRITICAL (SCAM)",
        "model_used": "Triplet Transformer Dynamic GNN",
        "x_raw": [3.5] + [0.0] * 165,
        "x_scaled": [3.5] + [0.0] * 165,
    }

    exp_res = explainer.explain(pred_res, top_k=5)

    assert exp_res["tx_id"] == 999999
    assert len(exp_res["top_indicators"]) == 5
    assert exp_res["top_indicators"][0]["feature_id"] == "feat_1"
    assert exp_res["top_indicators"][0]["scaled_z_score"] == 3.5
    assert "anomalous deviations" in exp_res["explanation_summary"]


def test_dataset_simulator_and_subgraph():
    """Test EllipticDatasetSimulator data querying, random tx generation, and subgraph extraction."""
    sim = EllipticDatasetSimulator()
    assert sim.is_loaded

    # Test tx listing
    tx_list = sim.get_transaction_list(time_step=1, limit=10)
    assert isinstance(tx_list, list)
    if len(tx_list) > 0:
        assert "tx_id" in tx_list[0]
        assert tx_list[0]["time_step"] == 1

    # Test random transaction
    rand_tx = sim.get_random_transaction(time_step=1)
    assert "tx_id" in rand_tx
    assert "understandable_properties" in rand_tx
    assert "feat_1" in rand_tx["understandable_properties"]

    # Test custom properties conversion
    custom_props = {"feat_1": 5.5, "feat_2": 0.05}
    feat_vec = sim.custom_props_to_features(custom_props)
    assert len(feat_vec) == 166
    assert feat_vec[0] == 5.5
    assert feat_vec[1] == 0.05

    # Test subgraph extraction
    subgraph = sim.get_subgraph(tx_id=rand_tx["tx_id"], max_nodes=20)
    assert subgraph["target_tx_id"] == rand_tx["tx_id"]
    assert "nodes" in subgraph
    assert "edges" in subgraph
    assert len(subgraph["nodes"]) <= 20


def test_fastapi_health_route():
    """Test GET /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model_name" in data
    assert "simulator_loaded" in data


def test_fastapi_simulator_routes():
    """Test simulator endpoints."""
    # List transactions
    res_list = client.get("/api/simulator/transactions?time_step=1&limit=5")
    assert res_list.status_code == 200
    assert isinstance(res_list.json(), list)

    # Properties metadata
    res_meta = client.get("/api/simulator/properties-meta")
    assert res_meta.status_code == 200
    assert len(res_meta.json()) > 0

    # Random tx
    res_rand = client.get("/api/simulator/random?time_step=1")
    assert res_rand.status_code == 200
    tx_data = res_rand.json()
    assert "tx_id" in tx_data

    # Tx details
    res_tx = client.get(f"/api/simulator/transaction/{tx_data['tx_id']}")
    assert res_tx.status_code in [200, 404]


def test_fastapi_predict_single_route():
    """Test POST /api/predict/single endpoint with custom properties."""
    payload = {
        "tx_id": 888888,
        "time_step": 45,
        "custom_properties": {"feat_1": 2.5, "feat_2": 0.01}
    }
    response = client.post("/api/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == 888888
    assert data["time_step"] == 45
    assert data["prediction"] in ["Illicit", "Licit"]
    assert "explanation_summary" in data
    assert len(data["top_indicators"]) == 5
    assert "understandable_properties" in data


def test_fastapi_graph_routes():
    """Test subgraph and timestep network API routes."""
    # Subgraph route
    res_sub = client.get("/api/graph/subgraph/100001?max_nodes=15")
    assert res_sub.status_code == 200
    data_sub = res_sub.json()
    assert "nodes" in data_sub
    assert "edges" in data_sub

    # Timestep network route
    res_ts = client.get("/api/graph/timestep/1?max_nodes=15")
    assert res_ts.status_code == 200
    data_ts = res_ts.json()
    assert "nodes" in data_ts


def test_fastapi_dashboard_stats_and_analytics():
    """Test GET /api/dashboard/stats, GET /api/analytics/summary, GET /api/history, and DELETE /api/history."""
    # Clean history
    client.delete("/api/history")

    # Add single prediction
    client.post("/api/predict/single", json={"tx_id": 777777})

    # Stats check
    stats_res = client.get("/api/dashboard/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_transactions"] >= 1

    # Analytics check
    analytics_res = client.get("/api/analytics/summary")
    assert analytics_res.status_code == 200
    analytics_data = analytics_res.json()
    assert "dataset_name" in analytics_data

    # History check
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert len(hist_data) >= 1

    # Clear history
    del_res = client.delete("/api/history")
    assert del_res.status_code == 200

    empty_stats = client.get("/api/dashboard/stats").json()
    assert empty_stats["total_transactions"] == 0


def test_serve_frontend_index():
    """Test GET / serves HTML frontend dashboard without CSV upload UI."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Bitcoin Risk Intelligence" in response.text
    assert "Built-in Transaction Simulator" in response.text
    # Ensure CSV Upload UI is NOT present in frontend
    assert "CSV Upload" not in response.text
    assert 'type="file"' not in response.text
