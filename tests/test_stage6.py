"""
Unit tests for Stage 6: FastAPI Backend, Inference Engine, Explainability Engine, and REST Routes.
"""

import pytest
import numpy as np
import io
import pandas as pd
from fastapi.testclient import TestClient

from src.backend.inference import ModelInferenceEngine
from src.backend.explainability import TransactionExplainer
from src.backend.main import app, session_history

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
        "model_used": "Stage 5 Triplet Transformer GNN",
        "x_raw": [3.5] + [0.0] * 165,
        "x_scaled": [3.5] + [0.0] * 165,
    }

    exp_res = explainer.explain(pred_res, top_k=5)

    assert exp_res["tx_id"] == 999999
    assert len(exp_res["top_indicators"]) == 5
    assert exp_res["top_indicators"][0]["feature_id"] == "feat_1"
    assert exp_res["top_indicators"][0]["scaled_z_score"] == 3.5
    assert "anomalous deviations" in exp_res["explanation_summary"]


def test_fastapi_health_route():
    """Test GET /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model_name" in data


def test_fastapi_predict_single_route():
    """Test POST /api/predict/single endpoint."""
    payload = {
        "tx_id": 888888,
        "time_step": 45,
        "feature_dict": {"feat_1": 2.5, "feat_2": 0.01}
    }
    response = client.post("/api/predict/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == 888888
    assert data["time_step"] == 45
    assert data["prediction"] in ["Illicit", "Licit"]
    assert "explanation_summary" in data
    assert len(data["top_indicators"]) == 5


def test_fastapi_predict_csv_batch_route():
    """Test POST /api/predict/csv batch file upload endpoint."""
    # Create synthetic 3-row CSV dataframe
    df_data = {
        "txId": [101, 102, 103],
        "time_step": [40, 41, 42],
    }
    for i in range(1, 167):
        df_data[f"feat_{i}"] = np.random.randn(3)

    df = pd.DataFrame(df_data)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/api/predict/csv",
        files={"file": ("test_transactions.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_processed"] == 3
    assert len(data["predictions"]) == 3


def test_fastapi_dashboard_stats_and_history():
    """Test GET /api/dashboard/stats, GET /api/history, and DELETE /api/history."""
    # Clean history
    client.delete("/api/history")

    # Add single prediction
    client.post("/api/predict/single", json={"tx_id": 777777})

    # Stats check
    stats_res = client.get("/api/dashboard/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert stats_data["total_transactions"] >= 1

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
    """Test GET / serves HTML frontend dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Bitcoin Scam Detection Dashboard" in response.text
