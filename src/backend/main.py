"""
FastAPI REST API Backend for Stage 6 Bitcoin Scam Detection Web Application.
Provides endpoints for single/batch inference, explainability, monitoring dashboard metrics,
session prediction history, figure assets, and static frontend hosting.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import pandas as pd
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
)
from src.backend.inference import ModelInferenceEngine
from src.backend.explainability import TransactionExplainer
from src.utils.logger import get_logger

logger = get_logger("backend_api")

app = FastAPI(
    title="Bitcoin Scam Detection API",
    description="REST API Backend powered by Stage 5 Triplet Transformer Dynamic GNN",
    version="1.0.0",
)

# Enable CORS for local development & dashboard frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Inference Engine & Explainer singletons
engine = ModelInferenceEngine(processed_dir=PROCESSED_DATA_DIR, models_dir=MODELS_DIR)
explainer = TransactionExplainer(scaler_mean=engine.scaler_mean, scaler_scale=engine.scaler_scale)

# In-memory prediction session history store
session_history: List[Dict[str, Any]] = []


# --- Pydantic Data Transfer Objects ---

class SingleTransactionRequest(BaseModel):
    tx_id: Optional[int] = Field(default=100001, description="Transaction ID")
    time_step: Optional[int] = Field(default=1, description="Transaction Timestep (1-49)")
    features: Optional[List[float]] = Field(default=None, description="Raw feature vector of length 166")
    feature_dict: Optional[Dict[str, float]] = Field(default=None, description="Key-value mapping of features")


class SingleTransactionResponse(BaseModel):
    status: str
    tx_id: int
    time_step: int
    prediction: str
    is_illicit: bool
    risk_score_pct: float
    probability: float
    risk_level: str
    risk_color: str
    model_used: str
    explanation_summary: str
    top_indicators: List[Dict[str, Any]]


# --- API Routes ---

@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    """Check API health and loaded model status."""
    return {
        "status": "online",
        "model_loaded": engine.is_loaded,
        "model_name": engine.model_name,
        "model_type": engine.model_type,
        "history_count": len(session_history),
    }


@app.post("/api/predict/single", response_model=SingleTransactionResponse)
def predict_single_transaction(req: SingleTransactionRequest):
    """Run Stage 5 GNN prediction and explainability on a single transaction."""
    tx_id = req.tx_id if req.tx_id is not None else 100001
    time_step = req.time_step if req.time_step is not None else 1

    # Extract feature array
    if req.features is not None and len(req.features) == 166:
        x_raw = np.array(req.features, dtype=np.float32)
    elif req.feature_dict is not None:
        feats = [req.feature_dict.get(f"feat_{i}", 0.0) for i in range(1, 167)]
        x_raw = np.array(feats, dtype=np.float32)
    else:
        # Default sample features for quick demo test
        logger.info("No feature vector passed; generating baseline transaction feature vector.")
        x_raw = np.random.randn(166).astype(np.float32)

    # Inference
    pred_res = engine.predict_single(x_raw, tx_id=tx_id, time_step=time_step)

    # Explainability
    exp_res = explainer.explain(pred_res, top_k=5)

    full_result = {
        "status": "success",
        "tx_id": pred_res["tx_id"],
        "time_step": pred_res["time_step"],
        "prediction": pred_res["prediction"],
        "is_illicit": pred_res["is_illicit"],
        "risk_score_pct": pred_res["risk_score_pct"],
        "probability": pred_res["probability"],
        "risk_level": pred_res["risk_level"],
        "risk_color": pred_res["risk_color"],
        "model_used": pred_res["model_used"],
        "explanation_summary": exp_res["explanation_summary"],
        "top_indicators": exp_res["top_indicators"],
    }

    # Append to session history
    session_history.append(full_result)

    return full_result


@app.post("/api/predict/csv")
async def predict_csv_batch(file: UploadFile = File(...)):
    """Upload CSV file of transaction features, run batch inference and explainability."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Check txId and time_step columns if present
        tx_ids = df["txId"].tolist() if "txId" in df.columns else [100000 + i for i in range(len(df))]
        time_steps = df["time_step"].tolist() if "time_step" in df.columns else [1] * len(df)

        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        if len(feat_cols) != 166:
            # Drop non-feature columns
            non_feat = [c for c in df.columns if c not in ["txId", "time_step", "class", "binary_label"]]
            if len(non_feat) == 166:
                feat_cols = non_feat
            else:
                # Pad or slice to 166 features
                X_raw = df.select_dtypes(include=[np.number]).values.astype(np.float32)
                if X_raw.shape[1] < 166:
                    pad = np.zeros((X_raw.shape[0], 166 - X_raw.shape[1]), dtype=np.float32)
                    X_raw = np.hstack([X_raw, pad])
                else:
                    X_raw = X_raw[:, :166]
        else:
            X_raw = df[feat_cols].values.astype(np.float32)

        # Batch Inference
        batch_res = engine.predict_batch(X_raw, tx_ids=tx_ids, time_steps=time_steps)

        # Explain top items
        batch_output = []
        for res in batch_res:
            exp = explainer.explain(res, top_k=3)
            res_dict = {
                "tx_id": res["tx_id"],
                "time_step": res["time_step"],
                "prediction": res["prediction"],
                "is_illicit": res["is_illicit"],
                "risk_score_pct": res["risk_score_pct"],
                "risk_level": res["risk_level"],
                "risk_color": res["risk_color"],
                "explanation_summary": exp["explanation_summary"],
                "top_indicator": exp["top_indicators"][0]["feature_name"] if exp["top_indicators"] else "N/A",
            }
            batch_output.append(res_dict)
            session_history.append(res_dict)

        illicit_count = sum(1 for r in batch_output if r["is_illicit"])
        licit_count = len(batch_output) - illicit_count

        return {
            "status": "success",
            "filename": file.filename,
            "total_processed": len(batch_output),
            "illicit_count": illicit_count,
            "licit_count": licit_count,
            "predictions": batch_output,
        }

    except Exception as e:
        logger.error(f"Error processing CSV batch upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")


@app.get("/api/dashboard/stats")
def get_dashboard_stats() -> Dict[str, Any]:
    """Get aggregate monitoring metrics and risk distribution for current session."""
    total = len(session_history)
    if total == 0:
        return {
            "total_transactions": 0,
            "illicit_count": 0,
            "licit_count": 0,
            "illicit_pct": 0.0,
            "avg_risk_score": 0.0,
            "risk_distribution": {
                "CRITICAL": 0,
                "HIGH": 0,
                "MODERATE": 0,
                "LOW": 0,
            }
        }

    illicit_cnt = sum(1 for r in session_history if r.get("is_illicit", False))
    licit_cnt = total - illicit_cnt
    mean_risk = float(np.mean([r.get("risk_score_pct", 0.0) for r in session_history]))

    crit_cnt = sum(1 for r in session_history if r.get("risk_score_pct", 0) >= 80)
    high_cnt = sum(1 for r in session_history if 50 <= r.get("risk_score_pct", 0) < 80)
    mod_cnt = sum(1 for r in session_history if 20 <= r.get("risk_score_pct", 0) < 50)
    low_cnt = sum(1 for r in session_history if r.get("risk_score_pct", 0) < 20)

    return {
        "total_transactions": total,
        "illicit_count": illicit_cnt,
        "licit_count": licit_cnt,
        "illicit_pct": round((illicit_cnt / total) * 100, 2),
        "avg_risk_score": round(mean_risk, 2),
        "risk_distribution": {
            "CRITICAL": crit_cnt,
            "HIGH": high_cnt,
            "MODERATE": mod_cnt,
            "LOW": low_cnt,
        }
    }


@app.get("/api/history")
def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent prediction history."""
    return session_history[-limit:][::-1]


@app.delete("/api/history")
def clear_history():
    """Clear session prediction history."""
    global session_history
    session_history = []
    return {"status": "success", "message": "Prediction history cleared."}


@app.get("/api/figures/{filename}")
def get_figure(filename: str):
    """Serve pipeline PNG figures from reports/figures/."""
    fig_path = FIGURES_DIR / filename
    if not fig_path.exists():
        raise HTTPException(status_code=404, detail=f"Figure asset {filename} not found.")
    return FileResponse(fig_path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def serve_frontend_dashboard():
    """Serve single-page frontend application dashboard."""
    index_path = PROJECT_ROOT / "src" / "frontend" / "index.html"
    if not index_path.exists():
        return HTMLResponse(content="<h1>Bitcoin Scam Detection Dashboard API Online</h1><p>Frontend file not found.</p>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
