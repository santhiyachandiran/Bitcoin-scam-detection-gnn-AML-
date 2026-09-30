"""
Unit tests for Stage 3: Baseline Models, PyG GCN, Evaluation Metrics, and Visualizations.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

from src.models import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    GCNClassifier,
    GCNTrainer,
)
from src.evaluation import (
    compute_evaluation_metrics,
    generate_metrics_summary_table,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_metrics_comparison,
)


@pytest.fixture(scope="module")
def synthetic_tabular_data():
    """Generate synthetic tabular feature matrix and target labels."""
    np.random.seed(42)
    n_samples = 150
    n_features = 20

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]).astype(np.int64)

    train_mask = np.zeros(n_samples, dtype=bool)
    train_mask[:100] = True

    val_mask = np.zeros(n_samples, dtype=bool)
    val_mask[100:125] = True

    test_mask = np.zeros(n_samples, dtype=bool)
    test_mask[125:] = True

    return X, y, train_mask, val_mask, test_mask


@pytest.fixture(scope="module")
def synthetic_pyg_data(synthetic_tabular_data):
    """Generate synthetic PyG Data object."""
    X, y, train_mask, val_mask, test_mask = synthetic_tabular_data
    n_nodes = X.shape[0]

    # Simple ring + random graph edge list
    src = np.arange(n_nodes)
    dst = (np.arange(n_nodes) + 1) % n_nodes
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        labelled_mask=torch.ones(n_nodes, dtype=torch.bool),
        num_nodes=n_nodes,
    )
    return data


def test_logistic_regression_baseline(synthetic_tabular_data, tmp_path):
    """Test Logistic Regression baseline training, prediction, and persistence."""
    X, y, train_mask, val_mask, test_mask = synthetic_tabular_data

    lr = LogisticRegressionBaseline(random_state=42)
    assert not lr.is_fitted

    lr.fit(X[train_mask], y[train_mask])
    assert lr.is_fitted

    preds = lr.predict(X[test_mask])
    probs = lr.predict_proba(X[test_mask])

    assert len(preds) == int(test_mask.sum())
    assert len(probs) == int(test_mask.sum())
    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test Save & Load
    save_path = tmp_path / "lr_model.joblib"
    lr.save(save_path)
    assert save_path.exists()

    lr_reloaded = LogisticRegressionBaseline().load(save_path)
    assert lr_reloaded.is_fitted
    reloaded_preds = lr_reloaded.predict(X[test_mask])
    np.testing.assert_array_equal(preds, reloaded_preds)


def test_random_forest_baseline(synthetic_tabular_data, tmp_path):
    """Test Random Forest baseline training, prediction, and persistence."""
    X, y, train_mask, val_mask, test_mask = synthetic_tabular_data

    rf = RandomForestBaseline(n_estimators=10, max_depth=5, random_state=42)
    assert not rf.is_fitted

    rf.fit(X[train_mask], y[train_mask])
    assert rf.is_fitted

    preds = rf.predict(X[test_mask])
    probs = rf.predict_proba(X[test_mask])

    assert len(preds) == int(test_mask.sum())
    assert len(probs) == int(test_mask.sum())
    assert set(preds).issubset({0, 1})
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test Save & Load
    save_path = tmp_path / "rf_model.joblib"
    rf.save(save_path)
    assert save_path.exists()

    rf_reloaded = RandomForestBaseline().load(save_path)
    assert rf_reloaded.is_fitted
    reloaded_preds = rf_reloaded.predict(X[test_mask])
    np.testing.assert_array_equal(preds, reloaded_preds)


def test_gcn_classifier_architecture(synthetic_pyg_data):
    """Test GCNClassifier forward pass and shape."""
    data = synthetic_pyg_data
    model = GCNClassifier(in_channels=20, hidden_channels=32, num_layers=2)

    logits = model(data.x, data.edge_index)
    assert logits.dim() == 1
    assert logits.size(0) == data.num_nodes


def test_gcn_trainer_fit_and_predict(synthetic_pyg_data, tmp_path):
    """Test GCNTrainer training loop, prediction, and checkpoint persistence."""
    data = synthetic_pyg_data

    trainer = GCNTrainer(in_channels=20, hidden_channels=16, num_layers=2)
    assert not trainer.is_fitted

    trainer.fit(data, epochs=5, lr=0.01, patience=5)
    assert trainer.is_fitted

    probs = trainer.predict_proba(data, mask=data.test_mask)
    preds = trainer.predict(data, mask=data.test_mask)

    assert len(probs) == int(data.test_mask.sum().item())
    assert len(preds) == int(data.test_mask.sum().item())
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test Save & Load
    checkpoint_path = tmp_path / "gcn_model.pth"
    trainer.save(checkpoint_path)
    assert checkpoint_path.exists()

    reloaded_trainer = GCNTrainer(in_channels=20).load(checkpoint_path)
    assert reloaded_trainer.is_fitted
    reloaded_probs = reloaded_trainer.predict_proba(data, mask=data.test_mask)
    np.testing.assert_allclose(probs, reloaded_probs, rtol=1e-4)


def test_evaluation_metrics_calculator():
    """Test evaluation metrics computation and edge case handling."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.3, 0.85, 0.6, 0.95, 0.05])

    metrics = compute_evaluation_metrics(y_true, y_pred, y_prob, model_name="TestModel", split_name="Test")

    assert metrics["model_name"] == "TestModel"
    assert metrics["split_name"] == "Test"
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["roc_auc"] is not None
    assert len(metrics["confusion_matrix"]) == 2

    # Check summary table generation
    df = generate_metrics_summary_table([metrics])
    assert len(df) == 1
    assert "Model" in df.columns
    assert "Accuracy" in df.columns


def test_evaluation_visualizer(tmp_path):
    """Test visualization functions generate PNG files without error."""
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    y_prob_lr = np.array([0.2, 0.8, 0.1, 0.4, 0.2, 0.3, 0.7, 0.6, 0.9, 0.1])
    y_prob_rf = np.array([0.1, 0.9, 0.2, 0.3, 0.1, 0.2, 0.8, 0.5, 0.95, 0.05])

    eval_dict = {
        "LR": compute_evaluation_metrics(y_true, (y_prob_lr >= 0.5).astype(int), y_prob_lr, "LR"),
        "RF": compute_evaluation_metrics(y_true, (y_prob_rf >= 0.5).astype(int), y_prob_rf, "RF"),
    }
    y_trues = {"LR": y_true, "RF": y_true}
    y_probs = {"LR": y_prob_lr, "RF": y_prob_rf}

    p1 = plot_roc_curves(eval_dict, y_trues, y_probs, tmp_path / "roc.png")
    p2 = plot_precision_recall_curves(eval_dict, y_trues, y_probs, tmp_path / "pr.png")
    p3 = plot_confusion_matrices(eval_dict, tmp_path / "cm.png")
    p4 = plot_metrics_comparison(eval_dict, tmp_path / "comp.png")

    assert p1.exists()
    assert p2.exists()
    assert p3.exists()
    assert p4.exists()
