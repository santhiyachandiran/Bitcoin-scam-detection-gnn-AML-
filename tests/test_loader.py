"""
Unit test suite for Elliptic dataset loader, downloader, and validator.
"""

import pytest
from pathlib import Path
from src.config import SAMPLE_DATA_DIR
from src.data.download import generate_sample_dataset
from src.data.loader import EllipticDataLoader
from src.data.validator import DatasetValidator
from src.eda.analyzer import EllipticEDAAnalyzer


@pytest.fixture(scope="module")
def sample_data_dir(tmp_path_factory):
    """Fixture to generate a temporary sample dataset for testing."""
    tmp_dir = tmp_path_factory.mktemp("sample_data")
    generate_sample_dataset(tmp_dir, num_nodes=100, num_edges=150)
    return tmp_dir


def test_sample_dataset_generation(sample_data_dir):
    """Test that sample dataset files are created."""
    assert (sample_data_dir / "elliptic_txs_features.csv").exists()
    assert (sample_data_dir / "elliptic_txs_classes.csv").exists()
    assert (sample_data_dir / "elliptic_txs_edgelist.csv").exists()


def test_data_loader(sample_data_dir):
    """Test data loader functionality."""
    loader = EllipticDataLoader(data_dir=sample_data_dir)
    data_dict = loader.load_all()

    assert "features" in data_dict
    assert "classes" in data_dict
    assert "edges" in data_dict
    assert "merged" in data_dict

    assert len(data_dict["features"]) == 100
    assert len(data_dict["classes"]) == 100
    assert len(data_dict["edges"]) > 0


def test_validator(sample_data_dir):
    """Test dataset validator functionality."""
    loader = EllipticDataLoader(data_dir=sample_data_dir)
    data_dict = loader.load_all()

    validator = DatasetValidator(data_dict)
    res = validator.validate_all()

    assert res["is_valid"] is True
    assert len(res["issues"]) == 0


def test_analyzer(sample_data_dir):
    """Test EDA analyzer module."""
    loader = EllipticDataLoader(data_dir=sample_data_dir)
    data_dict = loader.load_all()

    analyzer = EllipticEDAAnalyzer(data_dict)
    analysis = analyzer.run_full_analysis()

    assert "missing_values" in analysis
    assert "class_distribution" in analysis
    assert "graph_statistics" in analysis
    assert analysis["missing_values"]["features"]["total_missing_cells"] == 0
