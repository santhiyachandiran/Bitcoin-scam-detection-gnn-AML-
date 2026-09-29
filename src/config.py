"""
Global Configuration and Constants for Bitcoin Scam Detection Project.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

# Raw File Names
FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"

# Dataset Constants
CLASS_MAP = {
    "1": "Illicit",
    "2": "Licit",
    "unknown": "Unknown"
}

NUM_LOCAL_FEATURES = 94
NUM_AGG_FEATURES = 72
TOTAL_FEATURES = NUM_LOCAL_FEATURES + NUM_AGG_FEATURES # 166
TOTAL_TIMESTEPS = 49

# Kaggle Dataset Slug
KAGGLE_DATASET_SLUG = "ellipticco/elliptic-data-set"

def ensure_directories_exist():
    """Ensure that essential directories exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        DOCS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Auto-ensure on import
ensure_directories_exist()
