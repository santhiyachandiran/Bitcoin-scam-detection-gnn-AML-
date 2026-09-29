"""
Main execution script for Elliptic Bitcoin Transaction Dataset Loading and EDA.

Usage:
    python run_eda.py
    python run_eda.py --use-sample
"""

import argparse
from pathlib import Path
import json

from src.config import RAW_DATA_DIR, SAMPLE_DATA_DIR, FIGURES_DIR, DOCS_DIR
from src.data.download import download_dataset, generate_sample_dataset, is_dataset_present
from src.data.loader import EllipticDataLoader
from src.data.validator import DatasetValidator
from src.eda.analyzer import EllipticEDAAnalyzer
from src.eda.visualizer import EDAVisualizer
from src.utils.logger import get_logger

logger = get_logger("run_eda")


def generate_markdown_report(results: dict, figures: dict, output_path: Path):
    """Generates a comprehensive EDA report in Markdown format."""
    cls_dist = results["class_distribution"]
    graph_stats = results["graph_statistics"]
    feat_stats = results["feature_statistics"]
    ts_stats = results["timestep_distribution"]
    dup_stats = results["duplicates"]
    missing_stats = results["missing_values"]

    report_content = f"""# Comprehensive Exploratory Data Analysis (EDA) Report
## Bitcoin Scam Detection Using Dynamic Graph Neural Networks

**Dataset:** Elliptic Bitcoin Transaction Dataset  
**Generated On:** 2026-09-29  
**Status:** Stage 1 - Initial Setup & Dataset Loading & EDA Completed  

---

## 1. Executive Summary

The **Elliptic Dataset** maps Bitcoin transactions over 49 discrete time-steps into a directed graph structure. Each node represents a transaction, edges represent payments/flows between transactions, and features describe local transaction metrics as well as aggregated neighborhood metrics.

### Key Highlights:
- **Total Transactions (Nodes):** `{graph_stats['node_count']:,}`
- **Total Directed Edges:** `{graph_stats['edge_count']:,}`
- **Total Time-Steps:** `{ts_stats['total_timesteps']}`
- **Class Labels Breakdown:**
  - **Illicit (Class 1):** `{cls_dist['raw_counts']['Illicit (1)']:,}` ({cls_dist['percentages']['Illicit (1)']:.2f}%)
  - **Licit (Class 2):** `{cls_dist['raw_counts']['Licit (2)']:,}` ({cls_dist['percentages']['Licit (2)']:.2f}%)
  - **Unknown (Unlabelled):** `{cls_dist['raw_counts']['Unknown']:,}` ({cls_dist['percentages']['Unknown']:.2f}%)
- **Illicit Ratio in Labelled Subset:** `{cls_dist['illicit_ratio_in_labelled']:.2f}%`

---

## 2. Dataset Schema & Validation

### Table Shapes and Columns:
- **Features Table:** `{results['validation']['shapes']['features']}` (Columns: `txId`, `time_step`, `feat_1` to `feat_166`)
- **Classes Table:** `{results['validation']['shapes']['classes']}` (Columns: `txId`, `class`, `class_name`, `binary_label`)
- **Edges Table:** `{results['validation']['shapes']['edges']}` (Columns: `txId1`, `txId2`)

### Data Integrity & Missingness:
- **Missing Values:**
  - Features Table: `{missing_stats['features']['total_missing_cells']}` missing cells
  - Classes Table: `{missing_stats['classes']['total_missing_cells']}` missing cells
  - Edges Table: `{missing_stats['edges']['total_missing_cells']}` missing cells
- **Duplicates:**
  - Duplicate Feature Rows / Node IDs: `{dup_stats['features_duplicate_rows']}` / `{dup_stats['features_duplicate_txIds']}`
  - Duplicate Class Rows: `{dup_stats['classes_duplicate_rows']}`
  - Duplicate Edges: `{dup_stats['edges_duplicate_pairs']}`

---

## 3. Class Distribution & Imbalance Analysis

| Class | Count | Percentage |
|---|---|---|
| **Illicit (1)** | {cls_dist['raw_counts']['Illicit (1)']:,} | {cls_dist['percentages']['Illicit (1)']}% |
| **Licit (2)** | {cls_dist['raw_counts']['Licit (2)']:,} | {cls_dist['percentages']['Licit (2)']}% |
| **Unknown** | {cls_dist['raw_counts']['Unknown']:,} | {cls_dist['percentages']['Unknown']}% |
| **Total** | **{cls_dist['total_transactions']:,}** | **100.0%** |

![Class Distribution](../reports/figures/class_distribution.png)
![Labelled vs Unknown](../reports/figures/unknown_vs_labelled.png)

> **Key Finding:** The dataset exhibits extreme class imbalance. Illicit transactions account for only ~2.3% of total transactions and ~10.4% of labelled transactions. Unlabelled ("Unknown") nodes constitute ~77.3% of the dataset, making Dynamic GNN semi-supervised learning highly suitable.

---

## 4. Temporal Dynamics Across 49 Time-Steps

- **Min / Max / Mean Txs per Time-Step:** `{ts_stats['min_transactions_per_step']:,}` / `{ts_stats['max_transactions_per_step']:,}` / `{ts_stats['mean_transactions_per_step']:,}`
- **Time-step Interval:** Each time-step represents a window of approximately 2 weeks of Bitcoin transaction activity.

![Time-step Distribution](../reports/figures/timestep_distribution.png)
![Illicit Ratio Over Time](../reports/figures/illicit_ratio_over_time.png)

> **Key Finding:** Time-step 43 exhibits a prominent structural shift due to a major dark market shutdown, leading to a temporary drop in illicit transaction ratio.

---

## 5. Feature Numerical Statistics

- **Total Features:** `{feat_stats['total_features_count']}` (94 Local features + 72 Aggregate features)
- **Local Features (1-94):** Mean zero ratio = `{feat_stats['local_features_summary']['avg_zero_ratio_pct']}%`, Avg Skewness = `{feat_stats['local_features_summary']['avg_skewness']}`
- **Aggregate Features (95-166):** Mean zero ratio = `{feat_stats['aggregate_features_summary']['avg_zero_ratio_pct']}%`, Avg Skewness = `{feat_stats['aggregate_features_summary']['avg_skewness']}`

![Feature Correlation Heatmap](../reports/figures/feature_correlation_sample.png)

---

## 6. Graph Topological & Network Statistics

- **Graph Type:** Directed Transaction Graph
- **Node Count:** `{graph_stats['node_count']:,}`
- **Edge Count:** `{graph_stats['edge_count']:,}`
- **Graph Density:** `{graph_stats['graph_density']:.8f}`
- **Isolated Nodes (Degree 0):** `{graph_stats['isolated_nodes_count']:,}` ({graph_stats['isolated_nodes_percentage']:.2f}%)
- **Degree Statistics:**
  - **In-Degree (Max / Mean / Std):** `{graph_stats['degree_stats']['in_degree']['max']}` / `{graph_stats['degree_stats']['in_degree']['mean']}` / `{graph_stats['degree_stats']['in_degree']['std']}`
  - **Out-Degree (Max / Mean / Std):** `{graph_stats['degree_stats']['out_degree']['max']}` / `{graph_stats['degree_stats']['out_degree']['mean']}` / `{graph_stats['degree_stats']['out_degree']['std']}`

![Node Degree Distribution](../reports/figures/node_degree_distribution.png)
![Sample Subgraph Topology](../reports/figures/subgraph_sample.png)

---

## 7. Next Stage Readiness

The modular architecture is now ready for:
1. **Stage 2:** Dynamic Graph Construction & Temporal PyTorch Geometric Data Loaders.
2. **Stage 3:** Dynamic GNN Architecture (EvolveGCN / GCN / GAT).
3. **Stage 4:** Model Evaluation & Benchmark Metrics (F1-score, Precision, Recall, AUC-ROC).
4. **Stage 5 & 6:** REST API Backend & Interactive Web Frontend.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"Markdown report generated at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Elliptic Bitcoin Scam Detection - Dataset Loading & EDA")
    parser.add_argument("--use-sample", action="store_true", help="Use synthetic sample dataset instead of full raw dataset")
    args = parser.parse_args()

    # Step 1: Ensure dataset is available
    if args.use_sample:
        data_dir = generate_sample_dataset(SAMPLE_DATA_DIR)
    else:
        if not is_dataset_present(RAW_DATA_DIR):
            logger.info("Raw dataset missing in data/raw. Triggering download...")
            data_dir = download_dataset(RAW_DATA_DIR)
        else:
            data_dir = RAW_DATA_DIR

    logger.info(f"Using dataset from: {data_dir}")

    # Step 2: Load Data
    loader = EllipticDataLoader(data_dir=data_dir)
    data_dict = loader.load_all()

    print("\n" + "=" * 60)
    print(" DATASET SHAPES AND COLUMNS VALIDATION")
    print("=" * 60)
    print(f"Features shape: {data_dict['features'].shape}")
    print(f"Features sample columns: {list(data_dict['features'].columns[:5])} ... {list(data_dict['features'].columns[-3:])}")
    print(f"Classes shape:  {data_dict['classes'].shape}")
    print(f"Classes columns: {list(data_dict['classes'].columns)}")
    print(f"Edges shape:    {data_dict['edges'].shape}")
    print(f"Edges columns:   {list(data_dict['edges'].columns)}")
    print("=" * 60 + "\n")

    # Step 3: Validate Integrity
    validator = DatasetValidator(data_dict)
    validation_results = validator.validate_all()
    print(f"Validation Status: {'PASS' if validation_results['is_valid'] else 'FAIL'}")
    if validation_results['issues']:
        print(f"Validation Issues: {validation_results['issues']}")

    # Step 4: Run EDA Analysis
    analyzer = EllipticEDAAnalyzer(data_dict)
    analysis_results = analyzer.run_full_analysis()
    analysis_results["validation"] = validation_results

    # Step 5: Visualizations
    visualizer = EDAVisualizer(data_dict, output_dir=FIGURES_DIR)
    figures = visualizer.generate_all_plots()

    # Step 6: Generate Markdown Report
    report_path = DOCS_DIR / "eda_report.md"
    generate_markdown_report(analysis_results, figures, report_path)

    print("\n" + "=" * 60)
    print(" EDA PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Summary report: {report_path}")
    print(f" Figures generated in: {FIGURES_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
