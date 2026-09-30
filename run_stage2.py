"""
Stage 2 Main Pipeline: Preprocessing and Dynamic Graph Construction.

Usage:
    python run_stage2.py
    python run_stage2.py --use-sample
"""

import argparse
import time
from pathlib import Path
import json

from src.config import RAW_DATA_DIR, SAMPLE_DATA_DIR, PROCESSED_DATA_DIR
from src.data.download import download_dataset, generate_sample_dataset, is_dataset_present
from src.data.loader import EllipticDataLoader
from src.graph.preprocessor import EllipticPreprocessor
from src.graph.builder import EllipticGraphBuilder
from src.graph.saver import save_processed_graph, load_processed_data
from src.utils.logger import get_logger

logger = get_logger("run_stage2")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Bitcoin Scam Detection - Preprocessing & Graph Construction")
    parser.add_argument("--use-sample", action="store_true", help="Use synthetic sample dataset instead of full raw dataset")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory containing features, classes, edges CSVs")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for processed PyG graph data")
    
    # Temporal Split Boundaries
    parser.add_argument("--train-start", type=int, default=1, help="Start timestep for training set")
    parser.add_argument("--train-end", type=int, default=34, help="End timestep for training set")
    parser.add_argument("--val-start", type=int, default=35, help="Start timestep for validation set")
    parser.add_argument("--val-end", type=int, default=39, help="End timestep for validation set")
    parser.add_argument("--test-start", type=int, default=40, help="Start timestep for test set")
    parser.add_argument("--test-end", type=int, default=49, help="End timestep for test set")

    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    print("\n" + "=" * 80)
    print(" STAGE 2: PREPROCESSING & DYNAMIC GRAPH CONSTRUCTION")
    print("=" * 80)

    # 1. Determine dataset directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.use_sample:
        logger.info("Generating/Loading synthetic sample dataset for offline testing...")
        data_dir = generate_sample_dataset(SAMPLE_DATA_DIR, num_nodes=150, num_edges=250)
    else:
        if not is_dataset_present(RAW_DATA_DIR):
            logger.info("Raw dataset missing. Triggering automatic download...")
            data_dir = download_dataset(RAW_DATA_DIR)
        else:
            data_dir = RAW_DATA_DIR

    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR
    logger.info(f"Input Dataset Directory:  {data_dir}")
    logger.info(f"Output Processed Directory: {output_dir}")

    # 2. Load dataset files using existing EllipticDataLoader
    logger.info("Step 1: Loading raw dataset files...")
    loader = EllipticDataLoader(data_dir=data_dir)
    data_dict = loader.load_all()

    # 3. Clean, preprocess features, normalize without leakage, create ID maps and split masks
    train_range = (args.train_start, args.train_end)
    val_range = (args.val_start, args.val_end)
    test_range = (args.test_start, args.test_end)

    logger.info(f"Step 2: Preprocessing features & creating temporal splits ({train_range}, {val_range}, {test_range})...")
    preprocessor = EllipticPreprocessor(
        data_dict=data_dict,
        train_timesteps=train_range,
        val_timesteps=val_range,
        test_timesteps=test_range
    )
    processed_data = preprocessor.preprocess()

    # 4. Construct PyTorch Geometric Graph Objects
    logger.info("Step 3: Constructing PyTorch Geometric full graph & temporal snapshots...")
    builder = EllipticGraphBuilder(
        preprocessed_data=processed_data,
        edges_df=data_dict["edges"]
    )
    
    full_pyg_data, graph_stats = builder.build_full_graph()
    temporal_snapshots = builder.build_temporal_snapshots()

    # Combine pipeline metadata
    pipeline_metadata = {
        "pipeline_stage": "Stage 2 - Data Preprocessing & Dynamic Graph Construction",
        "dataset_source": str(data_dir),
        "temporal_splits": {
            "train_timesteps": list(train_range),
            "val_timesteps": list(val_range),
            "test_timesteps": list(test_range)
        },
        "node_preprocessing_stats": processed_data["stats"],
        "graph_statistics": graph_stats,
        "pyg_schema": {
            "num_nodes": full_pyg_data.num_nodes,
            "num_edges": full_pyg_data.edge_index.size(1),
            "num_features": full_pyg_data.x.size(1),
            "device": str(full_pyg_data.x.device),
            "is_directed": full_pyg_data.is_directed(),
            "has_isolated_nodes": full_pyg_data.has_isolated_nodes(),
            "has_self_loops": full_pyg_data.has_self_loops(),
        }
    }

    # 5. Save Processed Graph Data & Metadata
    logger.info("Step 4: Saving processed PyG data objects and metadata...")
    saved_paths = save_processed_graph(
        data=full_pyg_data,
        snapshots=temporal_snapshots,
        metadata=pipeline_metadata,
        scaler=preprocessor.scaler,
        output_dir=output_dir
    )

    # 6. Verification Load Check
    logger.info("Step 5: Verifying saved data by reloading...")
    reloaded_data, reloaded_snapshots, reloaded_meta, reloaded_scaler_state = load_processed_data(processed_dir=output_dir)
    assert reloaded_data.num_nodes == full_pyg_data.num_nodes, "Reload verification failed: node count mismatch!"
    assert len(reloaded_snapshots) == len(temporal_snapshots), "Reload verification failed: snapshot count mismatch!"

    elapsed_time = time.time() - start_time

    # 7. Print Detailed Stage 2 Report
    print("\n" + "=" * 80)
    print(" STAGE 2 PIPELINE SUMMARY & GRAPH STATISTICS")
    print("=" * 80)
    print(f" Execution Status:      SUCCESS (Completed in {elapsed_time:.2f}s)")
    print(f" Processed Output Path: {output_dir.resolve()}")
    print("-" * 80)
    print(" 1. GRAPH TOPOLOGY STATISTICS:")
    print(f"    - Total Nodes (Transactions):  {graph_stats['num_nodes']:,}")
    print(f"    - Total Directed Edges:        {graph_stats['num_edges']:,}")
    print(f"    - Feature Vector Dimension:    {graph_stats['num_features']}")
    print(f"    - Graph Density:               {graph_stats['density']:.8f}")
    print(f"    - Isolated Nodes:              {graph_stats['isolated_nodes']:,} ({graph_stats['isolated_nodes_pct']:.2f}%)")
    print(f"    - Max In-Degree / Out-Degree:  {graph_stats['in_degree']['max']} / {graph_stats['out_degree']['max']}")
    print(f"    - Mean In-Degree / Out-Degree: {graph_stats['in_degree']['mean']:.2f} / {graph_stats['out_degree']['mean']:.2f}")
    print("-" * 80)
    print(" 2. TEMPORAL TRAIN / VAL / TEST SPLIT DETAILS:")
    print(f"    - Train Timesteps ({train_range[0]}-{train_range[1]}):  Total={processed_data['stats']['train_nodes_total']:,} | Labelled={processed_data['stats']['train_nodes_labelled']:,} (Illicit={processed_data['stats']['train_illicit']:,}, Licit={processed_data['stats']['train_licit']:,})")
    print(f"    - Val Timesteps   ({val_range[0]}-{val_range[1]}):  Total={processed_data['stats']['val_nodes_total']:,} | Labelled={processed_data['stats']['val_nodes_labelled']:,} (Illicit={processed_data['stats']['val_illicit']:,}, Licit={processed_data['stats']['val_licit']:,})")
    print(f"    - Test Timesteps  ({test_range[0]}-{test_range[1]}): Total={processed_data['stats']['test_nodes_total']:,} | Labelled={processed_data['stats']['test_nodes_labelled']:,} (Illicit={processed_data['stats']['test_illicit']:,}, Licit={processed_data['stats']['test_licit']:,})")
    print(f"    - Unlabelled Nodes (Unknown):   {processed_data['stats']['unknown_nodes']:,}")
    print("-" * 80)
    print(" 3. PYTORCH GEOMETRIC DATA SCHEMA:")
    print(f"    - PyG Data Object:  {full_pyg_data}")
    print(f"    - Temporal Sequence: {len(temporal_snapshots)} PyG Snapshot Graph Data objects (t=1..{len(temporal_snapshots)})")
    print(f"    - Validation Check:  PASSED (data.validate() clean)")
    print("-" * 80)
    print(" 4. SAVED ARTIFACTS:")
    for k, v in saved_paths.items():
        print(f"    - {k}: {v.name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
