"""
Graph Builder Module for Elliptic Bitcoin Transaction Dataset.
Constructs PyTorch Geometric Data objects for full static/temporal dynamic graph,
performs edge index mapping, calculates graph topology statistics, and validates PyG objects.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.config import TOTAL_TIMESTEPS
from src.utils.logger import get_logger

logger = get_logger("graph_builder")


class EllipticGraphBuilder:
    """
    Constructs PyTorch Geometric graph objects from preprocessed node data and transaction edges.
    """

    def __init__(self, preprocessed_data: Dict[str, Any], edges_df: pd.DataFrame):
        """
        Initialize the graph builder.
        
        Args:
            preprocessed_data: Output dictionary from EllipticPreprocessor.preprocess().
            edges_df: DataFrame containing raw transaction edges (columns: txId1, txId2).
        """
        self.preprocessed_data = preprocessed_data
        self.edges_df = edges_df
        self.txId_to_idx = preprocessed_data["txId_to_idx"]

    def build_edge_index(self) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Map raw txId1 -> txId2 edges to continuous 0..N-1 node indices.
        
        Returns:
            Tuple[np.ndarray, Dict[str, int]]: 
                - edge_index array of shape (2, num_valid_edges)
                - edge mapping statistics
        """
        logger.info("Mapping raw transaction edges to continuous node indices...")
        
        tx1 = self.edges_df["txId1"].values
        tx2 = self.edges_df["txId2"].values
        
        # Filter valid edge endpoints present in node mapping
        valid_mask = np.vectorize(lambda x1, x2: (x1 in self.txId_to_idx) and (x2 in self.txId_to_idx))(tx1, tx2)
        
        filtered_tx1 = tx1[valid_mask]
        filtered_tx2 = tx2[valid_mask]

        src_indices = np.array([self.txId_to_idx[tx] for tx in filtered_tx1], dtype=np.int64)
        dst_indices = np.array([self.txId_to_idx[tx] for tx in filtered_tx2], dtype=np.int64)

        edge_index_np = np.stack([src_indices, dst_indices], axis=0) # shape: (2, E)
        
        edge_stats = {
            "total_raw_edges": len(self.edges_df),
            "valid_mapped_edges": edge_index_np.shape[1],
            "dropped_edges": len(self.edges_df) - edge_index_np.shape[1]
        }
        
        logger.info(f"Mapped {edge_stats['valid_mapped_edges']} valid edges (dropped {edge_stats['dropped_edges']} invalid).")
        return edge_index_np, edge_stats

    def build_full_graph(self) -> Tuple[Data, Dict[str, Any]]:
        """
        Construct the full PyTorch Geometric Data object containing all nodes, features, and edges.
        
        Returns:
            Tuple[Data, Dict[str, Any]]: Full PyG Data object and detailed graph statistics.
        """
        logger.info("Building full PyTorch Geometric graph data object...")
        
        edge_index_np, edge_stats = self.build_edge_index()
        
        # Extract numpy arrays from preprocessed dictionary
        x = torch.tensor(self.preprocessed_data["x"], dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        y = torch.tensor(self.preprocessed_data["y"], dtype=torch.long)
        time_step = torch.tensor(self.preprocessed_data["time_step"], dtype=torch.long)
        tx_id = torch.tensor(self.preprocessed_data["tx_id"], dtype=torch.long)
        
        train_mask = torch.tensor(self.preprocessed_data["train_mask"], dtype=torch.bool)
        val_mask = torch.tensor(self.preprocessed_data["val_mask"], dtype=torch.bool)
        test_mask = torch.tensor(self.preprocessed_data["test_mask"], dtype=torch.bool)
        labelled_mask = torch.tensor(self.preprocessed_data["labelled_mask"], dtype=torch.bool)

        num_nodes = x.size(0)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            time_step=time_step,
            tx_id=tx_id,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            labelled_mask=labelled_mask,
            num_nodes=num_nodes
        )

        # PyG Data Validation Check
        data.validate(raise_on_error=True)
        
        # Calculate graph topological statistics
        graph_stats = self.compute_graph_statistics(data, edge_stats)
        logger.info(f"Full graph constructed successfully: {data}")
        
        return data, graph_stats

    def build_temporal_snapshots(self, total_timesteps: int = TOTAL_TIMESTEPS) -> List[Data]:
        """
        Construct a list of PyTorch Geometric Data objects, one per time-step (t=1..49).
        Nodes and edges are scoped to each specific time-step snapshot.
        
        Args:
            total_timesteps: Number of timesteps to construct (default: TOTAL_TIMESTEPS = 49).
            
        Returns:
            List[Data]: Sequence of 49 PyG Data snapshot objects.
        """
        logger.info(f"Building {total_timesteps} temporal graph snapshots (one PyG Data per time-step)...")
        
        full_data, _ = self.build_full_graph()
        timesteps = self.preprocessed_data["time_step"]
        num_features = full_data.x.size(1)

        # Full edge index in global indices
        full_edges = full_data.edge_index.numpy()
        src_global = full_edges[0]
        dst_global = full_edges[1]

        snapshots = []

        for ts in range(1, total_timesteps + 1):
            # Global node indices for timestep ts
            ts_global_mask = (timesteps == ts)
            ts_global_indices = np.where(ts_global_mask)[0]
            
            if len(ts_global_indices) == 0:
                # Handle timestep with 0 nodes gracefully
                snapshot_t = Data(
                    x=torch.empty((0, num_features), dtype=torch.float32),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    y=torch.empty((0,), dtype=torch.long),
                    tx_id=torch.empty((0,), dtype=torch.long),
                    time_step=torch.empty((0,), dtype=torch.long),
                    train_mask=torch.empty((0,), dtype=torch.bool),
                    val_mask=torch.empty((0,), dtype=torch.bool),
                    test_mask=torch.empty((0,), dtype=torch.bool),
                    labelled_mask=torch.empty((0,), dtype=torch.bool),
                    global_node_idx=torch.empty((0,), dtype=torch.long),
                    num_nodes=0
                )
                snapshots.append(snapshot_t)
                continue

            # Map global node index -> local snapshot index 0..N_t-1
            global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(ts_global_indices)}
            
            # Extract features, labels, and masks for timestep ts
            x_t = full_data.x[ts_global_indices]
            y_t = full_data.y[ts_global_indices]
            tx_id_t = full_data.tx_id[ts_global_indices]
            train_mask_t = full_data.train_mask[ts_global_indices]
            val_mask_t = full_data.val_mask[ts_global_indices]
            test_mask_t = full_data.test_mask[ts_global_indices]
            labelled_mask_t = full_data.labelled_mask[ts_global_indices]

            # Intra-timestep edges (both endpoints belong to timestep ts)
            edge_ts_mask = np.vectorize(lambda s, d: (s in global_to_local) and (d in global_to_local))(src_global, dst_global)
            
            if np.any(edge_ts_mask):
                src_local = np.array([global_to_local[s] for s in src_global[edge_ts_mask]], dtype=np.int64)
                dst_local = np.array([global_to_local[d] for d in dst_global[edge_ts_mask]], dtype=np.int64)
                edge_index_t = torch.tensor(np.stack([src_local, dst_local], axis=0), dtype=torch.long)
            else:
                edge_index_t = torch.empty((2, 0), dtype=torch.long)

            snapshot_t = Data(
                x=x_t,
                edge_index=edge_index_t,
                y=y_t,
                tx_id=tx_id_t,
                time_step=torch.full((len(ts_global_indices),), ts, dtype=torch.long),
                train_mask=train_mask_t,
                val_mask=val_mask_t,
                test_mask=test_mask_t,
                labelled_mask=labelled_mask_t,
                global_node_idx=torch.tensor(ts_global_indices, dtype=torch.long),
                num_nodes=len(ts_global_indices)
            )
            snapshot_t.validate(raise_on_error=True)
            snapshots.append(snapshot_t)


        logger.info(f"Built {len(snapshots)} temporal snapshot PyG graph objects.")
        return snapshots

    def compute_graph_statistics(self, data: Data, edge_stats: Dict[str, int]) -> Dict[str, Any]:
        """
        Compute graph statistics including degrees, isolated nodes, density, and class distributions.
        
        Args:
            data: Unified PyG Data object.
            edge_stats: Basic edge mapping statistics.
            
        Returns:
            Dict[str, Any]: Detailed dictionary of graph metrics.
        """
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)
        
        # Degree metrics
        src, dst = data.edge_index[0], data.edge_index[1]
        out_degrees = torch.bincount(src, minlength=num_nodes).float()
        in_degrees = torch.bincount(dst, minlength=num_nodes).float()
        total_degrees = in_degrees + out_degrees

        isolated_nodes = int((total_degrees == 0).sum().item())
        density = float(num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0

        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_features": data.x.size(1),
            "density": density,
            "isolated_nodes": isolated_nodes,
            "isolated_nodes_pct": float((isolated_nodes / num_nodes) * 100),
            "in_degree": {
                "max": int(in_degrees.max().item()),
                "mean": float(in_degrees.mean().item()),
                "std": float(in_degrees.std().item())
            },
            "out_degree": {
                "max": int(out_degrees.max().item()),
                "mean": float(out_degrees.mean().item()),
                "std": float(out_degrees.std().item())
            },
            "edge_mapping_stats": edge_stats,
            "splits": {
                "train_labelled": int(data.train_mask.sum().item()),
                "val_labelled": int(data.val_mask.sum().item()),
                "test_labelled": int(data.test_mask.sum().item()),
                "total_labelled": int(data.labelled_mask.sum().item()),
                "unlabelled": int((data.y == -1).sum().item())
            }
        }
        return stats
