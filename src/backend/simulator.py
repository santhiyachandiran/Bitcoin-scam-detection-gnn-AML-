"""
Dataset Simulator and Subgraph Provider for Backend API.
Provides fast lookups, random transaction generation, custom property mappings,
and graph topological subgraphs using the local Elliptic dataset.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SAMPLE_DATA_DIR
from src.data.loader import EllipticDataLoader
from src.utils.logger import get_logger

logger = get_logger("backend_simulator")

# Understandable transaction property metadata & mapping definitions
UNDERSTANDABLE_PROPERTIES_META = [
    {"key": "feat_1", "name": "Output Amount / Volume", "unit": "BTC", "min": 0.0, "max": 100.0, "step": 0.1, "default": 1.5, "desc": "Total value transferred in transaction"},
    {"key": "feat_2", "name": "Transaction Fee Ratio", "unit": "ratio", "min": 0.0, "max": 0.5, "step": 0.001, "default": 0.01, "desc": "Transaction fee as fraction of total volume"},
    {"key": "feat_3", "name": "Input Address Count", "unit": "inputs", "min": 1, "max": 50, "step": 1, "default": 2, "desc": "Number of input Bitcoin addresses funding transaction"},
    {"key": "feat_4", "name": "Output Address Count", "unit": "outputs", "min": 1, "max": 50, "step": 1, "default": 2, "desc": "Number of destination output addresses"},
    {"key": "feat_5", "name": "CoinJoin / Anonymity Mix Ratio", "unit": "score", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.0, "desc": "Measure of mixing/anonymization pattern"},
    {"key": "feat_6", "name": "Fee Variance Across Inputs", "unit": "variance", "min": 0.0, "max": 1.0, "step": 0.01, "default": 0.02, "desc": "Spread of transaction fees across input addresses"},
    {"key": "feat_7", "name": "Average Input Value", "unit": "BTC", "min": 0.0, "max": 50.0, "step": 0.1, "default": 0.8, "desc": "Mean Bitcoin value of input UTXOs"},
    {"key": "feat_8", "name": "Max Output Value Ratio", "unit": "ratio", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.75, "desc": "Fraction of total value sent to single largest output"},
    {"key": "feat_9", "name": "Parent Time Delay", "unit": "blocks", "min": 0, "max": 144, "step": 1, "default": 6, "desc": "Confirmation delay from parent transaction"},
    {"key": "feat_10", "name": "Address Reuse Count", "unit": "reuses", "min": 0, "max": 30, "step": 1, "default": 0, "desc": "Number of times participating addresses were reused"},
]


class EllipticDatasetSimulator:
    """
    In-memory dataset simulator and subgraph builder using local Elliptic dataset PyG data.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DATA_DIR, raw_dir: Path = RAW_DATA_DIR):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)

        self.pyg_data = None
        self.tx_ids: np.ndarray = np.array([], dtype=int)
        self.time_steps: np.ndarray = np.array([], dtype=int)
        self.labels: np.ndarray = np.array([], dtype=int)
        self.features: np.ndarray = np.empty((0, 166), dtype=np.float32)
        self.edge_index: Optional[np.ndarray] = None
        self.txid_to_idx: Dict[int, int] = {}
        self.is_loaded: bool = False

        self.load_data()

    def load_data(self):
        """Load processed PyG data object or fallback to dataset loader."""
        pyg_path = self.processed_dir / "elliptic_pyg_data.pt"
        if pyg_path.exists():
            try:
                logger.info(f"Simulator loading PyG dataset from {pyg_path}...")
                data = torch.load(pyg_path, weights_only=False)
                self.pyg_data = data
                self.tx_ids = data.tx_id.cpu().numpy()
                self.time_steps = data.time_step.cpu().numpy()
                self.labels = data.y.cpu().numpy()
                self.features = data.x.cpu().numpy()
                self.edge_index = data.edge_index.cpu().numpy()
                self.txid_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(self.tx_ids)}
                self.is_loaded = True
                logger.info(f"Simulator loaded {len(self.tx_ids)} transactions and {self.edge_index.shape[1]} edges.")
                return
            except Exception as e:
                logger.warning(f"Failed to load PyG data in simulator: {e}")

        # Fallback to CSV loader if PyG file is unavailable
        try:
            logger.info("Simulator falling back to raw CSV dataset loader...")
            loader = EllipticDataLoader(data_dir=self.raw_dir if self.raw_dir.exists() else SAMPLE_DATA_DIR)
            df_merged = loader.load_merged_data()
            df_edges = loader.load_edges()

            self.tx_ids = df_merged["txId"].values.astype(int)
            self.time_steps = df_merged["time_step"].values.astype(int)
            self.labels = df_merged["binary_label"].values.astype(int)
            
            feat_cols = [f"feat_{i}" for i in range(1, 167)]
            if all(c in df_merged.columns for c in feat_cols):
                self.features = df_merged[feat_cols].values.astype(np.float32)
            else:
                self.features = np.random.randn(len(df_merged), 166).astype(np.float32)

            self.txid_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(self.tx_ids)}
            
            # Edges
            valid_src = [self.txid_to_idx[tx] for tx in df_edges["txId1"] if tx in self.txid_to_idx]
            valid_dst = [self.txid_to_idx[tx] for tx in df_edges["txId2"] if tx in self.txid_to_idx]
            self.edge_index = np.stack([valid_src, valid_dst], axis=0) if valid_src else np.zeros((2, 0), dtype=int)
            self.is_loaded = True
            logger.info(f"Simulator loaded {len(self.tx_ids)} transactions via CSV loader.")
        except Exception as e:
            logger.error(f"Simulator failed fallback loading: {e}")
            self.is_loaded = False

    def get_understandable_properties(self, x_raw: np.ndarray) -> Dict[str, Any]:
        """Convert raw 166-feature array to user-understandable property dict."""
        x_flat = np.nan_to_num(x_raw.flatten(), nan=0.0)
        props = {}
        for item in UNDERSTANDABLE_PROPERTIES_META:
            idx = int(item["key"].split("_")[1]) - 1
            raw_val = float(x_flat[idx]) if idx < len(x_flat) else float(item["default"])
            # Format nicely
            props[item["key"]] = {
                "name": item["name"],
                "unit": item["unit"],
                "value": round(raw_val, 4),
                "desc": item["desc"],
                "min": item["min"],
                "max": item["max"],
                "step": item["step"],
            }
        return props

    def custom_props_to_features(
        self, custom_props: Dict[str, float], base_tx_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Build 166-feature array by combining base transaction features (or zeros)
        with user custom property overrides.
        """
        if base_tx_id is not None and base_tx_id in self.txid_to_idx:
            idx = self.txid_to_idx[base_tx_id]
            x_out = self.features[idx].copy()
        else:
            x_out = np.zeros(166, dtype=np.float32)

        for key, val in custom_props.items():
            if key.startswith("feat_"):
                try:
                    feat_idx = int(key.split("_")[1]) - 1
                    if 0 <= feat_idx < 166:
                        x_out[feat_idx] = float(val)
                except Exception:
                    pass
        return x_out

    def get_transaction_list(
        self,
        time_step: Optional[int] = None,
        filter_class: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List dataset transactions filtered by timestep, class, or search query."""
        if not self.is_loaded or len(self.tx_ids) == 0:
            return []

        mask = np.ones(len(self.tx_ids), dtype=bool)

        if time_step is not None:
            mask &= (self.time_steps == time_step)

        if filter_class == "illicit":
            mask &= (self.labels == 1)
        elif filter_class == "licit":
            mask &= (self.labels == 0)
        elif filter_class == "unknown":
            mask &= (self.labels == -1)

        indices = np.where(mask)[0]

        if search_query:
            try:
                q_int = int(search_query)
                indices = [i for i in indices if q_int == self.tx_ids[i] or str(q_int) in str(self.tx_ids[i])]
            except ValueError:
                pass

        indices = indices[:limit]

        results = []
        for idx in indices:
            tx_id = int(self.tx_ids[idx])
            ts = int(self.time_steps[idx])
            lbl = int(self.labels[idx])
            lbl_name = "Known Illicit" if lbl == 1 else ("Known Licit" if lbl == 0 else "Unlabelled")

            results.append({
                "tx_id": tx_id,
                "time_step": ts,
                "dataset_label": lbl_name,
                "binary_label": lbl,
            })
        return results

    def get_transaction_by_id(self, tx_id: int) -> Optional[Dict[str, Any]]:
        """Fetch detailed dataset record for tx_id."""
        if not self.is_loaded or tx_id not in self.txid_to_idx:
            return None

        idx = self.txid_to_idx[tx_id]
        x_raw = self.features[idx]
        ts = int(self.time_steps[idx])
        lbl = int(self.labels[idx])
        lbl_name = "Known Illicit" if lbl == 1 else ("Known Licit" if lbl == 0 else "Unlabelled")

        return {
            "tx_id": tx_id,
            "time_step": ts,
            "dataset_label": lbl_name,
            "binary_label": lbl,
            "x_raw": x_raw.tolist(),
            "understandable_properties": self.get_understandable_properties(x_raw),
        }

    def get_random_transaction(
        self, time_step: Optional[int] = None, filter_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Select a random transaction from the local dataset."""
        if not self.is_loaded or len(self.tx_ids) == 0:
            # Fallback synthetic transaction
            return {
                "tx_id": int(np.random.randint(100000, 999999)),
                "time_step": time_step if time_step is not None else int(np.random.randint(1, 50)),
                "dataset_label": "Simulated",
                "binary_label": -1,
                "x_raw": np.random.randn(166).astype(np.float32).tolist(),
                "understandable_properties": self.get_understandable_properties(np.random.randn(166)),
            }

        mask = np.ones(len(self.tx_ids), dtype=bool)

        if time_step is not None:
            mask &= (self.time_steps == time_step)

        if filter_class == "illicit":
            mask &= (self.labels == 1)
        elif filter_class == "licit":
            mask &= (self.labels == 0)

        matching_indices = np.where(mask)[0]
        if len(matching_indices) == 0:
            matching_indices = np.arange(len(self.tx_ids))

        chosen_idx = int(np.random.choice(matching_indices))
        tx_id = int(self.tx_ids[chosen_idx])
        return self.get_transaction_by_id(tx_id)

    def get_subgraph(
        self, tx_id: int, max_nodes: int = 40, inference_engine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract k-hop local subgraph around tx_id and run model inference for each node.
        Returns node topology and directed edges.
        """
        if not self.is_loaded or tx_id not in self.txid_to_idx:
            # Generate local fallback cluster if transaction not found
            return self._generate_synthetic_subgraph(tx_id, max_nodes=max_nodes, inference_engine=inference_engine)

        target_idx = self.txid_to_idx[tx_id]
        target_ts = int(self.time_steps[target_idx])

        # Find 1-hop & 2-hop connected nodes in edge_index
        src, dst = self.edge_index[0], self.edge_index[1]
        
        # 1-hop
        out_neighbors = dst[src == target_idx]
        in_neighbors = src[dst == target_idx]
        direct_neighbors = np.unique(np.concatenate([out_neighbors, in_neighbors]))

        # Combine target + 1-hop neighbors
        subgraph_nodes = np.unique(np.concatenate([[target_idx], direct_neighbors]))

        # If subgraph is small, add sibling nodes from same timestep
        if len(subgraph_nodes) < max_nodes:
            ts_indices = np.where(self.time_steps == target_ts)[0]
            extra_needed = max_nodes - len(subgraph_nodes)
            extra_nodes = np.random.choice(ts_indices, size=min(extra_needed, len(ts_indices)), replace=False)
            subgraph_nodes = np.unique(np.concatenate([subgraph_nodes, extra_nodes]))

        subgraph_nodes = subgraph_nodes[:max_nodes]
        subgraph_set = set(subgraph_nodes)

        # Build nodes list with model predictions
        nodes = []
        for idx in subgraph_nodes:
            node_tx = int(self.tx_ids[idx])
            node_ts = int(self.time_steps[idx])
            x_raw = self.features[idx]
            is_target = (node_tx == tx_id)

            if inference_engine is not None:
                pred = inference_engine.predict_single(x_raw, tx_id=node_tx, time_step=node_ts)
                prob = pred["probability"]
                risk_level = pred["risk_level"]
                risk_color = pred["risk_color"]
                is_illicit = pred["is_illicit"]
            else:
                prob = 0.5
                risk_level = "MODERATE"
                risk_color = "#ffea00"
                is_illicit = False

            nodes.append({
                "id": str(node_tx),
                "tx_id": node_tx,
                "time_step": node_ts,
                "risk_probability": prob,
                "risk_score_pct": round(prob * 100, 1),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "is_illicit": is_illicit,
                "is_target": is_target,
                "dataset_label": "Known Illicit" if self.labels[idx] == 1 else ("Known Licit" if self.labels[idx] == 0 else "Unknown"),
            })

        # Build subgraph edges
        edges = []
        edge_mask = np.array([(s in subgraph_set and d in subgraph_set) for s, d in zip(src, dst)])
        sub_src = src[edge_mask]
        sub_dst = dst[edge_mask]

        for s_idx, d_idx in zip(sub_src, sub_dst):
            s_tx = int(self.tx_ids[s_idx])
            d_tx = int(self.tx_ids[d_idx])
            edges.append({
                "source": str(s_tx),
                "target": str(d_tx),
            })

        return {
            "target_tx_id": tx_id,
            "target_time_step": target_ts,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def _generate_synthetic_subgraph(
        self, tx_id: int, max_nodes: int = 15, inference_engine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Generate a clean synthetic graph if requested tx_id is not in local index."""
        nodes = []
        edges = []
        target_ts = 1

        for i in range(max_nodes):
            curr_tx = tx_id if i == 0 else (tx_id + i * 17 % 10000 + 1)
            is_target = (i == 0)
            x_raw = np.random.randn(166).astype(np.float32)

            if inference_engine is not None:
                pred = inference_engine.predict_single(x_raw, tx_id=curr_tx, time_step=target_ts)
                prob = pred["probability"]
                risk_level = pred["risk_level"]
                risk_color = pred["risk_color"]
                is_illicit = pred["is_illicit"]
            else:
                prob = 0.85 if i in [0, 2] else 0.15
                risk_level = "CRITICAL (SCAM)" if prob > 0.8 else "LOW RISK (LICIT)"
                risk_color = "#ff1744" if prob > 0.8 else "#00e676"
                is_illicit = prob > 0.5

            nodes.append({
                "id": str(curr_tx),
                "tx_id": curr_tx,
                "time_step": target_ts,
                "risk_probability": prob,
                "risk_score_pct": round(prob * 100, 1),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "is_illicit": is_illicit,
                "is_target": is_target,
                "dataset_label": "Simulated",
            })

            if i > 0:
                parent_idx = 0 if i <= 3 else np.random.randint(0, i)
                edges.append({
                    "source": str(nodes[parent_idx]["tx_id"]),
                    "target": str(curr_tx),
                })

        return {
            "target_tx_id": tx_id,
            "target_time_step": target_ts,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def get_timestep_network(
        self, time_step: int = 1, max_nodes: int = 50, inference_engine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Get network graph for a given timestep (t=1..49)."""
        if not self.is_loaded or len(self.tx_ids) == 0:
            return self._generate_synthetic_subgraph(100001, max_nodes=max_nodes, inference_engine=inference_engine)

        ts_indices = np.where(self.time_steps == time_step)[0]
        if len(ts_indices) == 0:
            ts_indices = np.arange(min(max_nodes, len(self.tx_ids)))

        selected_indices = ts_indices[:max_nodes]
        first_tx = int(self.tx_ids[selected_indices[0]])
        return self.get_subgraph(first_tx, max_nodes=max_nodes, inference_engine=inference_engine)
