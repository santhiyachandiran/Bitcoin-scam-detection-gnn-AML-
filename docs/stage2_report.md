# Stage 2 Report: Data Preprocessing & Dynamic Graph Construction
## Bitcoin Scam Detection Using Dynamic Graph Neural Networks

**Dataset:** Elliptic Bitcoin Transaction Dataset  
**Status:** Stage 2 Completed  
**Generated Artifacts Location:** `data/processed/`

---

## 1. Executive Summary

In Stage 2, we built an end-to-end data preprocessing and dynamic graph construction pipeline that transforms raw tabular Elliptic transaction features, node classes, and transaction flow edge lists into PyTorch Geometric (`torch_geometric.data.Data`) graph structures.

Key highlights of the Stage 2 implementation:
- **Code Reuse:** Directly reused Stage 1 `EllipticDataLoader` and `DatasetValidator` without code duplication.
- **Leak-Free Normalization:** Standardized node features (`StandardScaler`) by fitting mean and variance **exclusively on the training time-steps (1-34)** and transforming validation and testing timesteps.
- **Graph Topology:** Mapped all 203,769 transactions to continuous 0..N-1 node indices and constructed 234,355 directed payment edges (`edge_index`).
- **Temporal Graph Snapshots:** Built a sequence of 49 discrete temporal graph snapshots (`List[Data]`), enabling dynamic GNN architectures such as EvolveGCN.
- **Strict Masking:** Created mutually exclusive `train_mask`, `val_mask`, `test_mask`, and `labelled_mask` tensors guaranteeing zero temporal data leakage.
- **PyG Integration:** Verified that built PyG graph objects pass schema validation (`data.validate()`).

---

## 2. Graph Topology Statistics

| Metric | Value |
|---|---|
| **Total Transaction Nodes ($N$)** | 203,769 |
| **Total Directed Edges ($E$)** | 234,355 |
| **Feature Dimensions ($D$)** | 166 (94 Local + 72 Aggregate) |
| **Graph Density** | $5.64 \times 10^{-6}$ |
| **Isolated Nodes (Degree 0)** | 0 (0.00%) |
| **Max In-Degree / Out-Degree** | 284 / 472 |
| **Mean In-Degree / Out-Degree** | 1.15 / 1.15 |

---

## 3. Temporal Train / Validation / Test Splits

The dataset spans **49 discrete time-steps** (each representing ~2 weeks). To evaluate GNN model generalization across time without temporal lookahead, nodes were split chronologically:

| Split | Time-Steps | Total Nodes | Labelled Nodes | Illicit (1) | Licit (0) | Unknown (-1) |
|---|---|---|---|---|---|---|
| **Train** | Steps 1 – 34 | 136,265 | 29,894 | 3,462 | 26,432 | 106,371 |
| **Validation** | Steps 35 – 39 | 20,857 | 5,486 | 447 | 5,039 | 15,371 |
| **Test** | Steps 40 – 49 | 46,647 | 11,184 | 636 | 10,548 | 35,463 |
| **Total / Unlabelled** | Steps 1 – 49 | **203,769** | **46,564** | **4,545** | **42,019** | **157,205** |

---

## 4. PyTorch Geometric Data Schema

The saved `elliptic_pyg_data.pt` PyG object contains:

```python
Data(
    x=[203769, 166],           # Normalized float32 node feature matrix
    edge_index=[2, 234355],    # LongTensor directed graph edge pairs
    y=[203769],                # Class labels (1=Illicit, 0=Licit, -1=Unknown)
    time_step=[203769],        # Node time-step values (1..49)
    tx_id=[203769],            # Original raw Bitcoin transaction IDs
    train_mask=[203769],       # Boolean training mask
    val_mask=[203769],         # Boolean validation mask
    test_mask=[203769],        # Boolean test mask
    labelled_mask=[203769],    # Boolean mask for labelled nodes
    num_nodes=203769           # Total node count
)
```

---

## 5. Saved Artifacts in `data/processed/`

1. `elliptic_pyg_data.pt` - Full unified PyG `Data` graph object.
2. `elliptic_temporal_snapshots.pt` - List of 49 discrete PyG `Data` graph snapshots.
3. `scaler.pt` - Fitted `StandardScaler` mean and scale parameters.
4. `preprocessing_metadata.json` - Complete metadata, split breakdown, and graph metrics.

---

## 6. How to Run Stage 2

Execute the complete Stage 2 pipeline on raw data:
```bash
python run_stage2.py
```

Execute on synthetic sample data for fast offline testing:
```bash
python run_stage2.py --use-sample
```

Run test suite:
```bash
python -m pytest tests/
```
