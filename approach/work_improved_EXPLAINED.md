# Overview and Loss Function: gnn_attention_density_prediction_work_improved.ipynb

## Overview
This notebook trains a Graph Attention Network (GAT) to predict the next electron-density state from recent density history on a fixed 3D grid.

Main idea:
- Use the last 5 timesteps as input per node.
- Predict 1-step-ahead density change (delta), not absolute density.
- Work in transformed space (`log1p`) for numerical stability.

Pipeline summary:
1. Load density time-series and grid coordinates.
2. Transform density with `log1p`.
3. Build train/val/test splits in chronological order.
4. Construct graph edges (spatial kNN plus feature-similarity edges).
5. Compute per-node weights from temporal variability.
6. Train GAT with a physics-aware composite loss.
7. Convert predictions back to original space and evaluate.

Why delta prediction is used:
- Predicting change is easier than predicting full absolute density.
- It stabilizes training and helps capture local dynamics.

## Graph Construction
The graph is built by combining two complementary edge sets so the model can learn both physical locality and behavioral similarity.

### 1) Spatial kNN edges
- Build k-nearest-neighbor edges from 3D grid coordinates (`rvlab.tdscf.xyz`).
- Each node is connected to its nearest spatial neighbors (`K_NEIGHBORS = 10`).
- Edges are added bidirectionally.

Why:
- Preserves local physical structure in space.

### 2) Feature-similarity edges
- Build per-node temporal signatures from training timesteps only.
- Normalize each node signature.
- Use cosine-distance kNN (`FEAT_NEIGHBORS = 8`) to connect behaviorally similar nodes.
- Edges are added bidirectionally.

Why:
- Connects nodes that evolve similarly over time, even if far apart in index or geometry.

### 3) Final graph
- Concatenate spatial edges and feature edges.
- Remove duplicates to get final `graph_edges`.

Why this hybrid graph helps:
- Spatial edges improve physical consistency.
- Feature edges improve long-range relational learning.
- Together they provide a stronger inductive bias than sequential index edges.

## Loss Function (PhysicsAwareLoss)
The notebook uses a composite loss with three terms:

`L_total = L_mse + lambda_mass * L_mass + lambda_smooth * L_smooth`

Default weights:
- `lambda_mass = 0.20`
- `lambda_smooth = 0.02`

### 1) Soft-weighted MSE (`L_mse`)
Compares predicted and target standardized deltas with per-node weights:

`L_mse = sum( w_i * (pred_i - target_i)^2 ) / sum(w_i)`

Node weights are built from train-only temporal variability:
- `per_node_std = std(train_deltas, axis=0)`
- `std_norm = per_node_std / mean(per_node_std)`
- `node_weight = sqrt(std_norm + eps)`
- `node_weight = clip(node_weight, STATIC_WEIGHT_FLOOR, inf)`
- `node_weight = node_weight / mean(node_weight)`

Meaning:
- Dynamic nodes get larger weight.
- Static nodes are not ignored because of `STATIC_WEIGHT_FLOOR`.

### 2) Conservation loss (`L_mass`)
Enforces electron-count conservation after mapping predictions back to original density space:
1. Unscale predicted delta: `pred_delta = pred_z * DELTA_STD + DELTA_MEAN`
2. Reconstruct next transformed density: `pred_next_logtf = last_input + pred_delta`
3. Convert to raw density: `pred_next_raw = expm1(pred_next_logtf)`
4. Sum electrons per sample and compare with target total.

Relative squared error form:

`L_mass = mean( ((pred_total - target_total) / (abs(target_total) + eps))^2 )`

Meaning:
- Penalizes physically inconsistent predictions that violate total electron conservation.

### 3) Graph smoothness loss (`L_smooth`)
Encourages neighboring nodes connected by graph edges to have similar predicted delta:

`L_smooth = mean( (pred_src - pred_dst)^2 )` over all edges `(src, dst)`

Meaning:
- Reduces noisy spikes and improves spatial coherence.

## Practical interpretation
- `L_mse` drives fit to ground-truth dynamics.
- `L_mass` injects global physical correctness.
- `L_smooth` regularizes local spatial behavior.

Together, this gives a model that is both accurate and more physically plausible.

## Results (Provided)
Absolute density, all nodes:

Metric summary:
- R2: 1.000000
- MAE: 0.000096
- RMSE: 0.000379
- NRMSE_pct: 0.000193
- MAPE_pct: 451.249002
- Cosine_Sim: 1.000000
- Within_0.1pct: 73.69%
- Within_0.5pct: 85.56%
- Within_1.0pct: 87.11%
- Within_5.0pct: 90.46%

Notes:
- Very high `R2` and `Cosine_Sim` indicate strong global shape agreement.
- Large `MAPE_pct` can still occur when true values are very close to zero (small denominator effect).
