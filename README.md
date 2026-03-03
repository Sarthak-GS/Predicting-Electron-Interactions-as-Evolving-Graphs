# Electron Density Evolution - GNN Project

This project uses Graph Neural Networks (GNNs) to predict electron density evolution in molecular systems.

## Setup

1. **Activate virtual environment:**

   ```bash
   source gnn_env/bin/activate
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

## Project Structure

- `data/` - Raw datasets and archives (gitignored)
- `models/` - Saved PyTorch model weights
- `notebooks/` - Jupyter notebooks for training and evaluation
  - **`electron_density_gnn_multi_input.ipynb`** - Multi-input → Single-output prediction
  - **`electron_density_gnn_multi_future.ipynb`** - Single-input → Multi-output prediction
- `reports/` - Model performance summaries
- `requirements.txt` - Python dependencies
- `gnn_env/` - Virtual environment (gitignored)
- `.gitignore` - Excludes large data and virtual env from git

## Data Format

Each `rvlab.tdscf.rho.XXXXX` file contains:

- Column 1: Grid point index
- Column 2: Electron density value (scientific notation)

Time series spans from timestep 0 to 2000 (401 files with increment of 5).

## Two Approaches

### 1. Multi-Input Prediction (`electron_density_gnn_multi_input.ipynb`)

**Input:** 3 past timesteps (t-2, t-1, t)  
**Output:** 1 future timestep (t+5)

**Advantages:**

- Uses historical context
- Higher accuracy for near-term predictions
- More stable predictions

**Use when:** You have sequential observations and need high short-term accuracy

### 2. Multi-Future Prediction (`electron_density_gnn_multi_future.ipynb`) ⭐ NEW

**Input:** Current timestep only (t)  
**Output:** 3 future timesteps (t+5, t+10, t+15)

**Advantages:**

- Simpler input requirements
- Predicts full trajectory at once
- Useful for long-term planning

**Use when:** You only have current state or need to predict multiple future states

## Model Architecture

### Shared Components:

- **Graph Construction:** k-NN + sequential neighbors
- **GNN Backbone:** Multiple Graph Convolutional layers with batch normalization
- **Node Features:** Electron density values at grid points

### Multi-Input Model:

- Input: (num_nodes, 3) - 3 historical timesteps
- Output: (num_nodes, 1) - 1 future timestep

### Multi-Future Model:

- Input: (num_nodes, 1) - current timestep only
- Output: (num_nodes, 3) - 3 future timesteps
- Architecture: Shared backbone + 3 separate output heads

## Understanding Your Results

Both notebooks include comprehensive evaluation sections:

### Built-in Performance Analysis:

- **Baseline Comparison** - Compares your model against simple baselines
- **Automated Interpretation** - Explains what your metrics mean
- **Visual Diagnostics** - Charts showing prediction quality
- **Performance Grade** - Clear A-F grade based on R² and MAE
- **Improvement Roadmap** - Prioritized list of what to try next

### Key Metrics to Watch:

- **R² Score**: >0.85 = Excellent, 0.7-0.85 = Good, <0.7 = Needs work
- **MAE**: Should be <2% of your data range
- **Baseline Improvement**: Should beat simple predictions by >10%

The notebooks automatically tell you:

- ✓ Is your model actually learning?
- ✓ How good are your results compared to baselines?
- ✓ What specific improvements to try next?
- ✓ Which changes will have the biggest impact?

## Next Steps

**After running the notebook:**

1. Check Section 12 (Executive Summary) for your performance grade
2. Follow the Top 3 Action Items listed there
3. Make ONE change at a time and re-run
4. Compare before/after to measure improvement

**Common first improvements:**

1. Use all 401 timesteps instead of 100
2. Increase model size (hidden_dim=256, num_layers=6)
3. Train for 100+ epochs
4. Try Graph Attention Networks (GAT)
5. Add feature normalization
