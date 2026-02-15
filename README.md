# Electron Density Evolution - GNN Project

This project uses Graph Neural Networks (GNNs) to predict electron density evolution in molecular systems.

## Setup

1. **Activate virtual environment:**

   ```bash
   source gnn_env/bin/activate
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook electron_density_gnn.ipynb
   ```

## Project Structure

- `ammonia_x/` - Ammonia electron density time series data
- `water_x/` - Water electron density time series data
- `electron_density_gnn.ipynb` - Main experimentation notebook
- `requirements.txt` - Python dependencies
- `gnn_env/` - Virtual environment

## Data Format

Each `rvlab.tdscf.rho.XXXXX` file contains:

- Column 1: Grid point index
- Column 2: Electron density value (scientific notation)

Time series spans from timestep 0 to 2000 (401 files with increment of 5).

## Approach

**Problem:** Predict electron density at future timesteps from current state

**Solution:**

- Treat each grid point as a node in a graph
- Model spatial relationships as edges (k-NN + sequential neighbors)
- Use GNN to learn temporal evolution patterns
- Predict density at t+Δt given densities at t, t-1, t-2, ...

## Model Architecture

- Input: Multi-timestep density values per node
- GNN: Multiple Graph Convolutional layers
- Output: Predicted density at future timestep

## Next Steps

1. Run the notebook cells sequentially
2. Experiment with different hyperparameters
3. Try different GNN architectures (GAT, GraphSAGE)
4. Implement multi-step prediction
5. Add physical constraints to the model
