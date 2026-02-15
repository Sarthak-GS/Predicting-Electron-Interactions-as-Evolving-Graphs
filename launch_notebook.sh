#!/bin/bash
# Launcher script for the GNN project

echo "=== Electron Density GNN Project ==="
echo ""

# Activate virtual environment
source gnn_env/bin/activate

# Launch Jupyter notebook
echo "Launching Jupyter Notebook..."
jupyter notebook electron_density_gnn.ipynb
