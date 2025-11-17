# Benchmarks Directory

This directory contains scripts and resources for running and evaluating graph learning models and tasks. Below is a description of each file and related submodules.

## Related Submodules

The project includes several key submodules that work together with these benchmarks:

### autograph/
- A framework for graph-based machine learning
- Includes data modules, evaluation tools, and model implementations
- Contains configuration files for training and testing

### graph-token/
- Tools for graph generation and task creation
- Supports various graph types (BA, Complete, ER, Path, SBM, SFN, Star)
- Implements different graph tasks:
  - Connected nodes
  - Cycle checking
  - Edge counting/existence
  - Node counting/degree
  - Shortest path

### graphgps/
- Graph Processing System implementation
- Includes various model architectures:
  - GatedGCN
  - GINE
  - GPS
  - Graphormer
  - SAN
- Provides comprehensive testing and evaluation tools

### reassesed_LRGB/
- Reassessed Long-Range Graph Benchmark
- Contains fine-tuned configurations
- Shares common architecture with graphgps

### SMP/
- Includes configurations for cycles and multi-task learning
- Supports ZINC dataset experiments
- Contains model implementations and utilities
- Dataset generation tools included

## Files

- **autograph_transformer.py**
  - Implements the Autograph Transformer model for graph-based tasks.

- **graph_task_generator_autograph.py**
  - Script for generating graph tasks using the Autograph framework.

- **graph_transformer.py**
  - Contains the implementation of a Graph Transformer model.

- **requirements.txt**
  - Lists the Python dependencies required to run the scripts in this directory.

- **mpnn.py**
  - Implements a GIN-style MPNN baseline using PyTorch Geometric.
  - Loads synthetic AutoGraph task files, rebuilds each graph, and trains a message-passing model for regression or classification.

## Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run scripts**
   - Use the provided Python scripts to train models, generate tasks, or evaluate performance. Refer to each script's docstring or comments for specific usage instructions.

## Notes
- Ensure you have the correct environment (Python version, CUDA, etc.) as specified in `requirements.txt` and project documentation.

For more details on each script, please refer to the inline documentation or comments within the respective files.