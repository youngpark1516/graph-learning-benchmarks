# Graph Learning Benchmarks

A comprehensive benchmarking framework for graph neural networks and transformer-based models on molecular property prediction and graph classification tasks.

## 🎯 Quick Start

### Run a Quick Test

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,autograph_transformer,graphgps \
  --task cycle_check --algorithm er \
  --epochs 1 --device cuda \
  --project graph-benchmarks --run_name test_smoke
```

## 📚 Project Structure

### Directory Layout

```
graph-learning-benchmarks/
├── benchmarks/                          # Core training framework
│   ├── run_benchmarks.py               # Main CLI entry point
│   ├── builders.py                     # Model/dataset builders
│   ├── mpnn.py                         # GIN-style MPNN implementation
│   ├── graph_transformer.py            # Transformer for tokenized graphs
│   ├── autograph_transformer.py        # AutoGraph trail transformer
│   ├── graphgps.py                     # GraphGPS baseline
│   ├── train_utils.py                  # Training utilities
│   ├── common.py                       # Common helpers
│   ├── unified_dataset.py              # Graph classification dataset
│   ├── zinc_dataset.py                 # ZINC dataset with tokenization
│   ├── graph_task_generator_autograph.py # AutoGraph task generation
│   ├── downsample_shortest_path_per_algorithm.py # Downsample shortest_path per algorithm
│   ├── model_configs.yaml              # Per-model config overrides
│   ├── requirements.txt                # Python dependencies
│   └── data/                           # Dataset cache
│
├── submodules/                          # External frameworks
│   ├── autograph/                      # AutoGraph tokenization framework
│   ├── graph-token/                    # Graph generation utilities
│   ├── graphgps/                       # GraphGPS reference implementations
│   ├── SMP/                            # Supplementary experiments
│   └── reassesed_LRGB/                 # Additional experimental modules
│
├── notebooks/                           # Jupyter notebooks for analysis
│   ├── dataset_exploration.ipynb       # Dataset visualization
│   ├── model_analysis.ipynb            # Model behavior analysis
│   ├── tokenization_inspection.ipynb   # Token sequence inspection
│   └── results_visualization.ipynb     # Result comparison plots
│
├── docs/                                # Feature-specific documentation
│   ├── AUTOGRAPH_ZINC_TOKENIZATION.md  # AutoGraph tokenization guide
│   └── [Feature].md                    # TBD: Feature guides
│
└── README.md                            # This file (high-level overview)
```

### `/benchmarks/` — Core Training Framework

**Main Entry Point:**
- `run_benchmarks.py` — Unified CLI runner supporting multiple models, datasets, and configurations

**Models:**
- `mpnn.py` — GIN-style Message Passing Neural Network
- `graph_transformer.py` — Transformer-based tokenized graph model
- `autograph_transformer.py` — Transformer for AutoGraph token sequences
- `graphgps.py` — Graph Position-aware Spectral (GPS) baseline

**Data Handling:**
- `unified_dataset.py` — Unified interface for graph classification tasks
- `zinc_dataset.py` — ZINC molecular property prediction dataset with multiple tokenization strategies
- `graph_task_generator_autograph.py` — AutoGraph-compatible task generation

**Utilities & Scripts:**
- `builders.py` — Model and dataset construction helpers
- `train_utils.py` — Training and evaluation utilities
- `common.py` — Common path and logging helpers
- `downsample_shortest_path_per_algorithm.py` — Downsample shortest_path examples per algorithm for balanced datasets
- `requirements.txt` — Dependencies
- `model_configs.yaml` — Per-model configuration overrides

### `/submodules/` — External Frameworks

- **`autograph/`** — AutoGraph framework for trail-based graph tokenization and model training
- **`graph-token/`** — Graph generation utilities (BA, ER, Path, SBM, SFN, Star graphs)
- **`graphgps/`** — Reference implementations of Graph Position-aware Spectral networks
- **`SMP/`**, **`reassesed_LRGB/`** — Supplementary experimental modules

### `/notebooks/` — Analysis & Exploration

Interactive Jupyter notebooks for:
- Dataset exploration and visualization
- Model analysis and debugging
- Results visualization and comparison
- Tokenization inspection

### `/docs/` — Detailed Documentation

High-level guides for specific features:
- `AUTOGRAPH_ZINC_TOKENIZATION.md` — Complete guide to AutoGraph tokenization for ZINC
  - 4 tokenization approaches explained
  - Performance comparisons
  - Implementation details
  - Troubleshooting

## 🔬 Supported Models

| Model | Location | Features |
|-------|----------|----------|
| **MPNN (GIN)** | `benchmarks/mpnn.py` | Graph classification, message passing |
| **Graph Transformer** | `benchmarks/graph_transformer.py` | Sequence tokenization, attention-based |
| **AutoGraph Transformer** | `benchmarks/autograph_transformer.py` | Random walk trails, chemical features |
| **GraphGPS** | `benchmarks/graphgps.py` | Positional encoding, spectral methods |

## 📊 Supported Datasets

| Dataset | Type | Task | Command |
|---------|------|------|---------|
| **Graph Classification** | Various graphs (BA, ER, etc.) | Cycle detection, etc. | `--task cycle_check --algorithm er` |
| **ZINC** | Molecular graphs | Property prediction (regression) | `--task zinc` |

## 🧪 Tokenization Strategies (ZINC)

AutoGraph supports 4 tokenization approaches for molecular graphs:

1. **Topology-Only** — Graph structure only
2. **Atoms Interspersed** — Structure + atomic types embedded in trail
3. **Atoms+Bonds Interleaved** — Full chemical context (atoms + bonds)
4. **Atoms Appended** — Structure + features appended (legacy)

→ See [`docs/AUTOGRAPH_ZINC_TOKENIZATION.md`](docs/AUTOGRAPH_ZINC_TOKENIZATION.md) for detailed guide

## 🚀 CLI Usage

### Basic Syntax

```bash
python benchmarks/run_benchmarks.py \
  --models MODEL_NAME \
  --task TASK_NAME \
  --epochs N_EPOCHS \
  --device cuda \
  [optional flags]
```

### Common Flags

```bash
# Model selection
--models mpnn,graph_transformer,autograph_transformer,graphgps

# Dataset configuration
--task zinc                          # Molecular property prediction
--task cycle_check --algorithm er    # Graph classification

# Training parameters
--epochs 30                          # Number of training epochs
--batch_size 32                      # Batch size
--learning_rate 1e-3                 # Learning rate
--device cuda                        # cuda or cpu

# Data limits (useful for testing)
--max_samples_train 10000
--max_samples_valid 1000
--max_samples_test 1000

# Logging & config
--project zinc-benchmark             # wandb project name
--run_name my_experiment             # wandb run name
--model_config path/to/config.yaml   # Model config overrides
--log_model                          # Save model artifacts

# AutoGraph-specific (ZINC only)
--use_autograph                      # Topology-only
--use_autograph_interspersed         # Atoms interspersed
--use_autograph_interleaved_edges    # Atoms + bonds interleaved
--use_autograph_with_features        # Atoms appended (legacy)
```

### Configuration Files

Auto-discovery (set one, or auto-detect from `benchmarks/`):
```bash
python benchmarks/run_benchmarks.py --model_config custom_config.yaml [...]
```

Override specific models in config:
```yaml
mpnn:
  hidden_dim: 128
  num_layers: 4

graph_transformer:
  d_model: 256
  n_heads: 8
  n_layers: 6
```

## 📈 Output & Results

### Logging

Results are automatically logged to **Weights & Biases (wandb)**:
- Training/validation metrics over time
- Model architecture and hyperparameters
- Per-run configuration saved to `./models/`

### Checkpoints

Saved under `./checkpoints/`:
- `checkpoint_epoch_N.pt` — Intermediate checkpoints
- `best_model.pt` — Best validation model

### Models

Saved under `./models/`:
- Per-run JSON config for reproducibility
- Model artifacts (if `--log_model` flag used)

## 🔧 Development & Extension

### Adding a New Model

1. Create model file in `benchmarks/model_name.py`
2. Add builder function in `builders.py`
3. Register in `run_benchmarks.py`

### Adding a New Dataset

1. Create dataset class in `benchmarks/dataset_name.py`
2. Implement loader in corresponding builder
3. Add CLI flags and handling in `run_benchmarks.py`

## 📖 Documentation Structure

- **High-level overview**: This README
- **Feature-specific guides**: `/docs/` folder
  - `AUTOGRAPH_ZINC_TOKENIZATION.md` — AutoGraph tokenization for ZINC
  - `[Feature].md` — Feature-specific documentation (TBD)

For implementation details, see docstrings in corresponding source files.

## 🛠️ Requirements

- Python 3.8+
- PyTorch + PyG (PyTorch Geometric)
- wandb (Weights & Biases)
- See `benchmarks/requirements.txt` for full dependencies

## 📝 References

Key papers and frameworks:
- **AutoGraph**: Trail-based graph tokenization
- **ZINC Dataset**: Commercially available compounds for virtual screening
- **Graph Transformers**: Attention-based graph learning
- **GraphGPS**: Graph Position-aware Spectral networks

## 📞 Quick Help

**Configuration precedence** (lowest to highest):
1. Code defaults
2. CLI arguments
3. Config file (YAML/JSON)
4. Environment variables

**Debugging**:
- Check `./models/` for config JSONs to understand what ran
- Review wandb logs for training curves and errors
- See `docs/` for feature-specific troubleshooting

**Common Issues**:
- See `docs/AUTOGRAPH_ZINC_TOKENIZATION.md` for ZINC-specific troubleshooting
- Check `requirements.txt` for dependency versions