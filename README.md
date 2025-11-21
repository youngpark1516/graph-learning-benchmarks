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

# Benchmarks Directory

This directory contains scripts and resources for running and evaluating graph learning models and tasks. Below is a concise reference to the important files and how to run a quick smoke benchmark.

## Related Submodules

The repository uses several submodules for data and model implementations (briefly):

- `autograph/` — data modules, utilities and model helpers used by the autograph transformer code.
- `graph-token/` — graph generators and task datasets (BA, ER, Path, SBM, SFN, Star, etc.).
- `graphgps/` — reference GPS-style implementations and utilities.
- `SMP/`, `reassesed_LRGB/` — supplementary experiments and configs used elsewhere in the repo.

## Benchmarks files

The following files are in `benchmarks/`. Descriptions are short — read the files' docstrings for details.

- `run_benchmarks.py` — Unified runner that trains/evaluates models and logs to Weights & Biases (wandb). Supports running multiple models sequentially, per-model wandb runs (grouped), per-model config overrides (JSON/YAML), and automatic model-config discovery under `benchmarks/`.
- `mpnn.py` — GIN-style MPNN baseline and dataset utilities (`GraphTaskDataset`, trainer, collate function).
- `graph_transformer.py` — Transformer-based token model and `GraphDataset` used for tokenized graph tasks.
- `autograph_transformer.py` — Alternative transformer implementation compatible with autograph token datasets.
- `graphgps.py` — Lightweight in-repo GPS-like baseline (SimpleGPS + Trainer) that integrates with the unified runner.
- `builders.py` — helpers that construct datasets, loaders and models for each supported model (keeps `run_benchmarks.py` modular).
- `train_utils.py` — small training/evaluation utilities for transformer-style models.
- `common.py` — common helpers (path setup, standardized wandb log dicts).
- `model_configs.yaml` / `model_configs.json` — example per-model override configs; `run_benchmarks.py` will auto-detect `model_configs.yaml` (preferred) if `--model_config` is not provided.
- `requirements.txt` — package dependencies for the benchmarks.

## Quick example (smoke run)

Run a 1-epoch smoke test for all four models (uses `benchmarks/model_configs.yaml` automatically if present):

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,autograph_transformer,graphgps \
  --task cycle_check --algorithm er \
  --epochs 1 --device cuda \
  --project graph-benchmarks --run_name test_yaml
```

Notes:
- If you have not explicitly passed `--model_config`, the runner prefers `benchmarks/model_configs.yaml` then `.yml` then `.json` and will auto-load the first match.
- Per-model overrides from the config file are shallow-merged with the CLI defaults so unspecified values fall back to the CLI.
- Each model run creates a separate wandb run (grouped) and saves a per-run config JSON under `./models/` for reproducibility.

For more details and advanced usage (parallel GPU runs, optimizer or scheduler changes, adding new model-specific flags), see the docstrings in each file and the inline comments in `run_benchmarks.py`.