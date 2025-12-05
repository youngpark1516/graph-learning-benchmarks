# Tags, Flags, and Tests Reference

Complete reference for all available command-line flags, parameters, and how to run different benchmarks in the graph learning framework.

## Table of Contents

1. [Core Tags](#core-tags)
2. [Model-Specific Tags](#model-specific-tags)
3. [Dataset & Task Tags](#dataset--task-tags)
4. [Training Tags](#training-tags)
5. [Output & Logging Tags](#output--logging-tags)
6. [Complete Examples](#complete-examples)
7. [Running Specific Tests](#running-specific-tests)

---

## Core Tags

Essential command-line flags used in most runs.

### `--models` (Multi-Model)

**Type**: `str` (comma-separated list)  
**Default**: `None` (must specify at least one)  
**Description**: Run multiple models in sequence

**Syntax**:
```bash
--models mpnn,graph_transformer,autograph_transformer,graphgps
```

**Available models**:
- `mpnn` — Graph neural network (GIN-based message passing)
- `graph_transformer` — Transformer-based model for tokenized graphs
- `autograph_transformer` — AutoGraph trail tokenization transformer
- `graphgps` — Graph Position-aware Spectral network

**Notes**:
- Each model runs independently with separate wandb logs
- Use `--group` to group runs together for comparison
- Can specify single model: `--models mpnn` or multiple: `--models mpnn,graphgps`

### `--model` (Single Model)

**Type**: `str`  
**Default**: `None`  
**Description**: Run a single model (overridden by `--models`)

**Syntax**:
```bash
--model mpnn
```

**Note**: Use `--models` for new scripts; `--model` is legacy

### `--task` (Task Selection)

**Type**: `str`  
**Default**: `edge_count`  
**Description**: Specifies the task to train/evaluate on

**Available tasks**:

#### Graph Classification Tasks (from `graph-token`)

```bash
--task cycle_check              # Detect cycles in graphs
--task shortest_path            # Find if path exists between nodes
--task shortest_path_downsampled_10k    # Balanced version, ~10k samples
--task shortest_path_downsampled_per_algo_5k  # Balanced version, ~5k/algo
```

#### Molecular Property Prediction

```bash
--task zinc                     # ZINC dataset (regression: log solubility)
```

**Naming convention for task variants**:
- `shortest_path` — Original (class-imbalanced)
- `shortest_path_downsampled_10k` — Downsampled to 10k samples
- `shortest_path_downsampled_per_algo_5k` — Downsampled per algorithm, ~5k/algo

### `--algorithm` (Single Graph Type)

**Type**: `str`  
**Default**: `ba`  
**Description**: Graph generation algorithm for graph classification tasks

**Available algorithms** (for graph-token tasks):
```bash
--algorithm ba              # Barabási-Albert (preferential attachment)
--algorithm er              # Erdős-Rényi (random, uniform edges)
--algorithm sbm             # Stochastic Block Model (community structure)
--algorithm path            # Path graph (linear chain)
--algorithm complete        # Complete graph (all edges)
--algorithm star            # Star graph (hub + leaves)
--algorithm sfn             # Scale-Free Network (power-law degrees)
```

**Graph properties**:

| Algorithm | Structure | Use Case |
|-----------|-----------|----------|
| **ba** | Power-law degree distribution | Preferential attachment networks |
| **er** | Random, uniform edge probability | Baseline random graphs |
| **sbm** | Community structure | Modular/clustered graphs |
| **path** | Linear chain (trivial) | Baseline simple structure |
| **complete** | All edges (trivial) | Fully connected baseline |
| **star** | Hub + leaves | Extreme hierarchy |
| **sfn** | Scale-free, realistic | Complex real-world networks |

### `--algorithms` (Multi-Algorithm)

**Type**: `str` (comma-separated list)  
**Default**: `None` (overrides `--algorithm`)  
**Description**: Train on multiple graph types simultaneously

**Syntax**:
```bash
--algorithms ba,er,sbm                  # Train on BA, ER, SBM
--algorithms path,er,sbm                # Path, ER, SBM (common for shortest_path)
--algorithms ba,er,sbm,star,complete    # All algorithms (maximal diversity)
```

**Benefits**:
- Improved generalization across different graph structures
- Better out-of-distribution (OOD) robustness
- Larger effective training set

### `--algorithm` vs `--test_algorithm`

**Training vs Testing**:
```bash
# Train on BA, test on BA
--algorithm ba

# Train on BA, test on different algorithm (OOD evaluation)
--algorithm ba --test_algorithm er
```

**For multi-algorithm**:
```bash
# Train on path/ER/SBM, test on SFN (OOD evaluation)
--algorithms path,er,sbm --test_algorithms sfn
```

### `--test_algorithm` / `--test_algorithms`

**Type**: `str` / `str` (comma-separated)  
**Default**: Same as `--algorithm` / `--algorithms`  
**Description**: Override test set graph type(s) for OOD evaluation

**Use cases**:

1. **In-distribution evaluation** (default):
   ```bash
   --algorithms path,er,sbm --test_algorithms path,er,sbm
   ```

2. **Out-of-distribution evaluation** (generalization):
   ```bash
   --algorithms path,er,sbm --test_algorithms sfn
   ```

3. **Single OOD test**:
   ```bash
   --algorithms ba,er --test_algorithm complete
   ```

---

## Model-Specific Tags

Flags that apply only to certain models.

### AutoGraph-Specific Tags (ZINC Only)

These tags control tokenization strategies for ZINC molecular graphs.

#### `--use_autograph_interleaved_edges`

**Type**: `bool` (flag, no value)  
**Model**: AutoGraph Transformer  
**Task**: ZINC only  
**Description**: Intersperse both atoms and bonds into the trail sequence

**What it does**:
- Embeds atomic types after each node
- Embeds bond types between nodes
- Creates enriched token sequences with chemical context

**Example output**:
```
<pad> ?6 atom_0 bond_1 ?7 atom_1 bond_1 ?8 atom_2 ...
       ↑    ↑      ↑      ↑    ↑      ↑
     node atom   bond   node  atom  bond
```

**Syntax**:
```bash
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --use_autograph_interleaved_edges
```

**Benefits**:
- Captures complete chemical context (atoms + bonds)
- Improves molecular property prediction
- Largest effective vocabulary (323 tokens)

#### `--use_autograph_interspersed`

**Type**: `bool` (flag, no value)  
**Model**: AutoGraph Transformer  
**Task**: ZINC only  
**Description**: Intersperse atomic types (not bonds) into trail

**What it does**:
- Embeds atomic types after each node
- Preserves graph topology via AutoGraph trail
- Simpler than bonds version

**Example output**:
```
<pad> ?6 atom_0 ?7 atom_1 ?8 atom_2 ...
       ↑    ↑    ↑    ↑    ↑    ↑
     node atom node atom node atom
```

**Syntax**:
```bash
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --use_autograph_interspersed
```

#### `--use_autograph_with_features`

**Type**: `bool` (flag, no value)  
**Model**: AutoGraph Transformer  
**Task**: ZINC only  
**Description**: Append node features after trail (legacy approach)

**What it does**:
- Generates AutoGraph trail without features
- Concatenates node feature vectors at end
- Older approach, less effective

**Syntax**:
```bash
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --use_autograph_with_features
```

#### No flag (Baseline)

**Type**: Default  
**Description**: Topology-only (no chemical features)

**What it does**:
- Pure graph structure from random walk trails
- No atomic or bond information
- Simplest baseline

**Syntax**:
```bash
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc
```

**Feature comparison**:

| Flag | Content | Vocab Size | Use Case |
|------|---------|-----------|----------|
| (none) | Topology only | 39 | Baseline |
| `--use_autograph_interspersed` | Atoms | 219 | Simple feature integration |
| `--use_autograph_interleaved_edges` | Atoms + Bonds | 323 | Full chemical context |
| `--use_autograph_with_features` | Appended features | 260+ | Legacy approach |

### GraphGPS-Specific Tags

#### `--use_lap_pe`

**Type**: `bool` (flag, no value)  
**Model**: GraphGPS  
**Description**: Enable Laplacian Positional Encoding

**What it does**:
- Computes eigenvalues/eigenvectors of graph Laplacian
- Encodes positional information for GNNs
- Improves model capacity for structural patterns

**Syntax**:
```bash
python benchmarks/run_benchmarks.py \
  --models graphgps \
  --task cycle_check \
  --use_lap_pe
```

#### `--lap_pe_dim`

**Type**: `int`  
**Default**: `16`  
**Model**: GraphGPS  
**Description**: Dimension of Laplacian positional encoding

**Recommended values**:
- `4-8` — Lightweight
- `16` — Balanced (default)
- `32` — High-capacity

**Syntax**:
```bash
--lap_pe_dim 32
```

#### `--norm_type`

**Type**: `str`  
**Default**: `batch`  
**Model**: GraphGPS  
**Description**: Normalization strategy

**Available options**:
```bash
--norm_type batch       # Batch normalization (default)
--norm_type layer       # Layer normalization
--norm_type graph       # Graph-wise normalization
--norm_type instance    # Instance normalization
--norm_type none        # No normalization
```

**When to use**:
- `batch` — Most common, works well with large batches
- `layer` — Better with small batches
- `graph` — Graph-level statistics (rare)
- `instance` — Per-instance normalization
- `none` — Disabled (rarely better)

**Syntax**:
```bash
python benchmarks/run_benchmarks.py \
  --models graphgps \
  --task cycle_check \
  --norm_type layer
```

---

## Dataset & Task Tags

### `--data_dir`

**Type**: `str` (file path)  
**Default**: `/data/young/capstone/graph-learning-benchmarks/submodules/graph-token`  
**Description**: Root directory containing task datasets

**Typical locations**:
```bash
# Default (submodule)
--data_dir submodules/graph-token

# Custom local directory
--data_dir ./data/custom_graphs

# Absolute path
--data_dir /absolute/path/to/graphs
```

**Directory structure**:
```
data_dir/
├── tasks_autograph/
│   ├── cycle_check/
│   │   ├── ba/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   ├── er/
│   │   └── ...
│   └── shortest_path/
│       ├── path/
│       ├── er/
│       └── ...
└── ZINC/                    (if ZINC downloaded)
    ├── train.csv
    ├── valid.csv
    └── test.csv
```

### `--max_samples_train` / `--max_samples_valid` / `--max_samples_test`

**Type**: `int`  
**Default**: `None` (use all samples)  
**Description**: Limit dataset sizes (useful for quick testing)

**Syntax**:
```bash
# Limit to 1000 training, 500 validation, 500 test
--max_samples_train 1000 --max_samples_valid 500 --max_samples_test 500

# Quick smoke test
--max_samples_train 100 --max_samples_valid 50 --max_samples_test 50
```

**Benefits**:
- Fast iteration during development
- Memory-efficient testing
- Reproducible subset experiments

---

## Training Tags

### `--epochs`

**Type**: `int`  
**Default**: `10`  
**Description**: Number of training epochs

**Typical values**:
```bash
--epochs 1          # Smoke test
--epochs 10         # Quick test
--epochs 50         # Standard training
--epochs 100        # Full training
--epochs 200        # Extended training
```

**Task-specific recommendations**:

| Task | Typical Epochs |
|------|----------------|
| cycle_check | 10-30 |
| shortest_path | 50-100 |
| ZINC | 50-100 |

### `--batch_size`

**Type**: `int`  
**Default**: `32`  
**Description**: Number of samples per batch

**Common values**:
```bash
--batch_size 8      # Small batches (low memory)
--batch_size 16     # Small
--batch_size 32     # Standard (default)
--batch_size 64     # Large
--batch_size 128    # Very large
```

**Memory usage**:
- Larger batch → faster training, more memory
- Smaller batch → slower, less memory

### `--learning_rate`

**Type**: `float`  
**Default**: `1e-3` (0.001)  
**Description**: Optimizer learning rate

**Common values**:
```bash
--learning_rate 1e-4    # Small (conservative)
--learning_rate 1e-3    # Standard (default)
--learning_rate 5e-3    # Medium
--learning_rate 1e-2    # Large (risky)
```

**Model-specific recommendations**:

| Model | Recommended LR |
|-------|----------------|
| MPNN | 1e-3 - 5e-3 |
| Graph Transformer | 5e-4 - 1e-3 |
| AutoGraph Transformer | 1e-3 - 5e-3 |
| GraphGPS | 1e-3 - 1e-2 |

### `--device`

**Type**: `str`  
**Default**: `cuda` (if available, else `cpu`)  
**Description**: Computation device

**Options**:
```bash
--device cuda       # GPU acceleration
--device cpu        # CPU (slower, no GPU requirement)
--device cuda:0     # Specific GPU (if multi-GPU)
```

---

## Output & Logging Tags

### `--project`

**Type**: `str`  
**Default**: `graph-benchmarks`  
**Description**: Weights & Biases (wandb) project name

**Syntax**:
```bash
--project zinc-benchmark
--project shortest-path-experiments
--project model-comparison
```

**Effect**: All runs logged to `wandb.ai/[username]/[project]`

### `--run_name`

**Type**: `str`  
**Default**: `None` (auto-generated)  
**Description**: Custom name for this run (wandb run name)

**Syntax**:
```bash
--run_name my_experiment_v1
--run_name zinc_atoms_interleaved_100ep
```

**Auto-generated format** (if not specified):
```
{model_name}-{task}-{timestamp}
```

### `--group`

**Type**: `str`  
**Default**: `None`  
**Description**: wandb group name to organize multiple runs

**Use case**: Compare models on same task

**Syntax**:
```bash
# Run 4 models with shared group
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,autograph_transformer,graphgps \
  --task zinc \
  --group zinc-all-models \
  --project zinc-benchmark
```

**Effect**: All 4 runs grouped together in wandb UI for easy comparison

### `--output_dir`

**Type**: `str` (file path)  
**Default**: `./models`  
**Description**: Directory to save model checkpoints

**Syntax**:
```bash
--output_dir ./checkpoints
--output_dir /tmp/models
```

**Saved files**:
- `{model_name}_best.pt` — Best model weights
- `{model_name}_config.json` — Model configuration

### `--model_config`

**Type**: `str` (file path)  
**Default**: Auto-detected (`benchmarks/model_configs.yaml`)  
**Description**: YAML/JSON file with per-model configuration overrides

**Syntax**:
```bash
--model_config custom_config.yaml
--model_config configs/models.json
```

**Example YAML**:
```yaml
mpnn:
  hidden_dim: 128
  num_layers: 5
  learning_rate: 1e-3

graph_transformer:
  d_model: 256
  n_heads: 8
  n_layers: 6

graphgps:
  use_lap_pe: true
  lap_pe_dim: 32
```

### `--log_model`

**Type**: `bool` (flag, no value)  
**Default**: `False`  
**Description**: Save best model artifact to wandb

**Syntax**:
```bash
--log_model
```

**Effect**: Best model checkpoint uploaded to wandb (enables model versioning)

### `--loss`

**Type**: `str`  
**Default**: `None` (auto-detect from task)  
**Description**: Loss function override

**Available losses**:
```bash
--loss mse          # Mean Squared Error (regression)
--loss mae          # Mean Absolute Error (regression)
--loss bce          # Binary Cross-Entropy (binary classification)
--loss cross_entropy # Cross-Entropy (multi-class)
```

**Auto-detection**:
- `shortest_path`, `cycle_check` → cross_entropy (classification)
- `zinc` → mse (regression)

### `--eval_metrics`

**Type**: `str` (comma-separated list)  
**Default**: `None`  
**Description**: Metrics to compute and log

**Available metrics**:
```bash
--eval_metrics accuracy         # Classification accuracy
--eval_metrics mae              # Mean Absolute Error
--eval_metrics rmse             # Root Mean Squared Error
--eval_metrics f1               # F1 score
--eval_metrics precision,recall # Multiple metrics
```

**Common combinations**:

Classification:
```bash
--eval_metrics accuracy,f1,precision
```

Regression:
```bash
--eval_metrics mae,rmse
```

---

## Complete Examples

### Example 1: Quick Smoke Test

**Purpose**: Verify setup works (< 1 minute)

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn \
  --task cycle_check \
  --algorithm er \
  --epochs 1 \
  --device cuda \
  --max_samples_train 50 \
  --max_samples_valid 20 \
  --max_samples_test 20 \
  --project smoke-test \
  --run_name quick_check
```

### Example 2: Single Task, All Models

**Purpose**: Compare models on one task

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,autograph_transformer,graphgps \
  --task cycle_check \
  --algorithm ba \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --device cuda \
  --project graph-benchmarks \
  --group cycle_check_ba \
  --eval_metrics accuracy,f1
```

**What it does**:
- Trains 4 different models
- All on cycle detection task
- BA graphs only
- Logs all runs to same wandb group

### Example 3: Multi-Algorithm Training

**Purpose**: Improve generalization with diverse graphs

```bash
python benchmarks/run_benchmarks.py \
  --models graphgps,mpnn \
  --task shortest_path_downsampled_per_algo_5k/shortest_path \
  --algorithms path,er,sbm \
  --test_algorithms sfn \
  --epochs 100 \
  --batch_size 32 \
  --device cuda \
  --data_dir submodules/graph-token \
  --use_lap_pe \
  --lap_pe_dim 32 \
  --project shortest-path \
  --group multi-algo-ood \
  --eval_metrics accuracy
```

**What it does**:
- Trains on path, ER, SBM graphs (in-distribution)
- Tests on SFN (out-of-distribution)
- Measures OOD generalization

### Example 4: ZINC With Different Tokenizations

**Purpose**: Compare AutoGraph tokenization strategies

```bash
# Baseline (topology only)
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --project zinc-tokenization \
  --run_name baseline_topology

# Atoms interspersed
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --project zinc-tokenization \
  --run_name atoms_interspersed \
  --use_autograph_interspersed

# Atoms + Bonds interleaved
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --project zinc-tokenization \
  --run_name atoms_bonds_interleaved \
  --use_autograph_interleaved_edges
```

**Expected results**:
- Topology only: ~0.15-0.20 MAE
- Atoms interspersed: ~0.12-0.15 MAE
- Atoms + bonds: ~0.10-0.13 MAE

### Example 5: GraphGPS With Positional Encoding

**Purpose**: Evaluate GraphGPS with spectral features

```bash
python benchmarks/run_benchmarks.py \
  --models graphgps \
  --task shortest_path \
  --algorithm sbm \
  --epochs 30 \
  --batch_size 32 \
  --device cuda \
  --use_lap_pe \
  --lap_pe_dim 32 \
  --norm_type batch \
  --project graphgps-experiments \
  --eval_metrics accuracy,f1
```

### Example 6: Full Benchmark Suite

**Purpose**: Comprehensive comparison across models and tasks

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,graphgps \
  --task cycle_check \
  --algorithms ba,er,sbm \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --project full-benchmark \
  --group cycle_check_multi_algo \
  --eval_metrics accuracy \
  --log_model
```

---

## Running Specific Tests

### Test 1: Cycle Detection

**What it tests**: Can models detect cycles in graphs?

**Quick test**:
```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graphgps \
  --task cycle_check \
  --algorithm er \
  --epochs 10 \
  --device cuda \
  --eval_metrics accuracy
```

**Full test** (all algorithms):
```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,graphgps \
  --task cycle_check \
  --algorithms ba,er,sbm \
  --epochs 50 \
  --device cuda
```

### Test 2: Shortest Path

**What it tests**: Can models predict path connectivity?

**Basic**:
```bash
python benchmarks/run_benchmarks.py \
  --models mpnn \
  --task shortest_path \
  --algorithm ba \
  --epochs 20 \
  --device cuda
```

**Balanced (recommended)**:
```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graphgps \
  --task shortest_path_downsampled_per_algo_5k/shortest_path \
  --algorithms path,er,sbm \
  --epochs 100 \
  --device cuda
```

**With OOD evaluation**:
```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graphgps \
  --task shortest_path_downsampled_per_algo_5k/shortest_path \
  --algorithms path,er,sbm \
  --test_algorithms sfn \
  --epochs 100 \
  --device cuda
```

### Test 3: ZINC Property Prediction

**What it tests**: Can models predict molecular property (solubility)?

**Quick test**:
```bash
python benchmarks/run_benchmarks.py \
  --models mpnn \
  --task zinc \
  --epochs 10 \
  --max_samples_train 500 \
  --device cuda
```

**Standard benchmark**:
```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,autograph_transformer,graphgps \
  --task zinc \
  --epochs 50 \
  --batch_size 32 \
  --device cuda
```

**With feature comparison**:
```bash
for features in "" "--use_autograph_interspersed" "--use_autograph_interleaved_edges"; do
  python benchmarks/run_benchmarks.py \
    --models autograph_transformer \
    --task zinc \
    --epochs 50 \
    --device cuda \
    $features
done
```

### Test 4: Model Architecture Comparison

**What it tests**: Which model architecture works best?

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,autograph_transformer,graphgps \
  --task cycle_check \
  --algorithm er \
  --epochs 30 \
  --batch_size 32 \
  --device cuda \
  --group model_comparison \
  --project benchmarks \
  --eval_metrics accuracy,f1
```

### Test 5: Out-of-Distribution Generalization

**What it tests**: Does model generalize to unseen graph structures?

```bash
python benchmarks/run_benchmarks.py \
  --models graphgps,mpnn \
  --task shortest_path_downsampled_per_algo_5k/shortest_path \
  --algorithms path,er,sbm,ba \
  --test_algorithms sfn,complete \
  --epochs 100 \
  --device cuda \
  --use_lap_pe \
  --eval_metrics accuracy
```

**Interpretation**:
- High in-distribution accuracy + High OOD accuracy = Good generalization
- High in-distribution + Low OOD = Overfitting to specific structures

---

## Troubleshooting

### Issue: "Task not found"

**Cause**: Task directory doesn't exist  
**Check**:
```bash
ls submodules/graph-token/tasks_autograph/
# Should see: cycle_check, shortest_path, shortest_path_downsampled_10k, ...
```

**Fix**: Verify `--task` and `--data_dir` are correct

### Issue: CUDA out of memory

**Cause**: Batch too large for GPU memory  
**Solutions**:
```bash
# Reduce batch size
--batch_size 8

# Limit samples
--max_samples_train 1000

# Use CPU instead
--device cpu
```

### Issue: Poor accuracy/loss not decreasing

**Check**:
1. Learning rate too low: `--learning_rate 1e-2`
2. Too few epochs: `--epochs 100`
3. Task/model mismatch: Verify `--task` and `--models`
4. Bad data: Check with `--max_samples_train 100` smoke test

### Issue: wandb not logging

**Cause**: Not authenticated  
**Fix**:
```bash
wandb login
# Enter your API key
```

---

## Quick Reference Table

| Flag | Type | Default | Example |
|------|------|---------|---------|
| `--models` | str | None | `mpnn,graphgps` |
| `--task` | str | edge_count | `cycle_check` |
| `--algorithm` | str | ba | `er` |
| `--algorithms` | str | None | `ba,er,sbm` |
| `--epochs` | int | 10 | `50` |
| `--batch_size` | int | 32 | `64` |
| `--learning_rate` | float | 1e-3 | `5e-3` |
| `--device` | str | cuda | `cpu` |
| `--project` | str | graph-benchmarks | `zinc-benchmark` |
| `--use_autograph_interleaved_edges` | bool | False | (flag) |
| `--use_lap_pe` | bool | False | (flag) |
| `--lap_pe_dim` | int | 16 | `32` |

---

## References

- **Tasks**: Generated by `submodules/graph-token/`
- **Models**: Implemented in `benchmarks/`
- **Logging**: Weights & Biases (wandb)
- **Config**: See `benchmarks/model_configs.yaml`
