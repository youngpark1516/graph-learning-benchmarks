# Shortest Path Task Guide

## Overview

The **Shortest Path** task is a graph classification problem where the model learns to predict whether a path exists between two nodes in a graph. This is a fundamental graph algorithm task useful for evaluating how well models capture graph connectivity and structural properties.

## Task Variants

### 1. Standard Shortest Path
Basic binary classification: does a path exist between two target nodes?

**Command**:
```bash
python benchmarks/run_benchmarks.py \
  --models graphgps \
  --task shortest_path \
  --algorithms path,er,sbm \
  --epochs 50 \
  --device cuda
```

### 2. Downsampled Shortest Path (Recommended)
**Label-downsampled variant that balances dataset distribution** to address class imbalance.

The standard shortest path datasets often suffer from class imbalance (too many "path exists" vs "no path" examples). The downsampled variant:
- Randomly samples negative examples to match positive class frequency
- Creates balanced datasets (50/50 split)
- Improves model generalization and training stability
- Uses per-algorithm downsampling (each graph type independently balanced)

**Command**:
```bash
python benchmarks/run_benchmarks.py \
  --models graphgps \
  --task shortest_path_downsampled_per_algo_5k/shortest_path \
  --data_dir submodules/graph-token \
  --algorithms path,er,sbm \
  --epochs 100 \
  --device cuda
```

**Dataset size in name**: The `5k` refers to ~5,000 samples per algorithm after downsampling.

---

## Key Parameters Explained

### Task Specification

```bash
--task shortest_path_downsampled_per_algo_5k/shortest_path
```

**Structure**: `task_variant/subtask`
- **`shortest_path_downsampled_per_algo_5k`** — Dataset variant with downsampling
- **`shortest_path`** — Specific prediction task (path existence)

**Naming convention**:
- `_downsampled` — Labels are downsampled for balance
- `_per_algo` — Downsampling applied per algorithm/graph type
- `_5k` — Approximate size after downsampling

### Graph Algorithms

```bash
--algorithms path,er,sbm
--test_algorithms sfn
```

**Training algorithms** (comma-separated):
- **`path`** — Path graphs (line structures: 1-2-3-4-...)
- **`er`** — Erdős-Rényi random graphs (uniform edge probability)
- **`sbm`** — Stochastic Block Model (community structure)
- **`ba`** — Barabási-Albert (preferential attachment)
- **`complete`** — Complete graphs (all edges)
- **`star`** — Star graphs (hub-and-spoke)
- **`sfn`** — Scale-Free Network (power-law degree distribution)

**Test algorithm** (separate flag):
- Model trained on `--algorithms` is tested on `--test_algorithms`
- Useful for **out-of-distribution (OOD) evaluation**
- In the example: trained on path/ER/SBM, tested on SFN
- Measures generalization to unseen graph structures

### Data Source

```bash
--data_dir submodules/graph-token
```

- Points to where graph datasets are generated/cached
- `graph-token` submodule provides graph generators for all algorithms
- Can also use local datasets: `--data_dir ./data/graphs`

---

## Example: Complete Experiment

Train GraphGPS on shortest path with full experimental setup:

```bash
python benchmarks/run_benchmarks.py \
  --models graphgps \
  --task shortest_path_downsampled_per_algo_5k/shortest_path \
  --data_dir submodules/graph-token \
  --algorithms path,er,sbm \
  --test_algorithms sfn \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --device cuda \
  --project shortest_path_experiment \
  --run_name gps_downsampled_sfn_ood \
  --max_samples_train 5000 \
  --max_samples_valid 1000 \
  --max_samples_test 1000
```

**Breakdown**:
- **Model**: GraphGPS (positional encoding + spectral methods)
- **Task**: Shortest path with balanced labels, ~5k samples
- **Train graphs**: Path, Erdős-Rényi, Stochastic Block Model
- **Test graphs**: Scale-Free Networks (OOD evaluation)
- **Training**: 100 epochs, batch size 32
- **Logging**: wandb project `shortest_path_experiment`

---

## Dataset Characteristics

### Graph Properties

| Algorithm | Structure | Typical Use |
|-----------|-----------|------------|
| **path** | Linear chain | Baseline (trivial connectivity) |
| **er** | Random, uniform | General random graphs |
| **sbm** | Community structure | Modular/clustered graphs |
| **ba** | Power-law degree | Scale-free growth |
| **complete** | All edges | Trivial (always connected) |
| **star** | Hub + leaves | Extreme hierarchy |
| **sfn** | Heavy-tailed degrees | Realistic networks |

### Task Characteristics

**Positive class** (path exists):
- Random pair of nodes with guaranteed shortest path
- ~50% of examples after downsampling

**Negative class** (no path):
- Random pair of nodes with no path
- Downsampled to match positive class
- ~50% of examples after downsampling

**Graph sizes**:
- Typical: 50-1000 nodes (task-dependent)
- Edges: varies by algorithm and density parameter

---

## Understanding Downsampling

### Why Downsampling?

Original shortest path datasets are **highly imbalanced**:
- **Connected graphs**: Most node pairs are connected (large positive class)
- **Imbalanced distribution**: 80%+ positive examples
- **Training issues**: Model biased toward predicting "path exists"

### What Downsampling Does

```
Before: [++++++++++--------] (80% positive, 20% negative)
After:  [+++++-----]         (50% positive, 50% negative)
         ↑ randomly sampled negative examples to match positive count
```

**Benefits**:
1. ✅ Balanced loss signal during training
2. ✅ Better generalization to both classes
3. ✅ More meaningful evaluation metrics (precision/recall not biased)
4. ✅ Faster convergence

**Per-algorithm downsampling** means each graph type (path, ER, SBM) is balanced independently, preserving algorithm-specific distribution characteristics.

---

## Expected Performance

### Typical Baselines

| Model | Path | ER | SBM | SFN (OOD) |
|-------|------|----|----|-----------|
| **MPNN** | ~98% | ~85% | ~92% | ~60% |
| **GraphGPS** | ~99% | ~88% | ~94% | ~70% |
| **Graph Transformer** | ~97% | ~83% | ~91% | ~58% |

*Accuracy on downsampled test sets (approximate)*

### OOD Generalization

- **In-distribution** (test on path/ER/SBM): Usually 85-99% accuracy
- **Out-of-distribution** (test on SFN): Often 50-75% accuracy
- OOD drop indicates structural overfitting to training graph types

---

## Advanced Usage

### Multi-Algorithm Training

Train on ALL algorithms simultaneously:
```bash
--algorithms path,er,sbm,ba,complete,star
--test_algorithms sfn
```

This increases dataset diversity and improves OOD generalization.

### Custom Train/Test Split

```bash
--algorithms path,er
--test_algorithms sbm,ba
```

Train on simpler structures (path, ER), test on more complex (SBM, BA).

### Ablation Study

Compare with standard (non-downsampled) variant:
```bash
# Non-downsampled (class-imbalanced)
--task shortest_path/shortest_path

# vs. downsampled (balanced)
--task shortest_path_downsampled_per_algo_5k/shortest_path
```

Check if downsampling improves OOD generalization.

---

## Troubleshooting

### Low Performance on OOD Test Set

**Cause**: Model overfits to training graph structure

**Solutions**:
1. Use more training algorithms: `--algorithms path,er,sbm,ba,star`
2. Increase regularization or dropout
3. Use positional encodings (GraphGPS)
4. Train for more epochs with lower learning rate

### Class Imbalance Warning

**Cause**: Using standard (non-downsampled) task variant

**Fix**: Use downsampled variant:
```bash
--task shortest_path_downsampled_per_algo_5k/shortest_path
```

### Out of Memory

**Cause**: Graph size or batch size too large

**Solution**:
```bash
--batch_size 16    # Reduce from 32
--max_samples_train 2500   # Reduce samples
```

---

## Dataset Location

Datasets are typically stored/generated in:
```
submodules/graph-token/data/
  ├── path/
  ├── er/
  ├── sbm/
  ├── ba/
  └── ...
```

Each contains:
- `graphs.pt` — Graph structures
- `labels.pt` — Binary labels (path exists?)
- `node_pairs.pt` — Target node pairs for evaluation

---

## References

- **Shortest Path**: Fundamental graph algorithm (BFS/DFS)
- **Class Imbalance**: Common in graph tasks with random sampling
- **OOD Generalization**: Evaluates if model learns invariant properties vs. dataset artifacts
- **Graph Types**: Standard benchmark generators used in graph ML literature

---

## Quick Reference

```bash
# Basic shortest path
--task shortest_path/shortest_path --algorithms path,er,sbm

# Balanced (recommended)
--task shortest_path_downsampled_per_algo_5k/shortest_path --algorithms path,er,sbm

# With OOD evaluation
--task shortest_path_downsampled_per_algo_5k/shortest_path --algorithms path,er,sbm --test_algorithms sfn

# Multi-model comparison
--models mpnn,graph_transformer,graphgps --task shortest_path_downsampled_per_algo_5k/shortest_path
```
