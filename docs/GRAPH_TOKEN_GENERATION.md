# Graph Token Generation & Task Creation Guide

Complete step-by-step guide to create graphs, generate tasks, tokenize them with AutoGraph, and use them for benchmarking in the graph learning framework.

## Table of Contents

1. [Overview](#overview)
2. [Step 1: Set Up Environment](#step-1-set-up-environment)
3. [Step 2: Generate Base Graphs](#step-2-generate-base-graphs)
4. [Step 3: Generate Tasks from Graphs](#step-3-generate-tasks-from-graphs)
5. [Step 4: Tokenize Tasks with AutoGraph](#step-4-tokenize-tasks-with-autograph)
6. [Step 5: Run Benchmarks](#step-5-run-benchmarks)
7. [Directory Structure](#directory-structure)
8. [Advanced: Custom Graph Generation](#advanced-custom-graph-generation)
9. [Advanced: Custom Tasks](#advanced-custom-tasks)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The workflow has four main stages:

```
┌─────────────────────┐
│ Generate Base Graphs│  (random graphs using graph algorithms)
│   (networkx)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Generate Tasks      │  (create task-specific samples from graphs)
│  (cycle check,      │
│   shortest path)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Tokenize with       │  (convert to sequences for transformers)
│ AutoGraph           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Benchmark Models    │  (train/evaluate on tokenized tasks)
│  (MPNN, Transformer,│
│   GraphGPS)         │
└─────────────────────┘
```

**Key tools**:
- `graph_generator.py` — Creates random graphs
- `graph_task_generator_autograph.py` — Creates task samples with AutoGraph tokenization
- `run_benchmarks.py` — Trains models on tasks

---

## Step 1: Set Up Environment

### Navigate to graph-token directory

```bash
cd submodules/graph-token
```

### Create virtual environment (if not already done)

```bash
python3 -m venv graphenv
source graphenv/bin/activate  # On Windows: graphenv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `networkx` — Graph generation and manipulation
- `torch` — PyTorch (required for tokenization)
- `torch-geometric` — Graph neural network operations
- `absl-py` — Command-line flags

---

## Step 2: Generate Base Graphs

### What this does

Creates random graphs using different generation algorithms (Barabási-Albert, Erdős-Rényi, etc.) and saves them as GraphML files.

### Graph algorithms available

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **ba** | Barabási-Albert | Preferential attachment, power-law |
| **er** | Erdős-Rényi | Random uniform edges |
| **sbm** | Stochastic Block Model | Community structure |
| **sfn** | Scale-Free Network | Realistic power-law |
| **path** | Path graph | Linear chain (trivial) |
| **complete** | Complete graph | All edges (trivial) |
| **star** | Star graph | Hub + leaves |

### Command: Generate graphs for a single algorithm

```bash
python graph_generator.py \
  --algorithm er \
  --number_of_graphs 500 \
  --split train \
  --output_path graphs
```

**Parameters**:
- `--algorithm` — Graph type (ba, er, sbm, sfn, path, complete, star)
- `--number_of_graphs` — How many graphs to generate
- `--split` — train, valid, or test
- `--output_path` — Where to save graphs

**Output structure**:
```
graphs/
├── er/
│   ├── train/
│   │   ├── 0.graphml
│   │   ├── 1.graphml
│   │   └── ... (500 files)
│   ├── valid/
│   └── test/
├── ba/
│   ├── train/
│   │   └── ... (500 files)
│   ├── valid/
│   └── test/
└── ... (other algorithms)
```

### Complete: Generate all algorithms and splits

```bash
# Generate 500 graphs per algorithm, for train/valid/test splits
for algorithm in er ba sbm sfn path complete star; do
  for split in train valid test; do
    python graph_generator.py \
      --algorithm "$algorithm" \
      --number_of_graphs 500 \
      --split "$split" \
      --output_path graphs
  done
done
```

**Approximate time**: ~2-5 minutes for all 21 combinations (7 algorithms × 3 splits)

**Disk usage**: ~500MB for 10,500 graphs

### Verify graphs were created

```bash
ls -R graphs/
# Should show: er/ ba/ sbm/ sfn/ path/ complete/ star/
# Each with: train/ valid/ test/ subdirectories

# Count graphs for ER algorithm
find graphs/er -name "*.graphml" | wc -l
# Should show: 1500 (500 × 3 splits)
```

---

## Step 3: Generate Tasks from Graphs

### What this does

Takes generated graphs and creates task-specific samples. For example:
- **Cycle Check**: "Does graph contain a cycle?" (binary label)
- **Shortest Path**: "Is there a path from node X to node Y?" (binary label)
- **Edge Count**: "How many edges?" (regression label)

### Available tasks

```bash
er, ba, sbm, sfn, complete, star, path, all
```

| Task | Type | Description |
|------|------|------------|
| `cycle_check` | Classification | Detect cycles |
| `shortest_path` | Classification | Path connectivity |
| `edge_existence` | Classification | Edge between nodes |
| `node_degree` | Regression | Node degree value |
| `edge_count` | Regression | Total edges |
| `node_count` | Regression | Total nodes |
| `connected_nodes` | Classification | Nodes connected? |
| `disconnected_nodes` | Classification | Nodes disconnected? |
| `triangle_counting` | Regression | Triangle count |
| `maximum_flow` | Regression | Flow value |
| `reachability` | Classification | Can reach? |
| `node_classification` | Classification | Node label |

### Command: Generate cycle_check task for ER graphs

```bash
python graph_task_generator_autograph.py \
  --task cycle_check \
  --algorithm er \
  --graphs_dir graphs \
  --task_dir tasks \
  --split train \
  --random_seed 1234
```

**Parameters**:
- `--task` — Task type (see available tasks above)
- `--algorithm` — Which graphs to use (or `all` for all algorithms)
- `--graphs_dir` — Where to read graphs (output from Step 2)
- `--task_dir` — Where to save task samples
- `--split` — train, valid, or test
- `--random_seed` — For reproducibility

**Output structure**:
```
tasks_autograph/
├── cycle_check/
│   ├── er/
│   │   ├── train/
│   │   │   ├── {id}_train_0.json
│   │   │   ├── {id}_train_1.json
│   │   │   └── ...
│   │   ├── valid/
│   │   └── test/
│   ├── ba/
│   │   └── ...
│   └── ... (other algorithms)
└── shortest_path/
    ├── er/
    │   └── ...
    └── ... (other tasks)
```

### Complete: Generate all tasks and splits

```bash
TASKS=(
  "cycle_check"
  "shortest_path"
  "edge_count"
  "node_degree"
  "edge_existence"
)

for task in "${TASKS[@]}"; do
  for split in train valid test; do
    python graph_task_generator_autograph.py \
      --task "$task" \
      --algorithm all \
      --graphs_dir graphs \
      --task_dir tasks \
      --split "$split" \
      --random_seed 1234
  done
done
```

**Approximate time**: ~5-10 minutes

### Verify tasks were created

```bash
ls -R tasks_autograph/
# Should show: cycle_check/, shortest_path/, edge_count/, ...

# Count samples in cycle_check
find tasks_autograph/cycle_check -name "*.json" | wc -l
# Should be many (500 graphs × number of samples per graph × 7 algorithms)

# Inspect a sample
cat tasks_autograph/cycle_check/er/train/er_train_0.json | python -m json.tool | head -30
# Should show: graph_id, text, tokens, label
```

---

## Step 4: Tokenize Tasks with AutoGraph

### What this does

Already done in Step 3! The `graph_task_generator_autograph.py` script automatically:
1. Reads graphs
2. Tokenizes them with AutoGraph's `Graph2TrailTokenizer`
3. Saves both text and tokenized representations

### Understanding the output format

Each task sample is a JSON object with:

```json
{
  "graph_id": "er_train_0",
  "text": "cycle_check: Does graph have a cycle? Answer: Yes",
  "tokens": [2, 6, 7, 8, 3],
  "label": 1
}
```

**Fields**:
- `graph_id` — Identifier for the graph
- `text` — Human-readable task description
- `tokens` — AutoGraph tokenized representation (integers)
- `label` — Target label (0/1 for classification, numeric for regression)

**Token interpretation**:
- Token `2` = `<bos>` (beginning of sequence)
- Tokens `6, 7, 8, ...` = node trail (random walk)
- Token `3` = `<eos>` (end of sequence)

### Verify tokenization

```bash
# Check token ranges
python -c "
import json
with open('tasks_autograph/cycle_check/er/train/er_train_0.json') as f:
    samples = json.load(f)
    for sample in samples[:3]:
        print(f'Tokens: {sample[\"tokens\"]}')
        print(f'Label: {sample[\"label\"]}')
"
```

---

## Step 5: Run Benchmarks

### What this does

Trains models (MPNN, GraphGPS, Transformers) on the generated and tokenized tasks.

### From project root (go back first)

```bash
cd ../..  # Back to project root
```

### Quick test: Run on cycle_check

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graphgps \
  --task cycle_check \
  --algorithm er \
  --epochs 10 \
  --batch_size 32 \
  --device cuda \
  --project cycle-check-benchmark \
  --data_dir submodules/graph-token
```

**What happens**:
1. Looks for tasks in `submodules/graph-token/tasks_autograph/cycle_check/er/`
2. Loads train/valid/test splits
3. Builds MPNN and GraphGPS models
4. Trains for 10 epochs
5. Logs to wandb project `cycle-check-benchmark`

### Multi-algorithm run: Better generalization

```bash
python benchmarks/run_benchmarks.py \
  --models graphgps,mpnn \
  --task shortest_path \
  --algorithms er,ba,sbm \
  --test_algorithms sfn \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --data_dir submodules/graph-token \
  --eval_metrics accuracy
```

**What this does**:
- Trains on 3 algorithms (ER, BA, SBM)
- Tests on different algorithm (SFN) for OOD evaluation
- Measures generalization to unseen graph types

### Full benchmark: All models

```bash
python benchmarks/run_benchmarks.py \
  --models mpnn,graph_transformer,autograph_transformer,graphgps \
  --task cycle_check \
  --algorithms ba,er,sbm \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --project full-cycle-check \
  --eval_metrics accuracy,f1
```

---

## Directory Structure

After completing all steps, you'll have:

```
graph-learning-benchmarks/
├── submodules/graph-token/
│   ├── graphs/                     # Step 2 output
│   │   ├── er/
│   │   │   ├── train/ (500 GraphML files)
│   │   │   ├── valid/ (500 GraphML files)
│   │   │   └── test/ (500 GraphML files)
│   │   ├── ba/, sbm/, sfn/, path/, complete/, star/
│   │   └── ...
│   │
│   └── tasks_autograph/            # Step 3-4 output (already tokenized!)
│       ├── cycle_check/
│       │   ├── er/
│       │   │   ├── train/
│       │   │   │   ├── er_train_0.json (graph_id, text, tokens, label)
│       │   │   │   ├── er_train_1.json
│       │   │   │   └── ...
│       │   │   ├── valid/
│       │   │   └── test/
│       │   ├── ba/, sbm/, sfn/, ...
│       │   └── ...
│       │
│       ├── shortest_path/
│       │   └── (same structure)
│       │
│       ├── edge_count/, node_degree/, ...
│       │   └── (same structure for each task)
│       │
│       └── models/                 # Saved configs & checkpoints
│           ├── mpnn_best.pt
│           ├── graphgps_best.pt
│           └── ...
│
└── benchmarks/
    ├── run_benchmarks.py           # Step 5: runs models
    └── ...
```

---

## Advanced: Custom Graph Generation

### Generate custom graph parameters

```bash
# ER graphs with specific sparsity range
python graph_generator.py \
  --algorithm er \
  --number_of_graphs 500 \
  --split train \
  --output_path graphs \
  --min_sparsity 0.1 \
  --max_sparsity 0.5
```

**Sparsity** = edge density (0 = sparse, 1 = dense)

### Generate directed graphs

```bash
python graph_generator.py \
  --algorithm ba \
  --number_of_graphs 500 \
  --split train \
  --output_path graphs \
  --directed true
```

### Generate smaller/larger graphs

Graph size is determined by the algorithm's internal parameters. To control size:

**Edit `graph_generator_utils.py`** to modify `generate_graphs()` function parameters like:
- `n` — Number of nodes (typically 50-1000)
- `m` — Number of edges to attach (for BA)
- `p` — Edge probability (for ER)

Example:
```python
# Line in graph_generator_utils.py
def generate_graphs(...):
    # Change these parameters:
    n_nodes = 100  # was 50, now larger
    m_edges = 3    # was 2, more edges
```

---

## Advanced: Custom Tasks

### Create a custom task class

Create file `submodules/graph-token/graph_task_custom.py`:

```python
import networkx as nx
from graph_task_utils import Task

class MyCustomTask(Task):
    """Detect if graph has exactly 3 triangles."""
    
    def tokenize_graph(self, graph: nx.Graph, graph_id: str):
        """Return dict[graph_id] -> list of samples."""
        
        # Count triangles
        triangles = sum(nx.triangles(graph).values()) // 3
        
        # Create 5 samples per graph
        samples = [
            f"triangle_count: Count triangles. Answer: {triangles}"
            for _ in range(5)
        ]
        
        # For graph-level tasks, all samples share same graph
        return {graph_id: samples}
```

### Register and use custom task

Edit `graph_task_generator_autograph.py`:

```python
from graph_task_custom import MyCustomTask

TASK_CLASS = {
    # ... existing tasks ...
    'my_custom_task': MyCustomTask,
}

_TASK = flags.DEFINE_enum(
    "task",
    None,
    list(TASK_CLASS.keys()),  # Auto includes 'my_custom_task'
    "The task to generate datapoints.",
    required=True,
)
```

### Generate custom task

```bash
python graph_task_generator_autograph.py \
  --task my_custom_task \
  --algorithm all \
  --graphs_dir graphs \
  --task_dir tasks \
  --split train
```

---

## Troubleshooting

### Issue: "No graphs found" when generating tasks

**Cause**: Graphs directory doesn't exist or is empty

**Fix**:
```bash
# Verify graphs exist
ls graphs/er/train/ | wc -l
# Should show: 500

# Regenerate if needed
python graph_generator.py --algorithm er --number_of_graphs 500 --split train --output_path graphs
```

### Issue: "Cannot import autograph" when generating tasks

**Cause**: AutoGraph submodule not initialized

**Fix**:
```bash
# Check if autograph submodule exists
ls submodules/autograph/
# If missing, init submodule
git submodule update --init --recursive
```

### Issue: Task generation is very slow

**Cause**: Tokenizing large graphs or generating many samples

**Solutions**:
```bash
# Use fewer graphs for testing
python graph_generator.py --algorithm er --number_of_graphs 50 --split train --output_path graphs

# Or process in parallel (requires manual setup)
# Generate tasks in separate terminals for different algorithms
```

### Issue: Out of memory during tokenization

**Cause**: Graphs too large or batch processing

**Fix**:
```bash
# Generate smaller graphs
# Edit graph_generator_utils.py and reduce n_nodes
# Then regenerate graphs with fewer nodes
```

### Issue: Task samples are empty

**Cause**: Task class not properly implemented

**Check**:
```bash
# Inspect sample
cat tasks_autograph/cycle_check/er/train/*.json | python -m json.tool | head -50
# Should show: graph_id, text, tokens, label

# If tokens are empty:
python -c "import graph_task_generator_autograph; print('Import OK')"
# If error, check AutoGraph submodule
```

### Issue: Benchmarking fails with "Task directory not found"

**Cause**: `--data_dir` doesn't point to right location

**Fix**:
```bash
# From project root, use correct path
python benchmarks/run_benchmarks.py \
  --task cycle_check \
  --data_dir submodules/graph-token \
  ...
```

---

## Complete End-to-End Example

Quick workflow to generate, tokenize, and benchmark in ~10 minutes:

```bash
# 1. Go to graph-token
cd submodules/graph-token
source graphenv/bin/activate

# 2. Generate small graph dataset (quick)
for algorithm in er ba; do
  for split in train valid test; do
    python graph_generator.py \
      --algorithm "$algorithm" \
      --number_of_graphs 50 \
      --split "$split" \
      --output_path graphs
  done
done

# 3. Generate cycle_check task with AutoGraph tokenization
for split in train valid test; do
  python graph_task_generator_autograph.py \
    --task cycle_check \
    --algorithm all \
    --graphs_dir graphs \
    --task_dir tasks \
    --split "$split"
done

# 4. Return to project root
cd ../..

# 5. Benchmark on the generated task
python benchmarks/run_benchmarks.py \
  --models mpnn,graphgps \
  --task cycle_check \
  --algorithms er,ba \
  --epochs 10 \
  --batch_size 32 \
  --device cuda \
  --data_dir submodules/graph-token \
  --project quick-benchmark
```

**Output**: 
- Models trained and evaluated on cycle detection
- Results logged to wandb
- Best models saved to `./models/`

---

## References

**Framework files**:
- `submodules/graph-token/graph_generator.py` — Base graph generation
- `submodules/graph-token/graph_task_generator_autograph.py` — Task generation + AutoGraph tokenization
- `benchmarks/run_benchmarks.py` — Benchmark runner
- `benchmarks/mpnn.py` — MPNN dataset loading

**Configuration**:
- `submodules/graph-token/graph_generator_utils.py` — Graph generation parameters
- `benchmarks/model_configs.yaml` — Model hyperparameters

**Related documentation**:
- See `docs/TAGS_AND_TESTS.md` for benchmark CLI reference
- See `docs/AUTOGRAPH_ZINC_TOKENIZATION.md` for ZINC-specific tokenization
- See `docs/SHORTEST_PATH_TASK.md` for shortest path task details
