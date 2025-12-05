# AutoGraph ZINC Tokenization Guide

## Overview

AutoGraph provides trail-based molecular graph tokenization for the ZINC dataset. This guide explains the four tokenization approaches and how to use them.

## Tokenization Approaches

### 1. Topology-Only (Baseline)
Uses AutoGraph's random walk trail without any chemical features.

**Pattern**: `<pad> node_0 node_1 node_5 node_2 ... <eos>`

**Vocabulary**: 39 tokens (AutoGraph special tokens + node indices)

**Use case**: Baseline to measure impact of chemical features

**Command**:
```bash
python benchmarks/run_benchmarks.py --models autograph_transformer --task zinc \
  --epochs 30 --batch_size 32 --device cuda
```

---

### 2. Atoms Interspersed ⭐ Most Efficient
AutoGraph trail with atomic types embedded after each node occurrence.

**Pattern**: `<pad> node_0 atom_C node_1 atom_N node_5 atom_O ... <eos>`

**Vocabulary**: 219 tokens
- 0-3: Special tokens
- 6+: AutoGraph node indices (pass-through)
- 200-318: Atom types (119 elements)

**Sequence length**: ~112 tokens for 29-node graph (30% shorter than appended)

**Advantages**:
- Preserves graph topology perfectly
- Atoms embedded contextually in the trail
- Efficient sequence length
- Model sees: "when traversing from C to N, what's the chemical context?"

**Use case**: Best balance of chemical information and efficiency

**Command**:
```bash
python benchmarks/run_benchmarks.py --models autograph_transformer --task zinc \
  --epochs 30 --batch_size 32 --device cuda --use_autograph_interspersed
```

---

### 3. Atoms+Bonds Interleaved ⭐ Most Contextual
AutoGraph trail with BOTH atomic types AND bond types fully interleaved.

**Pattern**: `<pad> node_0 atom_C bond_single node_1 atom_N bond_double node_5 atom_O ... <eos>`

**Vocabulary**: 323 tokens
- 0-3: Special tokens
- 6+: AutoGraph node indices (pass-through)
- 200-318: Atom types (119 elements)
- 319-322: Bond types (4 types: single, double, triple, aromatic)

**Sequence length**: ~119 tokens for 29-node graph (only +7 tokens vs atoms-only!)

**Advantages**:
- Full chemical context: "C --[single]-- N --[double]-- O"
- Bonds appear exactly where traversed in the trail
- Minimal overhead (+6% tokens vs atoms-only)
- Model sees complete molecular structure

**Bond types in ZINC**:
- `bond_1`: Single bond (C-C, C-N, etc.)
- `bond_2`: Double bond (C=C, C=N, etc.)
- `bond_3`: Triple bond (C≡C)
- `bond_4`: Aromatic bond

**Use case**: Maximum chemical information with reasonable efficiency

**Command**:
```bash
python benchmarks/run_benchmarks.py --models autograph_transformer --task zinc \
  --epochs 30 --batch_size 32 --device cuda --use_autograph_interleaved_edges
```

---

### 4. Atoms Appended (Legacy - Not Recommended)
AutoGraph trail followed by all node-atom pairs appended at end.

**Pattern**: `<pad> [trail] <n> node_0 atom_C node_1 atom_N ... <eos>`

**Vocabulary**: 324 tokens

**Sequence length**: ~130+ tokens for 29-node graph (inefficient)

**Disadvantages**:
- Atoms far from where they appear in trail
- Loss of sequential locality
- 30% longer sequences
- Model doesn't see chemical context during traversal

**Use case**: Not recommended; prefer interspersed or interleaved approaches

**Command**:
```bash
python benchmarks/run_benchmarks.py --models autograph_transformer --task zinc \
  --epochs 30 --batch_size 32 --device cuda --use_autograph_with_features
```

---

## Comparison Table

| Feature | Topology | Atoms Interspersed | Atoms+Bonds Interleaved | Atoms Appended |
|---------|----------|-------------------|------------------------|----------------|
| **Vocab Size** | 39 | 219 | 323 | 324 |
| **Avg Sequence Length** | 71 | 112 | 119 | 130+ |
| **Graph Topology** | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Atom Info** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Bond Info** | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Efficiency** | High | ⭐⭐⭐ | ⭐⭐ | Low |
| **Chemical Context** | Low | Medium | ⭐⭐⭐ | Low |

---

## Example Token Sequences

For a small benzene-like molecule: C-C(=C)-C-N-O

### Topology-Only
```
<bos> 0 1 2 3 4 5 4 3 <eos>
```

### Atoms Interspersed
```
<bos> 0 C 1 C 2 C 3 C 4 N 5 O 4 N 3 C <eos>
```

### Atoms+Bonds Interleaved
```
<bos> 0 C single 1 C double 2 C single 3 C single 4 N double 5 O single 4 N single 3 C <eos>
```

### Atoms Appended
```
<bos> 0 1 2 3 4 5 4 3 <n> 0 C 1 C 2 C 3 C 4 N 5 O <eos>
```

---

## Implementation Details

### Vocabulary Offset Strategy

To avoid conflicts between AutoGraph's node tokens and our custom tokens, we use high token IDs:

- **0-3**: Special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`)
- **4-5**: AutoGraph internal markers
- **6+**: AutoGraph node indices (preserved as-is)
- **200-318**: Atom types (moved to high range)
- **319-322**: Bond types (moved to high range)

This ensures:
1. AutoGraph's topological information is never corrupted
2. No token ID collisions
3. Embedding layer automatically handles all token ranges

### Bond Lookup

Bonds are stored in canonical form for undirected graphs:
```python
edge_key = (min(node_u, node_v), max(node_u, node_v))
```

This ensures:
- Each undirected edge stored only once
- O(1) lookup when interleaving bonds into trail
- No duplicate or reversed edges

### Atom Extraction

Supports multiple node feature formats:

```python
# Scalar atomic number
atom_id = int(atom_feat.item())

# One-hot encoded vector (finds nonzero index)
atom_id = int(torch.nonzero(atom_feat).item())

# Single-element vector
atom_id = int(atom_feat[0].item())
```

---

## Performance Recommendations

### For Production Models
**Use: `--use_autograph_interleaved_edges`**
- Provides full chemical context (atoms + bonds)
- Only 8% longer sequences than atoms-only
- Should improve molecular property prediction accuracy

### For Efficiency-Critical Models
**Use: `--use_autograph_interspersed`**
- Best sequence length/information trade-off
- 30% shorter than appended approach
- Sufficient for many downstream tasks

### For Baseline Comparisons
**Use: No flag (topology-only)**
- Minimal baseline to measure chemical feature contribution
- Smallest vocabulary and fastest inference

---

## Benchmarking Commands

### Compare All Three Main Approaches

```bash
# Topology-only (baseline)
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --project zinc_topology_baseline

# Atoms interspersed (efficient)
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --use_autograph_interspersed \
  --project zinc_atoms_interspersed

# Atoms + bonds interleaved (full context)
python benchmarks/run_benchmarks.py \
  --models autograph_transformer \
  --task zinc \
  --epochs 50 \
  --batch_size 32 \
  --device cuda \
  --use_autograph_interleaved_edges \
  --project zinc_atoms_bonds_interleaved
```

---

## Troubleshooting

### Model Shows NaN Losses
**Cause**: Likely token ID out of range for embedding layer

**Fix**: Ensure vocab size is calculated as `max(token_ids) + 1`, not just `len(vocab_dict)`

### Sequences Truncated
**Check**: `max_seq_length` parameter (default 512)

**Fix**: Increase if graphs are particularly large:
```bash
--max_seq_length 1024
```

### Memory Issues
**Tip**: Interleaved edges uses ~323 vocab tokens vs 219 for interspersed

**Fix**: Use atoms-only (`--use_autograph_interspersed`) if memory-constrained

---

## Technical Details

### AutoGraph Trail Structure

AutoGraph generates random walks using a depth-first search with backtracking:

```
trail = [start_node, n1, n2, ..., backtrack, n3, n4, ..., end]
```

Special markers indicate backtracking and structure. Our tokenization preserves this entire structure while adding chemical information.

### Embedding Layer Configuration

For transformers using these tokenizations:

```python
# Minimum embedding vocabulary
vocab_size = max(all_token_ids_in_dataset) + 1

# For interleaved edges: ~323 minimum
# For interspersed atoms: ~219 minimum
# For topology-only: ~39 minimum (+ max_nodes)
```

The embedding layer will have unused tokens for lower approaches (e.g., atoms-only won't use bond tokens), but this is fine and doesn't hurt performance.

---

## References

- **AutoGraph Paper**: Graph2Vec tokenization for molecular graphs
- **ZINC Dataset**: A Free Database of Commercially Available Compounds for Virtual Screening
- **Bond Types**: Standard SMILES bond notation (single, double, triple, aromatic)

---

## FAQ

**Q: Which approach should I use for new projects?**
A: Start with `--use_autograph_interleaved_edges` for maximum information. If memory/inference speed is critical, fall back to `--use_autograph_interspersed`.

**Q: Can I combine multiple flags?**
A: No, only one AutoGraph flag at a time. Precedence: `interleaved_edges` > `interspersed` > `with_features` > base `autograph`.

**Q: What if my molecules have aromatic bonds?**
A: ZINC includes aromatic bonds as `bond_4`. They're automatically handled by the interleaving logic.

**Q: How are disconnected components handled?**
A: AutoGraph will generate separate trails for each component. The tokenization preserves this structure.

**Q: Can I use this with other datasets?**
A: Yes, any PyG dataset with `x` (node features) and `edge_index`, `edge_attr` (connectivity). ZINC is just the tested case.
