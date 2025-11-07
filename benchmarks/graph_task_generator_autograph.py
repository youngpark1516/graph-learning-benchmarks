#!/usr/bin/env python3
# coding=utf-8

r"""Generate task-specific tokenized graph samples using AutoGraph's tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json
import sys

from absl import app
from absl import flags
from absl import logging
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

# Add AutoGraph to path
sys.path.append(str(Path(__file__).parent.parent / 'submodules' / 'autograph'))
from autograph.datamodules.data.tokenizer import Graph2TrailTokenizer

# Ensure graph-token utilities are importable. The folder name contains a hyphen
# ("graph-token") so it cannot be imported as a package; add its directory
# to sys.path so `import graph_task` works.
_GRAPH_TOKEN_DIR = Path(__file__).parent.parent / 'submodules' / 'graph-token'
sys.path.insert(0, str(_GRAPH_TOKEN_DIR))

from graph_task import EdgeExistence, NodeDegree, NodeCount, EdgeCount
from graph_task import ConnectedNodes, CycleCheck, DisconnectedNodes
from graph_task import Reachability, ShortestPath, MaximumFlow
from graph_task import TriangleCounting, NodeClassification
import graph_task_utils as utils

_TASK = flags.DEFINE_enum(
    "task",
    None,
    [
        "edge_existence",
        "node_degree",
        "node_count",
        "edge_count",
        "connected_nodes",
        "cycle_check",
        "disconnected_nodes",
        "reachability",
        "shortest_path",
        "maximum_flow",
        "triangle_counting",
        "node_classification",
    ],
    "The task to generate datapoints.",
    required=True,
)
_ALGORITHM = flags.DEFINE_enum(
    "algorithm",
    None,
    ["er", "ba", "sbm", "sfn", "complete", "star", "path", "all"],
    "The graph generator algorithm(s) to read from.",
    required=True,
)
_TASK_DIR = flags.DEFINE_string("task_dir", None, "Root directory to write tasks.", required=True)
_GRAPHS_DIR = flags.DEFINE_string("graphs_dir", None, "Root directory containing graphs.", required=True)
_SPLIT = flags.DEFINE_enum("split", "test", ["train", "valid", "test"], "Which split to read from.")
_RANDOM_SEED = flags.DEFINE_integer("random_seed", 1234, "Random seed for any sampling.")

TASK_CLASS = {
    'edge_existence': EdgeExistence,
    'node_degree': NodeDegree,
    'node_count': NodeCount,
    'edge_count': EdgeCount,
    'connected_nodes': ConnectedNodes,
    'cycle_check': CycleCheck,
    'disconnected_nodes': DisconnectedNodes,
    'reachability': Reachability,
    'shortest_path': ShortestPath,
    'maximum_flow': MaximumFlow,
    'triangle_counting': TriangleCounting,
    'node_classification': NodeClassification,
}

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _graph_id_from_index(algorithm: str, split: str, idx: int) -> str:
    return f"{algorithm}_{split}_{idx}"

def _regenerate_sbm_graphs_like(graphs, rng: np.random.RandomState):
    """Regenerate SBM graphs of similar size (for NodeClassification)."""
    regenerated = []
    for g in graphs:
        n = g.number_of_nodes()
        if n < 4:
            n = 6
        sizes = [n // 2, n - n // 2]
        p_in = rng.uniform(0.6, 0.8)
        p_out = rng.uniform(0.0, 0.05)
        probs = [[p_in, p_out], [p_out, p_in]]
        sbm = nx.stochastic_block_model(sizes, probs, seed=rng)
        regenerated.append(sbm)
    return regenerated

def _write_json(file_path: Path, records: list[dict]) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

def convert_nx_to_pyg(g: nx.Graph) -> Data:
    """Convert NetworkX graph to PyG Data object."""
    num_nodes = g.number_of_nodes()
    
    # Ensure nodes are numbered consecutively from 0 to n-1
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    
    # Handle empty graphs
    if g.number_of_edges() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        # Sort edges for consistency
        edges = sorted(g.edges())
        # Create edge index tensor ensuring proper node indexing
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # For undirected graphs, add reverse edges
        if not g.is_directed():
            edge_index_rev = edge_index.flip(0)
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    
    # Create and return PyG Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    # Ensure edge_index is coalesced
    if data.edge_index.numel() > 0:
        data = data.coalesce()
    
    return data

def tokenize_graph(g: nx.Graph, tokenizer: Graph2TrailTokenizer) -> torch.Tensor:
    """Convert a graph to tokens using AutoGraph's tokenizer."""
    # Convert to PyG format
    data = convert_nx_to_pyg(g)
    # Tokenize
    return tokenizer.tokenize(data)

class AutoGraphTokenizer:
    """Wrapper for Graph2TrailTokenizer to maintain compatibility with task classes."""
    def __init__(self, rng: np.random.RandomState):
        # Convert numpy RandomState to integer seed for consistency
        seed = rng.randint(0, 2**32 - 1)
        self.tokenizer = Graph2TrailTokenizer(
            max_length=-1,  # No maximum length
            labeled_graph=False,  # Our graphs are unlabeled
            undirected=True,  # Assuming undirected graphs
            append_eos=True,  # Add end of sequence token
            rng=seed  # Pass integer seed instead of RandomState
        )
        self.max_nodes = 0

    def update_max_nodes(self, graphs: list[nx.Graph]) -> None:
        """Update maximum number of nodes for the tokenizer."""
        for g in graphs:
            n = g.number_of_nodes()
            if n > self.max_nodes:
                self.max_nodes = n
        self.tokenizer.set_num_nodes(self.max_nodes)

    def tokenize(self, g: nx.Graph) -> torch.Tensor:
        """Tokenize a single graph."""
        return tokenize_graph(g, self.tokenizer)

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    logging.set_verbosity(logging.INFO)

    rng = np.random.RandomState(_RANDOM_SEED.value)

    algorithms = (
        ["er", "ba", "sbm", "sfn", "complete", "star", "path"]
        if _ALGORITHM.value == "all"
        else [_ALGORITHM.value]
    )

    # Load graphs per algorithm & split
    graphs: list[nx.Graph] = []
    algs_for_graph: list[str] = []
    for alg in algorithms:
        loaded = utils.load_graphs(_GRAPHS_DIR.value, alg, _SPLIT.value)
        graphs.extend(loaded)
        algs_for_graph.extend([alg] * len(loaded))
        logging.info("Loaded %d graph(s) for algorithm=%s split=%s", len(loaded), alg, _SPLIT.value)

    if not graphs:
        raise app.UsageError(
            f"No graphs found in {_GRAPHS_DIR.value} for algorithms={algorithms} split={_SPLIT.value}"
        )

    # Initialize AutoGraph tokenizer
    auto_tokenizer = AutoGraphTokenizer(rng)
    auto_tokenizer.update_max_nodes(graphs)
    logging.info("Initialized AutoGraph tokenizer with max_nodes=%d", auto_tokenizer.max_nodes)

    # Instantiate task
    TaskCls = TASK_CLASS[_TASK.value]
    task = TaskCls()

    # Special case: node_classification requires SBM regeneration
    if _TASK.value == "node_classification":
        graphs = _regenerate_sbm_graphs_like(graphs, rng)

    # Create a separate directory structure for AutoGraph tokenizations
    base_dir = Path(_TASK_DIR.value).parent / "tasks_autograph"  # Creates a parallel directory to 'tasks'
    out_root = base_dir / _TASK.value
    _ensure_dir(out_root)

    total_graphs = 0
    total_samples = 0

    for idx, (g, alg) in enumerate(zip(graphs, algs_for_graph)):
        graph_id = _graph_id_from_index(alg, _SPLIT.value, idx)
        
        # Get task-specific tokenization using original method first
        task_map = task.tokenize_graph(g, graph_id)
        samples = task_map[graph_id]
        
        # For each sample, get the task-specific tokenization
        tokens_list = []
        try:
            # Tokenize the base graph once
            base_tokens = auto_tokenizer.tokenize(g)
            # Use the same tokens for all samples since EdgeExistence doesn't modify the graph
            for _ in samples:
                tokens_list.append(base_tokens.tolist())  # Convert tensor to list for JSON serialization
        except Exception as e:
            logging.warning(f"Failed to tokenize graph {graph_id}: {str(e)}")
            # Use empty token list as fallback
            tokens_list.extend([[] for _ in samples])
        
        # Create records with task-specific format
        records = [
            {
                "graph_id": graph_id,
                "text": sample,  # Keep original text format
                "tokens": tokens  # Add AutoGraph tokens
            }
            for sample, tokens in zip(samples, tokens_list)
        ]

        out_dir = out_root / alg / _SPLIT.value
        _ensure_dir(out_dir)
        out_fp = out_dir / f"{graph_id}.json"
        _write_json(out_fp, records)

        total_graphs += 1
        total_samples += len(records)

    logging.info(
        "Task=%s | Split=%s | Algorithms=%s | %d graphs | %d samples | Output=%s",
        _TASK.value,
        _SPLIT.value,
        algorithms,
        total_graphs,
        total_samples,
        out_root,
    )

if __name__ == "__main__":
    app.run(main)
