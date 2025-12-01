"""Unified dataset loader for all benchmark models.

This module provides a flexible dataset loader that works with various directory structures
and can be used by MPNN, transformers, and other models.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import networkx as nx
import torch
from torch.utils.data import Dataset


class UnifiedGraphDataset(Dataset):
    """Unified dataset for loading graph tasks with flexible path resolution."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
    ):
        """
        Initialize dataset with flexible path resolution.
        
        Args:
            data_path: Can be:
                - Direct path to split dir: /path/to/algorithm/train
                - Path to algorithm dir: /path/to/algorithm (will append split)
                - Path to task dir: /path/to/task (will append algorithm/split)
                - Path to base dir: /path/to/base (will append tasks_autograph/task/algorithm/split)
            split: Data split ('train', 'valid', 'test')
            task: Task name (optional, used for path construction)
            algorithm: Algorithm name (optional, used for path construction)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.task = task
        self.algorithm = algorithm
        
        # Try to locate the actual data directory
        self.split_dir = self._resolve_split_directory()
        
        # Load all JSON files from the directory
        self.samples = self._load_samples()
        
    def _resolve_split_directory(self) -> Path:
        """Intelligently resolve the split directory from various possible paths."""
        candidates = []
        
        # Direct path (already points to split dir)
        if (self.data_path / "0.json").exists() or any(self.data_path.glob("*.json")):
            candidates.append(self.data_path)
        
        # Path + split
        split_path = self.data_path / self.split
        if split_path.exists():
            candidates.append(split_path)
        
        # Path + algorithm + split
        if self.algorithm:
            algo_split = self.data_path / self.algorithm / self.split
            if algo_split.exists():
                candidates.append(algo_split)
        
        # Path + task + algorithm + split
        if self.task and self.algorithm:
            task_algo_split = self.data_path / self.task / self.algorithm / self.split
            if task_algo_split.exists():
                candidates.append(task_algo_split)
        
        # Path + tasks_autograph + task + algorithm + split (legacy structure)
        if self.task and self.algorithm:
            legacy = self.data_path / "tasks_autograph" / self.task / self.algorithm / self.split
            if legacy.exists():
                candidates.append(legacy)
        
        # Return first valid candidate
        for candidate in candidates:
            json_files = list(candidate.glob("*.json"))
            if json_files:
                print(f"Using data directory: {candidate}")
                return candidate
        
        # If no candidates found, raise error with helpful message
        tried_paths = "\n  - ".join(str(c) for c in candidates) if candidates else "No paths tried"
        raise ValueError(
            f"Could not find data split '{self.split}' for path '{self.data_path}'\n"
            f"Tried:\n  - {tried_paths}\n"
            f"Please check that JSON files exist in one of these locations."
        )
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from JSON files in the split directory."""
        samples = []
        json_files = sorted(self.split_dir.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {self.split_dir}")
        
        for json_file in json_files:
            with json_file.open() as f:
                data = json.load(f)
                # Handle both list format and dict format
                if isinstance(data, list):
                    samples.extend(data)
                elif isinstance(data, dict):
                    samples.append(data)
        
        print(f"Loaded {len(samples)} samples from {len(json_files)} files in {self.split_dir}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return raw sample - subclasses can override to parse specific formats."""
        return self.samples[idx]


class TokenizedGraphDataset(UnifiedGraphDataset):
    """Dataset that parses tokenized graph format for transformers."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        max_seq_len: int = 512,
    ):
        super().__init__(data_path, split, task, algorithm)
        self.max_seq_len = max_seq_len
        self.vocab = vocab
        self.token2idx = None
        self.idx2token = None
        self.label_vocab = None
        
        if vocab is None:
            self._build_vocabulary()
        else:
            self.token2idx = vocab
            self.idx2token = {v: k for k, v in vocab.items()}
    
    def _build_vocabulary(self):
        """Build vocabulary from all samples."""
        tokens = set()
        labels = set()
        
        for sample in self.samples:
            text = sample.get("text", "")
            token_list = text.split()
            
            # Find prediction token (after <p>)
            if "<p>" in token_list:
                p_idx = token_list.index("<p>")
                input_tokens = token_list[:p_idx]
                if p_idx + 1 < len(token_list):
                    label_token = token_list[p_idx + 1]
                    labels.add(label_token)
                tokens.update(input_tokens)
            else:
                tokens.update(token_list)
        
        # Build token vocab with special tokens
        special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<e>", "<n>", "<q>", "<p>"]
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        for token in sorted(tokens):
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
        
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        
        # Build label vocab
        self.label_vocab = {label: idx for idx, label in enumerate(sorted(labels))}
        
        print(f"Built vocabulary: {len(self.token2idx)} tokens, {len(self.label_vocab)} labels")
    
    def get_vocab(self):
        """Return vocabulary for sharing across splits."""
        return self.token2idx
    
    def set_vocab(self, vocab: Dict[str, int], label_vocab: Optional[Dict[str, int]] = None):
        """Set vocabulary from another dataset."""
        self.token2idx = vocab
        self.idx2token = {v: k for k, v in vocab.items()}
        if label_vocab:
            self.label_vocab = label_vocab
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return tokenized input and label."""
        sample = self.samples[idx]
        text = sample.get("text", "")
        token_list = text.split()
        
        # Split at <p> marker
        if "<p>" in token_list:
            p_idx = token_list.index("<p>")
            input_tokens = token_list[:p_idx]
            label_token = token_list[p_idx + 1] if p_idx + 1 < len(token_list) else "<unk>"
        else:
            input_tokens = token_list
            label_token = "<unk>"
        
        # Convert to indices
        input_ids = [self.token2idx.get(token, self.token2idx["<unk>"]) for token in input_tokens]
        
        # Truncate or pad
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [self.token2idx["<pad>"]] * (self.max_seq_len - len(input_ids))
        
        # Get label index
        if label_token in self.label_vocab:
            label_idx = self.label_vocab[label_token]
        else:
            label_idx = 0  # Default to first label if not found
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label_idx, dtype=torch.long)


class PyGGraphDataset(UnifiedGraphDataset):
    """Dataset that parses graphs into PyTorch Geometric format for GNNs."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        add_query_features: bool = True,
    ):
        super().__init__(data_path, split, task, algorithm)
        self.add_query_features = add_query_features
        self.task_name = task
        
        # Parse all samples
        self.graphs = []
        self.labels = []
        self.query_nodes = []
        self._parse_samples()
    
    def _parse_samples(self):
        """Parse samples into graphs and labels."""
        for sample in self.samples:
            text = sample.get("text", "")
            token_list = text.split()
            
            graph, query_nodes, label = self._parse_tokens(token_list)
            self.graphs.append(graph)
            self.query_nodes.append(query_nodes)
            self.labels.append(label)
    
    def _parse_tokens(self, tokens: List[str]) -> Tuple[nx.Graph, Optional[Tuple[int, int]], float]:
        """Parse tokenized graph into NetworkX graph, query nodes, and label."""
        graph = nx.Graph()
        idx = 0
        
        # Skip <bos>
        if idx < len(tokens) and tokens[idx] == "<bos>":
            idx += 1
        
        # Parse edges (before <n>)
        while idx < len(tokens) and tokens[idx] not in ["<n>", "<q>", "<p>"]:
            token = tokens[idx]
            
            if token.startswith("<"):
                idx += 1
                continue
            
            # Try to parse edge
            try:
                if idx + 1 < len(tokens):
                    u = int(token)
                    v_token = tokens[idx + 1]
                    if not v_token.startswith("<"):
                        v = int(v_token)
                        graph.add_edge(u, v)
                        idx += 2
                        continue
            except ValueError:
                pass
            
            idx += 1
        
        # Parse query nodes for shortest_path task
        query_nodes = None
        if self.add_query_features and "<q>" in tokens:
            try:
                q_idx = tokens.index("<q>")
                # Format: <q> shortest_distance u v
                if q_idx + 3 < len(tokens) and tokens[q_idx + 1].startswith("shortest"):
                    u = int(tokens[q_idx + 2])
                    v = int(tokens[q_idx + 3])
                    query_nodes = (u, v)
            except (ValueError, IndexError):
                pass
        
        # Parse label (after <p>)
        label = 0.0
        if "<p>" in tokens:
            p_idx = tokens.index("<p>")
            if p_idx + 1 < len(tokens):
                label_token = tokens[p_idx + 1]
                # Try to extract numeric value from token like "len3" -> 3
                if label_token.startswith("len"):
                    try:
                        label = float(label_token[3:])
                    except ValueError:
                        label = 0.0
                else:
                    try:
                        label = float(label_token)
                    except ValueError:
                        label = 0.0
        
        return graph, query_nodes, label
    
    def _nx_to_pyg(self, graph: nx.Graph, query_nodes: Optional[Tuple[int, int]] = None):
        """Convert NetworkX graph to PyTorch Geometric Data object."""
        from torch_geometric.data import Data
        
        # Build node features
        node_features = []
        nodes = sorted(graph.nodes())
        
        if len(nodes) == 0:
            # Empty graph fallback
            feature_dim = 3 if query_nodes else 1
            x = torch.zeros((1, feature_dim), dtype=torch.float)
            edge_index = torch.tensor([[], []], dtype=torch.long)
            return Data(x=x, edge_index=edge_index)
        
        # Create mapping from original node indices to contiguous indices [0, 1, 2, ...]
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        for node in nodes:
            features = [float(graph.degree(node))]
            
            # Add query encoding for shortest_path
            if query_nodes is not None and self.add_query_features:
                query_u, query_v = query_nodes
                is_source = 1.0 if node == query_u else 0.0
                is_target = 1.0 if node == query_v else 0.0
                features.extend([is_source, is_target])
            
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Build edge index with remapped node indices
        edge_list = []
        for u, v in graph.edges():
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])  # Undirected graph
        
        if len(edge_list) == 0:
            edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=x, edge_index=edge_index)
    
    def __getitem__(self, idx: int):
        """Return PyG Data object and label."""
        graph = self.graphs[idx]
        label = self.labels[idx]
        query_nodes = self.query_nodes[idx]
        
        data = self._nx_to_pyg(graph, query_nodes)
        return data, torch.tensor(label, dtype=torch.float)
