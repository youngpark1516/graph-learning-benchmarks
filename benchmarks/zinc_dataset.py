"""ZINC 12 dataset loader for molecular property prediction.

Provides dataset loaders compatible with MPNN, Transformer, and GraphGPS models.
Uses PyTorch Geometric's ZINC dataset with automatic downloading and caching.
Supports both custom tokenization and AutoGraph trail-based tokenization.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
from torch_geometric.datasets import ZINC
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch.utils.data import Dataset
import os
import sys
import numpy as np
import networkx as nx


class ZINCDataset(InMemoryDataset):
    """Wrapper around PyTorch Geometric's ZINC dataset with caching.
    
    This dataset wraps PyG Data objects and their labels for compatibility
    with the benchmarking framework. The __getitem__ returns a tuple (data, label).
    """
    
    def __init__(
        self,
        root: str = "./data/ZINC",
        split: str = "train",
        subset: bool = True,
        transform=None,
        pre_transform=None,
    ):
        """
        Args:
            root: Root directory for dataset caching
            split: One of 'train', 'val', 'test'
            subset: If True, use 12k version; else use full 250k
            transform: Data transformation
            pre_transform: Pre-processing transformation
        """
        self.split_name = split
        self.subset = subset
        self._split = "train" if split == "train" else ("val" if split == "val" else "test")
        super().__init__(root, transform, pre_transform)
        
        # Load from cache
        if split == "train":
            self.pyg_dataset = torch.load(self.processed_paths[0])
        elif split == "val":
            self.pyg_dataset = torch.load(self.processed_paths[1])
        else:  # test
            self.pyg_dataset = torch.load(self.processed_paths[2])
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        subset_str = "12k" if self.subset else "250k"
        return [
            f'zinc_{subset_str}_train.pt',
            f'zinc_{subset_str}_val.pt',
            f'zinc_{subset_str}_test.pt'
        ]
    
    def download(self):
        """PyG handles downloading automatically."""
        pass
    
    def process(self):
        """Process and cache ZINC dataset splits."""
        # Load splits from PyG
        train_dataset = ZINC(self.raw_dir, subset=self.subset, split='train')
        val_dataset = ZINC(self.raw_dir, subset=self.subset, split='val')
        test_dataset = ZINC(self.raw_dir, subset=self.subset, split='test')
        
        print(f"ZINC dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        # Save raw datasets (not collated) for per-sample access
        torch.save(train_dataset, self.processed_paths[0])
        torch.save(val_dataset, self.processed_paths[1])
        torch.save(test_dataset, self.processed_paths[2])
        
        print(f"ZINC splits cached to {Path(self.processed_dir)}")
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx: int):
        """Return PyG Data and label as tuple for collate_fn compatibility."""
        data = self.pyg_dataset[idx]
        label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        return data, label


class ZINCLoader:
    """Unified loader for ZINC dataset across model types."""
    
    @staticmethod
    def load_pyg_data(
        root: str = "./data/ZINC",
        split: str = "train",
        subset: bool = True,
        return_tuple: bool = False,
    ) -> Dataset:
        """Load ZINC as PyG dataset for GNN models (MPNN, GraphGPS).
        
        Args:
            root: Root directory for caching
            split: 'train', 'val', or 'test'
            subset: Use 12k subset if True, else full 250k
            return_tuple: If True, return (data, label) tuples for MPNN collate_fn
            
        Returns:
            PyG-compatible dataset
        """
        zinc_ds = ZINCDataset(root=root, split=split, subset=subset)
        # Always wrap with ZINCGraphDataset to apply one-hot encoding
        return ZINCGraphDataset(zinc_ds, return_tuple=return_tuple)
    
    @staticmethod
    def load_tokenized_data(
        root: str = "./data/ZINC",
        split: str = "train",
        subset: bool = True,
        max_seq_len: int = 512,
        use_autograph: bool = False,
        random_seed: int = 1234,
    ) -> Dataset:
        """Load ZINC and tokenize for transformer models.
        
        Args:
            root: Root directory for caching
            split: 'train', 'val', or 'test'
            subset: Use 12k subset if True, else full 250k
            max_seq_len: Maximum sequence length
            use_autograph: If True, use AutoGraph's trail-based tokenization
            random_seed: Random seed for AutoGraph tokenizer
            
        Returns:
            Tokenized dataset with sequence and label
        """
        pyg_dataset = ZINCDataset(root=root, split=split, subset=subset)
        if use_autograph:
            return AutoGraphTokenizedZINCDataset(
                pyg_dataset, 
                max_seq_len=max_seq_len, 
                random_seed=random_seed
            )
        else:
            return TokenizedZINCDataset(pyg_dataset, max_seq_len=max_seq_len)


class TokenizedZINCDataset(Dataset):
    """Tokenize ZINC molecular graphs for transformer models."""
    
    def __init__(self, pyg_dataset: ZINCDataset, max_seq_len: int = 512):
        """
        Args:
            pyg_dataset: ZINCDataset instance
            max_seq_len: Maximum sequence length
        """
        self.pyg_dataset = pyg_dataset
        self.max_seq_len = max_seq_len
        self.token2idx = self._build_vocab()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from atom and bond types.
        
        Format: <bos> u1 bond1 v1 <e> u2 bond2 v2 <e> ... <n> u1 atom1 u2 atom2 ... <eos>
        """
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<e>": 4,  # edge marker
            "<n>": 5,  # node marker (node list section)
            "<q>": 6,  # query marker
            "<p>": 7,  # prediction marker
        }
        idx = 8
        
        # Add atom types (atomic numbers 0-118)
        for i in range(119):
            vocab[f"atom_{i}"] = idx
            idx += 1
        
        # Add bond types (ZINC uses numeric: 1=single, 2=double, 3=triple)
        for bond_type in range(1, 5):
            vocab[f"bond_{bond_type}"] = idx
            idx += 1
        
        # Add node indices (for edge endpoints and node list)
        for node_id in range(100):  # Support up to 100 node IDs
            vocab[f"node_{node_id}"] = idx
            idx += 1
        
        return vocab
    
    def _tokenize_graph(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert PyG graph to token sequence using graph-token format.
        
        Format: <bos> u1 bond1 v1 <e> u2 bond2 v2 <e> ... <n> u1 atom1 u2 atom2 ... <eos>
        
        For undirected graphs, each bond is encoded only once (canonical form: min(u,v) max(u,v))
        
        Args:
            data: PyG Data object
            
        Returns:
            Tuple of (input_ids, attention_mask)
        """
        tokens = [self.token2idx["<bos>"]]
        
        # Add edges with bond features (undirected: encode each bond once)
        # Format: u bond_type v <e> u bond_type v <e> ...
        if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.shape[1] > 0:
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            
            # Track edges we've already added (canonical form)
            added_edges = set()
            
            for edge_idx in range(edge_index.shape[1]):
                u = edge_index[0, edge_idx].item()
                v = edge_index[1, edge_idx].item()
                
                # For undirected graphs, use canonical form (min, max) to avoid duplicates
                canonical_edge = (min(u, v), max(u, v))
                if canonical_edge in added_edges:
                    continue
                added_edges.add(canonical_edge)
                
                # Get node ID tokens
                u_token = self.token2idx.get(f"node_{u}", self.token2idx["<unk>"])
                v_token = self.token2idx.get(f"node_{v}", self.token2idx["<unk>"])
                
                # Get bond type
                if edge_attr is not None:
                    bond_type = edge_attr[edge_idx].item()
                    bond_token = self.token2idx.get(f"bond_{bond_type}", self.token2idx["<unk>"])
                else:
                    bond_token = self.token2idx.get("bond_1", self.token2idx["<unk>"])
                
                # Add: u bond v <e>
                tokens.extend([u_token, bond_token, v_token, self.token2idx["<e>"]])
        
        # Add node marker and node features
        # Format: <n> u atom_u v atom_v ...
        tokens.append(self.token2idx["<n>"])
        
        if hasattr(data, 'x') and data.x is not None:
            for node_id, atom_feat in enumerate(data.x):
                # Node ID token
                node_token = self.token2idx.get(f"node_{node_id}", self.token2idx["<unk>"])
                tokens.append(node_token)
                
                # Atom type token
                atom_id = atom_feat.item() if atom_feat.dim() == 0 else atom_feat[0].item()
                atom_token = self.token2idx.get(f"atom_{atom_id}", self.token2idx["<unk>"])
                tokens.append(atom_token)
        
        # End token
        tokens.append(self.token2idx["<eos>"])
        
        # Truncate and pad
        tokens = tokens[:self.max_seq_len]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        if len(input_ids) < self.max_seq_len:
            padding = torch.full(
                (self.max_seq_len - len(input_ids),),
                self.token2idx["<pad>"],
                dtype=torch.long
            )
            input_ids = torch.cat([input_ids, padding])
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.token2idx["<pad>"]] = 0
        
        return input_ids, attention_mask
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary for sharing across splits."""
        return self.token2idx
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get tokenized sample with label.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with 'input_ids', 'attention_mask', 'label'
        """
        # Get the data - may be tuple (data, label) from ZINCDataset
        item = self.pyg_dataset[idx]
        if isinstance(item, tuple):
            data, label = item
        else:
            data = item
            label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        
        input_ids, attention_mask = self._tokenize_graph(data)
        
        # Regression target (molecular property)
        if isinstance(label, torch.Tensor):
            label = label.item() if label.dim() == 0 else label[0].item()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(float(label), dtype=torch.float),
        }


class AutoGraphTokenizedZINCDataset(Dataset):
    """Tokenize ZINC molecular graphs using AutoGraph's trail-based tokenizer."""
    
    def __init__(self, pyg_dataset: ZINCDataset, max_seq_len: int = 512, random_seed: int = 1234):
        """
        Args:
            pyg_dataset: ZINCDataset instance
            max_seq_len: Maximum sequence length
            random_seed: Random seed for trail generation
        """
        self.pyg_dataset = pyg_dataset
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Initialize AutoGraph tokenizer
        try:
            _GRAPH_TOKEN_DIR = Path(__file__).parent.parent / 'submodules' / 'graph-token'
            sys.path.insert(0, str(_GRAPH_TOKEN_DIR))
            _AUTOGRAPH_DIR = Path(__file__).parent.parent / 'submodules' / 'autograph'
            sys.path.insert(0, str(_AUTOGRAPH_DIR))
            
            from autograph.datamodules.data.tokenizer import Graph2TrailTokenizer
            self.Graph2TrailTokenizer = Graph2TrailTokenizer
            self.tokenizer_available = True
        except ImportError as e:
            print(f"Warning: Could not import AutoGraph tokenizer: {e}")
            print("Falling back to simple tokenization")
            self.tokenizer_available = False
        
        # Build vocabulary from AutoGraph if available
        self.token2idx = self._build_vocab()
        self._init_autograph_tokenizer()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary for AutoGraph tokenization.
        
        AutoGraph uses token IDs 0-5 for special tokens and 6+ for node indices.
        We need to support node IDs up to the maximum graph size.
        """
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        # Note: AutoGraph uses IDs 0-5 for special tokens internally,
        # and node indices start at idx_offset=6. We'll let transformers
        # handle the expanded vocabulary dynamically.
        return vocab
    
    def _init_autograph_tokenizer(self) -> None:
        """Initialize AutoGraph tokenizer with sample graphs."""
        if not self.tokenizer_available:
            self.autograph_tokenizer = None
            return
        
        try:
            seed = self.rng.randint(0, 2**32 - 1)
            self.autograph_tokenizer = self.Graph2TrailTokenizer(
                max_length=self.max_seq_len,
                labeled_graph=False,
                undirected=True,
                append_eos=True,
                rng=seed
            )
            
            # Sample graphs to determine max nodes
            max_nodes = 0
            sample_size = min(100, len(self.pyg_dataset))
            for i in range(sample_size):
                data = self.pyg_dataset.pyg_dataset[i]
                n = data.num_nodes if hasattr(data, 'num_nodes') else (data.x.shape[0] if hasattr(data, 'x') else 0)
                max_nodes = max(max_nodes, n)
            
            self.autograph_tokenizer.set_num_nodes(max_nodes)
            
            # Build extended vocabulary to include all possible node IDs
            # AutoGraph uses: 0=sos, 1=reset, 2=ladj, 3=radj, 4=eos, 5=pad, 6+=node_indices
            idx_offset = 6  # AutoGraph's idx_offset
            for node_id in range(max_nodes):
                token_id = idx_offset + node_id
                self.token2idx[f"node_{node_id}"] = token_id
                # Also add raw token IDs for direct mapping
                if token_id not in self.token2idx.values():
                    # Create reverse mapping so we can recognize these IDs
                    pass
            
        except Exception as e:
            print(f"Failed to initialize AutoGraph tokenizer: {e}")
            self.autograph_tokenizer = None
    
    def _pyg_to_nx(self, data: Data) -> nx.Graph:
        """Convert PyG Data to NetworkX graph."""
        g = nx.Graph()
        
        # Add nodes
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else (data.x.shape[0] if hasattr(data, 'x') else 0)
        g.add_nodes_from(range(num_nodes))
        
        # Add edges
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            edges = data.edge_index.t().cpu().numpy()
            g.add_edges_from(edges)
        
        return g
    
    def _pyg_to_pyg_for_autograph(self, data: Data) -> Data:
        """Ensure PyG data is in correct format for AutoGraph tokenizer."""
        if not hasattr(data, 'num_nodes'):
            data.num_nodes = data.x.shape[0] if hasattr(data, 'x') else 0
        return data
    
    def _tokenize_graph_autograph(self, data: Data) -> torch.Tensor:
        """Tokenize using AutoGraph tokenizer."""
        if self.autograph_tokenizer is None:
            return self._tokenize_graph_fallback(data)
        
        try:
            # Prepare data for AutoGraph
            data_prepared = self._pyg_to_pyg_for_autograph(data)
            
            # Tokenize using AutoGraph
            tokens = self.autograph_tokenizer.tokenize(data_prepared)
            
            # Ensure it's a tensor
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            
            return tokens
        except Exception as e:
            print(f"AutoGraph tokenization failed: {e}, falling back to simple tokenization")
            return self._tokenize_graph_fallback(data)
    
    def _tokenize_graph_fallback(self, data: Data) -> torch.Tensor:
        """Fallback simple tokenization if AutoGraph fails."""
        tokens = [self.token2idx["<bos>"]]
        
        # Add node count
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else (data.x.shape[0] if hasattr(data, 'x') else 0)
        tokens.extend([2 + (num_nodes % 100)] * min(5, num_nodes))  # Simple node representation
        
        # Add edge count indicator
        num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') and data.edge_index is not None else 0
        tokens.append(3 + (num_edges % 50))
        
        tokens.append(self.token2idx["<eos>"])
        
        return torch.tensor(tokens[:self.max_seq_len], dtype=torch.long)
    
    def _pad_and_mask(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad tokens and create attention mask."""
        token_len = len(tokens)
        
        if token_len >= self.max_seq_len:
            input_ids = tokens[:self.max_seq_len]
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
        else:
            padding = torch.full(
                (self.max_seq_len - token_len,),
                self.token2idx["<pad>"],
                dtype=torch.long
            )
            input_ids = torch.cat([tokens, padding])
            attention_mask = torch.ones(token_len, dtype=torch.long)
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(self.max_seq_len - token_len, dtype=torch.long)
            ])
        
        return input_ids, attention_mask
    
    def get_vocab(self) -> Dict[str, int]:
        """Return vocabulary for sharing across splits."""
        return self.token2idx
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get AutoGraph-tokenized sample with label.
        
        Args:
            idx: Sample index
            
        Returns:
            Dict with 'input_ids', 'attention_mask', 'label'
        """
        # Get the data - may be tuple (data, label) from ZINCDataset
        item = self.pyg_dataset[idx]
        if isinstance(item, tuple):
            data, label = item
        else:
            data = item
            label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        
        # Tokenize using AutoGraph
        tokens = self._tokenize_graph_autograph(data)
        
        # Pad and create mask
        input_ids, attention_mask = self._pad_and_mask(tokens)
        
        # Regression target (molecular property)
        if isinstance(label, torch.Tensor):
            label = label.item() if label.dim() == 0 else label[0].item()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(float(label), dtype=torch.float),
        }


class ZINCGraphDataset(Dataset):
    """Wrapper for PyG-compatible access to ZINC (returns Data objects with y attached)."""
    
    def __init__(self, zinc_dataset, return_tuple: bool = False):
        """
        Args:
            zinc_dataset: ZINCDataset instance
            return_tuple: If True, return (data, label) tuples; else attach label to Data
        """
        self.zinc_dataset = zinc_dataset
        self.return_tuple = return_tuple
    
    def __len__(self):
        return len(self.zinc_dataset)
    
    def __getitem__(self, idx: int):
        """Get sample - either as tuple or with y attached to Data."""
        data = self.zinc_dataset.pyg_dataset[idx]
        
        # Convert node features to one-hot encoding
        if hasattr(data, 'x') and data.x is not None:
            # Node features are [num_nodes, 1] with atomic numbers 0-118
            # Convert to one-hot: [num_nodes, 119]
            atomic_nums = data.x.squeeze(-1).long()  # [num_nodes]
            num_nodes = atomic_nums.shape[0]
            
            # Create one-hot encoding with 119 dimensions (atomic numbers 0-118)
            one_hot = torch.zeros(num_nodes, 119, dtype=torch.float32, device=data.x.device)
            one_hot.scatter_(1, atomic_nums.unsqueeze(1), 1.0)
            
            data.x = one_hot
        
        label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        
        if self.return_tuple:
            return data, label
        else:
            # Attach label to data for PyG DataLoader
            data.y = label if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.float)
            return data
