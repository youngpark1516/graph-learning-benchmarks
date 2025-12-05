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
        if split == "train":
            self.pyg_dataset = torch.load(self.processed_paths[0])
        elif split == "val":
            self.pyg_dataset = torch.load(self.processed_paths[1])
        else:
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
        train_dataset = ZINC(self.raw_dir, subset=self.subset, split='train')
        val_dataset = ZINC(self.raw_dir, subset=self.subset, split='val')
        test_dataset = ZINC(self.raw_dir, subset=self.subset, split='test')
        
        print(f"ZINC dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
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
    
    @staticmethod
    def load_tokenized_data(
        root: str = "./data/ZINC",
        split: str = "train",
        subset: bool = True,
        max_seq_len: int = 512,
        use_autograph: bool = False,
        use_autograph_with_features: bool = False,
        use_autograph_interspersed: bool = False,
        use_autograph_interleaved_edges: bool = False,
        random_seed: int = 1234,
    ) -> Dataset:
        """Load ZINC and tokenize for transformer models.
        
        Args:
            root: Root directory for caching
            split: 'train', 'val', or 'test'
            subset: Use 12k subset if True, else full 250k
            max_seq_len: Maximum sequence length
            use_autograph: If True, use AutoGraph's trail-based tokenization (topology only)
            use_autograph_with_features: If True, use AutoGraph trail + node features appended
            use_autograph_interspersed: If True, use AutoGraph trail with atoms interspersed after nodes
            use_autograph_interleaved_edges: If True, use AutoGraph trail with atoms AND bonds interleaved
            random_seed: Random seed for AutoGraph tokenizer
            
        Returns:
            Tokenized dataset with sequence and label
        """
        pyg_dataset = ZINCDataset(root=root, split=split, subset=subset)
        
        if use_autograph_interleaved_edges:
            return AutoGraphInterleavedEdgesZINCDataset(
                pyg_dataset, 
                max_seq_len=max_seq_len, 
                random_seed=random_seed
            )
        elif use_autograph_interspersed:
            return AutoGraphInterspersedFeaturesZINCDataset(
                pyg_dataset, 
                max_seq_len=max_seq_len, 
                random_seed=random_seed
            )
        elif use_autograph_with_features:
            return AutoGraphWithNodeFeaturesZINCDataset(
                pyg_dataset, 
                max_seq_len=max_seq_len, 
                random_seed=random_seed,
                use_node_features=True
            )
        elif use_autograph:
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
        """Build vocabulary from atom and bond types."""
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
        
        for i in range(119):
            vocab[f"atom_{i}"] = idx
            idx += 1
        
        for bond_type in range(1, 5):
            vocab[f"bond_{bond_type}"] = idx
            idx += 1
        
        for node_id in range(100):  # Support up to 100 node IDs
            vocab[f"node_{node_id}"] = idx
            idx += 1
        
        return vocab
    
    def _tokenize_graph(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert PyG graph to token sequence."""
        tokens = [self.token2idx["<bos>"]]
        
        # Format: u bond_type v <e> u bond_type v <e> ...
        if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.shape[1] > 0:
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
            
            added_edges = set()
            
            for edge_idx in range(edge_index.shape[1]):
                u = edge_index[0, edge_idx].item()
                v = edge_index[1, edge_idx].item()
                
                # For undirected graphs, use canonical form (min, max) to avoid duplicates
                canonical_edge = (min(u, v), max(u, v))
                if canonical_edge in added_edges:
                    continue
                added_edges.add(canonical_edge)
                
                u_token = self.token2idx.get(f"node_{u}", self.token2idx["<unk>"])
                v_token = self.token2idx.get(f"node_{v}", self.token2idx["<unk>"])
                
                if edge_attr is not None:
                    bond_type = edge_attr[edge_idx].item()
                    bond_token = self.token2idx.get(f"bond_{bond_type}", self.token2idx["<unk>"])
                else:
                    bond_token = self.token2idx.get("bond_1", self.token2idx["<unk>"])
                tokens.extend([u_token, bond_token, v_token, self.token2idx["<e>"]])
        
        tokens.append(self.token2idx["<n>"])
        
        if hasattr(data, 'x') and data.x is not None:
            for node_id, atom_feat in enumerate(data.x):
                node_token = self.token2idx.get(f"node_{node_id}", self.token2idx["<unk>"])
                tokens.append(node_token)
                
                if atom_feat.dim() == 0:
                    atom_id = int(atom_feat.item())
                elif atom_feat.dim() == 1 and atom_feat.shape[0] > 1:
                    nonzero_indices = torch.nonzero(atom_feat)
                    if len(nonzero_indices) > 0:
                        atom_id = int(nonzero_indices.squeeze().item())
                    else:
                        atom_id = 0  # Fallback for all-zero case
                else:
                    # Single-element vector
                    atom_id = int(atom_feat[0].item())
                
                atom_token = self.token2idx.get(f"atom_{atom_id}", self.token2idx["<unk>"])
                tokens.append(atom_token)
        
        tokens.append(self.token2idx["<eos>"])
        tokens = tokens[:self.max_seq_len]
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        if len(input_ids) < self.max_seq_len:
            padding = torch.full(
                (self.max_seq_len - len(input_ids),),
                self.token2idx["<pad>"],
                dtype=torch.long
            )
            input_ids = torch.cat([input_ids, padding])
        
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
        item = self.pyg_dataset[idx]
        if isinstance(item, tuple):
            data, label = item
        else:
            data = item
            label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        
        input_ids, attention_mask = self._tokenize_graph(data)
        
        if isinstance(label, torch.Tensor):
            label = label.item() if label.dim() == 0 else label[0].item()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(float(label), dtype=torch.float),
        }


class AutoGraphTokenizedZINCDataset(Dataset):
    """Tokenize ZINC molecular graphs using AutoGraph's trail-based tokenizer (topology only)."""
    
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
            
            max_nodes = 0
            sample_size = min(100, len(self.pyg_dataset))
            for i in range(sample_size):
                data = self.pyg_dataset.pyg_dataset[i]
                n = data.num_nodes if hasattr(data, 'num_nodes') else (data.x.shape[0] if hasattr(data, 'x') else 0)
                max_nodes = max(max_nodes, n)
            
            self.autograph_tokenizer.set_num_nodes(max_nodes)
            
            idx_offset = 6
            for node_id in range(max_nodes):
                token_id = idx_offset + node_id
                self.token2idx[f"node_{node_id}"] = token_id
                if token_id not in self.token2idx.values():
                    pass
            
        except Exception as e:
            print(f"Failed to initialize AutoGraph tokenizer: {e}")
            self.autograph_tokenizer = None
    
    def _pyg_to_nx(self, data: Data) -> nx.Graph:
        """Convert PyG Data to NetworkX graph."""
        g = nx.Graph()
        
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else (data.x.shape[0] if hasattr(data, 'x') else 0)
        g.add_nodes_from(range(num_nodes))
        
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


class AutoGraphWithNodeFeaturesZINCDataset(Dataset):
    """AutoGraph trail-based tokenization with node features appended (like graph_token format).
    
    Combines:
    - AutoGraph's random walk trail for topology
    - Node feature tokens appended at the end (node ID + atom type pairs)
    """
    
    def __init__(self, pyg_dataset: ZINCDataset, max_seq_len: int = 512, random_seed: int = 1234, 
                 use_node_features: bool = True):
        """
        Args:
            pyg_dataset: ZINCDataset instance
            max_seq_len: Maximum sequence length
            random_seed: Random seed for trail generation
            use_node_features: If True, append node features to trail
        """
        self.pyg_dataset = pyg_dataset
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.use_node_features = use_node_features
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
            self.tokenizer_available = False
        
        # Build vocabulary
        self.token2idx = self._build_vocab()
        self._init_autograph_tokenizer()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary combining AutoGraph tokens + node features.
        
        AutoGraph uses: 0=sos, 1=reset, 2=ladj, 3=radj, 4=eos, 5=pad, 6+=node_indices
        We append: <n> node_id atom_type node_id atom_type ... <eos>
        """
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<n>": 4,  # Node feature section marker
        }
        idx = 5
        
        # Add atom types (atomic numbers 0-118)
        for atom_id in range(119):
            vocab[f"atom_{atom_id}"] = idx
            idx += 1
        
        # Add node indices (support up to 200 nodes for ZINC)
        for node_id in range(200):
            vocab[f"node_{node_id}"] = idx
            idx += 1
        
        return vocab
    
    def _init_autograph_tokenizer(self) -> None:
        """Initialize AutoGraph tokenizer."""
        if not self.tokenizer_available:
            self.autograph_tokenizer = None
            return
        
        try:
            seed = self.rng.randint(0, 2**32 - 1)
            self.autograph_tokenizer = self.Graph2TrailTokenizer(
                max_length=self.max_seq_len - 150,  # Reserve space for node features
                labeled_graph=False,
                undirected=True,
                append_eos=False,  # We'll add our own eos after node features
                rng=seed
            )
            
            # Determine max nodes from sample
            max_nodes = 0
            sample_size = min(100, len(self.pyg_dataset))
            for i in range(sample_size):
                data = self.pyg_dataset.pyg_dataset[i]
                n = data.num_nodes if hasattr(data, 'num_nodes') else (data.x.shape[0] if hasattr(data, 'x') else 0)
                max_nodes = max(max_nodes, n)
            
            self.autograph_tokenizer.set_num_nodes(max_nodes)
            
        except Exception as e:
            print(f"Failed to initialize AutoGraph tokenizer: {e}")
            self.autograph_tokenizer = None
    
    def _tokenize_graph_with_features(self, data: Data) -> torch.Tensor:
        """Tokenize using AutoGraph trail + node features.
        
        Format: [AutoGraph trail] <n> node_0 atom_0 node_1 atom_1 ... <eos>
        """
        if self.autograph_tokenizer is None:
            return self._tokenize_graph_fallback(data)
        
        try:
            # Get AutoGraph trail (topology)
            if not hasattr(data, 'num_nodes'):
                data.num_nodes = data.x.shape[0] if hasattr(data, 'x') else 0
            
            trail_tokens = self.autograph_tokenizer.tokenize(data)
            if not isinstance(trail_tokens, torch.Tensor):
                trail_tokens = torch.tensor(trail_tokens, dtype=torch.long)
            
            # Convert to list for appending
            tokens = trail_tokens.tolist() if isinstance(trail_tokens, torch.Tensor) else list(trail_tokens)
            
            # Append node features if requested
            if self.use_node_features and hasattr(data, 'x') and data.x is not None:
                tokens.append(self.token2idx["<n>"])  # Node feature marker
                
                for node_id, atom_feat in enumerate(data.x):
                    # Extract atom ID from one-hot or scalar
                    if atom_feat.dim() == 0:
                        atom_id = int(atom_feat.item())
                    elif atom_feat.dim() == 1 and atom_feat.shape[0] > 1:
                        # One-hot vector
                        nonzero_indices = torch.nonzero(atom_feat)
                        atom_id = int(nonzero_indices.squeeze().item()) if len(nonzero_indices) > 0 else 0
                    else:
                        # Single-element vector
                        atom_id = int(atom_feat[0].item())
                    
                    # Add node_id and atom_type tokens
                    node_token = self.token2idx.get(f"node_{node_id}", self.token2idx["<unk>"])
                    atom_token = self.token2idx.get(f"atom_{atom_id}", self.token2idx["<unk>"])
                    tokens.extend([node_token, atom_token])
            
            # Add end marker
            tokens.append(self.token2idx["<eos>"])
            
            return torch.tensor(tokens[:self.max_seq_len], dtype=torch.long)
            
        except Exception as e:
            print(f"Hybrid tokenization failed: {e}, falling back to simple tokenization")
            return self._tokenize_graph_fallback(data)
    
    def _tokenize_graph_fallback(self, data: Data) -> torch.Tensor:
        """Fallback tokenization."""
        tokens = [self.token2idx["<bos>"]]
        
        # Simple structure representation
        num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else (data.x.shape[0] if hasattr(data, 'x') else 0)
        num_edges = data.edge_index.shape[1] if hasattr(data, 'edge_index') and data.edge_index is not None else 0
        
        tokens.extend([2] * min(5, num_nodes))
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
        """Return vocabulary."""
        return self.token2idx
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get hybrid-tokenized sample with label."""
        item = self.pyg_dataset[idx]
        if isinstance(item, tuple):
            data, label = item
        else:
            data = item
            label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        
        # Tokenize with features
        tokens = self._tokenize_graph_with_features(data)
        input_ids, attention_mask = self._pad_and_mask(tokens)
        
        if isinstance(label, torch.Tensor):
            label = label.item() if label.dim() == 0 else label[0].item()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(float(label), dtype=torch.float),
        }


class AutoGraphInterspersedFeaturesZINCDataset(Dataset):
    """AutoGraph trail with atom types interspersed right after each node occurrence.
    
    Instead of appending all node features at the end, this interleaves atom types
    into the trail as: when node_i appears in trail, we follow it with atom_i token.
    
    This keeps topological structure while making atomic info contextual.
    """
    
    def __init__(self, pyg_dataset: ZINCDataset, max_seq_len: int = 512, random_seed: int = 1234):
        self.pyg_dataset = pyg_dataset
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Build mapping of node indices to atom IDs first
        self.node_to_atom = {}
        self._build_node_atom_mapping()
        
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
            self.tokenizer_available = False
        
        self.token2idx = self._build_vocab()
        self._init_autograph_tokenizer()
    
    def _build_node_atom_mapping(self) -> None:
        """Build mapping of node indices to atom IDs from sample graphs."""
        try:
            sample_size = min(10, len(self.pyg_dataset))
            for i in range(sample_size):
                item = self.pyg_dataset[i]
                data = item[0] if isinstance(item, tuple) else item
                
                if hasattr(data, 'x') and data.x is not None:
                    for node_id, atom_feat in enumerate(data.x):
                        if node_id not in self.node_to_atom:
                            # Extract atom ID
                            if atom_feat.dim() == 0:
                                atom_id = int(atom_feat.item())
                            elif atom_feat.dim() == 1 and atom_feat.shape[0] > 1:
                                nonzero_indices = torch.nonzero(atom_feat)
                                atom_id = int(nonzero_indices.squeeze().item()) if len(nonzero_indices) > 0 else 0
                            else:
                                atom_id = int(atom_feat[0].item())
                            self.node_to_atom[node_id] = atom_id
        except Exception as e:
            print(f"Failed to build node-atom mapping: {e}")
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary for AutoGraph + interspersed atom types.
        
        AutoGraph uses token IDs 0-5 for special markers and 6+ for node indices.
        We use high IDs for atoms to avoid conflicts:
        - Atoms: 200-318 (119 atomic numbers)
        - Node indices: 6+ (pass-through from AutoGraph)
        """
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        idx = 200  # Start atoms at high ID to avoid conflicts with AutoGraph
        
        # Add atom types (atomic numbers 0-118)
        for atom_id in range(119):
            vocab[f"atom_{atom_id}"] = idx
            idx += 1
        
        # Note: AutoGraph node indices (6+) are passed through unmapped
        return vocab
    
    def _init_autograph_tokenizer(self) -> None:
        """Initialize AutoGraph tokenizer."""
        if not self.tokenizer_available:
            self.autograph_tokenizer = None
            return
        
        try:
            seed = self.rng.randint(0, 2**32 - 1)
            self.autograph_tokenizer = self.Graph2TrailTokenizer(
                max_length=self.max_seq_len - 20,  # Reserve space for atom insertions
                labeled_graph=False,
                undirected=True,
                append_eos=True,
                rng=seed
            )
            
            max_nodes = 0
            sample_size = min(100, len(self.pyg_dataset))
            for i in range(sample_size):
                item = self.pyg_dataset.pyg_dataset[i]
                n = item.num_nodes if hasattr(item, 'num_nodes') else (item.x.shape[0] if hasattr(item, 'x') else 0)
                max_nodes = max(max_nodes, n)
            
            self.autograph_tokenizer.set_num_nodes(max_nodes)
        except Exception as e:
            print(f"Failed to initialize AutoGraph tokenizer: {e}")
            self.autograph_tokenizer = None
    
    def _intersperse_atom_types(self, trail_tokens: list, data: Data) -> torch.Tensor:
        """Insert atom type tokens right after each node occurrence in trail.
        
        AutoGraph trail contains node indices (6+). When we see a node index,
        we follow it with its corresponding atom type token.
        """
        if not (hasattr(data, 'x') and data.x is not None):
            return torch.tensor(trail_tokens[:self.max_seq_len], dtype=torch.long)
        
        node_to_atom = {}
        for node_id, atom_feat in enumerate(data.x):
            if atom_feat.dim() == 0:
                atom_id = int(atom_feat.item())
            elif atom_feat.dim() == 1 and atom_feat.shape[0] > 1:
                nonzero_indices = torch.nonzero(atom_feat)
                atom_id = int(nonzero_indices.squeeze().item()) if len(nonzero_indices) > 0 else 0
            else:
                atom_id = int(atom_feat[0].item())
            node_to_atom[node_id] = atom_id
        
        output_tokens = []
        idx_offset = 6
        
        for token in trail_tokens:
            output_tokens.append(token)
            if token >= idx_offset:
                node_id = token - idx_offset
                if node_id in node_to_atom:
                    atom_id = node_to_atom[node_id]
                    atom_token = self.token2idx.get(f"atom_{atom_id}", self.token2idx["<unk>"])
                    output_tokens.append(atom_token)
            if len(output_tokens) >= self.max_seq_len:
                break
        
        return torch.tensor(output_tokens[:self.max_seq_len], dtype=torch.long)
    
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
        """Return vocabulary."""
        return self.token2idx
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with interspersed atom features."""
        item = self.pyg_dataset[idx]
        if isinstance(item, tuple):
            data, label = item
        else:
            data = item
            label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        
        try:
            # Get AutoGraph trail
            if not hasattr(data, 'num_nodes'):
                data.num_nodes = data.x.shape[0] if hasattr(data, 'x') else 0
            
            trail_tokens = self.autograph_tokenizer.tokenize(data)
            if not isinstance(trail_tokens, torch.Tensor):
                trail_tokens = torch.tensor(trail_tokens, dtype=torch.long)
            
            # Intersperse atom types
            tokens = self._intersperse_atom_types(trail_tokens.tolist(), data)
            
        except Exception as e:
            print(f"Interspersed tokenization failed: {e}")
            tokens = torch.tensor([self.token2idx["<bos>"], self.token2idx["<eos>"]], dtype=torch.long)
        
        input_ids, attention_mask = self._pad_and_mask(tokens)
        
        if isinstance(label, torch.Tensor):
            label = label.item() if label.dim() == 0 else label[0].item()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(float(label), dtype=torch.float),
        }


class AutoGraphInterleavedEdgesZINCDataset(Dataset):
    """AutoGraph trail with atom types and bond types fully interleaved.
    
    Embeds both atoms and bonds contextually:
    - Atoms appear right after each node in trail
    - Bonds appear between consecutive nodes in trail
    
    Format: <bos> node0 atom0 bond(0,1) node1 atom1 bond(1,2) node2 atom2 ... <eos>
    
    This keeps maximum chemical context - model sees "C --[single]-- N --[double]-- O" patterns.
    """
    
    def __init__(self, pyg_dataset: ZINCDataset, max_seq_len: int = 512, random_seed: int = 1234):
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
            self.tokenizer_available = False
        
        self.token2idx = self._build_vocab()
        self._init_autograph_tokenizer()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary: special tokens, atoms, and bonds.
        
        AutoGraph uses token IDs 0-5 for special markers:
          0=sos, 1=reset, 2=ladj, 3=radj, 4=eos, 5=pad
        
        Node indices start at 6+.
        
        We'll use high token IDs for atoms and bonds to avoid conflicts:
        - Atoms: 200-318 (119 atomic numbers)
        - Bonds: 319-322 (4 bond types)
        """
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            # Note: token IDs 4-5 are AutoGraph specials (we don't need to define them)
            # Token IDs 6+ are node indices from AutoGraph (pass through as-is)
        }
        idx = 200  # Start atoms at high ID to avoid conflicts
        
        # Add atom types (atomic numbers 0-118)
        for atom_id in range(119):
            vocab[f"atom_{atom_id}"] = idx
            idx += 1
        
        # Add bond types (ZINC has 1=single, 2=double)
        for bond_id in range(1, 5):  # 1=single, 2=double, 3=triple, 4=aromatic
            vocab[f"bond_{bond_id}"] = idx
            idx += 1
        
        return vocab
    
    def _init_autograph_tokenizer(self) -> None:
        """Initialize AutoGraph tokenizer."""
        if not self.tokenizer_available:
            self.autograph_tokenizer = None
            return
        
        try:
            seed = self.rng.randint(0, 2**32 - 1)
            self.autograph_tokenizer = self.Graph2TrailTokenizer(
                max_length=self.max_seq_len - 50,  # Reserve space for interleaved features
                labeled_graph=False,
                undirected=True,
                append_eos=True,
                rng=seed
            )
            
            max_nodes = 0
            sample_size = min(100, len(self.pyg_dataset))
            for i in range(sample_size):
                item = self.pyg_dataset.pyg_dataset[i]
                n = item.num_nodes if hasattr(item, 'num_nodes') else (item.x.shape[0] if hasattr(item, 'x') else 0)
                max_nodes = max(max_nodes, n)
            
            self.autograph_tokenizer.set_num_nodes(max_nodes)
        except Exception as e:
            print(f"Failed to initialize AutoGraph tokenizer: {e}")
            self.autograph_tokenizer = None
    
    def _get_edge_bonds(self, data: Data) -> Dict[tuple, int]:
        """Build edge -> bond_type mapping (undirected, canonical form).
        
        Returns: {(min_node, max_node): bond_type}
        """
        edge_bonds = {}
        
        if not (hasattr(data, 'edge_index') and data.edge_index is not None):
            return edge_bonds
        
        if not (hasattr(data, 'edge_attr') and data.edge_attr is not None):
            return edge_bonds
        
        try:
            for u, v, bond_feat in zip(
                data.edge_index[0].tolist(), 
                data.edge_index[1].tolist(), 
                data.edge_attr
            ):
                edge_key = (min(u, v), max(u, v))
                if isinstance(bond_feat, torch.Tensor):
                    bond_id = int(bond_feat.item()) if bond_feat.dim() == 0 else int(bond_feat[0].item())
                else:
                    bond_id = int(bond_feat)
                if edge_key not in edge_bonds:
                    edge_bonds[edge_key] = bond_id
        except Exception as e:
            print(f"Error building edge bonds: {e}")
        
        return edge_bonds
    
    def _intersperse_atoms_and_bonds(self, trail_tokens: list, data: Data) -> torch.Tensor:
        """Interleave both atom and bond tokens into the trail.
        
        Process:
        For each consecutive pair of nodes in the trail:
        1. Add current node token
        2. Add current node's atom token
        3. If next node exists, add bond between them
        
        Result: node0 atom0 bond(0,1) node1 atom1 bond(1,2) node2 atom2 ...
        """
        if not (hasattr(data, 'x') and data.x is not None):
            return torch.tensor(trail_tokens[:self.max_seq_len], dtype=torch.long)
        
        # Build node -> atom mapping
        node_to_atom = {}
        for node_id, atom_feat in enumerate(data.x):
            if atom_feat.dim() == 0:
                atom_id = int(atom_feat.item())
            elif atom_feat.dim() == 1 and atom_feat.shape[0] > 1:
                nonzero_indices = torch.nonzero(atom_feat)
                atom_id = int(nonzero_indices.squeeze().item()) if len(nonzero_indices) > 0 else 0
            else:
                atom_id = int(atom_feat[0].item())
            node_to_atom[node_id] = atom_id
        
        # Build edge -> bond mapping
        edge_bonds = self._get_edge_bonds(data)
        
        # Extract sequence of node indices from trail
        idx_offset = 6  # AutoGraph's node index offset
        node_sequence = []
        for token in trail_tokens:
            if token >= idx_offset:
                node_sequence.append(int(token) - idx_offset)
        
        # Process trail with interleaved atoms and bonds
        output_tokens = []
        node_idx = 0  # Index into node_sequence
        
        for token in trail_tokens:
            if token >= idx_offset:
                curr_node = int(token) - idx_offset
                output_tokens.append(token)
                if curr_node in node_to_atom:
                    atom_id = node_to_atom[curr_node]
                    atom_token = self.token2idx.get(f"atom_{atom_id}", self.token2idx["<unk>"])
                    output_tokens.append(atom_token)
                node_idx += 1
                if node_idx < len(node_sequence):
                    next_node = node_sequence[node_idx]
                    edge_key = (min(curr_node, next_node), max(curr_node, next_node))
                    if edge_key in edge_bonds:
                        bond_id = edge_bonds[edge_key]
                        bond_token = self.token2idx.get(f"bond_{bond_id}", self.token2idx["<unk>"])
                        output_tokens.append(bond_token)
            else:
                output_tokens.append(token)
                node_idx = 0
            if len(output_tokens) >= self.max_seq_len:
                break
        
        return torch.tensor(output_tokens[:self.max_seq_len], dtype=torch.long)
    
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
        """Return vocabulary."""
        return self.token2idx
    
    def __len__(self):
        return len(self.pyg_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with interleaved atom and bond features."""
        item = self.pyg_dataset[idx]
        if isinstance(item, tuple):
            data, label = item
        else:
            data = item
            label = data.y if hasattr(data, 'y') and data.y is not None else 0.0
        
        try:
            # Get AutoGraph trail
            if not hasattr(data, 'num_nodes'):
                data.num_nodes = data.x.shape[0] if hasattr(data, 'x') else 0
            
            trail_tokens = self.autograph_tokenizer.tokenize(data)
            if not isinstance(trail_tokens, torch.Tensor):
                trail_tokens = torch.tensor(trail_tokens, dtype=torch.long)
            
            # Interleave atom and bond types
            tokens = self._intersperse_atoms_and_bonds(trail_tokens.tolist(), data)
            
        except Exception as e:
            print(f"Interleaved edge tokenization failed: {e}")
            tokens = torch.tensor([self.token2idx["<bos>"], self.token2idx["<eos>"]], dtype=torch.long)
        
        input_ids, attention_mask = self._pad_and_mask(tokens)
        
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


# Aliases for backward compatibility with builders.py
GraphTransformerZincDataset = TokenizedZINCDataset
AutoGraphTransformerZincDataset = AutoGraphTokenizedZINCDataset

