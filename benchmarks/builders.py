import torch
import torch.nn as nn
from typing import Dict
from torch.utils.data import DataLoader, Subset, ConcatDataset
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def build_mpnn(args, device):
    from mpnn import GraphTaskDataset, GIN, GraphMPNNTrainer, collate_fn

    # Allow `args.algorithm` to be a list/tuple to form a union of datasets
    def _maybe_concat(fn, split, algo=None, *extra_args, **kwargs):
        algo = algo or args.algorithm
        if isinstance(algo, (list, tuple)):
            ds_list = [fn(args.data_dir, args.task, a, split, *extra_args, **kwargs) for a in algo]
            concat = ConcatDataset(ds_list)
            first = ds_list[0] if len(ds_list) > 0 else None
            if first is not None:
                for attr in ('token2idx', 'idx2token', 'vocab_size'):
                    if hasattr(first, attr):
                        val = getattr(first, attr)
                        setattr(concat, attr, val)
                        for ds in ds_list:
                            try:
                                setattr(ds, attr, val)
                            except Exception:
                                pass
            return concat
        else:
            return fn(args.data_dir, args.task, algo, split, *extra_args, **kwargs)

    train_dataset = _maybe_concat(GraphTaskDataset, "train")
    valid_dataset = _maybe_concat(GraphTaskDataset, "valid")
    test_algo = args.test_algorithm or args.algorithm
    test_dataset = _maybe_concat(GraphTaskDataset, "test", test_algo)

    try:
        import random
        
        def _sample_subset(dataset, max_samples, seed=42):
            """Sample uniformly from dataset."""
            if max_samples is None or max_samples <= 0:
                return dataset
            
            n = min(len(dataset), int(max_samples))
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(len(dataset)), n))
            return Subset(dataset, indices)
        
        train_dataset = _sample_subset(train_dataset, getattr(args, 'max_samples_train', None), seed=42)
        valid_dataset = _sample_subset(valid_dataset, getattr(args, 'max_samples_valid', None), seed=43)
        test_dataset = _sample_subset(test_dataset, getattr(args, 'max_samples_test', None), seed=44)
    except Exception:
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    task_name = args.task if isinstance(args.task, str) else str(args.task)
    is_shortest_path = "shortest_path" in task_name.lower()
    in_features = 3 if is_shortest_path else 1
    
    task_type = "classification" if args.task in ["cycle_check", "shortest_path"] or "shortest_path" in args.task else "regression"
    
    out_features = 1
    if task_type == "classification":
        try:
            import math
            labels = []
            base_train = train_dataset.dataset if hasattr(train_dataset, 'dataset') else train_dataset
            sample_size = min(len(base_train), 5000)
            for i in range(sample_size):
                try:
                    _, label = base_train[i]
                    label_val = float(label)
                    # Skip infinity labels
                    if not math.isinf(label_val):
                        labels.append(int(label_val))
                except Exception:
                    pass
            if labels:
                max_label = max(labels)
                num_classes = max_label + 1 + 1
                out_features = max(2, num_classes)
            else:
                out_features = 2
        except Exception as e:
            print(f"Warning: Could not determine number of classes: {e}")
            out_features = 2
    
    model = GIN(in_features=in_features, hidden_dim=args.hidden_dim, num_layers=args.num_layers, out_features=out_features, dropout=0.1)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"[MPNN] Model Architecture")
    print(f"{'='*70}")
    print(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    trainer = GraphMPNNTrainer(
        model,
        learning_rate=args.learning_rate,
        device=device,
        task_type=task_type,
        loss=getattr(args, 'loss', None),
        task_name=args.task,
    )
    try:
        trainer.eval_metrics = getattr(args, 'eval_metrics', None)
    except Exception:
        trainer.eval_metrics = None

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": task_type,
    }


def build_transformer(args, device, which):
    # Lazy import
    if which == "graph_transformer":
        from graph_transformer import GraphDataset, GraphTransformer as Transformer
    else:
        from autograph_transformer import GraphDataset as AGDataset, GraphTransformer as AGTransformer
        GraphDataset = AGDataset
        Transformer = AGTransformer

    def _maybe_concat_graphdataset(split, algo=None):
        algo = algo or args.algorithm
        common_kwargs = dict(
            max_seq_length=args.max_seq_length,
            n_samples_per_file=getattr(args, 'n_samples_per_file', -1),
            sampling_seed=getattr(args, 'sampling_seed', None),
        )
        if isinstance(algo, (list, tuple)):
            ds_list = [GraphDataset(args.data_dir, args.task, a, split, **common_kwargs) for a in algo]
            concat = ConcatDataset(ds_list)
            first = ds_list[0] if len(ds_list) > 0 else None
            if first is not None:
                for attr in ('token2idx', 'idx2token', 'vocab_size'):
                    if hasattr(first, attr):
                        val = getattr(first, attr)
                        setattr(concat, attr, val)
                        for ds in ds_list:
                            try:
                                setattr(ds, attr, val)
                            except Exception:
                                pass
            return concat
        else:
            return GraphDataset(args.data_dir, args.task, algo, split, **common_kwargs)

    train_dataset = _maybe_concat_graphdataset("train")
    valid_dataset = _maybe_concat_graphdataset("valid")
    test_algo = args.test_algorithm or args.algorithm
    test_dataset = _maybe_concat_graphdataset("test", test_algo)

    try:
        import random
        
        def _sample_subset(dataset, max_samples, seed=42):
            """Sample uniformly from dataset (important for ConcatDataset with multiple algos)."""
            if max_samples is None or max_samples <= 0:
                return dataset
            
            n = min(len(dataset), int(max_samples))
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(len(dataset)), n))
            return Subset(dataset, indices)
        
        train_dataset = _sample_subset(train_dataset, getattr(args, 'max_samples_train', None), seed=42)
        valid_dataset = _sample_subset(valid_dataset, getattr(args, 'max_samples_valid', None), seed=43)
        test_dataset = _sample_subset(test_dataset, getattr(args, 'max_samples_test', None), seed=44)
    except Exception:
        pass

    # Share vocabulary if available
    def _base(ds):
        try:
            from torch.utils.data import Subset as _Subset, ConcatDataset as _Concat
            if isinstance(ds, _Subset):
                return ds.dataset
            if isinstance(ds, _Concat):
                return ds.datasets[0] if len(ds.datasets) > 0 else ds
            return ds
        except Exception:
            return ds

    base_train = _base(train_dataset)
    base_valid = _base(valid_dataset)
    base_test = _base(test_dataset)

    try:
        if hasattr(base_train, 'token2idx'):
            base_valid.token2idx = base_train.token2idx
            base_valid.idx2token = base_train.idx2token
            base_valid.vocab_size = base_train.vocab_size
            base_test.token2idx = base_train.token2idx
            base_test.idx2token = base_train.idx2token
            base_test.vocab_size = base_train.vocab_size
    except Exception:
        pass

    # Ensure the authoritative train vocab is propagated into any inner
    # datasets that may be wrapped in ConcatDataset or Subset so that
    # __getitem__ uses the same mapping across splits.
    def _propagate_to_inner(ds, attr_name, value):
        try:
            from torch.utils.data import Subset as _Subset, ConcatDataset as _Concat
            if isinstance(ds, _Subset):
                inner = ds.dataset
            else:
                inner = ds

            if isinstance(inner, _Concat):
                for sub in inner.datasets:
                    try:
                        setattr(sub, attr_name, value)
                    except Exception:
                        pass
            else:
                try:
                    setattr(inner, attr_name, value)
                except Exception:
                    pass
        except Exception:
            pass

    if hasattr(base_train, 'token2idx'):
        for attr in ('token2idx', 'idx2token', 'vocab_size'):
            val = getattr(base_train, attr)
            _propagate_to_inner(train_dataset, attr, val)
            _propagate_to_inner(valid_dataset, attr, val)
            _propagate_to_inner(test_dataset, attr, val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    vocab_size = getattr(base_train, 'vocab_size', None)
    model = Transformer(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, max_seq_length=args.max_seq_length).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "Graph Transformer" if which == "graph_transformer" else "AutoGraph Transformer"
    print(f"\n{'='*70}")
    print(f"[{model_name.upper()}] Model Architecture")
    print(f"{'='*70}")
    print(model)
    print(f"\nVocab Size: {vocab_size}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*70}\n")

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "model": model,
    }


def build_graphgps(args, device):
    """Attempt to build a GraphGPS run using the repository submodule first,
    then the `benchmarks/graphgps.py` lightweight fallback.

    This mirrors the behaviour expected by `run_benchmarks.py`: prefer a
    full-featured `graphgps` package (if available under `submodules/graphgps`
    or on PYTHONPATH), otherwise use the bundled lightweight implementation.
    """
    try:
        device_arg = str(device)
    except Exception:
        device_arg = device

    repo_root = Path(__file__).resolve().parents[1]
    submodule_path = repo_root / "submodules" / "graphgps"

    benchmarks_dir = repo_root / "benchmarks"
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))

    if submodule_path.exists():
        sp = str(submodule_path)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    try:
        mod = importlib.import_module("benchmarks.graphgps")
        if hasattr(mod, "build_graphgps"):
            return mod.build_graphgps(args, device_arg)
    except Exception:
        try:
            fallback = Path(__file__).resolve().parent / "graphgps.py"
            spec = importlib.util.spec_from_file_location("benchmarks_graphgps_fallback", str(fallback))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.build_graphgps(args, device_arg)
        except Exception as e:
            raise ImportError("Could not import any GraphGPS builder (submodule or fallback).\n"
                              f"Last error: {e}")


def build_mpnn_zinc(args, device):
    """Build MPNN for ZINC dataset."""
    from zinc_dataset import ZincGraphTaskDataset
    from mpnn import GIN, GraphMPNNTrainer, collate_fn
    
    train_dataset = ZincGraphTaskDataset(args.data_dir, split="train", subset=True)
    valid_dataset = ZincGraphTaskDataset(args.data_dir, split="valid", subset=True)
    test_dataset = ZincGraphTaskDataset(args.data_dir, split="test", subset=True)
    
    try:
        if getattr(args, 'max_samples_train', None) and args.max_samples_train > 0:
            n = min(len(train_dataset), int(args.max_samples_train))
            train_dataset = Subset(train_dataset, list(range(n)))
        if getattr(args, 'max_samples_valid', None) and args.max_samples_valid > 0:
            n = min(len(valid_dataset), int(args.max_samples_valid))
            valid_dataset = Subset(valid_dataset, list(range(n)))
        if getattr(args, 'max_samples_test', None) and args.max_samples_test > 0:
            n = min(len(test_dataset), int(args.max_samples_test))
            test_dataset = Subset(test_dataset, list(range(n)))
    except Exception:
        pass
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = GIN(in_features=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, out_features=1, dropout=0.1)
    trainer = GraphMPNNTrainer(
        model,
        learning_rate=args.learning_rate,
        device=device,
        task_type="regression",
        loss=getattr(args, 'loss', None),
    )
    
    try:
        trainer.eval_metrics = getattr(args, 'eval_metrics', None)
    except Exception:
        trainer.eval_metrics = None
    
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": "regression",
    }


def build_transformer_zinc(args, device, which):
    """Build Transformer for ZINC dataset with model-specific tokenization.
    
    Tries to load from preprocessed data first, falls back to on-the-fly tokenization.
    """
    if which == "graph_transformer":
        from graph_transformer import GraphTransformer as Transformer
    else:
        from autograph_transformer import GraphTransformer as AGTransformer
        Transformer = AGTransformer
    
    preprocessed_dir = getattr(args, 'preprocessed_zinc_dir', None)
    if preprocessed_dir and Path(preprocessed_dir).exists():
        from preprocess_zinc_loader import PreprocessedGraphTransformerZincDataset, PreprocessedAutoGraphTransformerZincDataset
        
        if which == "graph_transformer":
            DatasetClass = PreprocessedGraphTransformerZincDataset
        else:
            DatasetClass = PreprocessedAutoGraphTransformerZincDataset
        
        try:
            train_dataset = DatasetClass(preprocessed_dir, split="train")
            valid_dataset = DatasetClass(preprocessed_dir, split="valid")
            test_dataset = DatasetClass(preprocessed_dir, split="test")
        except Exception as e:
            print(f"Warning: Could not load preprocessed data: {e}")
            print("Falling back to on-the-fly tokenization...")
            preprocessed_dir = None
    
    if not preprocessed_dir or not Path(preprocessed_dir).exists():
        from zinc_dataset import GraphTransformerZincDataset, AutoGraphTransformerZincDataset
        
        if which == "graph_transformer":
            DatasetClass = GraphTransformerZincDataset
        else:
            DatasetClass = AutoGraphTransformerZincDataset
        
        train_dataset = DatasetClass(
            args.data_dir,
            split="train",
            subset=True,
            max_seq_length=args.max_seq_length,
        )
        valid_dataset = DatasetClass(
            args.data_dir,
            split="valid",
            subset=True,
            max_seq_length=args.max_seq_length,
        )
        test_dataset = DatasetClass(
            args.data_dir,
            split="test",
            subset=True,
            max_seq_length=args.max_seq_length,
        )
    
    try:
        if getattr(args, 'max_samples_train', None) and args.max_samples_train > 0:
            n = min(len(train_dataset), int(args.max_samples_train))
            train_dataset = Subset(train_dataset, list(range(n)))
        if getattr(args, 'max_samples_valid', None) and args.max_samples_valid > 0:
            n = min(len(valid_dataset), int(args.max_samples_valid))
            valid_dataset = Subset(valid_dataset, list(range(n)))
        if getattr(args, 'max_samples_test', None) and args.max_samples_test > 0:
            n = min(len(test_dataset), int(args.max_samples_test))
            test_dataset = Subset(test_dataset, list(range(n)))
    except Exception:
        pass
    
    def _base(ds):
        try:
            if isinstance(ds, Subset):
                return ds.dataset
            if isinstance(ds, ConcatDataset):
                return ds.datasets[0] if len(ds.datasets) > 0 else ds
            return ds
        except Exception:
            return ds
    
    base_train = _base(train_dataset)
    vocab_size = getattr(base_train, 'vocab_size', 1000)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
    ).to(device)
    
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "model": model,
    }


def build_graphgps_zinc(args, device):
    """Build GraphGPS for ZINC dataset - simplified version for PyG data."""
    from zinc_dataset import ZincPyGDataset
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv, global_mean_pool, GINConv
    
    train_dataset = ZincPyGDataset(args.data_dir, split="train", subset=True)
    valid_dataset = ZincPyGDataset(args.data_dir, split="valid", subset=True)
    test_dataset = ZincPyGDataset(args.data_dir, split="test", subset=True)
    
    try:
        if getattr(args, 'max_samples_train', None) and args.max_samples_train > 0:
            n = min(len(train_dataset), int(args.max_samples_train))
            train_dataset = Subset(train_dataset, list(range(n)))
        if getattr(args, 'max_samples_valid', None) and args.max_samples_valid > 0:
            n = min(len(valid_dataset), int(args.max_samples_valid))
            valid_dataset = Subset(valid_dataset, list(range(n)))
        if getattr(args, 'max_samples_test', None) and args.max_samples_test > 0:
            n = min(len(test_dataset), int(args.max_samples_test))
            test_dataset = Subset(test_dataset, list(range(n)))
    except Exception:
        pass
    
    from torch_geometric.data import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = PyGDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    batch_size = getattr(args, "batch_size", 32)
    learning_rate = getattr(args, "learning_rate", 1e-3)
    hidden_dim = getattr(args, "d_model", 64)
    n_layers = max(1, getattr(args, "n_layers", 3))
    n_heads = getattr(args, "n_heads", 4)
    dropout = getattr(args, "dropout", 0.1)
    
    class SimpleGPS(nn.Module):
        def __init__(self, in_channels: int = 1, hidden_dim: int = 64, n_layers: int = 3, n_heads: int = 4, out_dim: int = 1, dropout: float = 0.1):
            super().__init__()
            self.input_lin = nn.Linear(in_channels, hidden_dim)
            self.convs = nn.ModuleList()
            for _ in range(n_layers):
                self.convs.append(TransformerConv(hidden_dim, max(1, hidden_dim // n_heads), heads=n_heads, dropout=dropout))
            self.pool = global_mean_pool
            self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
        
        def forward(self, batch):
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            x = x.float()
            x = self.input_lin(x)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
            x = self.pool(x, batch_idx)
            out = self.mlp(x)
            return out
    
    model = SimpleGPS(in_channels=1, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads, out_dim=1, dropout=dropout)
    model = model.to(device)
    
    class Trainer:
        def __init__(self, model, lr=1e-3, device="cpu", task_type="regression", loss: str | None = None):
            self.model = model
            self.device = torch.device(device)
            self.model.to(self.device)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)
            self.task_type = task_type
            self.loss_fn = nn.MSELoss()
        
        def train_epoch(self, loader: PyGDataLoader) -> float:
            self.model.train()
            total = 0.0
            n = 0
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                label = batch.y.view(-1, 1)
                loss = self.loss_fn(pred, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                total += loss.item()
                n += 1
            return total / n if n > 0 else 0.0
        
        @torch.no_grad()
        def evaluate(self, loader: PyGDataLoader) -> dict:
            self.model.eval()
            total = 0.0
            total_mae = 0.0
            n = 0
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                label = batch.y.view(-1, 1)
                loss = self.loss_fn(pred, label)
                mae = torch.mean(torch.abs(pred - label)).item()
                total += loss.item()
                total_mae += mae
                n += 1
            return {
                "loss": total / n if n > 0 else float('nan'),
                "mae": total_mae / n if n > 0 else float('nan'),
            }
        
        def save_checkpoint(self, path: str):
            torch.save(self.model.state_dict(), path)
        
        def load_checkpoint(self, path: str):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    trainer = Trainer(model, lr=learning_rate, device=device, task_type="regression")
    
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": "regression",
    }


def build_zinc_mpnn(args, device):
    """Build MPNN (GIN) for ZINC molecular property prediction.
    
    ZINC is a regression task predicting molecular properties.
    """
    from zinc_dataset import ZINCLoader
    from mpnn import GIN, GraphMPNNTrainer, collate_fn
    
    subset = not ("250" in str(getattr(args, 'task', 'zinc')))
    data_dir = getattr(args, 'data_dir', None) or "./data/ZINC"
    
    train_dataset = ZINCLoader.load_pyg_data(root=data_dir, split="train", subset=subset, return_tuple=True)
    valid_dataset = ZINCLoader.load_pyg_data(root=data_dir, split="val", subset=subset, return_tuple=True)
    test_dataset = ZINCLoader.load_pyg_data(root=data_dir, split="test", subset=subset, return_tuple=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    sample_item = train_dataset[0]
    if isinstance(sample_item, tuple):
        sample, _ = sample_item
    else:
        sample = sample_item
    in_features = sample.x.shape[1] if hasattr(sample, 'x') and sample.x is not None else 1
    
    model = GIN(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        out_features=1,
        dropout=0.1,
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"[ZINC MPNN] Model Architecture")
    print(f"{'='*70}")
    print(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    trainer = GraphMPNNTrainer(
        model,
        learning_rate=args.learning_rate,
        device=device,
        task_type="regression",
        loss=getattr(args, 'loss', None) or 'mae',
        task_name="zinc"
    )
    trainer.eval_metrics = getattr(args, 'eval_metrics', None)
    
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": "regression",
    }


def build_zinc_transformer(args, device, which="graph_transformer", use_autograph: bool = False, 
                           use_autograph_with_features: bool = False, use_autograph_interspersed: bool = False,
                           use_autograph_interleaved_edges: bool = False):
    """Build Transformer for ZINC molecular property prediction.
    
    Tokenizes molecular graphs for transformer processing.
    
    Args:
        args: Configuration arguments
        device: Torch device (cuda or cpu)
        which: Model type identifier
        use_autograph: If True, use AutoGraph's trail-based tokenization (topology only)
        use_autograph_with_features: If True, use AutoGraph's trail with node features appended
        use_autograph_interspersed: If True, use AutoGraph's trail with atoms interspersed after nodes
        use_autograph_interleaved_edges: If True, use AutoGraph's trail with atoms and bonds interleaved
    """
    from zinc_dataset import ZINCLoader
    from graph_transformer import GraphTransformer as Transformer
    
    subset = not ("250" in str(getattr(args, 'task', 'zinc')))
    data_dir = getattr(args, 'data_dir', None) or "./data/ZINC"
    
    train_dataset = ZINCLoader.load_tokenized_data(
        root=data_dir,
        split="train",
        subset=subset,
        max_seq_len=args.max_seq_length,
        use_autograph=use_autograph,
        use_autograph_with_features=use_autograph_with_features,
        use_autograph_interspersed=use_autograph_interspersed,
        use_autograph_interleaved_edges=use_autograph_interleaved_edges,
        random_seed=getattr(args, 'random_seed', 1234)
    )
    valid_dataset = ZINCLoader.load_tokenized_data(
        root=data_dir,
        split="val",
        subset=subset,
        max_seq_len=args.max_seq_length,
        use_autograph=use_autograph,
        use_autograph_with_features=use_autograph_with_features,
        use_autograph_interspersed=use_autograph_interspersed,
        use_autograph_interleaved_edges=use_autograph_interleaved_edges,
        random_seed=getattr(args, 'random_seed', 1235)
    )
    test_dataset = ZINCLoader.load_tokenized_data(
        root=data_dir,
        split="test",
        subset=subset,
        max_seq_len=args.max_seq_length,
        use_autograph=use_autograph,
        use_autograph_with_features=use_autograph_with_features,
        use_autograph_interspersed=use_autograph_interspersed,
        use_autograph_interleaved_edges=use_autograph_interleaved_edges,
        random_seed=getattr(args, 'random_seed', 1236)
    )
    
    try:
        if hasattr(train_dataset, 'token2idx'):
            valid_dataset.token2idx = train_dataset.token2idx
            test_dataset.token2idx = train_dataset.token2idx
    except Exception:
        pass
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if hasattr(train_dataset, 'token2idx'):
        vocab_size = max(train_dataset.token2idx.values()) + 1
    else:
        vocab_size = 200
    
    base_model = Transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
    ).to(device)
    
    class TransformerRegressor(torch.nn.Module):
        def __init__(self, transformer, d_model):
            super().__init__()
            self.transformer = transformer
            self.head = torch.nn.Linear(d_model, 1)
        
        def forward(self, input_ids, attention_mask=None):
            x = self.transformer.token_embedding(input_ids)
            x = x + self.transformer.position_embedding[:input_ids.size(1)].unsqueeze(0)
            x = self.transformer.dropout(x)
            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            x = x.transpose(0, 1)
            for layer in self.transformer.encoder_layers:
                x = layer(x, mask=key_padding_mask)
            x = x.transpose(0, 1)
            
            if attention_mask is not None:
                x_masked = x * attention_mask.unsqueeze(-1)
                x_mean = x_masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                x_mean = x.mean(dim=1)
            
            out = self.head(x_mean)
            return out
    
    model = TransformerRegressor(base_model, args.d_model).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if use_autograph_interleaved_edges:
        tokenizer_type = "AutoGraph+Interleaved(Atoms+Bonds)"
    elif use_autograph_interspersed:
        tokenizer_type = "AutoGraph+Interspersed"
    elif use_autograph_with_features:
        tokenizer_type = "AutoGraph+Features"
    elif use_autograph:
        tokenizer_type = "AutoGraph"
    else:
        tokenizer_type = "Simple"
    print(f"\n{'='*70}")
    print(f"[ZINC {which.upper()}] Model Architecture (Tokenizer: {tokenizer_type})")
    print(f"{'='*70}")
    print(model)
    print(f"\nVocab Size: {vocab_size}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "model": model,
        "task_type": "regression",
    }


def build_zinc_graphgps(args, device):
    """Build GraphGPS for ZINC molecular property prediction."""
    from zinc_dataset import ZINCLoader
    from torch_geometric.data import DataLoader as PyGDataLoader
    
    subset = not ("250" in str(getattr(args, 'task', 'zinc')))
    data_dir = getattr(args, 'data_dir', None) or "./data/ZINC"
    
    train_dataset = ZINCLoader.load_pyg_data(root=data_dir, split="train", subset=subset, return_tuple=False)
    valid_dataset = ZINCLoader.load_pyg_data(root=data_dir, split="val", subset=subset, return_tuple=False)
    test_dataset = ZINCLoader.load_pyg_data(root=data_dir, split="test", subset=subset, return_tuple=False)
    
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = PyGDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    sample = train_dataset[0]
    in_features = sample.x.shape[1] if hasattr(sample, 'x') and sample.x is not None else 1
    
    from torch_geometric.nn import TransformerConv, global_mean_pool
    import torch.nn.functional as F
    
    class SimpleGPSRegressor(torch.nn.Module):
        """Simple GPS-style model for molecular regression."""
        
        def __init__(self, in_features, hidden_dim, n_layers, n_heads, dropout=0.1):
            super().__init__()
            self.in_proj = torch.nn.Linear(in_features, hidden_dim)
            self.layers = torch.nn.ModuleList([
                TransformerConv(hidden_dim, hidden_dim // n_heads, n_heads, dropout=dropout, edge_dim=None, concat=True)
                for _ in range(n_layers)
            ])
            self.out_proj = torch.nn.Linear(hidden_dim, 1)
        
        def forward(self, data):
            x = self.in_proj(data.x)
            for layer in self.layers:
                x = layer(x, data.edge_index)
                x = F.dropout(x, p=0.1, training=self.training)
            x = global_mean_pool(x, data.batch)
            return self.out_proj(x)
    
    model = SimpleGPSRegressor(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers if hasattr(args, 'num_layers') else args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*70}")
    print(f"[ZINC GraphGPS] Model Architecture")
    print(f"{'='*70}")
    print(model)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"{'='*70}\n")
    
    class SimpleTrainer:
        def __init__(self, model, lr, device, task_type="regression"):
            self.model = model
            self.device = device
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.loss_fn = torch.nn.L1Loss()
        
        def train_epoch(self, loader):
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader) if len(loader) > 0 else 0.0
        
        @torch.no_grad()
        def evaluate(self, loader):
            self.model.eval()
            total_loss = 0.0
            total_mae = 0.0
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                loss = self.loss_fn(pred, batch.y.view(-1, 1))
                mae = torch.mean(torch.abs(pred - batch.y.view(-1, 1))).item()
                total_loss += loss.item()
                total_mae += mae
            n = len(loader) if len(loader) > 0 else 1
            return {"loss": total_loss / n, "mae": total_mae / n}
        
        def save_checkpoint(self, path):
            torch.save(self.model.state_dict(), path)
        
        def load_checkpoint(self, path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    trainer = SimpleTrainer(model, lr=args.learning_rate, device=device)
    trainer.eval_metrics = getattr(args, 'eval_metrics', None)
    
    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": "regression",
    }

