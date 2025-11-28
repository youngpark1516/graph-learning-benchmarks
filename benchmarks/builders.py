import torch
import torch.nn as nn
from typing import Dict
from torch.utils.data import DataLoader, Subset, ConcatDataset
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def build_mpnn(args, device):
    # Import inside function to avoid heavy imports at module-import time
    from mpnn import GIN, GraphMPNNTrainer
    from unified_dataset import PyGGraphDataset
    from torch_geometric.data import Batch
    
    def collate_fn(batch):
        """Collate function for PyG Data objects that adds labels."""
        data_list = []
        for item in batch:
            data, label = item
            data.y = label.unsqueeze(0) if label.dim() == 0 else label
            data_list.append(data)
        return Batch.from_data_list(data_list)

    # Create datasets using unified loader
    train_dataset = PyGGraphDataset(
        data_path=args.data_dir,
        split="train",
        task=args.task,
        algorithm=args.algorithm if isinstance(args.algorithm, str) else args.algorithm[0],
        add_query_features=True
    )
    valid_dataset = PyGGraphDataset(
        data_path=args.data_dir,
        split="valid",
        task=args.task,
        algorithm=args.algorithm if isinstance(args.algorithm, str) else args.algorithm[0],
        add_query_features=True
    )
    test_dataset = PyGGraphDataset(
        data_path=args.data_dir,
        split="test",
        task=args.task,
        algorithm=args.algorithm if isinstance(args.algorithm, str) else args.algorithm[0],
        add_query_features=True
    )

    # Optionally limit dataset sizes via args (set in model config or CLI overrides)
    try:
        if getattr(args, 'max_samples_train', None) and args.max_samples_train > 0:
            n = min(len(train_dataset), int(args.max_samples_train))
            train_dataset = Subset(train_dataset, list(range(n)))
            print(f"Limited train dataset to {n} samples")
        if getattr(args, 'max_samples_valid', None) and args.max_samples_valid > 0:
            n = min(len(valid_dataset), int(args.max_samples_valid))
            valid_dataset = Subset(valid_dataset, list(range(n)))
            print(f"Limited valid dataset to {n} samples")
        if getattr(args, 'max_samples_test', None) and args.max_samples_test > 0:
            n = min(len(test_dataset), int(args.max_samples_test))
            test_dataset = Subset(test_dataset, list(range(n)))
            print(f"Limited test dataset to {n} samples")
    except Exception as e:
        print(f"Warning: Could not limit dataset sizes: {e}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Determine input features based on task
    # shortest_path adds 2 query encoding features (is_source, is_target) to degree
    if "shortest_path" in args.task.lower():
        in_features = 3
    else:
        in_features = 1
    
    model = GIN(in_features=in_features, hidden_dim=args.hidden_dim, num_layers=args.num_layers, out_features=1, dropout=0.5)
    task_type = "classification" if args.task in ["cycle_check"] else "regression"
    trainer = GraphMPNNTrainer(
        model,
        learning_rate=args.learning_rate,
        device=device,
        task_type=task_type,
        loss=getattr(args, 'loss', None),
    )
    # Propagate requested eval metrics (list or None) to trainer so it can
    # compute optional metrics like accuracy for regression tasks.
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
    # Lazy import to avoid heavy module loads during import-time checks
    if which == "graph_transformer":
        from graph_transformer import GraphTransformer as Transformer
    else:
        from autograph_transformer import GraphTransformer as AGTransformer
        Transformer = AGTransformer
    
    # Use unified tokenized dataset loader
    from unified_dataset import TokenizedGraphDataset

    def transformer_collate_fn(batch):
        """Collate function for transformer datasets - converts tuples to dicts."""
        input_ids_list = []
        labels_list = []
        
        for input_ids, label in batch:
            input_ids_list.append(input_ids)
            labels_list.append(label)
        
        input_ids_batch = torch.stack(input_ids_list)
        labels_batch = torch.stack(labels_list)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        # Assuming padding token idx is 0
        attention_mask = (input_ids_batch != 0).long()
        
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask,
            "label": labels_batch,
        }

    # Forward sampling args so datasets can sample `n_samples_per_file` per JSON
    def _maybe_concat_tokenized_dataset(split):
        algo = args.algorithm
        common_kwargs = dict(
            split=split,
            task=args.task,
            max_seq_len=getattr(args, 'max_seq_length', 512),
        )
        if isinstance(algo, (list, tuple)):
            ds_list = [TokenizedGraphDataset(args.data_dir, algorithm=a, **common_kwargs) for a in algo]
            concat = ConcatDataset(ds_list)
            first = ds_list[0] if len(ds_list) > 0 else None
            if first is not None:
                for attr in ('token2idx', 'idx2token', 'vocab_size', 'label_vocab'):
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
            return TokenizedGraphDataset(args.data_dir, algorithm=algo, **common_kwargs)

    train_dataset = _maybe_concat_tokenized_dataset("train")
    valid_dataset = _maybe_concat_tokenized_dataset("valid")
    test_dataset = _maybe_concat_tokenized_dataset("test")
    
    # Share vocabulary across splits
    try:
        if hasattr(train_dataset, 'token2idx'):
            # Extract base dataset if wrapped in Subset/Concat
            def _base(ds):
                if isinstance(ds, Subset):
                    return ds.dataset
                if isinstance(ds, ConcatDataset):
                    return ds.datasets[0] if len(ds.datasets) > 0 else ds
                return ds
            
            base_train = _base(train_dataset)
            base_valid = _base(valid_dataset)
            base_test = _base(test_dataset)
            
            train_vocab = base_train.token2idx
            train_labels = base_train.label_vocab if hasattr(base_train, 'label_vocab') else None
            
            # Propagate to valid/test
            if hasattr(base_valid, 'set_vocab'):
                base_valid.set_vocab(train_vocab, train_labels)
            else:
                base_valid.token2idx = train_vocab
                if train_labels:
                    base_valid.label_vocab = train_labels
            
            if hasattr(base_test, 'set_vocab'):
                base_test.set_vocab(train_vocab, train_labels)
            else:
                base_test.token2idx = train_vocab
                if train_labels:
                    base_test.label_vocab = train_labels
    except Exception as e:
        print(f"Warning: Could not share vocabulary across splits: {e}")

    # Optionally limit dataset sizes via args (set in model config or CLI overrides)
    try:
        if getattr(args, 'max_samples_train', None) and args.max_samples_train > 0:
            n = min(len(train_dataset), int(args.max_samples_train))
            train_dataset = Subset(train_dataset, list(range(n)))
            print(f"Limited train dataset to {n} samples")
        if getattr(args, 'max_samples_valid', None) and args.max_samples_valid > 0:
            n = min(len(valid_dataset), int(args.max_samples_valid))
            valid_dataset = Subset(valid_dataset, list(range(n)))
            print(f"Limited valid dataset to {n} samples")
        if getattr(args, 'max_samples_test', None) and args.max_samples_test > 0:
            n = min(len(test_dataset), int(args.max_samples_test))
            test_dataset = Subset(test_dataset, list(range(n)))
            print(f"Limited test dataset to {n} samples")
    except Exception as e:
        print(f"Warning: Could not limit dataset sizes: {e}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=transformer_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=transformer_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=transformer_collate_fn)

    # Get vocabulary size from dataset
    vocab_size = None
    if hasattr(train_dataset, 'token2idx'):
        vocab_size = len(train_dataset.token2idx)
    elif isinstance(train_dataset, Subset):
        base = train_dataset.dataset
        if hasattr(base, 'token2idx'):
            vocab_size = len(base.token2idx)
    
    if vocab_size is None:
        vocab_size = 1000  # fallback
    
    model = Transformer(vocab_size=vocab_size, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, max_seq_length=args.max_seq_length).to(device)

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
    # Normalize device to a string for downstream builders that expect it
    try:
        device_arg = str(device)
    except Exception:
        device_arg = device

    repo_root = Path(__file__).resolve().parents[1]
    submodule_path = repo_root / "submodules" / "graphgps"

    # Ensure local `benchmarks/` modules (e.g. `mpnn.py`) are importable
    benchmarks_dir = repo_root / "benchmarks"
    if str(benchmarks_dir) not in sys.path:
        sys.path.insert(0, str(benchmarks_dir))

    # Try to ensure the repo submodule is importable by adding it to sys.path
    if submodule_path.exists():
        sp = str(submodule_path)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    # Try importing a build_graphgps callable from the installed/submodule
    try:
        pkg = importlib.import_module("graphgps")
        if hasattr(pkg, "build_graphgps"):
            return getattr(pkg, "build_graphgps")(args, device_arg)
    except Exception:
        # ignore and fall through to fallback loader
        pass

    # Try importing a top-level module `graphgps` (if available on PYTHONPATH)
    try:
        pkg = importlib.import_module("graphgps")
        if hasattr(pkg, "build_graphgps"):
            return getattr(pkg, "build_graphgps")(args, device_arg)
    except Exception:
        pass

    # Finally, fall back to the lightweight implementation bundled under
    # `benchmarks/graphgps.py`.
    try:
        # First try a normal import relative to this package
        mod = importlib.import_module("benchmarks.graphgps")
        if hasattr(mod, "build_graphgps"):
            return mod.build_graphgps(args, device_arg)
    except Exception:
        # Last-resort: load by file path
        try:
            fallback = Path(__file__).resolve().parent / "graphgps.py"
            spec = importlib.util.spec_from_file_location("benchmarks_graphgps_fallback", str(fallback))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.build_graphgps(args, device_arg)
        except Exception as e:
            raise ImportError("Could not import any GraphGPS builder (submodule or fallback).\n"
                              f"Last error: {e}")
