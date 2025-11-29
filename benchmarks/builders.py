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
    from mpnn import GraphTaskDataset, GIN, GraphMPNNTrainer, collate_fn

    # Allow `args.algorithm` to be a list/tuple to form a union of datasets
    def _maybe_concat(fn, split, *extra_args, **kwargs):
        algo = args.algorithm
        if isinstance(algo, (list, tuple)):
            ds_list = [fn(args.data_dir, args.task, a, split, *extra_args, **kwargs) for a in algo]
            concat = ConcatDataset(ds_list)
            # propagate simple attributes (vocab) from first dataset if present
            first = ds_list[0] if len(ds_list) > 0 else None
            if first is not None:
                for attr in ('token2idx', 'idx2token', 'vocab_size'):
                    if hasattr(first, attr):
                        val = getattr(first, attr)
                        setattr(concat, attr, val)
                        # also propagate into each inner dataset so their
                        # __getitem__ uses the shared vocabulary
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
    test_dataset = _maybe_concat(GraphTaskDataset, "test")

    # Optionally limit dataset sizes via args (set in model config or CLI overrides)
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
        # Keep original datasets on any error
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GIN(in_features=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, out_features=1, dropout=0.5)
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
        from graph_transformer import GraphDataset, GraphTransformer as Transformer
    else:
        from autograph_transformer import GraphDataset as AGDataset, GraphTransformer as AGTransformer
        GraphDataset = AGDataset
        Transformer = AGTransformer

    # Forward sampling args so datasets can sample `n_samples_per_file` per JSON
    def _maybe_concat_graphdataset(split):
        algo = args.algorithm
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
    test_dataset = _maybe_concat_graphdataset("test")

    # Optionally limit dataset sizes via args (set in model config or CLI overrides)
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
        # Keep original datasets on any error
        pass

    # Share vocabulary if available. If datasets are wrapped in `Subset`,
    # extract the underlying base dataset to access `token2idx`/`vocab_size`.
    def _base(ds):
        try:
            from torch.utils.data import Subset as _Subset, ConcatDataset as _Concat
            if isinstance(ds, _Subset):
                return ds.dataset
            if isinstance(ds, _Concat):
                # return the first underlying dataset for vocab inspection
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
            # token-id mismatches between training and test time.
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

    # Ensure we pass the underlying dataset's vocab_size (handle Subset wrappers).
    vocab_size = getattr(base_train, 'vocab_size', None)
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
