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
    """Build GraphGPS using the submodule implementation when available.

    Falls back to the lightweight `benchmarks/graphgps.build_graphgps` if the
    submodule cannot be imported or instantiation fails.
    """
    # Reuse dataset and collate_fn from benchmarks/mpnn.py
    from mpnn import GraphTaskDataset, collate_fn

    batch_size = getattr(args, "batch_size", 32)
    # Allow `args.algorithm` to be a list/tuple to form a union of datasets
    def _maybe_concat_taskdataset(split):
        algo = args.algorithm
        if isinstance(algo, (list, tuple)):
            ds_list = [GraphTaskDataset(args.data_dir, args.task, a, split) for a in algo]
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
            return GraphTaskDataset(args.data_dir, args.task, algo, split)

    train_dataset = _maybe_concat_taskdataset("train")
    valid_dataset = _maybe_concat_taskdataset("valid")
    test_dataset = _maybe_concat_taskdataset("test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Try to use repository submodule first (deterministic). If not present
    # or loading fails, fall back to top-level import attempt. Print clear
    # diagnostics so users know why the lightweight fallback was chosen.
    try:
        import traceback
        repo_root = Path(__file__).resolve().parents[1]
        candidate = repo_root / "submodules" / "graphgps" / "graphgps" / "__init__.py"
        use_submodule = False
        if candidate.exists():
            print(f"Found graphgps submodule at: {candidate}; attempting to load it")
            try:
                spec = importlib.util.spec_from_file_location("graphgps", str(candidate))
                module = importlib.util.module_from_spec(spec)
                sys.modules["graphgps"] = module
                spec.loader.exec_module(module)  # type: ignore
                from graphgps.network.gps_model import GPSModel  # type: ignore
                use_submodule = True
                print("Loaded graphgps from repository submodule")
            except Exception:
                print("Failed to load graphgps repo submodule; traceback:")
                traceback.print_exc()

        if not use_submodule:
            # Try top-level import as a fallback
            try:
                from graphgps.network.gps_model import GPSModel  # type: ignore
                use_submodule = True
                print("Loaded graphgps from top-level package")
            except Exception:
                print("Top-level import of graphgps failed; will use lightweight fallback")
                # Print traceback for diagnostic purposes
                traceback.print_exc()
    except Exception:
        # Any unexpected error; ensure we fallback to lightweight implementation
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        use_submodule = False

    if not use_submodule:
        # Fall back to lightweight builder in benchmarks/graphgps.py
        try:
            print("Using lightweight GraphGPS (fallback)")
            from graphgps import build_graphgps as light_build  # type: ignore
            return light_build(args, device)
        except Exception:
            raise

    # Instantiate submodule GPSModel (simple instantiation compatible with our batches)
    # Honor explicit opt-out flag if present
    use_flag = getattr(args, 'use_submodule', None)
    if use_flag is not None and not bool(use_flag):
        print("use_submodule explicitly disabled in config; falling back to lightweight GraphGPS")
        try:
            from graphgps import build_graphgps as light_build  # type: ignore
            return light_build(args, device)
        except Exception:
            raise

    # Map model_args into GraphGym cfg if available so the submodule model
    # matches requested hyperparameters (hidden dim, n_layers, n_heads, etc.).
    try:
        from torch_geometric.graphgym.config import cfg as gym_cfg
    except Exception:
        gym_cfg = None

    # If we found a gym_cfg, try to auto-merge a sensible default YAML from
    # the graphgps submodule so the submodule has a full config to read.
    if gym_cfg is not None:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            cfg_candidates = [
                repo_root / "submodules" / "graphgps" / "tests" / "configs" / "graph" / "zinc-GPS-small.yaml",
                repo_root / "submodules" / "graphgps" / "tests" / "configs" / "graph" / "zinc-GPS.yaml",
            ]
            # Fallback: any yaml file under tests/configs/graph
            if not any(p.exists() for p in cfg_candidates):
                graph_cfg_dir = repo_root / "submodules" / "graphgps" / "tests" / "configs" / "graph"
                if graph_cfg_dir.exists():
                    first = next(graph_cfg_dir.glob("*.yaml"), None)
                    if first is not None:
                        cfg_candidates.append(first)

            for cand in cfg_candidates:
                try:
                    if cand.exists():
                        try:
                            gym_cfg.merge_from_file(str(cand))
                            print(f"Loaded GraphGPS config from: {cand}")
                        except Exception as e:
                            print(f"Failed to merge GraphGPS config {cand}: {e}")
                        break
                except Exception:
                    continue
        except Exception:
            pass

    if gym_cfg is not None:
        # Ensure `gym_cfg` has the expected subnodes used by the
        # submodule. Use yacs' CfgNode so we don't violate type
        # expectations; only populate the minimal fields required by
        # `GPSModel` to avoid AttributeError on access.
        try:
            try:
                from yacs.config import CfgNode as CN
            except Exception:
                CN = None

            # Create subnodes if they are missing and we have CN available
            if CN is not None:
                if not hasattr(gym_cfg, 'gnn') or getattr(gym_cfg, 'gnn') is None:
                    gym_cfg.gnn = CN()
                if not hasattr(gym_cfg, 'gt') or getattr(gym_cfg, 'gt') is None:
                    gym_cfg.gt = CN()
                if not hasattr(gym_cfg, 'dataset') or getattr(gym_cfg, 'dataset') is None:
                    gym_cfg.dataset = CN()

            # Map common keys into the cfg (create keys if missing)
            hid = getattr(args, 'hidden_dim', None) or getattr(args, 'd_model', None)
            if hid is not None:
                try:
                    gym_cfg.gnn.dim_inner = int(hid)
                except Exception:
                    pass
                try:
                    gym_cfg.gt.dim_hidden = int(hid)
                except Exception:
                    pass

            nl = getattr(args, 'n_layers', None) or getattr(args, 'num_layers', None)
            if nl is not None:
                try:
                    gym_cfg.gt.layers = int(nl)
                except Exception:
                    pass

            nh = getattr(args, 'n_heads', None)
            if nh is not None:
                try:
                    gym_cfg.gt.n_heads = int(nh)
                except Exception:
                    pass

            lt = getattr(args, 'gt_layer_type', None) or getattr(args, 'layer_type', None)
            if lt is not None:
                try:
                    gym_cfg.gt.layer_type = lt
                except Exception:
                    pass

            pre_mp = getattr(args, 'pre_mp', None)
            if pre_mp is not None:
                try:
                    gym_cfg.gnn.layers_pre_mp = int(pre_mp)
                except Exception:
                    pass

            # Print a short summary of the applied cfg for user visibility
            try:
                summary = []
                if hasattr(gym_cfg, 'gnn') and getattr(gym_cfg, 'gnn') is not None:
                    summary.append(f"gnn.dim_inner={getattr(gym_cfg.gnn, 'dim_inner', None)}")
                if hasattr(gym_cfg, 'gt') and getattr(gym_cfg, 'gt') is not None:
                    summary.append(f"gt.layers={getattr(gym_cfg.gt, 'layers', None)}")
                    summary.append(f"gt.n_heads={getattr(gym_cfg.gt, 'n_heads', None)}")
                    summary.append(f"gt.layer_type={getattr(gym_cfg.gt, 'layer_type', None)}")
                print(f"GraphGym cfg mapping: {' ,'.join(summary)}")
            except Exception:
                pass
        except Exception:
            # If anything goes wrong while mapping into the yacs cfg,
            # leave gym_cfg unchanged and continue with submodule defaults.
            pass

    print("Using submodule GraphGPS implementation")
    # Instantiate GPSModel with an input embedding size that matches the
    # configured inner dimension when available. The submodule expects
    # `cfg.gnn.dim_inner == cfg.gt.dim_hidden == dim_in` so prefer the
    # mapped `gym_cfg.gnn.dim_inner` value when present.
    try:
        # Populate a few additional defaults the submodule expects so
        # attribute access won't raise `AttributeError`. These are safe
        # defaults for small/debug runs and match typical graphgps configs.
        try:
            if not hasattr(gym_cfg, 'dataset') or getattr(gym_cfg, 'dataset') is None:
                from yacs.config import CfgNode as CN
                gym_cfg.dataset = CN()
        except Exception:
            pass

        try:
            # posenc equivstable lapPE flag
            if not hasattr(gym_cfg, 'posenc_EquivStableLapPE') or getattr(gym_cfg, 'posenc_EquivStableLapPE') is None:
                from yacs.config import CfgNode as CN
                gym_cfg.posenc_EquivStableLapPE = CN()
                gym_cfg.posenc_EquivStableLapPE.enable = False
        except Exception:
            pass

        # sensible gt defaults
        try:
            if getattr(gym_cfg.gt, 'layer_type', None) is None:
                gym_cfg.gt.layer_type = 'GINE+Transformer'
        except Exception:
            try:
                gym_cfg.gt.layer_type = 'GINE+Transformer'
            except Exception:
                pass

        try:
            if getattr(gym_cfg.gt, 'n_heads', None) is None:
                gym_cfg.gt.n_heads = int(getattr(args, 'n_heads', 1))
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gt, 'layers', None) is None:
                gym_cfg.gt.layers = int(getattr(args, 'n_layers', 1))
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gnn, 'layers_pre_mp', None) is None:
                gym_cfg.gnn.layers_pre_mp = 0
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gnn, 'act', None) is None:
                gym_cfg.gnn.act = 'relu'
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gt, 'dropout', None) is None:
                gym_cfg.gt.dropout = 0.0
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gt, 'attn_dropout', None) is None:
                gym_cfg.gt.attn_dropout = 0.0
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gt, 'layer_norm', None) is None:
                gym_cfg.gt.layer_norm = False
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gt, 'batch_norm', None) is None:
                gym_cfg.gt.batch_norm = False
        except Exception:
            pass

        try:
            if getattr(gym_cfg, 'train', None) is None:
                from yacs.config import CfgNode as CN
                gym_cfg.train = CN()
                gym_cfg.train.mode = ''
            elif getattr(gym_cfg.train, 'mode', None) is None:
                gym_cfg.train.mode = ''
        except Exception:
            pass

        try:
            if getattr(gym_cfg.gnn, 'head', None) is None:
                gym_cfg.gnn.head = 'san_graph'
        except Exception:
            pass

        # choose dim_in_for_model equal to configured inner dim when available
        dim_in_for_model = int(getattr(gym_cfg.gnn, 'dim_inner', 1) or 1)
    except Exception:
        dim_in_for_model = 1
    model = GPSModel(dim_in=dim_in_for_model, dim_out=1).to(device)

    class SubmoduleTrainer:
        def __init__(self, model, lr=1e-3, device="cpu", task_type="regression", loss: str | None = None):
            self.model = model
            self.device = torch.device(device)
            self.model.to(self.device)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)
            self.task_type = task_type
            loss_name = (loss or "").lower() if loss is not None else None
            if loss_name in ("bce", "bcewithlogits", "bce_with_logits"):
                self.loss_fn = nn.BCEWithLogitsLoss()
            elif loss_name in ("mse", "mse_loss"):
                self.loss_fn = nn.MSELoss()
            elif loss_name in ("mae", "l1", "l1loss"):
                self.loss_fn = nn.L1Loss()
            elif loss_name in ("rmse",):
                def _rmse(pred, target):
                    mse = nn.MSELoss()(pred, target)
                    return torch.sqrt(mse + 1e-8)
                self.loss_fn = _rmse
            else:
                self.loss_fn = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()

        def train_epoch(self, loader: DataLoader) -> float:
            self.model.train()
            total = 0.0
            n = 0
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                label = batch.y.view(-1, 1)
                loss = self.loss_fn(pred, label)
                self.opt.zero_grad()
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                else:
                    torch.tensor(float(loss), requires_grad=True).backward()
                self.opt.step()
                total += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                n += 1
            return total / n if n > 0 else 0.0

        @torch.no_grad()
        def evaluate(self, loader: DataLoader) -> Dict[str, float]:
            self.model.eval()
            total = 0.0
            total_mae = 0.0
            correct = 0
            samples = 0
            n = 0
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                label = batch.y.view(-1, 1)
                loss = self.loss_fn(pred, label)
                total += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                n += 1
                if self.task_type == "classification":
                    p = (torch.sigmoid(pred) > 0.5).float()
                    correct += (p == label).sum().item()
                    samples += label.size(0)
                else:
                    total_mae += torch.mean(torch.abs(pred - label)).item()
                    eval_metrics = getattr(self, 'eval_metrics', None) or []
                    if 'accuracy' in [m.lower() for m in (eval_metrics or [])]:
                        pred_round = torch.round(pred).to(torch.int64)
                        label_round = torch.round(label).to(torch.int64)
                        correct += (pred_round == label_round).sum().item()
                        samples += label.size(0)
            metrics = {"loss": total / n if n > 0 else float('nan')}
            if self.task_type == "classification":
                metrics["accuracy"] = correct / samples if samples > 0 else float('nan')
            else:
                metrics["mae"] = total_mae / n if n > 0 else float('nan')
                eval_metrics = getattr(self, 'eval_metrics', None) or []
                if 'accuracy' in [m.lower() for m in (eval_metrics or [])]:
                    metrics["accuracy"] = correct / samples if samples > 0 else float('nan')
            return metrics

    task_type = "classification" if args.task in ["cycle_check"] else "regression"
    trainer = SubmoduleTrainer(model, lr=getattr(args, 'learning_rate', 1e-3), device=device, task_type=task_type, loss=getattr(args, 'loss', None))
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
