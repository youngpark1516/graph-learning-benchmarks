"""Run and log benchmarks for available models using Weights & Biases (wandb).

Supported models: 'mpnn', 'graph_transformer', 'autograph_transformer'.

This script creates a unified CLI to train/evaluate a selected model, logs
epoch metrics to wandb, and saves the best model checkpoint.

Run example:
  python run_benchmarks.py --model mpnn --task edge_count --algorithm ba --project my-project
"""

from pathlib import Path
import argparse
import time
import json

import torch
import wandb

# Import modular helpers
from common import add_repo_path, _make_standard_log, _make_standard_test_log
from builders import build_mpnn as builders_build_mpnn, build_transformer as builders_build_transformer
from train_utils import train_transformer_epoch as tu_train_transformer_epoch, eval_transformer_epoch as tu_eval_transformer_epoch


def build_mpnn(args, device):
    return builders_build_mpnn(args, device)


def build_transformer(args, device, which):
    return builders_build_transformer(args, device, which)


def train_transformer_epoch(model, dataloader, optimizer, device, loss_name: str | None = None):
    return tu_train_transformer_epoch(model, dataloader, optimizer, device, loss_name=loss_name)


@torch.no_grad()
def eval_transformer_epoch(model, dataloader, device, loss_name: str | None = None):
    return tu_eval_transformer_epoch(model, dataloader, device, loss_name=loss_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["mpnn", "graph_transformer", "autograph_transformer", "graphgps"], required=False)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to run (overrides --model).")
    parser.add_argument("--group", type=str, default=None, help="WandB group name to group model runs for comparison.")
    parser.add_argument("--data_dir", type=str, default="/data/young/capstone/graph-learning-benchmarks/submodules/graph-token")
    parser.add_argument("--task", type=str, default="edge_count")
    parser.add_argument("--algorithm", type=str, default="ba")
    parser.add_argument("--project", type=str, default="graph-benchmarks")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss", type=str, default=None, help="Loss name to use for this run (e.g. mse, mae, rmse, bce, cross_entropy)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--model_config", type=str, default=None, help="Path to JSON file with per-model overrides")
    parser.add_argument("--log_model", action="store_true", help="Save best model artifact for each run")

    args = parser.parse_args()
    add_repo_path()

    # Default values used when a per-model config or CLI doesn't provide them.
    DEFAULT_MODEL_ARGS = {
        "hidden_dim": 64,
        "num_layers": 4,
        "d_model": 256,
        "n_heads": 8,
        "d_ff": 1024,
        "n_layers": 6,
        "dropout": 0.1,
        "max_seq_length": 512,
        "n_samples_per_file": -1,
        "sampling_seed": 1234,
        "loss": None,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 10,
        "log_model": False,
        "run_name": None,
    }

    # Auto-discover model config if not explicitly provided. Prefer YAML.
    if not args.model_config:
        # look for common locations under the benchmarks/ folder
        candidates = [
            Path(__file__).resolve().parent / "model_configs.yaml",
        ]
        for p in candidates:
            if p.exists():
                args.model_config = str(p)
                print(f"Auto-detected model_config: {args.model_config}")
                break

    # Load optional per-model config (JSON or YAML) (overrides per model).
    model_cfg = {}
    if args.model_config:
        try:
            with open(args.model_config, 'r') as f:
                txt = f.read()
        except Exception as e:
            print(f"Failed to open model_config {args.model_config}: {e}")
            txt = None

        if txt:
            # Try JSON first, then YAML if JSON parsing fails.
            try:
                model_cfg = json.loads(txt)
            except Exception:
                try:
                    import yaml  # lazy import to avoid hard dependency
                    model_cfg = yaml.safe_load(txt) or {}
                except ModuleNotFoundError:
                    print("PyYAML not installed; install with 'pip install pyyaml' to use YAML model configs")
                    model_cfg = {}
                except Exception as e:
                    print(f"Failed to parse model_config {args.model_config} as YAML: {e}")
                    model_cfg = {}
    # Extract top-level `global` config if present. These values will be merged
    # into each model's per-model overrides (per-model keys win on conflict).
    global_cfg = {}
    if isinstance(model_cfg, dict) and 'global' in model_cfg:
        try:
            global_cfg = model_cfg.get('global', {}) or {}
        except Exception:
            global_cfg = {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to run
    if args.models:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        models_to_run = [args.model]
    else:
        models_to_run = ["mpnn", "graph_transformer", "autograph_transformer", "graphgps"]

    # Group identifier for wandb comparisons
    group_id = args.group or f"bench-{int(time.time())}"

    for model_name in models_to_run:
        # Merge per-model overrides into a fresh args namespace so each model
        # run receives its own hyperparameters.
        # Merge `global` config with per-model overrides. Per-model keys override global ones.
        if isinstance(model_cfg, dict):
            per_model = model_cfg.get(model_name, {}) or {}
        else:
            per_model = {}
        overrides = dict(global_cfg) if isinstance(global_cfg, dict) else {}
        if isinstance(per_model, dict):
            overrides.update(per_model)
        model_vars = vars(args).copy()
        # Apply overrides (shallow merge)
        model_vars.update(overrides)
        # Ensure defaults exist so the per-model Namespace has all expected attributes
        for k, v in DEFAULT_MODEL_ARGS.items():
            model_vars.setdefault(k, v)
        # Create a Namespace for builder convenience
        model_args = argparse.Namespace(**model_vars)

        # Make per-model unique run names. If the caller provided --run_name (or
        # the model config provided one), append the model name and timestamp.
        if getattr(model_args, 'run_name', None):
            run_name = f"{model_args.run_name}-{model_name}-{int(time.time())}"
        else:
            run_name = f"{model_name}-{model_args.task}-{int(time.time())}"

        # Check that the task data exists before starting a run
        train_dir = Path(model_args.data_dir) / "tasks_autograph" / model_args.task / model_args.algorithm / "train"
        if not train_dir.exists():
            print(f"Data not found for task '{model_args.task}/{model_args.algorithm}' at: {train_dir}. Skipping model '{model_name}'.")
            continue

        # initialize a separate wandb run per model, grouped together
        wandb.init(project=model_args.project, name=run_name, group=group_id, config=vars(model_args))

        # Resolve device per-model (allow overrides in model config)
        device = torch.device(model_args.device)

        best_valid = float("inf")
        best_path = out_dir / f"{model_name}_best.pt"

        if model_name == "mpnn":
            v = build_mpnn(model_args, device)
            trainer = v["trainer"]
            train_loader = v["train_loader"]
            valid_loader = v["valid_loader"]
            test_loader = v["test_loader"]

            # Compute and log parameter counts for the MPNN model
            try:
                model_obj = trainer.model
                total_params = sum(p.numel() for p in model_obj.parameters())
                trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
                print(f"MPNN params: total={total_params}, trainable={trainable_params}")
                wandb.log({"param_count/total": total_params, "param_count/trainable": trainable_params})
            except Exception:
                pass

            for epoch in range(model_args.epochs):
                # train step
                _ = trainer.train_epoch(train_loader)

                # Gather consistent train/valid metrics dictionaries
                train_metrics = trainer.evaluate(train_loader)
                valid_metrics = trainer.evaluate(valid_loader)

                logd = _make_standard_log(epoch, model_name, v["task_type"], train_metrics, valid_metrics)
                wandb.log(logd)

                valid_loss = valid_metrics.get("loss", float("inf"))
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    trainer.save_checkpoint(str(best_path))

            # load best and evaluate test
            trainer.load_checkpoint(str(best_path))
            test_metrics = trainer.evaluate(test_loader)
            wandb.log(_make_standard_test_log(model_name, v["task_type"], test_metrics))

            if getattr(model_args, 'log_model', False):
                wandb.save(str(best_path))

        elif model_name == "graphgps":
            # lightweight GraphGPS baseline
            from graphgps import build_graphgps

            v = build_graphgps(model_args, device)
            trainer = v["trainer"]
            train_loader = v["train_loader"]
            valid_loader = v["valid_loader"]
            test_loader = v["test_loader"]

            # Compute and log parameter counts for the GraphGPS model
            try:
                model_obj = trainer.model
                total_params = sum(p.numel() for p in model_obj.parameters())
                trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
                print(f"GraphGPS params: total={total_params}, trainable={trainable_params}")
                wandb.log({"param_count/total": total_params, "param_count/trainable": trainable_params})
            except Exception:
                pass

            for epoch in range(model_args.epochs):
                # train step
                _ = trainer.train_epoch(train_loader)

                # Gather consistent train/valid metrics dictionaries
                train_metrics = trainer.evaluate(train_loader)
                valid_metrics = trainer.evaluate(valid_loader)

                logd = _make_standard_log(epoch, model_name, v["task_type"], train_metrics, valid_metrics)
                wandb.log(logd)

                valid_loss = valid_metrics.get("loss", float("inf"))
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    trainer.save_checkpoint(str(best_path))

            trainer.load_checkpoint(str(best_path))
            test_metrics = trainer.evaluate(test_loader)
            wandb.log(_make_standard_test_log(model_name, v["task_type"], test_metrics))

            if getattr(model_args, 'log_model', False):
                wandb.save(str(best_path))

        else:
            v = build_transformer(model_args, device, model_name)
            model = v["model"]
            train_loader = v["train_loader"]
            valid_loader = v["valid_loader"]

            # Compute and log parameter counts for transformer models
            try:
                model_obj = model
                total_params = sum(p.numel() for p in model_obj.parameters())
                trainable_params = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
                print(f"{model_name} params: total={total_params}, trainable={trainable_params}")
                wandb.log({"param_count/total": total_params, "param_count/trainable": trainable_params})
            except Exception:
                pass

            optimizer = torch.optim.Adam(model.parameters(), lr=model_args.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, model_args.epochs))

            for epoch in range(model_args.epochs):
                train_loss, train_acc = train_transformer_epoch(model, train_loader, optimizer, device, loss_name=getattr(model_args, 'loss', None))
                valid_loss, valid_acc = eval_transformer_epoch(model, valid_loader, device, loss_name=getattr(model_args, 'loss', None))
                scheduler.step()

                # Build standardized metric dicts for transformer
                train_metrics = {"loss": train_loss, "accuracy": train_acc}
                valid_metrics = {"loss": valid_loss, "accuracy": valid_acc}

                logd = _make_standard_log(epoch, model_name, "classification", train_metrics, valid_metrics)
                wandb.log(logd)

                # save best
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    torch.save(model.state_dict(), str(best_path))

            # load best and evaluate test set for transformer
            try:
                model.load_state_dict(torch.load(str(best_path)))
                test_loader = v.get("test_loader")
                if test_loader is not None:
                    test_loss, test_acc = eval_transformer_epoch(model, test_loader, device)
                    test_metrics = {"loss": test_loss, "accuracy": test_acc}
                    wandb.log(_make_standard_test_log(model_name, "classification", test_metrics))
            except Exception:
                # If loading or evaluation fails, log nothing but continue
                pass

            # final test/validation snapshot
            if getattr(model_args, 'log_model', False):
                wandb.save(str(best_path))

        # Save run config for this model and finish the wandb run
        cfg_path = out_dir / f"{run_name}_config.json"
        with cfg_path.open("w") as f:
            json.dump(vars(model_args), f, indent=2)
        wandb.save(str(cfg_path))
        wandb.finish()


if __name__ == "__main__":
    main()
