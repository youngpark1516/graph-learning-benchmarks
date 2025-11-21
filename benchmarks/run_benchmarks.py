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


def train_transformer_epoch(model, dataloader, optimizer, device):
    return tu_train_transformer_epoch(model, dataloader, optimizer, device)


@torch.no_grad()
def eval_transformer_epoch(model, dataloader, device):
    return tu_eval_transformer_epoch(model, dataloader, device)


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--model_config", type=str, default=None, help="Path to JSON file with per-model overrides")

    args = parser.parse_args()
    add_repo_path()

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
        overrides = model_cfg.get(model_name, {}) if model_cfg else {}
        model_vars = vars(args).copy()
        # Apply overrides (shallow merge)
        model_vars.update(overrides)
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

            for epoch in range(args.epochs):
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

            if args.log_model:
                wandb.save(str(best_path))

        elif model_name == "graphgps":
            # lightweight GraphGPS baseline
            from graphgps import build_graphgps

            v = build_graphgps(model_args, device)
            trainer = v["trainer"]
            train_loader = v["train_loader"]
            valid_loader = v["valid_loader"]
            test_loader = v["test_loader"]

            for epoch in range(args.epochs):
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

            if args.log_model:
                wandb.save(str(best_path))

        else:
            v = build_transformer(model_args, device, model_name)
            model = v["model"]
            train_loader = v["train_loader"]
            valid_loader = v["valid_loader"]

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

            for epoch in range(args.epochs):
                train_loss, train_acc = train_transformer_epoch(model, train_loader, optimizer, device)
                valid_loss, valid_acc = eval_transformer_epoch(model, valid_loader, device)
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
            if args.log_model:
                wandb.save(str(best_path))

        # Save run config for this model and finish the wandb run
        cfg_path = out_dir / f"{run_name}_config.json"
        with cfg_path.open("w") as f:
            json.dump(vars(model_args), f, indent=2)
        wandb.save(str(cfg_path))
        wandb.finish()


if __name__ == "__main__":
    main()
