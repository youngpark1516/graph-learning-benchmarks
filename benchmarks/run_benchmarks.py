"""Run and log benchmarks for available models using Weights & Biases (wandb).

Supported models: 'mpnn', 'graph_transformer', 'autograph_transformer'.

This script creates a unified CLI to train/evaluate a selected model, logs
epoch metrics to wandb, and saves the best model checkpoint.

Run example:
  python run_benchmarks.py --model mpnn --task edge_count --algorithm ba --project my-project
"""

from pathlib import Path
import sys
import argparse
import time
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb


def add_repo_path():
    # Ensure local benchmarks/ modules importable
    repo_dir = Path(__file__).resolve().parent
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def build_mpnn(args, device):
    from mpnn import GraphTaskDataset, GIN, GraphMPNNTrainer, collate_fn

    train_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "train")
    valid_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "valid")
    test_dataset = GraphTaskDataset(args.data_dir, args.task, args.algorithm, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GIN(in_features=1, hidden_dim=args.hidden_dim, num_layers=args.num_layers, out_features=1, dropout=0.5)
    task_type = "classification" if args.task in ["cycle_check"] else "regression"
    trainer = GraphMPNNTrainer(model, learning_rate=args.learning_rate, device=device, task_type=task_type)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": task_type,
    }


def build_transformer(args, device, which):
    if which == "graph_transformer":
        from graph_transformer import GraphDataset, GraphTransformer

        Dataset = GraphDataset
        Transformer = GraphTransformer
    else:
        # autograph_transformer
        from autograph_transformer import GraphDataset as AGDataset, GraphTransformer as AGTransformer

        Dataset = AGDataset
        Transformer = AGTransformer

    train_dataset = Dataset(args.data_dir, args.task, args.algorithm, "train", max_seq_length=args.max_seq_length)
    valid_dataset = Dataset(args.data_dir, args.task, args.algorithm, "valid", max_seq_length=args.max_seq_length)

    # Share vocabulary
    try:
        valid_dataset.token2idx = train_dataset.token2idx
        valid_dataset.idx2token = train_dataset.idx2token
        valid_dataset.vocab_size = train_dataset.vocab_size
    except Exception:
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    model = Transformer(vocab_size=train_dataset.vocab_size, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout, max_seq_length=args.max_seq_length).to(device)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "model": model,
    }


def train_transformer_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)  # (B, S, V)
        lengths = attention_mask.long().sum(dim=1)
        p_pos = lengths - 1
        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]

        loss = F.cross_entropy(logits_at_p, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits_at_p.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * bsz
        total_count += bsz

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    return avg_loss, acc


@torch.no_grad()
def eval_transformer_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        lengths = attention_mask.long().sum(dim=1)
        p_pos = lengths - 1
        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]

        loss = F.cross_entropy(logits_at_p, labels)
        preds = logits_at_p.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * bsz
        total_count += bsz

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    return avg_loss, acc


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
    # model hyperparams
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--log_model", action="store_true")

    args = parser.parse_args()
    add_repo_path()

    device = torch.device(args.device)
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

    def _make_standard_log(epoch, model_name, task_type, train_metrics, valid_metrics):
        # Ensure consistent keys for wandb across all models. Values may be None
        # when not applicable or not provided by the underlying trainer.
        return {
            "epoch": epoch + 1,
            "model": model_name,
            "train_loss": train_metrics.get("loss") if train_metrics else None,
            "valid_loss": valid_metrics.get("loss") if valid_metrics else None,
            "train_accuracy": train_metrics.get("accuracy") if (train_metrics and task_type == "classification") else None,
            "valid_accuracy": valid_metrics.get("accuracy") if (valid_metrics and task_type == "classification") else None,
            "train_mae": train_metrics.get("mae") if (train_metrics and task_type != "classification") else None,
            "valid_mae": valid_metrics.get("mae") if (valid_metrics and task_type != "classification") else None,
        }

    for model_name in models_to_run:
        # Make per-model unique run names. If the caller provided --run_name,
        # append the model name and a timestamp to avoid collisions. Otherwise
        # fall back to the default naming that includes task and timestamp.
        if args.run_name:
            run_name = f"{args.run_name}-{model_name}-{int(time.time())}"
        else:
            run_name = f"{model_name}-{args.task}-{int(time.time())}"

        # Check that the task data exists before starting a run
        train_dir = Path(args.data_dir) / "tasks_autograph" / args.task / args.algorithm / "train"
        if not train_dir.exists():
            print(f"Data not found for task '{args.task}/{args.algorithm}' at: {train_dir}. Skipping model '{model_name}'.")
            continue

        # initialize a separate wandb run per model, grouped together
        wandb.init(project=args.project, name=run_name, group=group_id, config=vars(args))

        best_valid = float("inf")
        best_path = out_dir / f"{model_name}_best.pt"

        if model_name == "mpnn":
            v = build_mpnn(args, device)
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
            wandb.log({
                "model": model_name,
                "test_loss": test_metrics.get("loss"),
                "test_mae": test_metrics.get("mae", None),
                "test_accuracy": test_metrics.get("accuracy", None),
            })

            if args.log_model:
                wandb.save(str(best_path))

        elif model_name == "graphgps":
            # lightweight GraphGPS baseline
            from graphgps import build_graphgps

            v = build_graphgps(args, device)
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
            wandb.log({
                "model": model_name,
                "test_loss": test_metrics.get("loss"),
                "test_mae": test_metrics.get("mae", None),
                "test_accuracy": test_metrics.get("accuracy", None),
            })

            if args.log_model:
                wandb.save(str(best_path))

        else:
            v = build_transformer(args, device, model_name)
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

            # final test/validation snapshot
            if args.log_model:
                wandb.save(str(best_path))

        # Save run config for this model and finish the wandb run
        cfg_path = out_dir / f"{run_name}_config.json"
        with cfg_path.open("w") as f:
            json.dump(vars(args), f, indent=2)
        wandb.save(str(cfg_path))
        wandb.finish()


if __name__ == "__main__":
    main()
