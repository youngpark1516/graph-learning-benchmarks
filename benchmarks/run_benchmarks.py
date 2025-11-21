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
    parser.add_argument("--model", type=str, choices=["mpnn", "graph_transformer", "autograph_transformer"], required=True)
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

    # Setup wandb
    run_name = args.run_name or f"{args.model}-{args.task}-{int(time.time())}"
    wandb.init(project=args.project, name=run_name, config=vars(args))

    best_valid = float("inf")
    best_path = out_dir / f"{args.model}_best.pt"

    if args.model == "mpnn":
        v = build_mpnn(args, device)
        trainer = v["trainer"]
        train_loader = v["train_loader"]
        valid_loader = v["valid_loader"]
        test_loader = v["test_loader"]

        for epoch in range(args.epochs):
            train_loss = trainer.train_epoch(train_loader)
            valid_metrics = trainer.evaluate(valid_loader)
            valid_loss = valid_metrics.get("loss", float("nan"))

            logd = {"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss}
            if v["task_type"] == "classification":
                logd["valid_accuracy"] = valid_metrics.get("accuracy")
            else:
                logd["valid_mae"] = valid_metrics.get("mae")

            wandb.log(logd)

            if valid_loss < best_valid:
                best_valid = valid_loss
                trainer.save_checkpoint(str(best_path))

        # load best and evaluate test
        trainer.load_checkpoint(str(best_path))
        test_metrics = trainer.evaluate(test_loader)
        wandb.log({"test_loss": test_metrics.get("loss"), "test_mae": test_metrics.get("mae", None), "test_accuracy": test_metrics.get("accuracy", None)})

        if args.log_model:
            wandb.save(str(best_path))

    else:
        v = build_transformer(args, device, args.model)
        model = v["model"]
        train_loader = v["train_loader"]
        valid_loader = v["valid_loader"]

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        for epoch in range(args.epochs):
            train_loss, train_acc = train_transformer_epoch(model, train_loader, optimizer, device)
            valid_loss, valid_acc = eval_transformer_epoch(model, valid_loader, device)
            scheduler.step()

            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "valid_loss": valid_loss, "valid_acc": valid_acc})

            # save best
            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save(model.state_dict(), str(best_path))

        # final test/validation snapshot
        wandb.save(str(best_path)) if args.log_model else None

    # Save run config and finish
    cfg_path = out_dir / f"{run_name}_config.json"
    with cfg_path.open("w") as f:
        json.dump(vars(args), f, indent=2)
    wandb.save(str(cfg_path))
    wandb.finish()


if __name__ == "__main__":
    main()
