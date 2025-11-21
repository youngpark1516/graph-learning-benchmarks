"""Clean lightweight GraphGPS-like baseline for graph-token data.

This file provides:
- `build_graphgps(args, device)` to construct DataLoaders and a Trainer
- `SimpleGPS` model based on TransformerConv + global pooling
- A CLI that logs to Weights & Biases (wandb)

Keep this file runnable and minimal so it can be imported by `run_benchmarks.py`.
"""

from pathlib import Path
import json
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.nn import TransformerConv, global_mean_pool
import wandb


def build_graphgps(args, device: str):
    """Build DataLoaders, model and trainer for a graph-token task.

    Args:
        args: parsed args with attributes used below
        device: device string like 'cpu' or 'cuda'

    Returns:
        dict with keys: train_loader, valid_loader, test_loader, trainer, task_type
    """
    # Reuse dataset and collate_fn from benchmarks/mpnn.py
    from mpnn import GraphTaskDataset, collate_fn

    data_dir = args.data_dir
    task = args.task
    algorithm = args.algorithm

    train_dataset = GraphTaskDataset(data_dir, task, algorithm, "train")
    valid_dataset = GraphTaskDataset(data_dir, task, algorithm, "valid")
    test_dataset = GraphTaskDataset(data_dir, task, algorithm, "test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    class SimpleGPS(nn.Module):
        def __init__(self, in_channels=1, hidden_dim=64, n_layers=3, n_heads=4, out_dim=1, dropout=0.1):
            super().__init__()
            self.input_lin = nn.Linear(in_channels, hidden_dim)
            self.convs = nn.ModuleList()
            for _ in range(n_layers):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim // n_heads, heads=n_heads, dropout=dropout))
            self.pool = global_mean_pool
            self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

        def forward(self, batch):
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            x = self.input_lin(x)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
            x = self.pool(x, batch_idx)
            out = self.mlp(x)
            return out

    model = SimpleGPS(in_channels=1, hidden_dim=getattr(args, "hidden_dim", 64), n_layers=max(2, getattr(args, "num_layers", 3)), n_heads=4, out_dim=1, dropout=0.1)
    model = model.to(device)

    class Trainer:
        def __init__(self, model, lr=1e-3, device="cpu", task_type="regression"):
            self.model = model
            self.device = torch.device(device)
            self.model.to(self.device)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)
            self.task_type = task_type
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
                loss.backward()
                self.opt.step()
                total += loss.item()
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
                total += loss.item()
                n += 1
                if self.task_type == "classification":
                    p = (torch.sigmoid(pred) > 0.5).float()
                    correct += (p == label).sum().item()
                    samples += label.size(0)
                else:
                    total_mae += torch.mean(torch.abs(pred - label)).item()

            metrics = {"loss": total / n if n > 0 else float('nan')}
            if self.task_type == "classification":
                metrics["accuracy"] = correct / samples if samples > 0 else float('nan')
            else:
                metrics["mae"] = total_mae / n if n > 0 else float('nan')
            return metrics

        def save_checkpoint(self, path: str):
            torch.save(self.model.state_dict(), path)

        def load_checkpoint(self, path: str):
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    task_type = "classification" if task in ["cycle_check"] else "regression"
    trainer = Trainer(model, lr=getattr(args, "learning_rate", 1e-3), device=device, task_type=task_type)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "trainer": trainer,
        "task_type": task_type,
    }


def _cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/young/capstone/graph-learning-benchmarks/submodules/graph-token")
    parser.add_argument("--task", type=str, default="edge_count")
    parser.add_argument("--algorithm", type=str, default="ba")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--project", type=str, default="graph-benchmarks")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--log_model", action="store_true")
    

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v = build_graphgps(args, args.device)
    trainer = v["trainer"]
    train_loader = v["train_loader"]
    valid_loader = v["valid_loader"]
    test_loader = v["test_loader"]

    best_valid = float('inf')
    best_path = out_dir / "graphgps_best.pt"

    run_name = args.run_name or f"graphgps-{args.task}-{int(time.time())}"
    wandb.init(project=args.project, name=run_name, group=args.group, config=vars(args))

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(train_loader)
        valid_metrics = trainer.evaluate(valid_loader)
        valid_loss = valid_metrics.get("loss", float('nan'))
        print(f"Epoch {epoch+1}/{args.epochs} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            trainer.save_checkpoint(str(best_path))

        logd = {"epoch": epoch + 1, "train_loss": train_loss, "valid_loss": valid_loss}
        if v["task_type"] == "classification":
            logd["valid_accuracy"] = valid_metrics.get("accuracy")
        else:
            logd["valid_mae"] = valid_metrics.get("mae")
        wandb.log(logd)

    trainer.load_checkpoint(str(best_path))
    test_metrics = trainer.evaluate(test_loader)
    print("Test metrics:", test_metrics)

    wandb.log({"test_loss": test_metrics.get("loss"), "test_mae": test_metrics.get("mae", None), "test_accuracy": test_metrics.get("accuracy", None)})

    cfg_path = out_dir / f"{run_name}_config.json"
    with cfg_path.open("w") as f:
        json.dump(vars(args), f, indent=2)
    wandb.save(str(cfg_path))

    if args.log_model:
        try:
            art = wandb.Artifact(f"{run_name}-model", type="model")
            art.add_file(str(best_path))
            wandb.log_artifact(art)
        except Exception:
            wandb.save(str(best_path))

    wandb.finish()


if __name__ == "__main__":
    _cli()