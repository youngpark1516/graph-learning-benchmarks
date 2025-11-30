
from pathlib import Path
import json
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.sparse import coo_matrix

from torch_geometric.nn import TransformerConv, global_mean_pool, GINConv
from torch_geometric.nn.norm import LayerNorm, GraphNorm, InstanceNorm
import wandb


def compute_laplacian_pe(edge_index, num_nodes, k=16):
    """Compute Laplacian Positional Encoding (LapPE).
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
        k: Number of eigenvectors to use
        
    Returns:
        Laplacian PE matrix [num_nodes, k]
    """
    try:
        from scipy.sparse.linalg import eigsh
        
        # Clamp k to a reasonable value - must be at least 1 and less than num_nodes - 2
        k = min(k, max(1, num_nodes - 2))
        
        # Build adjacency matrix
        row = edge_index[0].cpu().numpy()
        col = edge_index[1].cpu().numpy()
        data = np.ones(len(row))
        adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        adj = adj + adj.T  # Make symmetric (undirected)
        
        # Compute degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero
        inv_sqrt_deg = np.diag(1.0 / np.sqrt(degrees))
        
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        L = np.eye(num_nodes) - inv_sqrt_deg @ adj.toarray() @ inv_sqrt_deg
        
        # Get smallest k eigenvalues and eigenvectors
        try:
            if k > 0 and k < num_nodes - 1:
                eigenvalues, eigenvectors = eigsh(L, k=k, which='SM', maxiter=2000)
                pe = torch.from_numpy(eigenvectors).float()
            else:
                pe = torch.randn(num_nodes, k).float()
        except Exception:
            # Fallback: use random encoding if eigendecomposition fails
            pe = torch.randn(num_nodes, k).float()
            
    except Exception as e:
        # Fallback: random encoding
        k = max(1, min(k, num_nodes - 1))
        pe = torch.randn(num_nodes, k).float()
    
    return pe


def build_graphgps(args, device: str):
    """Build DataLoaders, model and trainer for a graph-token task.

    Args:
        args: parsed args with attributes used below
        device: device string like 'cpu' or 'cuda'

    Returns:
        dict with keys: train_loader, valid_loader, test_loader, trainer, task_type
    """
    # Use unified dataset loader
    from unified_dataset import PyGGraphDataset
    from torch_geometric.data import Batch

    data_dir = args.data_dir
    task = args.task
    algorithm = args.algorithm

    # Respect a broad set of configuration keys so `graphgps` can be
    # configured similarly to `mpnn` and transformer models.
    batch_size = getattr(args, "batch_size", 32)
    learning_rate = getattr(args, "learning_rate", 1e-3)
    hidden_dim = getattr(args, "hidden_dim", None) or getattr(args, "d_model", 64)
    # Accept either `num_layers` (mpnn style) or `n_layers` (transformer style).
    n_layers = max(1, getattr(args, "num_layers", None) or getattr(args, "n_layers", 3))
    n_heads = getattr(args, "n_heads", 4)
    dropout = getattr(args, "dropout", 0.1)
    
    # New PE and normalization options
    use_lap_pe = getattr(args, "use_lap_pe", False)
    lap_pe_dim = getattr(args, "lap_pe_dim", 16)
    norm_type = getattr(args, "norm_type", "batch")  # "batch", "layer", "graph", "instance", "none"

    def collate_fn(batch):
        """Collate function for PyG Data objects that adds labels."""
        data_list = []
        for item in batch:
            data, label = item
            data.y = label.unsqueeze(0) if label.dim() == 0 else label
            data_list.append(data)
        return Batch.from_data_list(data_list)

    train_dataset = PyGGraphDataset(
        data_path=data_dir,
        split="train",
        task=task,
        algorithm=algorithm if isinstance(algorithm, str) else algorithm[0],
        add_query_features=True
    )
    valid_dataset = PyGGraphDataset(
        data_path=data_dir,
        split="valid",
        task=task,
        algorithm=algorithm if isinstance(algorithm, str) else algorithm[0],
        add_query_features=True
    )
    test_dataset = PyGGraphDataset(
        data_path=data_dir,
        split="test",
        task=task,
        algorithm=algorithm if isinstance(algorithm, str) else algorithm[0],
        add_query_features=True
    )

    # Optionally limit dataset sizes
    from torch.utils.data import Subset
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"[GraphGPS] DataLoaders created:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Valid: {len(valid_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")

    class SimpleGPS(nn.Module):
        def __init__(
            self,
            in_channels: int = 1,
            hidden_dim: int = 64,
            n_layers: int = 3,
            n_heads: int = 4,
            out_dim: int = 1,
            dropout: float = 0.1,
            mpnn_layers: int = 0,
            mpnn_hidden_dim: int | None = None,
            use_lap_pe: bool = False,
            lap_pe_dim: int = 16,
            norm_type: str = "batch",
        ):
            super().__init__()
            self.use_lap_pe = use_lap_pe
            self.lap_pe_dim = lap_pe_dim
            self.norm_type = norm_type
            
            # Optional GIN MPNN front-end
            self.mpnn_layers_count = int(mpnn_layers or 0)
            self.mpnn_hidden_dim = int(mpnn_hidden_dim) if mpnn_hidden_dim is not None else None

            if self.mpnn_layers_count > 0:
                mp_h = self.mpnn_hidden_dim or hidden_dim
                self.mp_input_lin = nn.Linear(in_channels, mp_h)
                self.mpnn_gins = nn.ModuleList()
                self.mpnn_norms = nn.ModuleList()
                for _ in range(self.mpnn_layers_count):
                    nn_first = nn.Sequential(nn.Linear(mp_h, mp_h), nn.ReLU(), nn.Linear(mp_h, mp_h))
                    self.mpnn_gins.append(GINConv(nn_first))
                    self.mpnn_norms.append(self._get_norm_layer(mp_h))
                # project to transformer hidden dim if needed
                if mp_h != hidden_dim:
                    self.mp_to_tr = nn.Linear(mp_h, hidden_dim)
                else:
                    self.mp_to_tr = None
            else:
                self.mp_input_lin = None
                self.mpnn_gins = None
                self.mpnn_norms = None
                self.mp_to_tr = None

            # Positional Encoding projection if enabled
            if self.use_lap_pe:
                self.pe_lin = nn.Linear(lap_pe_dim, hidden_dim)
            
            # Transformer input projection (used when no mpnn front-end)
            self.input_lin = nn.Linear(in_channels, hidden_dim) if self.mpnn_layers_count == 0 else None

            # Transformer-style convolutions with normalization
            self.convs = nn.ModuleList()
            self.conv_norms = nn.ModuleList()
            for _ in range(n_layers):
                self.convs.append(TransformerConv(hidden_dim, max(1, hidden_dim // n_heads), heads=n_heads, dropout=dropout))
                self.conv_norms.append(self._get_norm_layer(hidden_dim))

            self.pool = global_mean_pool
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(), 
                nn.Dropout(dropout), 
                nn.Linear(hidden_dim, out_dim)
            )

        def _get_norm_layer(self, dim: int):
            """Get normalization layer based on norm_type."""
            if self.norm_type.lower() == "batch":
                return nn.BatchNorm1d(dim)
            elif self.norm_type.lower() == "layer":
                return LayerNorm(dim)
            elif self.norm_type.lower() == "graph":
                return GraphNorm(dim)
            elif self.norm_type.lower() == "instance":
                return InstanceNorm(dim)
            else:
                return nn.Identity()

        def forward(self, batch):
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

            # If MPNN front-end is configured, run GIN layers first
            if self.mpnn_layers_count > 0:
                x = self.mp_input_lin(x)
                for gin, norm in zip(self.mpnn_gins, self.mpnn_norms):
                    x_in = x
                    x = gin(x, edge_index)
                    x = norm(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=dropout, training=self.training)
                    x = x + x_in  # Residual connection
                if self.mp_to_tr is not None:
                    x = self.mp_to_tr(x)
            else:
                x = self.input_lin(x)
            
            # Add Laplacian PE if enabled
            if self.use_lap_pe:
                pe = compute_laplacian_pe(edge_index, batch.num_nodes, k=self.lap_pe_dim).to(x.device)
                # Only add PE for first batch (avoid dimension mismatch in batched inference)
                if pe.shape[0] == x.shape[0]:
                    pe_proj = self.pe_lin(pe)
                    x = x + pe_proj

            for conv, norm in zip(self.convs, self.conv_norms):
                x_in = x
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = x + x_in  # Residual connection
            
            x = self.pool(x, batch_idx)
            out = self.mlp(x)
            return out

    mpnn_layers = getattr(args, 'mpnn_layers', None) or getattr(args, 'mpnn_num_layers', None) or 0
    mpnn_hidden = getattr(args, 'mpnn_hidden_dim', None) or hidden_dim
    # Input features: degree (1) + query encoding (2) for shortest_path
    in_channels = 3 if "shortest_path" in task.lower() else 1
    
    print(f"[GraphGPS] Building model with:")
    print(f"  - Task: {task}")
    print(f"  - in_channels: {in_channels}")
    print(f"  - hidden_dim: {hidden_dim}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - n_heads: {n_heads}")
    print(f"  - mpnn_layers: {mpnn_layers}")
    print(f"  - use_lap_pe: {use_lap_pe}")
    print(f"  - lap_pe_dim: {lap_pe_dim if use_lap_pe else 'N/A'}")
    print(f"  - norm_type: {norm_type}")
    
    model = SimpleGPS(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        out_dim=1,
        dropout=dropout,
        mpnn_layers=mpnn_layers,
        mpnn_hidden_dim=mpnn_hidden,
        use_lap_pe=use_lap_pe,
        lap_pe_dim=lap_pe_dim,
        norm_type=norm_type,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[GraphGPS] Model created with {total_params} total params ({trainable_params} trainable)")

    class Trainer:
        def __init__(self, model, lr=1e-3, device="cpu", task_type="regression", loss: str | None = None):
            self.model = model
            self.device = torch.device(device)
            self.model.to(self.device)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=100)
            self.task_type = task_type

            # Configure loss function based on provided loss name or sensible defaults
            loss_name = (loss or "").lower() if loss is not None else None
            if loss_name:
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
                        # If caller requested accuracy for regression tasks, compute it
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
                # If accuracy was requested for regression tasks, include it
                eval_metrics = getattr(self, 'eval_metrics', None) or []
                if 'accuracy' in [m.lower() for m in (eval_metrics or [])]:
                    metrics["accuracy"] = correct / samples if samples > 0 else float('nan')
            return metrics

        def save_checkpoint(self, path: str):
            torch.save(self.model.state_dict(), path)

        def load_checkpoint(self, path: str):
            self.model.load_state_dict(torch.load(path, map_location=self.device))

    task_type = "classification" if task in ["cycle_check"] else "regression"
    trainer = Trainer(
        model,
        lr=getattr(args, "learning_rate", 1e-3),
        device=device,
        task_type=task_type,
        loss=getattr(args, "loss", None),
    )
    # Propagate eval_metrics from args so Trainer can compute optional metrics
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
    
    # New PE and normalization arguments
    parser.add_argument("--use_lap_pe", action="store_true", help="Enable Laplacian Positional Encoding")
    parser.add_argument("--lap_pe_dim", type=int, default=16, help="Dimension of Laplacian PE")
    parser.add_argument("--norm_type", type=str, default="batch", choices=["batch", "layer", "graph", "instance", "none"], 
                       help="Normalization type to use")

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