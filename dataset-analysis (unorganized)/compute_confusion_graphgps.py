#!/usr/bin/env python3
"""Compute per-label confusion (rounded predictions vs labels) for GraphGPS model.

Usage:
  python3 benchmarks/compute_confusion_graphgps.py --checkpoint models/graphgps_best.pt --task shortest_path --algorithm path --out viz/confusion_graphgps

Saves:
 - `confusion_counts.csv` (rows=labels, cols=predicted)
 - `confusion_heatmap.png`
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _P
# ensure repo root is on sys.path so we can import local modules
repo_root = str(_P(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from benchmarks.graphgps import build_graphgps


def compute_confusion(checkpoint: str, task: str, algorithm: str, out_dir: Path, device: str = 'cpu', hidden_dim: int | None = None, num_layers: int | None = None, data_dir: str = 'submodules/graph-token'):
    class Args:
        pass
    args = Args()
    args.data_dir = data_dir
    args.task = task
    args.algorithm = algorithm
    args.batch_size = 64
    args.learning_rate = 1e-3
    # allow overrides to match checkpoint architecture
    args.hidden_dim = hidden_dim or 64
    args.num_layers = num_layers or 3
    args.n_heads = 4
    args.dropout = 0.1
    args.mpnn_layers = 0
    args.mpnn_hidden_dim = None
    args.loss = None
    args.eval_metrics = ['accuracy']

    dev = device
    v = build_graphgps(args, dev)
    trainer = v['trainer']
    test_loader = v['test_loader']

    # load checkpoint
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
    trainer.load_checkpoint(str(ckpt))

    # run through test set and collect rounded preds
    preds = []
    labels = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(trainer.device)
            out = trainer.model(batch)
            # out shape: (B, 1)
            out_np = out.cpu().numpy().reshape(-1)
            lbl = batch.y.view(-1,1).cpu().numpy().reshape(-1)
            # round predictions to nearest int
            rounded = np.rint(out_np).astype(int)
            preds.extend(rounded.tolist())
            labels.extend(lbl.astype(int).tolist())

    preds = np.array(preds)
    labels = np.array(labels)

    # compute confusion counts
    min_label = min(labels.min(), preds.min())
    max_label = max(labels.max(), preds.max())
    # create mapping from label to index
    labels_range = list(range(int(min_label), int(max_label) + 1))
    idx_map = {v: i for i, v in enumerate(labels_range)}
    cm = np.zeros((len(labels_range), len(labels_range)), dtype=int)
    for t, p in zip(labels, preds):
        if t in idx_map and p in idx_map:
            cm[idx_map[t], idx_map[p]] += 1
        else:
            # if prediction outside range, expand? For simplicity clamp
            if p < min_label:
                cm[0,0] += 1
            elif p > max_label:
                cm[-1,-1] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    # save counts as dataframe
    df = pd.DataFrame(cm, index=labels_range, columns=labels_range)
    df.index.name = 'label_true'
    df.columns.name = 'label_pred'
    # sanitize task string for filenames (avoid embedding slashes as directories)
    safe_task = str(task).replace('/', '_')
    csv_path = out_dir / f'confusion_{safe_task}_{algorithm}_graphgps.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)

    # plot heatmap
    plt.figure(figsize=(10,8))
    plt.imshow(df.values, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted (rounded)')
    plt.ylabel('True label')
    plt.title(f'Confusion counts: {task}/{algorithm} (GraphGPS)')
    ticks = np.arange(len(labels_range))
    plt.xticks(ticks, labels_range, rotation=90)
    plt.yticks(ticks, labels_range)
    plt.tight_layout()
    png_path = out_dir / f'confusion_{safe_task}_{algorithm}_graphgps.png'
    plt.savefig(png_path, dpi=150)
    plt.close()
    print('Saved', csv_path, png_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--out', type=str, default='viz/confusion')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data-dir', type=str, default='submodules/graph-token', help='Root graph-token data directory (can point to a downsampled copy)')
    parser.add_argument('--hidden-dim', type=int, default=None, help='Override hidden dim for model construction')
    parser.add_argument('--num-layers', type=int, default=None, help='Override transformer num layers for model construction')
    args = parser.parse_args()
    # pass overrides into compute_confusion via environment: set attributes on a temporary args object inside function
    # We'll monkeypatch build_graphgps input by setting attributes on args namespace used in compute_confusion
    # For simplicity, supply overrides by writing them into a small config file or set module-level defaults.
    # Here we simply pass them by adding to the checkpoint filename parsing; instead, call compute_confusion and let it use CLI overrides
    # So set environment variables for now (compute_confusion reads only programmatic args). We'll instead re-run by importing the function.
    # call compute_confusion with explicit overrides
    compute_confusion(args.checkpoint, args.task, args.algorithm, Path(args.out), device=args.device, hidden_dim=args.hidden_dim, num_layers=args.num_layers, data_dir=args.data_dir)
