#!/usr/bin/env python3
"""CLI tool to analyze graph tasks (based on the notebook analysis).

Saves CSV summaries and a few plots per task+algorithm into an output directory.
"""
import argparse
from pathlib import Path
import math
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_graph(path: Path) -> dict:
    try:
        if path.suffix == '.graphml':
            G = nx.read_graphml(path)
        elif path.suffix == '.json':
            # parse JSON produced by the `tasks_autograph` dataset format
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                return {"error": f"json load error: {e}", "path": str(path)}
            # support either a list of graph objects or a single object
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
            elif isinstance(data, dict):
                item = data
            else:
                return {"error": "unexpected json format", "path": str(path)}
            text = item.get('text', '')
            # edges appear before the '<n>' token separated by '<e>' markers
            edges_part = text.split('<n>')[0]
            # remove common tokens
            for tok in ['<bos>', '<eos>']:
                edges_part = edges_part.replace(tok, ' ')
            parts = edges_part.split()
            edges = []
            i = 0
            while i < len(parts) - 1:
                # skip non-numeric tokens
                if parts[i].startswith('<'):
                    i += 1
                    continue
                try:
                    u = int(parts[i])
                    v = int(parts[i+1])
                except Exception:
                    i += 1
                    continue
                edges.append((u, v))
                # advance past the pair and optional '<e>' token
                if i + 2 < len(parts) and parts[i+2] == '<e>':
                    i += 3
                else:
                    i += 2
            G = nx.Graph()
            if edges:
                G.add_edges_from(edges)
            else:
                # fall back to empty graph
                G = nx.Graph()
        else:
            return {"error": f"unsupported filetype: {path.suffix}", "path": str(path)}
    except Exception as e:
        return {"error": str(e), "path": str(path)}
    n = G.number_of_nodes()
    m = G.number_of_edges()
    directed = nx.is_directed(G)
    if n > 0:
        degrees = [d for _, d in G.degree()]
        avg_deg = float(np.mean(degrees))
        max_deg = int(np.max(degrees))
        min_deg = int(np.min(degrees))
        deg_median = float(np.median(degrees))
        deg_std = float(np.std(degrees, ddof=0))
    else:
        degrees = []
        avg_deg = max_deg = min_deg = deg_median = deg_std = 0
    try:
        if directed:
            comp_iter = nx.weakly_connected_components(G)
        else:
            comp_iter = nx.connected_components(G)
        comp_sizes = [len(c) for c in comp_iter]
        comps = len(comp_sizes)
        largest_cc = int(max(comp_sizes)) if comp_sizes else 0
        avg_cc_size = float(np.mean(comp_sizes)) if comp_sizes else 0.0
    except Exception:
        comps = None
        largest_cc = None
        avg_cc_size = None
    try:
        if directed:
            clu = nx.average_clustering(G.to_undirected())
        else:
            clu = nx.average_clustering(G)
    except Exception:
        clu = None
    return dict(path=str(path), nodes=int(n), edges=int(m), avg_deg=avg_deg, max_deg=max_deg,
                min_deg=min_deg, deg_median=deg_median, deg_std=deg_std, components=comps,
                largest_cc=largest_cc, avg_cc_size=avg_cc_size, directed=directed, clustering=clu)


def discover_algorithms(base: Path, task: str):
    task_dir = base / task
    if not task_dir.exists():
        return []
    algs = sorted([p.name for p in task_dir.iterdir() if p.is_dir()])
    return algs


def scan_and_analyze(base: Path, task: str, alg: str, limit_per_split: int):
    results = []
    ds_dir = base / task / alg
    if not ds_dir.exists():
        return results
    for split in sorted([p for p in ds_dir.iterdir() if p.is_dir()]):
        # accept both GraphML and the JSON format used in `tasks_autograph`
        files = sorted(list(split.glob('*.graphml')) + list(split.glob('*.json')))
        if not files:
            continue
        to_take = files[:limit_per_split]
        for p in to_take:
            res = analyze_graph(p)
            res['task'] = task
            res['algorithm'] = alg
            res['split'] = split.name
            results.append(res)
    return results


def save_summary(df: pd.DataFrame, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{prefix}_graph_summary.csv"
    df.to_csv(out_csv, index=False)
    print('Saved summary CSV to', out_csv)


def plots_for_df(df: pd.DataFrame, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if 'error' in df.columns:
        mask = df['path'].notnull() & df['error'].isna()
    else:
        mask = df['path'].notnull()
    plot_df = df[mask]
    if plot_df.empty:
        print('No graph data available to plot for', prefix)
        return
    datasets = sorted(plot_df['algorithm'].unique())
    # Nodes boxplot by algorithm
    groups = [plot_df.loc[plot_df['algorithm'] == d, 'nodes'].dropna().values for d in datasets]
    plt.figure(figsize=(10,5))
    plt.boxplot(groups, labels=datasets, showfliers=False, patch_artist=True)
    plt.xticks(rotation=45)
    plt.ylabel('Nodes')
    plt.title(f'Nodes distribution by algorithm ({prefix})')
    plt.tight_layout()
    p1 = out_dir / f"{prefix}_nodes_boxplot.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print('Saved', p1)

    # Edges vs Nodes scatter
    plt.figure(figsize=(8,6))
    for d in datasets:
        sub = plot_df[plot_df['algorithm'] == d]
        plt.scatter(sub['nodes'], sub['edges'], alpha=0.6, label=d, s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Nodes (log)')
    plt.ylabel('Edges (log)')
    plt.title(f'Edges vs Nodes ({prefix})')
    plt.legend(fontsize='small')
    plt.tight_layout()
    p2 = out_dir / f"{prefix}_edges_vs_nodes.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print('Saved', p2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='submodules/graph-token/graphs', help='Graphs base directory')
    parser.add_argument('--tasks', nargs='+', required=True, help='Task names to analyze (e.g., cycle_check)')
    parser.add_argument('--algorithms', nargs='*', help='Algorithm names (if omitted, discover under task)')
    parser.add_argument('--limit-per-split', type=int, default=200, help='Max graphml files per split to read')
    parser.add_argument('--out-dir', type=str, default='viz/tasks_analysis', help='Output directory for CSVs and plots')
    args = parser.parse_args()

    base = Path(args.base)
    out_base = Path(args.out_dir)

    all_rows = []
    for task in args.tasks:
        task_base = base
        algs = args.algorithms if args.algorithms else discover_algorithms(task_base, task)
        # if nothing found under the provided base, try the alternate `tasks_autograph` layout
        if not algs:
            alt = base.parent / 'tasks_autograph'
            if (alt / task).exists():
                print(f'No algorithms under {task_base / task}; trying alternate base {alt}')
                task_base = alt
                algs = discover_algorithms(task_base, task)
        if not algs:
            print(f'No algorithms found for task {task} under {task_base / task}, skipping')
            continue
        for alg in algs:
            print(f'Analyzing task={task} algorithm={alg} (limit {args.limit_per_split} per split)')
            rows = scan_and_analyze(task_base, task, alg, args.limit_per_split)
            if not rows:
                print('  No data for', task, alg)
                continue
            all_rows.extend(rows)
            df = pd.DataFrame(rows)
            # Convert numeric columns
            num_cols = ['nodes','edges','avg_deg','max_deg','min_deg','components','clustering','deg_median','deg_std','largest_cc','avg_cc_size']
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            # derived
            df.loc[:, 'density'] = df.apply(lambda r: (r['edges'] / (r['nodes'] * (r['nodes'] - 1))) if pd.notnull(r['nodes']) and r['nodes'] > 1 else 0.0, axis=1)
            df.loc[:, 'edges_per_node'] = df.apply(lambda r: (r['edges'] / r['nodes']) if pd.notnull(r['nodes']) and r['nodes'] > 0 else 0.0, axis=1)
            out_dir = out_base / f"{task}_{alg}"
            save_summary(df, out_dir, f"{task}_{alg}")
            plots_for_df(df, out_dir, f"{task}_{alg}")

    # global aggregated outputs
    if all_rows:
        gdf = pd.DataFrame(all_rows)
        out_dir = out_base / 'combined'
        save_summary(gdf, out_dir, 'combined')
        plots_for_df(gdf, out_dir, 'combined')


if __name__ == '__main__':
    main()
