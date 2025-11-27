#!/usr/bin/env python3
"""Generate plots for query/prediction analysis outputs produced by
`benchmarks/analyze_query_prediction.py`.

Saves PNGs into the same output directory.
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_top_counts(counter_series, out_path: Path, title: str, xlabel: str='count', top_n: int=20):
    top = counter_series.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10,6))
    bars = plt.barh(range(len(top)), top.values[::-1], color='#4c72b0')
    plt.yticks(range(len(top)), top.index[::-1])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-csv', type=str, default='viz/query_analysis/query_prediction_by_graph.csv')
    parser.add_argument('--out-dir', type=str, default='viz/query_analysis')
    parser.add_argument('--top-n', type=int, default=20)
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_csv.exists():
        print('Input CSV not found:', in_csv)
        return

    df = pd.read_csv(in_csv)
    # normalize None/NaN
    df['query'] = df['query'].fillna('')
    df['prediction'] = df['prediction'].fillna('')

    # global top predictions
    pred_counts = df['prediction'].value_counts()
    plot_top_counts(pred_counts, out_dir / 'top_predictions_global.png', 'Top Predictions (global)', top_n=args.top_n)

    # global top queries
    q_counts = df['query'].value_counts()
    plot_top_counts(q_counts, out_dir / 'top_queries_global.png', 'Top Queries (global)', top_n=args.top_n)

    # per-task top predictions and queries
    for task, group in df.groupby('task'):
        pc = group['prediction'].value_counts()
        qc = group['query'].value_counts()
        if not pc.empty:
            plot_top_counts(pc, out_dir / f'top_predictions_{task}.png', f'Top Predictions ({task})', top_n=args.top_n)
        if not qc.empty:
            plot_top_counts(qc, out_dir / f'top_queries_{task}.png', f'Top Queries ({task})', top_n=args.top_n)

    # small stacked bar of prediction length distribution (lenN and INF)
    # detect length tokens starting with 'len' or 'INF'
    pred_lengths = df['prediction'].str.extract(r'^(len(\d+)|INF)$')
    # when matched, group by the captured group
    length_series = df['prediction'].where(df['prediction'].str.startswith('len', na=False) | (df['prediction']=='INF'), other='other')
    length_counts = length_series.value_counts()
    plt.figure(figsize=(10,6))
    length_counts.plot(kind='bar', color='#55a868')
    plt.title('Prediction length labels distribution')
    plt.xlabel('label')
    plt.ylabel('count')
    plt.tight_layout()
    out_len = out_dir / 'prediction_length_distribution.png'
    plt.savefig(out_len, dpi=150)
    plt.close()
    print('Saved', out_len)

    print('All plots generated in', out_dir)

if __name__ == '__main__':
    main()
