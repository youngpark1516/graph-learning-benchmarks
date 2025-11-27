#!/usr/bin/env python3
"""Plot a confusion matrix CSV as a heatmap (optionally row/col normalized).

Usage:
  python3 benchmarks/plot_confusion.py --csv <in.csv> --out <out.png> --normalize row
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def load_confusion(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    # Ensure numeric
    df = df.fillna(0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
    return df


def normalize_df(df, mode):
    if mode == 'row':
        denom = df.sum(axis=1).replace(0, 1)
        return df.div(denom, axis=0)
    if mode == 'col':
        denom = df.sum(axis=0).replace(0, 1)
        return df.div(denom, axis=1)
    return df


def plot_heatmap(df, out_path, title=None, annot=False, cmap='viridis'):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig_w = max(6, df.shape[1] * 0.6)
    fig_h = max(4, df.shape[0] * 0.4)
    plt.figure(figsize=(fig_w, fig_h))
    if _HAS_SNS:
        sns.heatmap(df, cmap=cmap, annot=annot, fmt='.2f' if df.values.max() <= 1.0 else '.0f')
    else:
        im = plt.imshow(df.values, aspect='auto', cmap=cmap, interpolation='nearest')
        plt.colorbar(im)
        # ticks
        plt.xticks(np.arange(len(df.columns)), df.columns, rotation=45, ha='right')
        plt.yticks(np.arange(len(df.index)), df.index)
        if annot:
            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    val = df.iat[i, j]
                    txt = f"{val:.2f}" if df.values.max() <= 1.0 else f"{int(val)}"
                    plt.text(j, i, txt, ha='center', va='center', color='white' if val > df.values.max() * 0.6 else 'black')

    plt.xlabel('predicted')
    plt.ylabel('true')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Input confusion CSV (index = true labels)')
    p.add_argument('--out', required=True, help='Output PNG path')
    p.add_argument('--normalize', choices=['none', 'row', 'col'], default='row', help='Normalization mode')
    p.add_argument('--annot', action='store_true', help='Annotate cells')
    p.add_argument('--title', default=None, help='Optional title')
    args = p.parse_args()

    df = load_confusion(args.csv)
    norm = normalize_df(df, args.normalize)

    if args.normalize == 'none':
        title = args.title or 'Confusion matrix (counts)'
    else:
        title = args.title or f'Confusion matrix (normalized={args.normalize})'

    plot_heatmap(norm, args.out, title=title, annot=args.annot)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
