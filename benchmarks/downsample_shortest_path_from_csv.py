#!/usr/bin/env python3
"""Downsample shortest_path examples by prediction token using the query_prediction CSV.

Reads `viz/query_analysis/query_prediction_by_graph.csv`, selects rows where
`task=='shortest_path'` and `prediction` matches `len\d+`, then samples up to
`--cap` examples per `len*` label and copies the JSON files into an output
directory preserving `<algo>/<split>/filename.json`.

Example:
  python3 benchmarks/downsample_shortest_path_from_csv.py --cap 10000 --out submodules/graph-token/tasks_autograph/shortest_path_downsampled_10k
"""
import argparse
import csv
import os
import random
import re
import shutil
from collections import defaultdict, Counter


def load_rows(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def group_by_label(rows):
    groups = defaultdict(list)
    pattern = re.compile(r'^len\d+$')
    for r in rows:
        if r.get('task') != 'shortest_path':
            continue
        pred = (r.get('prediction') or '').strip()
        if not pattern.match(pred):
            continue
        groups[pred].append(r)
    return groups


def sample_groups(groups, cap, seed=12345):
    random.seed(seed)
    sampled = {}
    for label, items in groups.items():
        if len(items) <= cap:
            sampled[label] = list(items)
        else:
            sampled[label] = random.sample(items, cap)
    return sampled


def copy_selected(sampled, out_root):
    counts = Counter()
    for label, items in sampled.items():
        for r in items:
            path = r.get('path')
            if not path:
                continue
            # deduce algo and split from path if possible, else copy into label dir
            # path example: submodules/graph-token/tasks_autograph/shortest_path/path/train/xxx.json
            rel = None
            try:
                parts = path.split('tasks_autograph')
                if len(parts) > 1:
                    rel = parts[1].lstrip(os.sep)
            except Exception:
                rel = None
            if rel:
                dst_dir = os.path.join(out_root, rel)
                # rel includes algo/split/filename
                dst_dir = os.path.dirname(dst_dir)
            else:
                dst_dir = os.path.join(out_root, label)
            os.makedirs(dst_dir, exist_ok=True)
            fname = os.path.basename(path)
            dst_path = os.path.join(dst_dir, fname)
            try:
                shutil.copy2(path, dst_path)
                counts[label] += 1
            except FileNotFoundError:
                # file may not exist locally; skip
                continue
    return counts


def write_summary(groups, sampled, counts, out_root):
    out_csv = os.path.join(out_root, 'downsample_summary.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['label', 'original_count', 'selected_count'])
        for label in sorted(groups.keys(), key=lambda x: int(x.replace('len',''))):
            orig = len(groups[label])
            sel = counts.get(label, 0)
            w.writerow([label, orig, sel])
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='viz/query_analysis/query_prediction_by_graph.csv')
    p.add_argument('--cap', type=int, default=10000, help='Maximum examples to keep per len label')
    p.add_argument('--out', required=True, help='Output directory to write downsampled copies')
    p.add_argument('--seed', type=int, default=12345)
    args = p.parse_args()

    print('Loading rows from', args.csv)
    rows = load_rows(args.csv)
    print('Total rows loaded:', len(rows))

    groups = group_by_label(rows)
    print('Found labels:', ','.join(sorted(groups.keys(), key=lambda x: int(x.replace('len','')))))
    for k, v in sorted(groups.items(), key=lambda x: int(x[0].replace('len',''))):
        print(k, len(v))

    sampled = sample_groups(groups, args.cap, seed=args.seed)
    out_root = args.out
    os.makedirs(out_root, exist_ok=True)
    counts = copy_selected(sampled, out_root)
    summary_csv = write_summary(groups, sampled, counts, out_root)

    total_selected = sum(counts.values())
    print('Wrote', total_selected, 'files to', out_root)
    print('Summary CSV:', summary_csv)


if __name__ == '__main__':
    main()
