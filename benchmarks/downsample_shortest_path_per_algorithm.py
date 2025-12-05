import argparse
import csv
import os
import random
import re
import shutil
from collections import defaultdict, Counter


def load_rows(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        return list(r)


def group_by_algo_label(rows):
    pattern = re.compile(r'^len\d+$')
    groups = defaultdict(list)
    for r in rows:
        if r.get('task') != 'shortest_path':
            continue
        pred = (r.get('prediction') or '').strip()
        if not pattern.match(pred):
            continue
        algo = (r.get('algorithm') or 'UNKNOWN')
        groups[(algo, pred)].append(r)
    return groups


def sample_per_group(groups, cap, seed=12345):
    random.seed(seed)
    sampled = {}
    for key, items in groups.items():
        if len(items) <= cap:
            sampled[key] = list(items)
        else:
            sampled[key] = random.sample(items, cap)
    return sampled


def copy_selected(sampled, out_root):
    counts = Counter()
    for (algo, label), items in sampled.items():
        for r in items:
            path = r.get('path')
            if not path:
                continue
            rel = None
            try:
                parts = path.split('tasks_autograph')
                if len(parts) > 1:
                    rel = parts[1].lstrip(os.sep)
            except Exception:
                rel = None
            if rel:
                dst_dir = os.path.join(out_root, rel)
                dst_dir = os.path.dirname(dst_dir)
            else:
                dst_dir = os.path.join(out_root, algo)
            os.makedirs(dst_dir, exist_ok=True)
            fname = os.path.basename(path)
            dst_path = os.path.join(dst_dir, fname)
            try:
                shutil.copy2(path, dst_path)
                counts[(algo, label)] += 1
            except FileNotFoundError:
                continue
    return counts


def write_summary(groups, sampled, counts, out_root):
    out_csv = os.path.join(out_root, 'downsample_per_algorithm_summary.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'label', 'original_count', 'selected_count'])
        for (algo, label), items in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1].replace('len','')))):
            orig = len(items)
            sel = counts.get((algo, label), 0)
            w.writerow([algo, label, orig, sel])
    return out_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='viz/query_analysis/query_prediction_by_graph.csv')
    p.add_argument('--cap', type=int, default=5000)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=12345)
    args = p.parse_args()

    print('Loading rows from', args.csv)
    rows = load_rows(args.csv)
    print('Total rows loaded:', len(rows))

    groups = group_by_algo_label(rows)
    print('Found groups (algorithm,label):', len(groups))

    for (algo, label), items in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1].replace('len',''))))[:10]:
        print('  sample group:', algo, label, len(items))

    sampled = sample_per_group(groups, args.cap, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    counts = copy_selected(sampled, args.out)
    summary = write_summary(groups, sampled, counts, args.out)
    total = sum(counts.values())
    print('Wrote', total, 'files to', args.out)
    print('Summary CSV:', summary)


if __name__ == '__main__':
    main()
