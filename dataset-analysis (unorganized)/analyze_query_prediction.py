#!/usr/bin/env python3
"""Scan tasks (JSON format) and extract query/prediction tokens from the `text` field.
Produces a CSV with one row per graph and a small summary count-by-query and count-by-prediction.

Usage:
  python3 benchmarks/analyze_query_prediction.py --tasks cycle_check shortest_path --base submodules/graph-token/tasks_autograph --out viz/query_analysis
"""
import argparse
from pathlib import Path
import json
import csv
from collections import Counter, defaultdict


def extract_qp_from_text(text: str):
    # return (query_text, prediction_text) or (None, None)
    q = None
    p = None
    try:
        # normalize spacing
        s = text.replace('\n', ' ')
        parts = s.split()
        # join then use simple find to capture segments between markers
        joined = ' '.join(parts)
        if '<q>' in joined:
            after_q = joined.split('<q>', 1)[1]
            # prediction marker starts the prediction
            if '<p>' in after_q:
                q = after_q.split('<p>', 1)[0].strip()
                after_p = after_q.split('<p>', 1)[1]
                # stop at <eos> if present
                p = after_p.split('<eos>', 1)[0].strip()
            else:
                # take what remains as query
                q = after_q.split('<eos>', 1)[0].strip()
        else:
            # fallback: try to find '<p>' and capture after it
            if '<p>' in joined:
                after_p = joined.split('<p>', 1)[1]
                p = after_p.split('<eos>', 1)[0].strip()
    except Exception:
        return None, None
    # normalize empty strings to None
    q = q if q and len(q) > 0 else None
    p = p if p and len(p) > 0 else None
    return q, p


def scan_json_file(path: Path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return []
    rows = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = [data]
    else:
        return []
    for item in entries:
        graph_id = item.get('graph_id') or item.get('id') or ''
        text = item.get('text', '')
        q, p = extract_qp_from_text(text)
        rows.append({'path': str(path), 'graph_id': graph_id, 'query': q, 'prediction': p})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--base', type=str, default='submodules/graph-token/tasks_autograph')
    parser.add_argument('--out', type=str, default='viz/query_analysis')
    args = parser.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    query_counter = Counter()
    pred_counter = Counter()
    per_task = defaultdict(lambda: {'query': Counter(), 'pred': Counter(), 'rows': []})

    for task in args.tasks:
        task_dir = base / task
        if not task_dir.exists():
            print('Task dir not found:', task_dir)
            continue
        for alg in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
            for split in sorted([p for p in alg.iterdir() if p.is_dir()]):
                for jf in sorted(split.glob('*.json')):
                    rows = scan_json_file(jf)
                    for r in rows:
                        q = r.get('query')
                        p = r.get('prediction')
                        all_rows.append({'task': task, 'algorithm': alg.name, 'split': split.name, **r})
                        if q:
                            query_counter[q] += 1
                            per_task[task]['query'][q] += 1
                        if p:
                            pred_counter[p] += 1
                            per_task[task]['pred'][p] += 1
                        per_task[task]['rows'].append(r)

    # write per-graph CSV
    csv_path = out / 'query_prediction_by_graph.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task','algorithm','split','path','graph_id','query','prediction'])
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print('Wrote', csv_path)

    # write summary counts
    summary_path = out / 'query_prediction_summary.txt'
    with open(summary_path, 'w') as f:
        f.write('Top queries (global):\n')
        for q, c in query_counter.most_common(50):
            f.write(f'{q}: {c}\n')
        f.write('\nTop predictions (global):\n')
        for p, c in pred_counter.most_common(50):
            f.write(f'{p}: {c}\n')
        f.write('\nPer-task breakdown:\n')
        for task, info in per_task.items():
            f.write(f'\nTask: {task}\n')
            f.write('  Top queries:\n')
            for q,c in info['query'].most_common(20):
                f.write(f'    {q}: {c}\n')
            f.write('  Top predictions:\n')
            for p,c in info['pred'].most_common(20):
                f.write(f'    {p}: {c}\n')
    print('Wrote', summary_path)


if __name__ == '__main__':
    main()
