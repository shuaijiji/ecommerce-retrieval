#!/usr/bin/env python3
"""Label-free candidate generation/RRF; separate label-only evaluation."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r['photo_id'] for r in rows]
    if len(set(ids)) != len(ids) or not all(isinstance(x, str) for x in ids):
        raise ValueError('duplicate or non-string photo IDs')
    return rows


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def generate(args):
    if args.top_k < 1 or args.expected_count < 1:
        raise ValueError('top-k and expected-count must be positive')
    # Deliberately never access positive_item_id, positive_score, rank or labels.
    modes = ('vision', 'multimodal')
    files = {m: {p.stem: p for p in (args.retrieval_dir / 'results' / m).glob('*.json')} for m in modes}
    if set(files['vision']) != set(files['multimodal']) or len(files['vision']) != args.expected_count:
        raise ValueError('two routes must have the same complete expected query set')
    rows = []
    source_hash = hashlib.sha256()
    for pid in sorted(files['vision']):
        route = {}
        for mode in modes:
            path = files[mode][pid]
            raw = path.read_bytes()
            source_hash.update(mode.encode() + pid.encode() + raw)
            r = json.loads(raw)
            ids, scores = r['top_item_ids'], r['top_scores']
            if r['photo_id'] != pid or len(ids) != len(scores) or len(ids) < args.top_k:
                raise ValueError(f'invalid retrieval record: {path}')
            if len(set(ids)) != len(ids) or not all(isinstance(i, str) for i in ids):
                raise ValueError(f'duplicate or non-string item IDs: {path}')
            if not all(math.isfinite(s) for s in scores):
                raise ValueError(f'nonfinite scores: {path}')
            if any(a < b for a, b in zip(scores, scores[1:])):
                raise ValueError(f'unsorted scores: {path}')
            route[mode] = {item: (rank, score) for rank, (item, score) in enumerate(zip(ids, scores), 1)}
        union = {item for mode in modes for item, (rank, _) in route[mode].items() if rank <= args.top_k}
        candidates = []
        for item in sorted(union):
            c = {'item_id': item}
            for mode in modes:
                rank, score = route[mode].get(item, (None, None))
                c[mode + '_rank'] = rank
                c[mode + '_similarity'] = score
            candidates.append(c)
        rows.append({'photo_id': pid, 'vision_top1': next(iter(route['vision'])), 'candidates': candidates})
    write_rows(rows, args.output)
    save({'count': len(rows), 'top_k_per_route': args.top_k,
          'mean_candidates': sum(len(r['candidates']) for r in rows) / len(rows),
          'source_sha256': source_hash.hexdigest(), 'candidate_sha256': digest(args.output),
          'missing_route_values': 'null means absent from saved Top-K; never invented',
          'labels_used': False}, str(args.output) + '.manifest.json')


def rrf(args):
    if args.rrf_k <= 0:
        raise ValueError('RRF k must be positive')
    rows = load_rows(args.candidates)
    out = []
    for row in rows:
        start = time.perf_counter()
        scored = []
        for c in row['candidates']:
            # Both saved route rankings (normally Top-100), not only ranks <=10.
            score = sum(1 / (args.rrf_k + c[m + '_rank']) for m in ('vision', 'multimodal') if c[m + '_rank'] is not None)
            scored.append({'item_id': c['item_id'], 'score': score})
        scored.sort(key=lambda c: (-c['score'], c['item_id']))
        out.append({'photo_id': row['photo_id'], 'ranking': scored, 'seconds': time.perf_counter() - start})
    write_rows(out, args.output)
    save({'candidate_sha256': digest(args.candidates), 'rrf_k': args.rrf_k,
          'weights': [1, 1], 'tie_break': 'item_id ascending', 'labels_used': False}, str(args.output) + '.manifest.json')


def evaluate(args):
    candidates = {r['photo_id']: r for r in load_rows(args.candidates)}
    raw_labels = json.loads(args.labels.read_text())
    labels = {str(r['photo_id']): str(r['item_id']) for r in raw_labels}
    if len(labels) != len(raw_labels):
        raise ValueError('duplicate labels')
    predictions = load_rows(args.predictions)
    if len(predictions) != args.expected_count:
        raise ValueError('prediction count differs from expected count')
    if args.expected_count == len(candidates) and {r['photo_id'] for r in predictions} != set(candidates):
        raise ValueError('full evaluation requires exact candidate query set')
    hits = correct = visual_correct = harmed = 0
    times, by_count = [], {}
    for p in predictions:
        pid = p['photo_id']
        c, truth = candidates[pid], labels[pid]
        ids = {v['item_id'] for v in c['candidates']}
        ranking = p['ranking']
        if len(ranking) != len(ids) or {v['item_id'] for v in ranking} != ids:
            raise ValueError(f'incomplete/invalid ranking: {pid}')
        scores = [v['score'] for v in ranking]
        if not all(math.isfinite(s) for s in scores) or any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError(f'invalid scores: {pid}')
        hit, ok, was_ok = truth in ids, ranking[0]['item_id'] == truth, c['vision_top1'] == truth
        hits += hit
        correct += ok
        visual_correct += was_ok
        harmed += was_ok and not ok
        sec = p['seconds']
        if not math.isfinite(sec) or sec < 0:
            raise ValueError('invalid timing')
        times.append(sec)
        by_count.setdefault(len(ids), []).append(sec)
    n = len(predictions)
    if not n:
        raise ValueError('no predictions')
    recall = hits / n
    if args.expect_recall is not None and abs(recall - args.expect_recall) > 1e-8:
        raise ValueError(f'candidate Recall mismatch: {recall} != {args.expect_recall}')
    ordered = sorted(times)
    result = {'count': n, 'correct': correct, 'candidate_hits': hits,
              'system_top1': correct / n, 'candidate_rerank_accuracy': correct / hits if hits else None,
              'candidate_recall': recall, 'reference_full_candidate_recall': .9039,
              'vision_top1': visual_correct / n, 'vision_correct_count': visual_correct,
              'vision_correct_changed_wrong_count': harmed,
              'vision_correct_changed_wrong_rate': harmed / visual_correct if visual_correct else None,
              'changed_wrong_fraction_all': harmed / n,
              'mean_seconds_per_video': sum(times) / n,
              'p50_seconds': ordered[math.ceil(n * .5) - 1],
              'p95_seconds': ordered[math.ceil(n * .95) - 1],
              'p99_seconds': ordered[math.ceil(n * .99) - 1],
              'latency_by_candidate_count': {k: {'count': len(v), 'mean_seconds': sum(v) / len(v)} for k, v in sorted(by_count.items())},
              'candidate_sha256': digest(args.candidates), 'prediction_sha256': digest(args.predictions),
              'label_sha256': digest(args.labels)}
    save(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('candidates')
    g.add_argument('--retrieval-dir', type=Path, required=True)
    g.add_argument('--top-k', type=int, default=10)
    g.add_argument('--expected-count', type=int, default=10000)
    g.add_argument('--output', type=Path, required=True)
    g.set_defaults(fn=generate)
    r = sub.add_parser('rrf')
    r.add_argument('--candidates', type=Path, required=True)
    r.add_argument('--rrf-k', type=int, default=60)
    r.add_argument('--output', type=Path, required=True)
    r.set_defaults(fn=rrf)
    e = sub.add_parser('evaluate')
    e.add_argument('--candidates', type=Path, required=True)
    e.add_argument('--predictions', type=Path, required=True)
    e.add_argument('--labels', type=Path, required=True)
    e.add_argument('--expected-count', type=int, required=True)
    e.add_argument('--expect-recall', type=float)
    e.add_argument('--output', type=Path, required=True)
    e.set_defaults(fn=evaluate)
    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
