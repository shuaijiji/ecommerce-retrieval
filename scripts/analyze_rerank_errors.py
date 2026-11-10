#!/usr/bin/env python3
"""Decompose reranker errors and compare two nested candidate runs."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from rerank_pipeline import digest, load_rows, save, write_rows


def load_labels(path):
    raw = json.loads(path.read_text())
    labels = {str(row['photo_id']): str(row['item_id']) for row in raw}
    if len(labels) != len(raw):
        raise ValueError('duplicate labels')
    return labels


def load_metadata(path, key):
    if path is None:
        return {}
    rows = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                value = str(row[key])
                if value in rows:
                    raise ValueError(f'duplicate {key}: {value}')
                rows[value] = row
    return rows


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[math.ceil(len(values) * q) - 1]


def rank_bucket(rank):
    if rank is None:
        return 'not_in_saved_retrieval'
    for upper in (1, 5, 10, 25, 50, 100):
        if rank <= upper:
            return f'01-{upper:03d}' if upper == 1 else f'{previous + 1:02d}-{upper:03d}'
        previous = upper
    return '101+'


def retrieval_truth_ranks(retrieval_dir, pid, truth):
    result = {}
    if retrieval_dir is None:
        return {'vision': None, 'multimodal': None}
    for mode in ('vision', 'multimodal'):
        path = retrieval_dir / 'results' / mode / f'{pid}.json'
        row = json.loads(path.read_text())
        if row['photo_id'] != pid:
            raise ValueError(f'photo mismatch: {path}')
        # Full retrieval outputs retain the exact truth rank even when only
        # Top-100 IDs are saved. This field is used only during post-evaluation.
        if 'positive_item_id' in row and 'rank' in row:
            if str(row['positive_item_id']) != truth or not isinstance(row['rank'], int) or row['rank'] < 1:
                raise ValueError(f'invalid truth rank: {path}')
            result[mode] = row['rank']
        else:
            try:
                result[mode] = row['top_item_ids'].index(truth) + 1
            except ValueError:
                result[mode] = None
    return result


def validate_run(candidates_path, predictions_path, labels):
    candidates = {row['photo_id']: row for row in load_rows(candidates_path)}
    predictions = {row['photo_id']: row for row in load_rows(predictions_path)}
    if set(candidates) != set(predictions) or not set(candidates) <= set(labels):
        raise ValueError('candidate, prediction and label query sets do not align')
    for pid, candidate in candidates.items():
        ids = {x['item_id'] for x in candidate['candidates']}
        ranking = predictions[pid]['ranking']
        if len(ranking) != len(ids) or {x['item_id'] for x in ranking} != ids:
            raise ValueError(f'incomplete ranking: {pid}')
        scores = [x['score'] for x in ranking]
        if not scores or not all(math.isfinite(x) for x in scores) or any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError(f'invalid ranking scores: {pid}')
    return candidates, predictions


def build_cases(candidates, predictions, labels, retrieval_dir=None, photos=None):
    photos = photos or {}
    cases = []
    for pid in sorted(candidates):
        candidate = candidates[pid]
        ranking = predictions[pid]['ranking']
        truth = labels[pid]
        candidate_by_id = {x['item_id']: x for x in candidate['candidates']}
        score_by_id = {x['item_id']: x['score'] for x in ranking}
        prediction = ranking[0]['item_id']
        hit = truth in candidate_by_id
        correct = prediction == truth
        status = 'correct' if correct else ('rerank_wrong' if hit else 'candidate_miss')
        truth_ranks = retrieval_truth_ranks(retrieval_dir, pid, truth)
        predicted_candidate = candidate_by_id[prediction]
        top_score = ranking[0]['score']
        second_score = ranking[1]['score'] if len(ranking) > 1 else None
        truth_score = score_by_id.get(truth)
        photo = photos.get(pid, {})
        cases.append({
            'photo_id': pid,
            'truth_item_id': truth,
            'predicted_item_id': prediction,
            'vision_top1': candidate['vision_top1'],
            'status': status,
            'candidate_count': len(candidate_by_id),
            'truth_vision_rank': truth_ranks['vision'],
            'truth_multimodal_rank': truth_ranks['multimodal'],
            'truth_best_route_rank': min((x for x in truth_ranks.values() if x is not None), default=None),
            'predicted_vision_rank': predicted_candidate['vision_rank'],
            'predicted_multimodal_rank': predicted_candidate['multimodal_rank'],
            'top_score': top_score,
            'second_score': second_score,
            'top1_top2_margin': top_score - second_score if second_score is not None else None,
            'truth_score': truth_score,
            'predicted_truth_score_gap': top_score - truth_score if truth_score is not None else None,
            'vision_top1_correct': candidate['vision_top1'] == truth,
            'frame_count': len(photo.get('photo_frames', [])) if photo else None,
            'text_present': {field: bool(photo.get(field)) for field in
                             ('photo_caption', 'photo_title', 'photo_text', 'photo_ocr', 'photo_asr')} if photo else None,
        })
    return cases


def summarize(cases, candidates_path, predictions_path, labels_path, retrieval_dir):
    n = len(cases)
    status = Counter(x['status'] for x in cases)
    transition = Counter(
        ('vision_correct' if x['vision_top1_correct'] else 'vision_wrong') + '__' +
        ('rerank_correct' if x['status'] == 'correct' else 'rerank_wrong') for x in cases)
    rank_buckets = defaultdict(Counter)
    candidate_counts = defaultdict(Counter)
    margins = defaultdict(list)
    frame_counts = defaultdict(Counter)
    text_presence = defaultdict(Counter)
    miss_routes = Counter()
    for row in cases:
        rank_buckets[row['status']][rank_bucket(row['truth_best_route_rank'])] += 1
        candidate_counts[row['status']][str(row['candidate_count'])] += 1
        frame_counts[row['status']][str(row['frame_count'])] += 1
        if row['text_present'] is not None:
            for field, present in row['text_present'].items():
                text_presence[row['status']][field + ('_present' if present else '_empty')] += 1
        if row['top1_top2_margin'] is not None:
            margins[row['status']].append(row['top1_top2_margin'])
        if row['status'] == 'candidate_miss':
            v, m = row['truth_vision_rank'], row['truth_multimodal_rank']
            miss_routes['rank_unavailable' if v is None and m is None else
                        'vision_best' if m is None or (v is not None and v < m) else
                        'multimodal_best' if v is None or m < v else 'equal_rank'] += 1
    return {
        'count': n,
        'status': {key: {'count': status[key], 'rate': status[key] / n} for key in
                   ('correct', 'candidate_miss', 'rerank_wrong')},
        'system_top1': status['correct'] / n,
        'candidate_recall': (n - status['candidate_miss']) / n,
        'candidate_rerank_accuracy': status['correct'] / (n - status['candidate_miss']) if n > status['candidate_miss'] else None,
        'vision_to_rerank_transition': dict(sorted(transition.items())),
        'truth_best_route_rank_bucket_by_status': {k: dict(sorted(v.items())) for k, v in sorted(rank_buckets.items())},
        'candidate_count_by_status': {k: dict(sorted(v.items(), key=lambda x: int(x[0]))) for k, v in sorted(candidate_counts.items())},
        'frame_count_by_status': {k: dict(sorted(v.items())) for k, v in sorted(frame_counts.items())},
        'text_presence_by_status': {k: dict(sorted(v.items())) for k, v in sorted(text_presence.items())},
        'top1_top2_margin_by_status': {k: {'count': len(v), 'mean': sum(v) / len(v),
                                             'p50': percentile(v, .5), 'p90': percentile(v, .9)}
                                        for k, v in sorted(margins.items())},
        'candidate_miss_best_route': dict(sorted(miss_routes.items())),
        'inputs': {'candidate_sha256': digest(candidates_path), 'prediction_sha256': digest(predictions_path),
                   'label_sha256': digest(labels_path), 'retrieval_dir': str(retrieval_dir) if retrieval_dir else None},
    }


def analyze(args):
    labels = load_labels(args.labels)
    candidates, predictions = validate_run(args.candidates, args.predictions, labels)
    photos = load_metadata(args.photo_info, 'photo_id')
    cases = build_cases(candidates, predictions, labels, args.retrieval_dir, photos)
    write_rows(cases, args.output_cases)
    summary = summarize(cases, args.candidates, args.predictions, args.labels, args.retrieval_dir)
    save(summary, args.output_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def compare(args):
    labels = load_labels(args.labels)
    base_c, base_p = validate_run(args.base_candidates, args.base_predictions, labels)
    new_c, new_p = validate_run(args.new_candidates, args.new_predictions, labels)
    if set(base_c) != set(new_c):
        raise ValueError('base and new query sets differ')
    transitions = Counter()
    restricted_transitions = Counter()
    score_diffs = []
    rows = []
    for pid in sorted(base_c):
        truth = labels[pid]
        base_ids = {x['item_id'] for x in base_c[pid]['candidates']}
        new_ids = {x['item_id'] for x in new_c[pid]['candidates']}
        if not base_ids <= new_ids:
            raise ValueError(f'new candidate set is not a superset: {pid}')
        base_hit, new_hit = truth in base_ids, truth in new_ids
        base_ok = base_p[pid]['ranking'][0]['item_id'] == truth
        new_ok = new_p[pid]['ranking'][0]['item_id'] == truth
        base_scores = {x['item_id']: x['score'] for x in base_p[pid]['ranking']}
        new_scores = {x['item_id']: x['score'] for x in new_p[pid]['ranking']}
        score_diffs.extend(abs(base_scores[item] - new_scores[item]) for item in base_ids)
        # Counterfactual: use scores from the new run, but allow only the old
        # candidate set. This separates run/batch drift from candidate expansion.
        restricted_prediction = next(x['item_id'] for x in new_p[pid]['ranking'] if x['item_id'] in base_ids)
        restricted_ok = restricted_prediction == truth
        key = f"{'hit' if base_hit else 'miss'}_to_{'hit' if new_hit else 'miss'}__{'correct' if base_ok else 'wrong'}_to_{'correct' if new_ok else 'wrong'}"
        transitions[key] += 1
        restricted_key = f"{'correct' if base_ok else 'wrong'}_to_{'correct' if restricted_ok else 'wrong'}"
        restricted_transitions[restricted_key] += 1
        rows.append({'photo_id': pid, 'truth_item_id': truth, 'base_candidate_hit': base_hit,
                     'new_candidate_hit': new_hit, 'base_correct': base_ok, 'new_correct': new_ok,
                     'new_run_restricted_to_base_correct': restricted_ok,
                     'new_run_restricted_to_base_prediction': restricted_prediction,
                     'base_candidate_count': len(base_ids), 'new_candidate_count': len(new_ids), 'transition': key})
    n = len(rows)
    base_hits = sum(x['base_candidate_hit'] for x in rows)
    new_hits = sum(x['new_candidate_hit'] for x in rows)
    base_correct = sum(x['base_correct'] for x in rows)
    new_correct = sum(x['new_correct'] for x in rows)
    restricted_correct = sum(x['new_run_restricted_to_base_correct'] for x in rows)
    gained = sum(not x['base_correct'] and x['new_correct'] for x in rows)
    lost = sum(x['base_correct'] and not x['new_correct'] for x in rows)
    result = {
        'count': n,
        'base': {'candidate_recall': base_hits / n, 'system_top1': base_correct / n,
                 'mean_candidates': sum(x['base_candidate_count'] for x in rows) / n},
        'new': {'candidate_recall': new_hits / n, 'system_top1': new_correct / n,
                'mean_candidates': sum(x['new_candidate_count'] for x in rows) / n},
        'delta': {'candidate_recall_points': (new_hits - base_hits) / n,
                  'system_top1_points': (new_correct - base_correct) / n,
                  'newly_correct_count': gained, 'newly_wrong_count': lost,
                  'net_correct_count': new_correct - base_correct,
                  'recovered_correct_per_new_candidate_hit': (new_correct - base_correct) / (new_hits - base_hits) if new_hits > base_hits else None},
        'controlled_decomposition': {
            'new_run_scores_restricted_to_base_system_top1': restricted_correct / n,
            'run_or_batch_drift_net_correct_count': restricted_correct - base_correct,
            'candidate_expansion_net_correct_count_same_run': new_correct - restricted_correct,
            'candidate_expansion_top1_points_same_run': (new_correct - restricted_correct) / n,
            'base_to_new_restricted_transitions': dict(sorted(restricted_transitions.items())),
        },
        'common_candidate_score_abs_diff': {
            'count': len(score_diffs), 'mean': sum(score_diffs) / len(score_diffs),
            'p50': percentile(score_diffs, .5), 'p95': percentile(score_diffs, .95),
            'p99': percentile(score_diffs, .99), 'max': max(score_diffs),
            'count_gt_1e-6': sum(x > 1e-6 for x in score_diffs),
            'count_gt_1e-4': sum(x > 1e-4 for x in score_diffs),
        },
        'paired_transitions': dict(sorted(transitions.items())),
        'inputs': {name: digest(getattr(args, name)) for name in
                   ('base_candidates', 'base_predictions', 'new_candidates', 'new_predictions', 'labels')},
    }
    write_rows(rows, args.output_cases)
    save(result, args.output_summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('analyze')
    one.add_argument('--candidates', type=Path, required=True)
    one.add_argument('--predictions', type=Path, required=True)
    one.add_argument('--labels', type=Path, required=True)
    one.add_argument('--retrieval-dir', type=Path)
    one.add_argument('--photo-info', type=Path)
    one.add_argument('--output-summary', type=Path, required=True)
    one.add_argument('--output-cases', type=Path, required=True)
    one.set_defaults(fn=analyze)
    two = sub.add_parser('compare')
    for name in ('base-candidates', 'base-predictions', 'new-candidates', 'new-predictions'):
        two.add_argument('--' + name, type=Path, required=True)
    two.add_argument('--labels', type=Path, required=True)
    two.add_argument('--output-summary', type=Path, required=True)
    two.add_argument('--output-cases', type=Path, required=True)
    two.set_defaults(fn=compare)
    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
