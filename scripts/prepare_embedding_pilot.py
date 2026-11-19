#!/usr/bin/env python3
"""Audit the entire training negative pool before preparing a small embedding pilot."""
import argparse
import json
from pathlib import Path
from prepare_reranker_sft import (
    iter_jsonl, read_ids, select_ids, load_duplicate_groups, load_metadata,
    mode_candidates, sha256_file, write_jsonl, write_json_atomic,
)


def check_row(row, positive, train_items, groups):
    if str(row['positive_item_id']) != positive:
        raise ValueError(f"label mismatch: {row['photo_id']}")
    if positive not in train_items:
        raise ValueError('positive outside training items')
    forbidden = set(groups.get(positive, ())) | {positive}
    for mode in ('vision', 'multimodal'):
        candidates = mode_candidates(row, mode)
        ids = [c['item_id'] for c in candidates]
        ranks = [c['raw_rank'] for c in candidates]
        if len(set(ids)) != len(ids) or ranks != sorted(set(ranks)):
            raise ValueError('duplicate candidates or invalid ranks')
        if forbidden.intersection(ids) or set(ids) - train_items:
            raise ValueError('false negative or non-training item in candidates')


def choose_negatives(row, count):
    """Round-robin routes, preserving rank and both routes' provenance."""
    routes = {m: mode_candidates(row, m) for m in ('vision', 'multimodal')}
    selected = {}
    for pair in zip(*routes.values()):
        for c in pair:
            selected.setdefault(c['item_id'], c)
        if len(selected) >= count:
            break
    return list(selected.values())[:count]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'split-dir', 'labels', 'candidates', 'duplicate-groups', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--num-queries', type=int, default=100)
    p.add_argument('--negatives', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261119)
    a = p.parse_args()
    if not 100 <= a.num_queries <= 500 or not 2 <= a.negatives <= 20:
        raise ValueError('pilot requires 100–500 queries and 2–20 negatives')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    paths = {n: a.split_dir / (n + '.txt') for n in
             ('train_photo_ids', 'dev_photo_ids', 'train_item_ids', 'dev_item_ids')}
    ids = {n: read_ids(path) for n, path in paths.items()}
    train_photos = ids['train_photo_ids']
    train_items = set(ids['train_item_ids'])
    if set(train_photos) & set(ids['dev_photo_ids']) or train_items & set(ids['dev_item_ids']):
        raise ValueError('train/dev overlap')
    for _, group in iter_jsonl(a.duplicate_groups):
        members = [str(m['item_id']) for m in group['members']]
        if len(members) != len(set(members)):
            raise ValueError('repeated member in duplicate group')
    groups = load_duplicate_groups(a.duplicate_groups)
    if set(groups) & set(ids['dev_item_ids']):
        raise ValueError('duplicate-group quarantine violated')
    labels = {}
    for _, row in iter_jsonl(a.labels):
        photo, item = str(row['photo_id']), str(row['item_id'])
        if photo in labels:
            raise ValueError('duplicate label')
        labels[photo] = item
    dev_items = set(ids['dev_item_ids'])
    if any(labels[p] not in dev_items for p in ids['dev_photo_ids']):
        raise ValueError('dev labels outside dev item split')
    selected = set(select_ids(train_photos, a.num_queries, a.seed))
    manifest = json.loads((a.candidates / 'manifest.json').read_text())
    config_path = a.candidates / manifest['config']
    config = json.loads(config_path.read_text())
    if not manifest.get('complete'):
        raise ValueError('incomplete candidates')
    for key, path in [('labels_sha256', a.labels), ('photo_ids_sha256', paths['train_photo_ids']),
                      ('duplicate_groups_sha256', a.duplicate_groups)]:
        if config[key] != sha256_file(path):
            raise ValueError(f'candidate provenance mismatch: {key}')
    rows, cursor, hashes = [], 0, {}
    for name in manifest['shards']:
        path = a.candidates / 'shards' / name
        hashes[name] = sha256_file(path)
        for _, row in iter_jsonl(path):
            photo = str(row['photo_id'])
            if cursor >= len(train_photos) or photo != train_photos[cursor]:
                raise ValueError('candidate coverage/order mismatch')
            check_row(row, labels[photo], train_items, groups)
            cursor += 1
            if photo in selected:
                rows.append({'photo_id': photo, 'positive_item_id': labels[photo],
                             'negatives': choose_negatives(row, a.negatives)})
        print(f'audited {cursor}/{len(train_photos)} queries', flush=True)
    if cursor != len(train_photos) or len(rows) != a.num_queries:
        raise ValueError('incomplete candidate coverage')
    photos = load_metadata(a.data_root / 'photo_info.jsonl', 'photo_id', selected)
    needed = {r['positive_item_id'] for r in rows} | {n['item_id'] for r in rows for n in r['negatives']}
    items = load_metadata(a.data_root / 'item_info.jsonl', 'item_id', needed)
    # Decode all selected media; preserve the existing retrieval preprocessing protocol.
    from PIL import Image
    for row in rows:
        photo = photos[row['photo_id']]
        if not photo.get('photo_frames'):
            raise ValueError('query has no frames')
        row['query'] = photo
        row['items'] = [items[i] for i in [row['positive_item_id']] + [n['item_id'] for n in row['negatives']]]
        media = photo['photo_frames'] + [i['item_image'] for i in row['items']]
        for relative in media:
            with Image.open(a.data_root / relative) as image:
                image.load()
    write_jsonl(sorted(rows, key=lambda r: r['photo_id']), a.output_dir / 'samples.jsonl')
    write_json_atomic({
        'complete': True, 'audited_queries': cursor, 'pilot_queries': len(rows),
        'seed': a.seed, 'negatives': a.negatives, 'data_root': str(a.data_root.resolve()),
        'source_hashes': {str(p): sha256_file(p) for p in [*paths.values(), a.labels,
                          a.duplicate_groups, config_path, a.candidates / 'manifest.json']},
        'candidate_shard_hashes': hashes, 'samples_sha256': sha256_file(a.output_dir / 'samples.jsonl'),
        'exclusion_scope': 'positive ID and all known proposed duplicate-group members',
        'limitation': 'perceptual groups do not prove semantic same-product completeness; review selected hard negatives',
        'in_batch_negatives': False,
    }, a.output_dir / 'manifest.json')


if __name__ == '__main__':
    main()
