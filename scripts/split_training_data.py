#!/usr/bin/env python3
"""Create deterministic item-disjoint train/dev ID manifests without copying media."""
import argparse
import hashlib
import json
from pathlib import Path


def bucket(seed, item_id):
    value = hashlib.sha256(f'{seed}:{item_id}'.encode()).digest()
    return int.from_bytes(value[:8], 'big') / 2 ** 64


def write_lines(values, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(value + '\n' for value in values))
    temporary.replace(path)


def read_pairs(path):
    if path.suffix == '.jsonl':
        pairs = []
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f'not an object at line {line_number}')
                pairs.append((str(row['photo_id']), str(row['item_id'])))
        return pairs
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise TypeError('label JSON must contain an array')
    return [(str(row['photo_id']), str(row['item_id'])) for row in raw]


def read_quarantine_groups(path):
    if path is None:
        return set(), 0
    item_ids, group_count = set(), 0
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            members = row.get('members')
            if not isinstance(members, list) or len(members) < 2:
                raise ValueError(f'invalid duplicate group at line {line_number}')
            group_items = {str(member['item_id']) for member in members}
            overlap = item_ids & group_items
            if overlap:
                raise ValueError(f'item appears in multiple duplicate groups: {sorted(overlap)[:5]}')
            item_ids.update(group_items)
            group_count += 1
    return item_ids, group_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dev-item-ratio', type=float, default=.02)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--quarantine-groups', type=Path,
                        help='proposed duplicate groups whose members must remain on the train side')
    args = parser.parse_args()
    if not 0 < args.dev_item_ratio < .5:
        raise ValueError('dev-item-ratio must be between 0 and 0.5')
    raw_pairs = read_pairs(args.labels)
    photo_to_item, duplicate_rows = {}, 0
    for photo, item in raw_pairs:
        if photo in photo_to_item:
            if photo_to_item[photo] != item:
                raise ValueError(f'photo has conflicting item labels: {photo}')
            duplicate_rows += 1
        else:
            photo_to_item[photo] = item
    pairs = sorted(photo_to_item.items())
    all_items = {item for _, item in pairs}
    quarantined_items, quarantine_group_count = read_quarantine_groups(args.quarantine_groups)
    unknown_quarantined = quarantined_items - all_items
    if unknown_quarantined:
        raise ValueError(f'quarantined items missing from labels: {sorted(unknown_quarantined)[:5]}')
    dev_items = {item for item in all_items
                 if item not in quarantined_items and bucket(args.seed, item) < args.dev_item_ratio}
    train_photos = sorted(photo for photo, item in pairs if item not in dev_items)
    dev_photos = sorted(photo for photo, item in pairs if item in dev_items)
    train_items = sorted({item for _, item in pairs if item not in dev_items})
    dev_items = sorted(dev_items)
    quarantined_photos = sorted(photo for photo, item in pairs if item in quarantined_items)
    if not train_photos or not dev_photos or set(train_items) & set(dev_items):
        raise ValueError('invalid empty or overlapping split')
    out = args.output_dir
    write_lines(train_photos, out / 'train_photo_ids.txt')
    write_lines(dev_photos, out / 'dev_photo_ids.txt')
    write_lines(train_items, out / 'train_item_ids.txt')
    write_lines(dev_items, out / 'dev_item_ids.txt')
    if args.quarantine_groups is not None:
        write_lines(sorted(quarantined_items), out / 'quarantined_item_ids.txt')
        write_lines(quarantined_photos, out / 'quarantined_photo_ids.txt')
    manifest = {'method': 'deterministic SHA-256 item-disjoint split', 'seed': args.seed,
                'requested_dev_item_ratio': args.dev_item_ratio,
                'counts': {'train_photos': len(train_photos), 'dev_photos': len(dev_photos),
                           'train_items': len(train_items), 'dev_items': len(dev_items),
                           'eligible_dev_pool_items': len(all_items - quarantined_items),
                           'quarantined_groups': quarantine_group_count,
                           'quarantined_items': len(quarantined_items),
                           'quarantined_photos': len(quarantined_photos),
                           'source_label_rows': len(raw_pairs), 'deduplicated_label_rows': len(pairs),
                           'removed_exact_duplicate_rows': duplicate_rows},
                'item_overlap': 0, 'labels_sha256': hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                'duplicate_policy': ('quarantine_all_proposed_groups_from_dev'
                                     if args.quarantine_groups is not None else 'not_applied'),
                'quarantine_groups_sha256': (hashlib.sha256(args.quarantine_groups.read_bytes()).hexdigest()
                                             if args.quarantine_groups is not None else None),
                'near_duplicate_grouping_confirmed': False,
                'near_duplicate_handling_done': args.quarantine_groups is not None,
                'warning': ('Candidate groups are unconfirmed and conservatively kept out of dev; '
                            'do not sample within-group negatives.'
                            if args.quarantine_groups is not None else
                            'Run duplicate handling before treating this split as final.')}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
