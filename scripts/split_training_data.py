#!/usr/bin/env python3
"""Create deterministic item-disjoint train/dev ID manifests without copying media."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def bucket(seed, item_id):
    value = hashlib.sha256(f'{seed}:{item_id}'.encode()).digest()
    return int.from_bytes(value[:8], 'big') / 2 ** 64


def write_lines(values, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(value + '\n' for value in values))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dev-item-ratio', type=float, default=.02)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    if not 0 < args.dev_item_ratio < .5:
        raise ValueError('dev-item-ratio must be between 0 and 0.5')
    raw = json.loads(args.labels.read_text())
    pairs = [(str(row['photo_id']), str(row['item_id'])) for row in raw]
    photo_counts = Counter(photo for photo, _ in pairs)
    if any(value != 1 for value in photo_counts.values()):
        raise ValueError('each photo must have exactly one label before splitting')
    dev_items = {item for _, item in pairs if bucket(args.seed, item) < args.dev_item_ratio}
    train_photos = sorted(photo for photo, item in pairs if item not in dev_items)
    dev_photos = sorted(photo for photo, item in pairs if item in dev_items)
    train_items = sorted({item for _, item in pairs if item not in dev_items})
    dev_items = sorted(dev_items)
    if not train_photos or not dev_photos or set(train_items) & set(dev_items):
        raise ValueError('invalid empty or overlapping split')
    out = args.output_dir
    write_lines(train_photos, out / 'train_photo_ids.txt')
    write_lines(dev_photos, out / 'dev_photo_ids.txt')
    write_lines(train_items, out / 'train_item_ids.txt')
    write_lines(dev_items, out / 'dev_item_ids.txt')
    manifest = {'method': 'deterministic SHA-256 item-disjoint split', 'seed': args.seed,
                'requested_dev_item_ratio': args.dev_item_ratio,
                'counts': {'train_photos': len(train_photos), 'dev_photos': len(dev_photos),
                           'train_items': len(train_items), 'dev_items': len(dev_items)},
                'item_overlap': 0, 'labels_sha256': hashlib.sha256(args.labels.read_bytes()).hexdigest(),
                'near_duplicate_grouping_done': False,
                'warning': 'Run near-duplicate image grouping before treating this split as final.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
