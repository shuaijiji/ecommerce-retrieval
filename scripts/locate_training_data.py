#!/usr/bin/env python3
"""Locate extracted ecommerce datasets without descending into image trees."""
import argparse
import json
import os
from pathlib import Path


PRUNE = {'item_images', 'photo_frames', '.git', 'artifacts', 'models', '__pycache__'}


def first_object(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'first non-empty row is not an object: {path}')
                return value
    raise ValueError(f'empty metadata file: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-root', type=Path, default=Path('/code/ecommerce-retrieval/data'))
    parser.add_argument('--max-depth', type=int, default=6)
    args = parser.parse_args()
    root = args.search_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    found = []
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        dirs[:] = [] if depth >= args.max_depth else [d for d in dirs if d not in PRUNE]
        if {'item_info.jsonl', 'photo_info.jsonl'} <= set(files):
            labels = current_path / 'label.json'
            item = current_path / 'item_info.jsonl'
            photo = current_path / 'photo_info.jsonl'
            found.append({
                'data_root': str(current_path),
                'label_exists': labels.is_file(),
                'metadata_bytes': {'item_info.jsonl': item.stat().st_size,
                                   'photo_info.jsonl': photo.stat().st_size,
                                   'label.json': labels.stat().st_size if labels.is_file() else None},
                'item_keys': sorted(first_object(item)),
                'photo_keys': sorted(first_object(photo)),
            })
    archives = [{'path': str(path), 'bytes': path.stat().st_size}
                for path in sorted(root.rglob('all_train_data.zip'))]
    result = {'search_root': str(root), 'dataset_roots': found, 'train_archives': archives}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not found:
        raise SystemExit('no extracted dataset root found')


if __name__ == '__main__':
    main()
