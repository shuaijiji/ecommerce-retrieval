#!/usr/bin/env python3
"""Resumably compute exact SHA-256 and a versioned 63-bit pHash for item images."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ALGORITHM = 'phash-dct32-low8-exclude-dc-v1'
_DCT32 = None


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def phash_image(path):
    global _DCT32
    import numpy as np
    from PIL import Image
    with Image.open(path) as source:
        width, height = source.size
        gray = source.convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.float32)
    if _DCT32 is None:
        n = 32
        positions = np.arange(n, dtype=np.float32)
        frequencies = positions[:, None]
        _DCT32 = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * positions + 1) * frequencies / (2 * n))
        _DCT32[0] /= np.sqrt(2.0)
    coefficients = _DCT32 @ pixels @ _DCT32.T
    low = coefficients[:8, :8].reshape(-1)[1:]
    median = float(np.median(low))
    value = 0
    for bit in low >= median:
        value = (value << 1) | int(bit)
    return f'{value:016x}', width, height


def hash_one(task):
    item_id, relative, root = task
    path = Path(root) / relative
    try:
        exact = sha256_file(path)
        perceptual, width, height = phash_image(path)
        return {'item_id': item_id, 'item_image': relative, 'bytes': path.stat().st_size,
                'width': width, 'height': height, 'file_sha256': exact,
                'phash63': perceptual, 'phash_algorithm': ALGORITHM, 'error': None}
    except Exception as exc:
        return {'item_id': item_id, 'item_image': relative, 'error': f'{type(exc).__name__}: {exc}'}


def read_items(path, excluded):
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row['item_id'])
            if item_id in excluded:
                continue
            relative = row.get('item_image')
            if not relative:
                raise ValueError(f'item has no image and was not excluded: {item_id} at line {line_number}')
            rows.append((item_id, str(relative)))
    rows.sort()
    if len({item for item, _ in rows}) != len(rows):
        raise ValueError('duplicate item IDs')
    return rows


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(path)


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--exclude-item-id', action='append', default=[])
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument('--shard-size', type=int, default=10000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.shard_size < 1:
        raise ValueError('workers and shard-size must be positive')
    root = args.data_root.resolve()
    item_info = root / 'item_info.jsonl'
    excluded = set(map(str, args.exclude_item_id))
    items = read_items(item_info, excluded)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('limit must be positive')
        items = items[:args.limit]
    config = {'data_root': str(root), 'item_info_sha256': sha256_file(item_info),
              'algorithm': ALGORITHM, 'excluded_item_ids': sorted(excluded),
              'shard_size': args.shard_size, 'item_count': len(items)}
    config_path = args.output_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('hash configuration changed; use a new output directory')
    write_json(config, config_path)
    shard_dir = args.output_dir / 'shards'
    errors, completed = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for start in range(0, len(items), args.shard_size):
            subset = items[start:start + args.shard_size]
            destination = shard_dir / f'shard_{start:07d}.jsonl'
            expected_ids = [item for item, _ in subset]
            if destination.exists():
                rows = load_rows(destination)
                if [row['item_id'] for row in rows] != expected_ids:
                    raise ValueError(f'invalid resume shard: {destination}')
            else:
                tasks = [(item, relative, str(root)) for item, relative in subset]
                rows = list(executor.map(hash_one, tasks, chunksize=32))
                write_rows(rows, destination)
            errors.extend(row for row in rows if row.get('error'))
            completed += len(rows)
            print(f'{completed}/{len(items)} shard={destination.name} errors={len(errors)}', flush=True)
    manifest = {'config': config, 'completed_items': completed, 'error_count': len(errors),
                'error_examples': errors[:100], 'shard_count': (len(items) + args.shard_size - 1) // args.shard_size}
    write_json(manifest, args.output_dir / 'manifest.json')
    if errors:
        raise SystemExit('image hashing found decode/read errors; inspect manifest before grouping')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
