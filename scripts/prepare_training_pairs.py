#!/usr/bin/env python3
"""Deduplicate labels and exclude unusable items without changing source data."""
import argparse
import hashlib
import json
from pathlib import Path


def iter_rows(path):
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f'not an object at line {line_number}')
                    yield row
    else:
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            raise TypeError('label JSON must contain an array')
        yield from raw


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-labels', type=Path, required=True)
    parser.add_argument('--output-manifest', type=Path, required=True)
    parser.add_argument('--exclude-item-id', action='append', default=[])
    parser.add_argument('--exclude-item-ids-file', type=Path)
    args = parser.parse_args()
    excluded_items = set(map(str, args.exclude_item_id))
    if args.exclude_item_ids_file:
        excluded_items.update(line.strip() for line in args.exclude_item_ids_file.read_text().splitlines() if line.strip())
    photo_to_item, raw_rows, duplicate_rows = {}, 0, 0
    excluded_rows, excluded_photos = 0, []
    for row in iter_rows(args.labels):
        raw_rows += 1
        photo, item = str(row['photo_id']), str(row['item_id'])
        if photo in photo_to_item:
            if photo_to_item[photo] != item:
                raise ValueError(f'photo has conflicting item labels: {photo}')
            duplicate_rows += 1
            continue
        photo_to_item[photo] = item
    cleaned = []
    for photo, item in sorted(photo_to_item.items()):
        if item in excluded_items:
            excluded_rows += 1
            if len(excluded_photos) < 100:
                excluded_photos.append(photo)
        else:
            cleaned.append({'photo_id': photo, 'item_id': item})
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_labels.with_suffix(args.output_labels.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for row in cleaned:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(args.output_labels)
    manifest = {
        'source_labels': str(args.labels.resolve()), 'source_sha256': file_sha256(args.labels),
        'raw_label_rows': raw_rows, 'unique_photo_item_pairs': len(photo_to_item),
        'removed_exact_duplicate_rows': duplicate_rows,
        'excluded_item_ids': sorted(excluded_items), 'excluded_unique_pairs': excluded_rows,
        'excluded_photo_examples': excluded_photos, 'clean_pairs': len(cleaned),
        'output_labels': str(args.output_labels.resolve()),
        'output_sha256': file_sha256(args.output_labels),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = args.output_manifest.with_suffix(args.output_manifest.suffix + '.tmp')
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    manifest_tmp.replace(args.output_manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
