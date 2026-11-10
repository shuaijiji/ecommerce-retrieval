#!/usr/bin/env python3
"""Streaming audit for training metadata, labels and image references."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

TEXT_FIELDS = ('photo_caption', 'photo_title', 'photo_text', 'photo_ocr', 'photo_asr')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def reservoir_add(sample, value, seen, limit, rng):
    if limit <= 0:
        return
    if len(sample) < limit:
        sample.append(value)
    else:
        position = rng.randrange(seen)
        if position < limit:
            sample[position] = value


def read_labels(path):
    errors, pairs, source_count, error_count = [], [], 0, 0
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as handle:
            source = ((line_number, line) for line_number, line in enumerate(handle, 1) if line.strip())
            for line_number, line in source:
                source_count += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError('not an object')
                    pairs.append((str(row['photo_id']), str(row['item_id'])))
                except Exception as exc:
                    error_count += 1
                    if len(errors) < 20:
                        errors.append({'line': line_number, 'error': str(exc)})
    else:
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            raise TypeError('label JSON must contain an array')
        source_count = len(raw)
        for index, row in enumerate(raw):
            try:
                if not isinstance(row, dict):
                    raise TypeError('not an object')
                pairs.append((str(row['photo_id']), str(row['item_id'])))
            except Exception as exc:
                error_count += 1
                if len(errors) < 20:
                    errors.append({'index': index, 'error': str(exc)})
    return source_count, pairs, error_count, errors


def read_jsonl(path, id_field, image_fields, sample_size, rng, data_root, full_path_check):
    ids, duplicates, errors, paths = set(), 0, [], []
    count, error_count, path_seen, missing_path_count = 0, 0, 0, 0
    missing_path_examples, text_counts, frame_counts = [], Counter(), Counter()
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError('not a JSON object')
                identity = row.get(id_field)
                if identity is None or not isinstance(identity, (str, int)):
                    raise ValueError(f'missing/invalid {id_field}')
                identity = str(identity)
                duplicates += identity in ids
                ids.add(identity)
                count += 1
                for field in TEXT_FIELDS:
                    text_counts[field] += bool(str(row.get(field) or '').strip())
                if 'photo_frames' in image_fields:
                    frames = row.get('photo_frames') or []
                    if not isinstance(frames, list):
                        raise TypeError('photo_frames is not a list')
                    frame_counts[len(frames)] += 1
                    for relative in frames:
                        path_seen += 1
                        if full_path_check:
                            if not relative or not (data_root / relative).is_file():
                                missing_path_count += 1
                                if len(missing_path_examples) < 20:
                                    missing_path_examples.append((identity, relative))
                        else:
                            reservoir_add(paths, (identity, relative), path_seen, sample_size, rng)
                if 'item_image' in image_fields:
                    relative = row.get('item_image')
                    path_seen += 1
                    if full_path_check:
                        if not relative or not (data_root / relative).is_file():
                            missing_path_count += 1
                            if len(missing_path_examples) < 20:
                                missing_path_examples.append((identity, relative))
                    else:
                        reservoir_add(paths, (identity, relative), path_seen, sample_size, rng)
            except Exception as exc:
                error_count += 1
                if len(errors) < 20:
                    errors.append({'line': line_number, 'error': str(exc)})
    return {'count': count, 'ids': ids, 'duplicates': duplicates, 'error_count': error_count, 'errors': errors,
            'text_counts': text_counts, 'frame_counts': frame_counts, 'path_sample': paths,
            'missing_path_count': missing_path_count, 'missing_path_examples': missing_path_examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--path-sample-size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--full-path-check', action='store_true')
    parser.add_argument('--allow-missing-item-images', type=int, default=0)
    args = parser.parse_args()
    root = args.data_root.resolve()
    labels_path = next((root / name for name in ('label.jsonl', 'label.json') if (root / name).is_file()), None)
    required = [root / 'item_info.jsonl', root / 'photo_info.jsonl']
    if labels_path is not None:
        required.append(labels_path)
    missing = [str(path) for path in required if not path.is_file()]
    if labels_path is None:
        missing.append(str(root / 'label.jsonl or label.json'))
    if missing:
        raise FileNotFoundError('missing required metadata: ' + ', '.join(missing))
    rng = random.Random(args.seed)
    sample_size = args.path_sample_size
    items = read_jsonl(required[0], 'item_id', {'item_image'}, sample_size, rng, root, args.full_path_check)
    photos = read_jsonl(required[1], 'photo_id', {'photo_frames'}, sample_size, rng, root, args.full_path_check)
    label_source_count, label_pairs, label_error_count, label_errors = read_labels(labels_path)
    photo_label_counts = Counter(photo for photo, _ in label_pairs)
    pair_counts = Counter(label_pairs)
    first_item_by_photo, conflicting_photos = {}, set()
    for photo, item in label_pairs:
        previous = first_item_by_photo.setdefault(photo, item)
        if previous != item:
            conflicting_photos.add(photo)
    missing_item_paths = items['missing_path_examples'] if args.full_path_check else [
        (identity, relative) for identity, relative in items['path_sample']
        if not relative or not (root / relative).is_file()]
    missing_frame_paths = photos['missing_path_examples'] if args.full_path_check else [
        (identity, relative) for identity, relative in photos['path_sample']
        if not relative or not (root / relative).is_file()]
    actual_missing_items = items['missing_path_count'] if args.full_path_check else len(missing_item_paths)
    actual_missing_frames = photos['missing_path_count'] if args.full_path_check else len(missing_frame_paths)
    critical = {
        'item_json_errors': items['error_count'], 'photo_json_errors': photos['error_count'],
        'label_errors': label_error_count, 'duplicate_item_ids': items['duplicates'],
        'duplicate_photo_ids': photos['duplicates'],
        'photos_with_conflicting_items': len(conflicting_photos),
        'missing_photo_references': sum(photo not in photos['ids'] for photo, _ in label_pairs),
        'missing_item_references': sum(item not in items['ids'] for _, item in label_pairs),
        'photos_without_labels': len(photos['ids'] - set(photo_label_counts)),
        'unexpected_missing_item_images': max(0, actual_missing_items - args.allow_missing_item_images),
        'missing_frames': actual_missing_frames,
    }
    result = {
        'data_root': str(root), 'seed': args.seed,
        'path_check': 'full' if args.full_path_check else f'reservoir_sample_{args.path_sample_size}',
        'counts': {'items': items['count'], 'photos': photos['count'], 'labels': label_source_count,
                   'valid_label_pairs': len(label_pairs)},
        'critical_checks': critical,
        'known_or_allowed_issues': {'actual_missing_item_images': actual_missing_items,
                                    'allowed_missing_item_images': args.allow_missing_item_images,
                                    'duplicate_label_rows': sum(value - 1 for value in pair_counts.values()),
                                    'photos_with_repeated_label_rows': sum(value > 1 for value in photo_label_counts.values()),
                                    'unique_label_pairs': len(pair_counts)},
        'frame_count_distribution': dict(sorted(photos['frame_counts'].items())),
        'nonempty_video_text': {field: {'count': photos['text_counts'][field],
                                        'rate': photos['text_counts'][field] / photos['count'] if photos['count'] else 0}
                                for field in TEXT_FIELDS},
        'examples': {'item_json_errors': items['errors'], 'photo_json_errors': photos['errors'],
                     'label_errors': label_errors, 'missing_item_images': missing_item_paths[:20],
                     'missing_frames': missing_frame_paths[:20],
                     'conflicting_photo_ids': sorted(conflicting_photos)[:20]},
        'metadata_sha256': {path.name: sha256(path) for path in required},
    }
    atomic_save(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(critical.values()):
        raise SystemExit('training audit found critical issues; inspect output before splitting/training')


if __name__ == '__main__':
    main()
