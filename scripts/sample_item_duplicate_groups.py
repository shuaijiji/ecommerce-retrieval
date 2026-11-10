#!/usr/bin/env python3
"""Create deterministic contact sheets for visual review of perceptual duplicate groups."""
import argparse
import hashlib
import json
from pathlib import Path


def stable_key(seed, group_id):
    return hashlib.sha256(f'{seed}:{group_id}'.encode()).digest()


def hamming(left, right):
    value = int(left, 16) ^ int(right, 16)
    return value.bit_count() if hasattr(value, 'bit_count') else bin(value).count('1')


def group_distance(members):
    values = sorted({row['phash63'] for row in members})
    if len(values) < 2:
        return 0
    return max(hamming(left, right) for number, left in enumerate(values)
               for right in values[number + 1:])


def load_candidates(path):
    exact_only, perceptual = [], []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            (exact_only if row['exact_file_hash_count'] == 1 else perceptual).append(row)
    return exact_only, perceptual


def sample_groups(groups, sample_size, largest_count, seed):
    largest = sorted(groups, key=lambda row: (-row['size'], row['proposed_group_id']))[:largest_count]
    selected_ids = {row['proposed_group_id'] for row in largest}
    remainder = [row for row in groups if row['proposed_group_id'] not in selected_ids]
    remainder.sort(key=lambda row: stable_key(seed, row['proposed_group_id']))
    return largest + remainder[:max(0, sample_size - len(largest))]


def select_members(members, limit):
    """Choose different files with greedy pHash diversity instead of adjacent item IDs."""
    by_file = {}
    for row in members:
        by_file.setdefault(row['file_sha256'], row)
    candidates = sorted(by_file.values(), key=lambda row: row['item_id'])
    if len(candidates) <= limit:
        return candidates
    selected = [candidates.pop(0)]
    while candidates and len(selected) < limit:
        next_row = max(candidates, key=lambda row: (
            min(hamming(row['phash63'], chosen['phash63']) for chosen in selected),
            row['item_id']))
        candidates.remove(next_row)
        selected.append(next_row)
    return selected


def fit_image(image, width, height):
    from PIL import Image
    image = image.convert('RGB')
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (width, height), 'white')
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def draw_sheet(groups, data_root, destination, members_per_group, cell_size):
    from PIL import Image, ImageDraw
    label_height, header_width = 34, 220
    row_height = cell_size + label_height
    sheet = Image.new('RGB', (header_width + members_per_group * cell_size,
                              len(groups) * row_height), 'white')
    draw = ImageDraw.Draw(sheet)
    review_rows = []
    for row_number, group in enumerate(groups):
        top = row_number * row_height
        members = select_members(group['members'], members_per_group)
        max_distance = group_distance(group['members'])
        draw.text((8, top + 8), f"{group['proposed_group_id']} size={group['size']}", fill='black')
        draw.text((8, top + 24), f"files={group['exact_file_hash_count']} max_h={max_distance}", fill='black')
        errors = []
        for column, member in enumerate(members):
            left = header_width + column * cell_size
            try:
                with Image.open(data_root / member['item_image']) as source:
                    tile = fit_image(source, cell_size, cell_size - label_height)
                sheet.paste(tile, (left, top))
            except Exception as exc:
                errors.append({'item_id': member['item_id'], 'error': f'{type(exc).__name__}: {exc}'})
                draw.rectangle((left, top, left + cell_size - 1, top + cell_size - label_height - 1),
                               outline='red', width=3)
            draw.text((left + 3, top + cell_size - label_height + 4), str(member['item_id'])[:24], fill='black')
        draw.line((0, top + row_height - 1, sheet.width, top + row_height - 1), fill='#aaaaaa')
        review_rows.append({'proposed_group_id': group['proposed_group_id'], 'size': group['size'],
                            'exact_file_hash_count': group['exact_file_hash_count'],
                            'max_pairwise_phash_distance': max_distance,
                            'shown_item_ids': [row['item_id'] for row in members],
                            'render_errors': errors, 'review_decision': None, 'review_note': None})
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)
    return review_rows


def write_json(value, path):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--groups', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--largest-count', type=int, default=20)
    parser.add_argument('--groups-per-sheet', type=int, default=10)
    parser.add_argument('--members-per-group', type=int, default=8)
    parser.add_argument('--cell-size', type=int, default=180)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    for name in ('sample_size', 'groups_per_sheet', 'members_per_group', 'cell_size'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if not 0 <= args.largest_count <= args.sample_size:
        raise ValueError('largest-count must be between zero and sample-size')
    exact_only, perceptual = load_candidates(args.groups)
    selected = sample_groups(perceptual, min(args.sample_size, len(perceptual)),
                             min(args.largest_count, len(perceptual)), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviews, sheet_names = [], []
    for start in range(0, len(selected), args.groups_per_sheet):
        number = start // args.groups_per_sheet + 1
        name = f'review_sheet_{number:03d}.jpg'
        reviews.extend(draw_sheet(selected[start:start + args.groups_per_sheet], args.data_root,
                                  args.output_dir / name, args.members_per_group, args.cell_size))
        sheet_names.append(name)
    review_path = args.output_dir / 'review_manifest.jsonl'
    with review_path.open('w', encoding='utf-8') as handle:
        for row in reviews:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    summary = {
        'status': 'awaiting_visual_review',
        'all_proposed_groups': len(exact_only) + len(perceptual),
        'exact_file_only_groups': len(exact_only),
        'exact_file_only_items': sum(row['size'] for row in exact_only),
        'perceptual_candidate_groups': len(perceptual),
        'perceptual_candidate_items': sum(row['size'] for row in perceptual),
        'sampled_perceptual_groups': len(selected),
        'largest_groups_in_sample': min(args.largest_count, len(selected)),
        'random_groups_in_sample': max(0, len(selected) - min(args.largest_count, len(selected))),
        'seed': args.seed, 'sheets': sheet_names, 'review_manifest': review_path.name,
        'decision_values': ['same_or_near_duplicate', 'different', 'uncertain'],
        'note': 'Only sampled visual review is performed; exact byte-identical groups need no visual confirmation.'
    }
    write_json(summary, args.output_dir / 'summary.json')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
