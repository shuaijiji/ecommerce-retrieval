#!/usr/bin/env python3
"""Propose exact/perceptual duplicate item groups; never edits a split automatically."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def hamming(left, right):
    value = left ^ right
    return value.bit_count() if hasattr(value, 'bit_count') else bin(value).count('1')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hash-dir', type=Path, required=True)
    parser.add_argument('--output-groups', type=Path, required=True)
    parser.add_argument('--output-summary', type=Path, required=True)
    parser.add_argument('--hamming-threshold', type=int, default=3)
    parser.add_argument('--max-band-bucket', type=int, default=5000)
    args = parser.parse_args()
    if not 0 <= args.hamming_threshold <= 3:
        raise ValueError('this four-band candidate scheme guarantees completeness only for threshold <= 3')
    rows = []
    shard_hash = hashlib.sha256()
    for path in sorted((args.hash_dir / 'shards').glob('shard_*.jsonl')):
        raw = path.read_bytes()
        shard_hash.update(path.name.encode() + raw)
        rows.extend(json.loads(line) for line in raw.splitlines() if line.strip())
    if not rows or any(row.get('error') for row in rows):
        raise ValueError('missing hash rows or hash errors present')
    if len({row['item_id'] for row in rows}) != len(rows):
        raise ValueError('duplicate item IDs in hash shards')
    dsu = DSU(len(rows))
    exact = defaultdict(list)
    phash = defaultdict(list)
    for index, row in enumerate(rows):
        exact[row['file_sha256']].append(index)
        phash[int(row['phash63'], 16)].append(index)
    for members in exact.values():
        for index in members[1:]:
            dsu.union(members[0], index)
    for members in phash.values():
        for index in members[1:]:
            dsu.union(members[0], index)
    representatives = [(value, members[0]) for value, members in phash.items()]
    buckets, comparisons, near_edges = defaultdict(list), 0, 0
    for value, index in representatives:
        candidates = set()
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xffff)
            members = buckets[key]
            if len(members) > args.max_band_bucket:
                raise ValueError(f'perceptual band bucket too large: {key} has {len(members)} entries')
            candidates.update(members)
        for other_value, other_index in candidates:
            comparisons += 1
            if hamming(value, other_value) <= args.hamming_threshold:
                dsu.union(index, other_index)
                near_edges += 1
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xffff)
            buckets[key].append((value, index))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[dsu.find(index)].append(row)
    proposed = [sorted(members, key=lambda row: row['item_id']) for members in groups.values() if len(members) > 1]
    proposed.sort(key=lambda members: (-len(members), members[0]['item_id']))
    args.output_groups.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_groups.with_suffix(args.output_groups.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        for number, members in enumerate(proposed, 1):
            exact_hashes = len({row['file_sha256'] for row in members})
            handle.write(json.dumps({'proposed_group_id': f'g{number:07d}', 'size': len(members),
                                     'exact_file_hash_count': exact_hashes, 'members': members},
                                    ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(args.output_groups)
    sizes = sorted((len(members) for members in proposed), reverse=True)
    summary = {'status': 'proposed_not_confirmed', 'item_count': len(rows),
               'singleton_items': sum(len(members) == 1 for members in groups.values()),
               'proposed_group_count': len(proposed),
               'items_in_proposed_groups': sum(sizes), 'largest_group_size': sizes[0] if sizes else 0,
               'exact_duplicate_group_count': sum(len(members) > 1 for members in exact.values()),
               'exact_phash_group_count': sum(len(members) > 1 for members in phash.values()),
               'near_hash_edges': near_edges, 'candidate_comparisons': comparisons,
               'hamming_threshold': args.hamming_threshold, 'bands': 4, 'bits_per_band': 16,
               'hash_shards_sha256': shard_hash.hexdigest(),
               'groups_sha256': digest(args.output_groups),
               'warning': 'Perceptual groups require sampled visual review before split enforcement.'}
    write_json(summary, args.output_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
