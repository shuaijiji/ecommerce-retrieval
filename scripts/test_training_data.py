"""Synthetic tests for training audit and deterministic split."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_training_data
import prepare_training_pairs
import propose_item_duplicate_groups
import split_training_data


class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'item_images').mkdir()
        (self.root / 'photo_frames').mkdir()
        items, photos, labels = [], [], []
        for number in range(1, 21):
            item_id, photo_id = f'i{number}', f'p{number}'
            item_path = f'item_images/{item_id}.jpg'
            frame_path = f'photo_frames/{photo_id}.jpg'
            (self.root / item_path).write_bytes(b'x')
            (self.root / frame_path).write_bytes(b'x')
            items.append({'item_id': item_id, 'item_title': item_id, 'item_image': item_path})
            photos.append({'photo_id': photo_id, 'photo_frames': [frame_path], 'photo_ocr': item_id})
            labels.append({'photo_id': photo_id, 'item_id': item_id})
        (self.root / 'item_info.jsonl').write_text('\n'.join(map(json.dumps, items)) + '\n')
        (self.root / 'photo_info.jsonl').write_text('\n'.join(map(json.dumps, photos)) + '\n')
        self.labels = self.root / 'label.jsonl'
        self.labels.write_text('\n'.join(map(json.dumps, labels + [labels[0]])) + '\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_audit_and_split(self):
        report = self.root / 'audit.json'
        with patch.object(sys, 'argv', ['audit', '--data-root', str(self.root),
                                       '--output', str(report), '--full-path-check']):
            audit_training_data.main()
        result = json.loads(report.read_text())
        self.assertEqual(result['counts'], {'items': 20, 'photos': 20, 'labels': 21, 'valid_label_pairs': 21})
        self.assertFalse(any(result['critical_checks'].values()))
        self.assertEqual(result['known_or_allowed_issues']['duplicate_label_rows'], 1)
        clean_labels = self.root / 'clean.jsonl'
        clean_manifest = self.root / 'clean_manifest.json'
        with patch.object(sys, 'argv', ['prepare', '--labels', str(self.labels),
                                       '--output-labels', str(clean_labels), '--output-manifest', str(clean_manifest),
                                       '--exclude-item-id', 'i20']):
            prepare_training_pairs.main()
        clean = json.loads(clean_manifest.read_text())
        self.assertEqual(clean['removed_exact_duplicate_rows'], 1)
        self.assertEqual(clean['excluded_unique_pairs'], 1)
        self.assertEqual(clean['clean_pairs'], 19)
        out = self.root / 'split'
        with patch.object(sys, 'argv', ['split', '--labels', str(clean_labels), '--output-dir', str(out),
                                       '--dev-item-ratio', '.25', '--seed', '1']):
            split_training_data.main()
        train_items = set((out / 'train_item_ids.txt').read_text().splitlines())
        dev_items = set((out / 'dev_item_ids.txt').read_text().splitlines())
        self.assertFalse(train_items & dev_items)
        self.assertEqual(len(train_items | dev_items), 19)
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['counts']['removed_exact_duplicate_rows'], 0)

    def test_duplicate_group_proposals(self):
        hash_dir = self.root / 'hashes'
        (hash_dir / 'shards').mkdir(parents=True)
        rows = [
            {'item_id': 'a', 'item_image': 'a.jpg', 'file_sha256': 'sha-a', 'phash63': '0000000000000000', 'error': None},
            {'item_id': 'b', 'item_image': 'b.jpg', 'file_sha256': 'sha-b', 'phash63': '0000000000000001', 'error': None},
            {'item_id': 'c', 'item_image': 'c.jpg', 'file_sha256': 'same', 'phash63': '7fffffffffffffff', 'error': None},
            {'item_id': 'd', 'item_image': 'd.jpg', 'file_sha256': 'same', 'phash63': '7ffffffffffffffe', 'error': None},
            {'item_id': 'e', 'item_image': 'e.jpg', 'file_sha256': 'sha-e', 'phash63': '3333333333333333', 'error': None},
        ]
        (hash_dir / 'shards/shard_0000000.jsonl').write_text('\n'.join(map(json.dumps, rows)) + '\n')
        groups, summary = self.root / 'groups.jsonl', self.root / 'groups_summary.json'
        with patch.object(sys, 'argv', ['groups', '--hash-dir', str(hash_dir),
                                       '--output-groups', str(groups), '--output-summary', str(summary),
                                       '--hamming-threshold', '3']):
            propose_item_duplicate_groups.main()
        result = json.loads(summary.read_text())
        self.assertEqual(result['item_count'], 5)
        self.assertEqual(result['proposed_group_count'], 2)
        self.assertEqual(result['items_in_proposed_groups'], 4)
        self.assertEqual(result['singleton_items'], 1)


if __name__ == '__main__':
    unittest.main()
