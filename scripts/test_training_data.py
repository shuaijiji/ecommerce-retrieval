"""Synthetic tests for training audit and deterministic split."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_training_data
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
        self.labels.write_text('\n'.join(map(json.dumps, labels)) + '\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_audit_and_split(self):
        report = self.root / 'audit.json'
        with patch.object(sys, 'argv', ['audit', '--data-root', str(self.root),
                                       '--output', str(report), '--full-path-check']):
            audit_training_data.main()
        result = json.loads(report.read_text())
        self.assertEqual(result['counts'], {'items': 20, 'photos': 20, 'labels': 20, 'valid_label_pairs': 20})
        self.assertFalse(any(result['critical_checks'].values()))
        out = self.root / 'split'
        with patch.object(sys, 'argv', ['split', '--labels', str(self.labels), '--output-dir', str(out),
                                       '--dev-item-ratio', '.25', '--seed', '1']):
            split_training_data.main()
        train_items = set((out / 'train_item_ids.txt').read_text().splitlines())
        dev_items = set((out / 'dev_item_ids.txt').read_text().splitlines())
        self.assertFalse(train_items & dev_items)
        self.assertEqual(len(train_items | dev_items), 20)


if __name__ == '__main__':
    unittest.main()
