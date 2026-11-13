"""Synthetic tests for deterministic reranker SFT sampling."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prepare_reranker_sft as prepare


class PrepareRerankerSFTTest(unittest.TestCase):
    def test_stratified_unique_sampling_and_false_negative_exclusion(self):
        vision_ids = [f"v{i}" for i in range(1, 101)]
        multimodal_ids = ["v1"] + [f"m{i}" for i in range(2, 101)]
        row = {
            "photo_id": "p1",
            "positive_item_id": "positive",
            "vision": {
                "item_ids": vision_ids,
                "scores": [1 - i / 200 for i in range(100)],
                "raw_ranks": list(range(1, 101)),
            },
            "multimodal": {
                "item_ids": multimodal_ids,
                "scores": [1 - i / 250 for i in range(100)],
                "raw_ranks": list(range(1, 101)),
            },
        }
        train_items = ["positive", "near_duplicate", *vision_ids, *multimodal_ids, "random"]
        groups = {
            "positive": frozenset({"positive", "near_duplicate"}),
            "near_duplicate": frozenset({"positive", "near_duplicate"}),
        }
        first = prepare.sample_negatives(row, train_items, groups, 7, 20261113)
        second = prepare.sample_negatives(row, train_items, groups, 7, 20261113)
        self.assertEqual(first, second)
        ids = [candidate["item_id"] for candidate in first]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse({"positive", "near_duplicate"} & set(ids))
        self.assertEqual(len(first), 7)
        self.assertEqual(first[-1]["source_mode"], "random")

    def test_candidate_shard_selection_uses_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shards").mkdir()
            photo_ids = ["p0", "p1", "p2"]
            rows = [{"photo_id": photo_id} for photo_id in photo_ids]
            (root / "shards/part-000000-000003.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "config": "config.json",
                        "shards": ["part-000000-000003.jsonl"],
                    }
                )
            )
            selected = prepare.load_selected_candidate_rows(root, photo_ids, ["p1"])
            self.assertEqual(set(selected), {"p1"})


if __name__ == "__main__":
    unittest.main()
