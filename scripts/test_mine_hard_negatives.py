"""Synthetic checks for hard-negative shard and exclusion invariants."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_module_without_gpu_dependencies():
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = object
    fake_common = types.ModuleType("retrieval_common")
    fake_common.QUERY_INSTRUCTION = "query"
    fake_common.TEXT_BUDGETS = ()
    for name in (
        "atomic_json_dump",
        "compose_video_text",
        "encode_checked",
        "load_embedder",
        "load_video_frames",
        "sha256_file",
    ):
        setattr(fake_common, name, lambda *args, **kwargs: None)
    path = Path(__file__).with_name("mine_hard_negatives.py")
    spec = importlib.util.spec_from_file_location("mine_hard_negatives_test_target", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake_torch, "retrieval_common": fake_common}):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


mine = load_module_without_gpu_dependencies()


class HardNegativeTest(unittest.TestCase):
    def test_shards_and_duplicate_exclusion_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups_path = root / "groups.jsonl"
            groups_path.write_text(
                json.dumps(
                    {
                        "proposed_group_id": "g1",
                        "members": [{"item_id": "a"}, {"item_id": "b"}],
                    }
                )
                + "\n"
            )
            groups, count = mine.load_duplicate_groups(groups_path)
            self.assertEqual(count, 1)
            self.assertEqual(groups["a"][1], frozenset({"a", "b"}))

            specs = mine.shard_specs(["p1", "p2", "p3"], 2)
            self.assertEqual(
                [(start, ids, name) for _, start, ids, name in specs],
                [(0, ["p1", "p2"], "part-000000-000002.jsonl"),
                 (2, ["p3"], "part-000002-000003.jsonl")],
            )

            result_path = root / "part.jsonl"
            valid = {
                "photo_id": "p1",
                "positive_item_id": "a",
                "vision": {
                    "item_ids": ["c", "d"],
                    "scores": [0.9, 0.8],
                    "raw_ranks": [2, 3],
                },
                "multimodal": {
                    "item_ids": ["d", "c"],
                    "scores": [0.7, 0.6],
                    "raw_ranks": [1, 3],
                },
            }
            result_path.write_text(json.dumps(valid) + "\n")
            self.assertTrue(mine.validate_result_shard(result_path, ["p1"], 2, groups))

            valid["vision"]["item_ids"][0] = "b"
            result_path.write_text(json.dumps(valid) + "\n")
            with self.assertRaises(ValueError):
                mine.validate_result_shard(result_path, ["p1"], 2, groups)

    def test_label_conflicts_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "labels.jsonl"
            labels.write_text(
                json.dumps({"photo_id": "p1", "item_id": "a"})
                + "\n"
                + json.dumps({"photo_id": "p1", "item_id": "b"})
                + "\n"
            )
            with self.assertRaises(ValueError):
                mine.load_positive_items(labels, {"p1"})


if __name__ == "__main__":
    unittest.main()
