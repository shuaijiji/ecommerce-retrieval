"""Synthetic paired comparison checks without model dependencies."""
import json
import tempfile
import unittest
from pathlib import Path

from compare_reranker_predictions import summarize
from rerank_pipeline import load_labels, load_rows, write_rows


class ComparePredictionsTest(unittest.TestCase):
    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_path = root / "candidates.jsonl"
            write_rows([
                {"photo_id": "p1", "vision_top1": "a", "candidates": [{"item_id": "a"}, {"item_id": "b"}]},
                {"photo_id": "p2", "vision_top1": "a", "candidates": [{"item_id": "a"}, {"item_id": "b"}]},
            ], candidates_path)
            labels_path = root / "labels.jsonl"
            labels_path.write_text(
                json.dumps({"photo_id": "p1", "item_id": "a"}) + "\n" +
                json.dumps({"photo_id": "p2", "item_id": "b"}) + "\n"
            )
            predictions_path = root / "predictions.jsonl"
            write_rows([
                {"photo_id": "p1", "ranking": [{"item_id": "a", "score": .9}, {"item_id": "b", "score": .1}], "seconds": 1},
                {"photo_id": "p2", "ranking": [{"item_id": "a", "score": .8}, {"item_id": "b", "score": .2}], "seconds": 3},
            ], predictions_path)
            candidates = {row["photo_id"]: row for row in load_rows(candidates_path)}
            _, labels = load_labels(labels_path)
            metrics, correctness = summarize(predictions_path, candidates, labels)
            self.assertEqual(metrics["system_top1"], .5)
            self.assertEqual(metrics["system_mrr"], .75)
            self.assertEqual(metrics["mean_seconds_per_video"], 2)
            self.assertEqual(correctness, {"p1": True, "p2": False})


if __name__ == "__main__":
    unittest.main()
