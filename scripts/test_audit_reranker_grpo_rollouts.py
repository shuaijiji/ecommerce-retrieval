import json
import tempfile
import unittest
from pathlib import Path

from audit_reranker_grpo_rollouts import audit, probability_to_logit, softmax


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class AuditRerankerGrpoRolloutsTest(unittest.TestCase):
    def test_probability_round_trip_and_softmax(self):
        logits = [probability_to_logit(0.8), probability_to_logit(0.2)]
        policy = softmax(logits, 1.0)
        self.assertAlmostEqual(sum(policy), 1.0)
        self.assertGreater(policy[0], 0.9)

    def test_audit_reports_candidate_miss_and_active_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            predictions = root / "predictions.jsonl"
            labels = root / "labels.json"
            write_jsonl(candidates, [
                {"photo_id": "1", "candidates": [{"item_id": "a"}, {"item_id": "b"}]},
                {"photo_id": "2", "candidates": [{"item_id": "a"}, {"item_id": "b"}]},
            ])
            write_jsonl(predictions, [
                {"photo_id": "1", "ranking": [
                    {"item_id": "a", "score": 0.8}, {"item_id": "b", "score": 0.2}
                ], "seconds": 0.1},
                {"photo_id": "2", "ranking": [
                    {"item_id": "b", "score": 0.6}, {"item_id": "a", "score": 0.4}
                ], "seconds": 0.1},
            ])
            labels.write_text(json.dumps([
                {"photo_id": "1", "item_id": "a"},
                {"photo_id": "2", "item_id": "c"},
            ]))

            result = audit(candidates, predictions, labels, [4], [1.0], 128, 7)

            self.assertEqual(result["count"], 2)
            self.assertEqual(result["candidate_hits"], 1)
            self.assertEqual(result["candidate_recall"], 0.5)
            config = result["configurations"][0]
            self.assertEqual(config["queries_with_positive"], 1)
            self.assertGreater(config["expected_active_group_rate"], 0)
            self.assertLess(config["expected_active_group_rate"], 1)
            self.assertGreater(config["simulated_mean_absolute_advantage"], 0)


if __name__ == "__main__":
    unittest.main()
