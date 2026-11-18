import json
import tempfile
import unittest
from pathlib import Path

from summarize_score_policy_grpo_stats import summarize


class SummarizeScorePolicyStatsTest(unittest.TestCase):
    def test_weighted_summary_and_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.jsonl"
            rows = [
                {
                    "groups": 2,
                    "active_groups": 1,
                    "mean_reward": 0.25,
                    "mean_absolute_advantage": 0.5,
                    "mean_policy_entropy": 1.0,
                    "mean_positive_policy_probability": 0.3,
                    "loss": 0.1,
                },
                {
                    "groups": 1,
                    "active_groups": 1,
                    "mean_reward": 0.5,
                    "mean_absolute_advantage": 0.8,
                    "mean_policy_entropy": 0.7,
                    "mean_positive_policy_probability": 0.6,
                    "loss": -0.2,
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = summarize([path])
            self.assertEqual(result["groups"], 3)
            self.assertEqual(result["active_groups"], 2)
            self.assertAlmostEqual(result["active_group_rate"], 2 / 3)
            self.assertTrue(result["all_losses_finite"])
            self.assertTrue(result["passes_training_signal_gate"])


if __name__ == "__main__":
    unittest.main()

