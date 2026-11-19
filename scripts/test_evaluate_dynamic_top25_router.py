"""Synthetic checks for the leakage-controlled dynamic Top-25 router."""
import unittest

from evaluate_dynamic_top25_router import evaluate, selected_ids, split_ids


class DynamicTop25RouterTest(unittest.TestCase):
    def setUp(self):
        self.examples = {}
        for index in range(20):
            low_margin = index % 2 == 0
            photo_id = f"p{index:02d}"
            self.examples[photo_id] = {
                "photo_id": photo_id,
                "top1_score": .8,
                "margin": .01 if low_margin else .5,
                "route_disagreement": low_margin,
                "top10_candidate_count": 2,
                "top25_candidate_count": 4,
                "top10_correct": not low_margin,
                "top25_correct": True,
            }

    def test_split_is_deterministic_and_disjoint(self):
        calibration, test = split_ids(self.examples, 7)
        self.assertEqual((calibration, test), split_ids(self.examples, 7))
        self.assertEqual(len(calibration), 10)
        self.assertEqual(len(test), 10)
        self.assertFalse(set(calibration) & set(test))

    def test_budget_is_exact_and_label_free(self):
        ids = list(self.examples)
        selected = selected_ids(ids, self.examples, "low_margin", .25)
        self.assertEqual(len(selected), 5)
        self.assertTrue(all(self.examples[photo_id]["margin"] == .01 for photo_id in selected))

    def test_evaluation_reports_held_out_curve(self):
        result = evaluate(self.examples, seed=7, budgets=(.5, 1.0))
        self.assertEqual(result["split"]["calibration_count"], 10)
        self.assertEqual(result["split"]["test_count"], 10)
        self.assertEqual(result["budget_curve"][0]["selected_policy_on_calibration"], "low_margin")
        self.assertGreaterEqual(
            result["budget_curve"][0]["test"]["correct"],
            result["test_same_run_top10"]["correct"],
        )
        self.assertEqual(
            result["budget_curve"][1]["test"]["correct"],
            result["test_full_top25"]["correct"],
        )


if __name__ == "__main__":
    unittest.main()
