"""Synthetic checks for the leakage-controlled dynamic Top-25 router."""
import unittest

from evaluate_dynamic_top25_router import (
    evaluate,
    fit_top1_score_threshold,
    selected_ids,
    split_ids,
    threshold_selected_ids,
)


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
        self.assertEqual(result["deployment_candidate"]["policy"], "low_top1_score")
        self.assertEqual(
            result["budget_curve"][1]["test"]["correct"],
            result["test_full_top25"]["correct"],
        )

    def test_fixed_threshold_uses_stable_hash_for_ties(self):
        ids = list(self.examples)
        for example in self.examples.values():
            example["top1_score"] = .5
        threshold = fit_top1_score_threshold(ids, self.examples, .25, tie_seed=11)
        first = threshold_selected_ids(ids, self.examples, threshold)
        second = threshold_selected_ids(list(reversed(ids)), self.examples, threshold)
        self.assertEqual(first, second)
        self.assertEqual(threshold["tie_probability"], .25)


if __name__ == "__main__":
    unittest.main()
