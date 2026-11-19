"""Synthetic checks for cached Top-10 plus new Top-25 candidate emulation."""
import unittest

from evaluate_incremental_top25_router import build_examples, evaluate, summarize


def candidate_row(photo_id, item_ids):
    return {
        "photo_id": photo_id,
        "vision_top1": item_ids[0],
        "candidates": [{"item_id": item_id} for item_id in item_ids],
    }


def prediction_row(photo_id, values):
    ranking = [
        {"item_id": item_id, "score": score}
        for item_id, score in sorted(values.items(), key=lambda value: (-value[1], value[0]))
    ]
    return {"photo_id": photo_id, "ranking": ranking, "seconds": 1.0}


class IncrementalTop25RouterTest(unittest.TestCase):
    def test_cached_score_can_differ_from_full_rerun(self):
        top10_candidates = {"p": candidate_row("p", ["a", "b"])}
        top25_candidates = {"p": candidate_row("p", ["a", "b", "c"])}
        top10_predictions = {"p": prediction_row("p", {"a": .9, "b": .2})}
        top25_predictions = {"p": prediction_row("p", {"a": .7, "b": .2, "c": .8})}
        examples, differences = build_examples(
            top10_candidates,
            top10_predictions,
            top25_candidates,
            top25_predictions,
            {"p": "a"},
        )
        self.assertEqual(examples["p"]["incremental_ranking"][0], "a")
        self.assertEqual(examples["p"]["full_top25_ranking"][0], "c")
        self.assertAlmostEqual(max(differences), .2)

    def test_summary_counts_rescue_and_harm(self):
        examples = {
            "p1": {
                "top10_ranking": ["x", "a"], "incremental_ranking": ["a", "x", "n"],
                "full_top25_ranking": ["a", "x", "n"], "truth": "a",
                "top10_candidate_count": 2, "top25_candidate_count": 3,
            },
            "p2": {
                "top10_ranking": ["b", "x"], "incremental_ranking": ["n", "b", "x"],
                "full_top25_ranking": ["n", "b", "x"], "truth": "b",
                "top10_candidate_count": 2, "top25_candidate_count": 3,
            },
        }
        result = summarize(["p1", "p2"], examples, {"p1", "p2"}, "incremental")
        self.assertEqual(result["rescued_from_top10"], 1)
        self.assertEqual(result["harmed_vs_top10"], 1)
        self.assertEqual(result["net_correct_vs_top10"], 0)

    def test_evaluate_fits_threshold_on_calibration(self):
        examples = {}
        for index in range(20):
            photo_id = f"p{index:02d}"
            hard = index % 2 == 0
            examples[photo_id] = {
                "photo_id": photo_id,
                "top1_score": .2 if hard else .9,
                "top10_candidate_count": 2,
                "top25_candidate_count": 3,
                "top10_ranking": ["x", "a"] if hard else ["a", "x"],
                "incremental_ranking": ["a", "x", "n"],
                "full_top25_ranking": ["a", "x", "n"],
                "truth": "a",
            }
        result = evaluate(examples, [0.0] * 40, seed=7, budget=.5)
        self.assertEqual(result["split"]["calibration_count"], 10)
        self.assertGreaterEqual(
            result["test_dynamic_incremental"]["correct"], result["test_top10"]["correct"]
        )


if __name__ == "__main__":
    unittest.main()
