"""Model-free checks for dynamic incremental reranking helpers."""
import unittest

from evaluate_dynamic_top25_router import _hash_fraction
from rerank_dynamic_top25_incremental import exact_p_value, hash_fraction, percentile, should_expand


class DynamicIncrementalRerankTest(unittest.TestCase):
    def test_threshold(self):
        self.assertTrue(should_expand("p", .8, .85, 0, 7))
        self.assertFalse(should_expand("p", .9, .85, 1, 7))

    def test_tie_hash_matches_offline_evaluator(self):
        for photo_id in ("1", "abc", "999"):
            self.assertEqual(hash_fraction(photo_id, 20261119), _hash_fraction(photo_id, 20261119))

    def test_percentile(self):
        values = [5, 1, 4, 2, 3]
        self.assertEqual(percentile(values, .5), 3)
        self.assertEqual(percentile(values, 1), 5)

    def test_exact_p_value(self):
        self.assertEqual(exact_p_value(0, 0), 1)
        self.assertAlmostEqual(exact_p_value(10, 1), .01171875)


if __name__ == "__main__":
    unittest.main()
