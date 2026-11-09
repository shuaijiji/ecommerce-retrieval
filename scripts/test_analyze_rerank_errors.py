"""Synthetic tests for error decomposition and paired candidate comparison."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

from analyze_rerank_errors import analyze, compare
from rerank_pipeline import write_rows


class ErrorAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.labels = self.root / 'labels.json'
        self.labels.write_text(json.dumps([
            {'photo_id': '1', 'item_id': 'a'},
            {'photo_id': '2', 'item_id': 'b'},
            {'photo_id': '3', 'item_id': 'c'},
        ]))
        self.base_candidates = self.root / 'base_candidates.jsonl'
        write_rows([
            self.candidate('1', 'a', ['a', 'x']),
            self.candidate('2', 'x', ['b', 'x']),
            self.candidate('3', 'x', ['x', 'y']),
        ], self.base_candidates)
        self.base_predictions = self.root / 'base_predictions.jsonl'
        write_rows([
            self.prediction('1', ['a', 'x']),
            self.prediction('2', ['x', 'b']),
            self.prediction('3', ['x', 'y']),
        ], self.base_predictions)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def candidate(pid, vision_top1, ids):
        return {'photo_id': pid, 'vision_top1': vision_top1,
                'candidates': [{'item_id': item, 'vision_rank': rank, 'vision_similarity': 1 / rank,
                                'multimodal_rank': rank, 'multimodal_similarity': 1 / rank}
                               for rank, item in enumerate(ids, 1)]}

    @staticmethod
    def prediction(pid, ids):
        return {'photo_id': pid, 'ranking': [{'item_id': item, 'score': .9 - rank * .1}
                                             for rank, item in enumerate(ids)], 'seconds': .1}

    def test_analyze_three_outcomes(self):
        summary = self.root / 'summary.json'
        cases = self.root / 'cases.jsonl'
        analyze(argparse.Namespace(candidates=self.base_candidates, predictions=self.base_predictions,
                                   labels=self.labels, retrieval_dir=None, photo_info=None,
                                   output_summary=summary, output_cases=cases))
        result = json.loads(summary.read_text())
        self.assertEqual(result['status']['correct']['count'], 1)
        self.assertEqual(result['status']['rerank_wrong']['count'], 1)
        self.assertEqual(result['status']['candidate_miss']['count'], 1)
        self.assertAlmostEqual(result['candidate_recall'], 2 / 3)

    def test_paired_comparison(self):
        new_candidates = self.root / 'new_candidates.jsonl'
        write_rows([
            self.candidate('1', 'a', ['a', 'x', 'z']),
            self.candidate('2', 'x', ['b', 'x', 'z']),
            self.candidate('3', 'x', ['x', 'y', 'c']),
        ], new_candidates)
        new_predictions = self.root / 'new_predictions.jsonl'
        write_rows([
            self.prediction('1', ['x', 'a', 'z']),  # regression
            self.prediction('2', ['b', 'x', 'z']),  # recovered rerank error
            self.prediction('3', ['c', 'x', 'y']),  # recovered candidate miss
        ], new_predictions)
        summary = self.root / 'compare.json'
        compare(argparse.Namespace(base_candidates=self.base_candidates, base_predictions=self.base_predictions,
                                   new_candidates=new_candidates, new_predictions=new_predictions,
                                   labels=self.labels, output_summary=summary,
                                   output_cases=self.root / 'compare_cases.jsonl'))
        result = json.loads(summary.read_text())
        self.assertEqual(result['delta']['newly_correct_count'], 2)
        self.assertEqual(result['delta']['newly_wrong_count'], 1)
        self.assertEqual(result['delta']['net_correct_count'], 1)
        self.assertAlmostEqual(result['new']['candidate_recall'], 1)


if __name__ == '__main__':
    unittest.main()
