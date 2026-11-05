"""Synthetic regression checks; no GPU or third-party packages required."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from rerank_pipeline import generate, rrf, evaluate, load_rows


class PipelineTest(unittest.TestCase):
    def test_no_truth_injection_and_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ('vision', 'multimodal'):
                (root / 'results' / mode).mkdir(parents=True)
            for pid in ('1', '2', '3'):
                for mode in ('vision', 'multimodal'):
                    ids = ['a', 'b'] if mode == 'vision' else ['b', 'a']
                    (root / 'results' / mode / f'{pid}.json').write_text(json.dumps({
                        'photo_id': pid, 'top_item_ids': ids, 'top_scores': [.9, .8],
                        'positive_item_id': 'NEVER_INJECT', 'positive_score': 999, 'rank': 1}))
            candidates = root / 'candidates.jsonl'
            generate(argparse.Namespace(retrieval_dir=root, expected_count=3, top_k=1, output=candidates))
            rows = load_rows(candidates)
            self.assertNotIn('NEVER_INJECT', candidates.read_text())
            self.assertEqual(rows[0]['candidates'][0]['multimodal_rank'], 2)
            predictions = root / 'rrf.jsonl'
            rrf(argparse.Namespace(candidates=candidates, rrf_k=60, output=predictions))
            self.assertEqual(load_rows(predictions)[0]['ranking'][0]['item_id'], 'a')
            labels = root / 'labels.json'
            labels.write_text(json.dumps([{'photo_id': p, 'item_id': i} for p, i in [('1', 'a'), ('2', 'b'), ('3', 'absent')]]))
            output = root / 'metrics.json'
            opts = argparse.Namespace(candidates=candidates, predictions=predictions, labels=labels,
                                      expected_count=3, expect_recall=2/3, output=output)
            evaluate(opts)
            metrics = json.loads(output.read_text())
            self.assertEqual(metrics['system_top1'], 1/3)
            self.assertEqual(metrics['candidate_rerank_accuracy'], .5)
            self.assertEqual(metrics['vision_correct_changed_wrong_rate'], 0)
            pred = load_rows(predictions)
            pred[0]['ranking'].reverse()  # equal RRF scores: valid order, now harms visual winner
            from rerank_pipeline import write_rows
            write_rows(pred, predictions)
            evaluate(opts)
            self.assertEqual(json.loads(output.read_text())['vision_correct_changed_wrong_rate'], 1)
            opts.expect_recall = .9039
            with self.assertRaises(ValueError):
                evaluate(opts)
            (root / 'results' / 'vision' / '1.json').unlink()
            with self.assertRaises(ValueError):
                generate(argparse.Namespace(retrieval_dir=root, expected_count=3, top_k=1, output=candidates))


if __name__ == '__main__':
    unittest.main()
