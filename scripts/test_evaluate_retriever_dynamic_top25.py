import json
from pathlib import Path
import tempfile
import unittest
from evaluate_retriever_dynamic_top25 import freeze_threshold, analyze_arm
from evaluate_incremental_top25_router import evaluate
from rerank_pipeline import write_rows, digest


class FrozenRouterTest(unittest.TestCase):
    def test_freezes_from_cached_scores_without_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            candidates=[dict(photo_id=str(i),candidates=[dict(item_id='a'),dict(item_id='b')]) for i in range(8)]
            predictions=[dict(photo_id=str(i),ranking=[dict(item_id='a',score=.1+i*.1),dict(item_id='b',score=.01)]) for i in range(8)]
            write_rows(candidates,root/'c.jsonl')
            write_rows(predictions,root/'p.jsonl')
            threshold=freeze_threshold(root/'c.jsonl',root/'p.jsonl',expected=8)
            self.assertEqual(threshold['calibration_target_count'],1)
            self.assertEqual(threshold['tie_seed'],20261119)
            self.assertEqual(threshold['feature'],'cached independent Top-10 top1 score')
            predictions.pop()
            write_rows(predictions,root/'p.jsonl')
            with self.assertRaises(ValueError):
                freeze_threshold(root/'c.jsonl',root/'p.jsonl',expected=8)

    def test_complete_cached_analysis_preserves_registered_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            roots=[Path(d)/'cached',Path(d)/'expanded']
            labels={str(i):('c' if i%2 else 'a') for i in range(2000)}
            for root in roots:
                ids=['a','b'] if root==roots[0] else ['c','a','b']
                candidates=[dict(photo_id=str(i),candidates=[dict(item_id=x) for x in ids]) for i in range(2000)]
                predictions=[dict(photo_id=str(i),ranking=[dict(item_id=x,score=.9-j*.1) for j,x in enumerate(ids)],seconds=.1) for i in range(2000)]
                write_rows(candidates,root/'candidates/pilot100.jsonl')
                write_rows(candidates,root/'shards/pilot100/worker_0.jsonl')
                write_rows(predictions,root/'pilot100/predictions_2000.jsonl')
                cp=root/'pilot100/worker_0/config.json'
                cp.parent.mkdir(parents=True)
                cp.write_text(json.dumps(dict(model_weights_sha256={'model':'fixed'},batch_size=4,
                              candidate_sha256=digest(root/'shards/pilot100/worker_0.jsonl'))))
            frozen=freeze_threshold(roots[0]/'candidates/pilot100.jsonl',roots[0]/'pilot100/predictions_2000.jsonl')
            result=analyze_arm('pilot100',roots[0],roots[1],labels)
            self.assertEqual(result['router']['threshold'],frozen)
            self.assertEqual(result['split']['test_count'],1000)
            self.assertEqual(result['test_all_incremental_top25']['system_top3'],1)
            self.assertIn('test_top1_benefit_gate',result)

    def test_router_unchanged_by_labels_and_top25_outcomes(self):
        examples={str(i):dict(top1_score=.1+i*.02,top10_ranking=['a','b'],
                    incremental_ranking=['a','b','c'],full_top25_ranking=['a','b','c'],
                    truth='a',top10_candidate_count=2,top25_candidate_count=3) for i in range(20)}
        before=evaluate(examples,[0]*20,20261119,.25)
        for row in examples.values():
            row['truth']='c'
            row['incremental_ranking']=['c','b','a']
            row['full_top25_ranking']=['c','b','a']
        after=evaluate(examples,[0]*20,20261119,.25)
        self.assertEqual(before['router'],after['router'])
        self.assertEqual(before['test_dynamic_incremental']['expanded_count'],after['test_dynamic_incremental']['expanded_count'])


if __name__=='__main__':
    unittest.main()
