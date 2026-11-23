import json
from pathlib import Path
import tempfile
import unittest
from compare_retriever_rerank import compare_cases
from rerank_pipeline import digest, write_rows


class RetrieverRerankTest(unittest.TestCase):
    def fixture(self,root):
        cases=[]
        ranks={'original':[2,None],'pilot100':[1,4],'train1k':[1,3]}
        for name,positions in ranks.items():
            candidates=[]
            predictions=[]
            for i,rank in enumerate(positions):
                ids=['a','b','c','d']
                if rank:
                    ids[rank-1]='p'
                candidates.append(dict(photo_id=f'q{i}',candidates=[dict(item_id=x) for x in ids]))
                predictions.append(dict(photo_id=f'q{i}',ranking=[dict(item_id=x,score=1-j*.1) for j,x in enumerate(ids)],seconds=.1))
            c=root/'candidates'/(name+'.jsonl')
            pred=root/name/'predictions_2000.jsonl'
            shard=root/'shards'/name/'worker_0.jsonl'
            config=root/name/'worker_0/config.json'
            write_rows(candidates,c)
            write_rows(candidates,shard)
            write_rows(predictions,pred)
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps(dict(candidate_sha256=digest(shard),model_weights_sha256={'weights':'fixed'},batch_size=4)))
            cases.append((name,c,pred,[config]))
        return cases

    def test_different_candidates_full_denominator_and_paired_top3(self):
        with tempfile.TemporaryDirectory() as d:
            cases=self.fixture(Path(d))
            result=compare_cases(cases,dict(q0='p',q1='p'),expected=2)
            self.assertEqual(result['arms']['original']['system_top1'],0)
            self.assertEqual(result['arms']['original']['system_top3'],.5)
            self.assertEqual(result['arms']['pilot100']['system_top3'],.5)
            self.assertEqual(result['arms']['train1k']['system_top3'],1)
            self.assertEqual(result['paired_comparisons']['train1k_vs_pilot100']['top3']['net'],1)
            self.assertEqual(result['paired_comparisons']['train1k_vs_pilot100']['top1']['net'],0)

    def test_reject_changed_scoring_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            cases=self.fixture(Path(d))
            cp=cases[-1][-1][0]
            cfg=json.loads(cp.read_text())
            cfg['batch_size']=8
            cp.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'protocol differs'):
                compare_cases(cases,dict(q0='p',q1='p'),expected=2)

    def test_reject_incomplete_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            cases=self.fixture(Path(d))
            path=cases[-1][2]
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['ranking'].pop()
            write_rows(rows,path)
            with self.assertRaisesRegex(ValueError,'invalid ranking'):
                compare_cases(cases,dict(q0='p',q1='p'),expected=2)


if __name__=='__main__':
    unittest.main()
