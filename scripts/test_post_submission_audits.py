import json
from pathlib import Path
import tempfile
import unittest
from audit_submission_readonly import read_submission,score_denominator_check
from audit_retriever_error_budget import classify,summarize_cases,selected_cases,BUCKETS,build_cases
from export_official_submission import write_xlsx


class PostSubmissionAuditTest(unittest.TestCase):
    def test_submission_roundtrip_and_wrong_count(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'book.xlsx'
            rows=[('0001234567890123456789',['100','101','102'])]
            write_xlsx(rows,path)
            self.assertEqual(read_submission(path,expected=1),dict(rows))
            with self.assertRaisesRegex(ValueError,'expected 2'):
                read_submission(path,expected=2)

    def test_denominator_is_not_assumed_from_screenshot(self):
        result=score_denominator_check('70.8750','82.9300',50000)
        self.assertTrue(result['needs_organizer_clarification'])
        self.assertEqual(result['checks']['top1']['implied_correct'],'70875/2')
        self.assertFalse(result['checks']['top1']['integer_correct_count'])
        self.assertTrue(score_denominator_check('70.8750','82.9300',40000)['checks']['top1']['integer_correct_count'])
        # Integer arithmetic consistency is a check, not evidence that 40k is actually scored.

    def test_saved_retrieval_and_frozen_routing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ids=[str(i) for i in range(100)]
            for mode in ('vision','multimodal'):
                directory=root/'results'/mode
                directory.mkdir(parents=True)
                (directory/'42.json').write_text(json.dumps(dict(photo_id='42',positive_item_id='20',
                    top_item_ids=ids,top_scores=[1-i/100 for i in range(100)])))
            example=dict(top10_ranking=ids[:10],full_top25_ranking=ids[:25],
                incremental_ranking=['20']+ids[:20]+ids[21:25],top1_score=.9)
            threshold=dict(cutoff=.8,tie_probability=0,tie_seed=20261119)
            result=build_cases({'42':example},root,{'42':'20'},threshold)[0]
            self.assertEqual(result['bucket'],'not_expanded_truth_only_in_top25_tail')
            self.assertFalse(result['expanded'])
            threshold['cutoff']=1
            result=build_cases({'42':example},root,{'42':'20'},threshold)[0]
            self.assertEqual(result['bucket'],'correct_top1')
            self.assertEqual(result['truth_rank_final'],1)
            example['top10_ranking']=ids[:9]
            with self.assertRaisesRegex(ValueError,'candidate truncation'):
                build_cases({'42':example},root,{'42':'20'},threshold)

    def test_error_categories_exhaustive_and_summary_denominator(self):
        cases=[]
        inputs=[('p',{'p'},{'p'},['p','a','b']),
                ('p',{'a'},{'a'},['a','b','c']),
                ('p',{'p','a'},{'a'},['a','b','c']),
                ('p',{'p','a'},{'p','a'},['a','b','c']),
                ('p',{'p','a'},{'p','a'},['a','p','b']),
                ('p',{'p','a'},{'p','a'},['a','b','c','p'])]
        for i,(truth,h100,h25,ranking) in enumerate(inputs):
            cases.append(dict(photo_id=str(i),bucket=classify(truth,h100,h25,ranking),
                hit_top100=truth in h100,hit_top25=truth in h25,
                truth_rank_final=ranking.index(truth)+1 if truth in ranking else None,expanded=False))
        self.assertEqual([r['bucket'] for r in cases],list(BUCKETS))
        report=summarize_cases(cases)
        self.assertEqual(report['error_count'],5)
        self.assertEqual(sum(v['count'] for v in report['error_buckets'].values()),5)
        self.assertAlmostEqual(report['top1'],1/6)
        self.assertAlmostEqual(report['top3'],2/6)
        self.assertEqual(selected_cases(cases,1),selected_cases(list(reversed(cases)),1))


if __name__=='__main__':
    unittest.main()
