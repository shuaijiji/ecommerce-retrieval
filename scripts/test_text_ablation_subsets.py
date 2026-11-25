import unittest
from summarize_text_ablation_subsets import analyze,summarize_subset


class TextSubsetTest(unittest.TestCase):
    def fixture(self):
        cases={str(q):dict(truth='p',raw_top3=['p','a','b'],dedup_top3=['p','a','b']) for q in range(4)}
        cases['0']['dedup_top3']=['a','p','b']  # Harm on changed text.
        cases['3']['dedup_top3']=['a','p','b']  # Same-text run drift must be separated.
        preview={str(q):dict(raw='text',dedup='new text' if q<2 else 'text') for q in range(4)}
        split={'calibration':['0','1'],'test':['2','3']}
        groups={**split,'all_2000':list(cases)}
        comparison=dict(text_changed=2,metrics={g:summarize_subset(ids,cases) for g,ids in groups.items()})
        return cases,preview,split,comparison

    def test_partition_and_unchanged_text_drift(self):
        result=analyze(*self.fixture())
        self.assertEqual(result['subsets']['all_2000']['changed']['top1']['lost'],1)
        self.assertEqual(result['subsets']['all_2000']['unchanged']['top1']['lost'],1)
        self.assertEqual(result['unchanged_text_top1_prediction_changes'],['3'])
        self.assertIsNone(result['subsets']['test']['changed']['top1'])

    def test_reject_inconsistent_completed_report(self):
        cases,preview,split,comparison=self.fixture()
        comparison['metrics']['test']['top1']['lost']=0
        with self.assertRaisesRegex(ValueError,'reproduce'):
            analyze(cases,preview,split,comparison)

    def test_reject_changed_text_count_and_overlapping_split(self):
        cases,preview,split,comparison=self.fixture()
        comparison['text_changed']=1
        with self.assertRaisesRegex(ValueError,'changed-text count'):
            analyze(cases,preview,split,comparison)
        comparison['text_changed']=2
        split['test']=['1','3']
        with self.assertRaisesRegex(ValueError,'overlaps'):
            analyze(cases,preview,split,comparison)


if __name__=='__main__':
    unittest.main()
