import copy
import unittest
import tempfile
import json
from pathlib import Path
from prepare_embedding_pilot import check_row, choose_negatives
from compare_embedding_recall import paired, compare


class EmbeddingPilotTest(unittest.TestCase):
    def setUp(self):
        self.row = {'photo_id': 'q', 'positive_item_id': 'p'}
        for mode, prefix in [('vision', 'v'), ('multimodal', 'm')]:
            self.row[mode] = {'item_ids': [prefix+str(i) for i in range(100)],
                              'scores': [1-i/100 for i in range(100)],
                              'raw_ranks': list(range(1,101))}
        self.items = {'p'} | set(self.row['vision']['item_ids']) | set(self.row['multimodal']['item_ids'])

    def test_valid_and_balanced_top_ranked_selection(self):
        check_row(self.row, 'p', self.items, {})
        self.assertEqual([n['item_id'] for n in choose_negatives(self.row,4)], ['v0','m0','v1','m1'])

    def test_reject_wrong_label_positive_duplicate_and_dev_item(self):
        with self.assertRaises(ValueError):
            check_row(self.row, 'other', self.items, {})
        for forbidden in ('p', 'near', 'dev'):
            row=copy.deepcopy(self.row)
            row['vision']['item_ids'][0]=forbidden
            with self.assertRaises(ValueError):
                check_row(row,'p',self.items|{'near'},{'p':frozenset({'p','near'})})

    def test_reject_nan_duplicate_and_bad_rank(self):
        for field, value in [('scores',float('nan')),('item_ids','v1'),('raw_ranks',2)]:
            row=copy.deepcopy(self.row)
            row['vision'][field][0]=value
            with self.assertRaises(ValueError):
                check_row(row,'p',self.items,{})

    def test_overlap_routes_deduplicated(self):
        self.row['multimodal']=copy.deepcopy(self.row['vision'])
        self.assertEqual([n['item_id'] for n in choose_negatives(self.row,4)], ['v0','v1','v2','v3'])

    def test_fixed_query_comparison_with_pilot_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            base, trained = [Path(directory)/n for n in ('base','trained')]
            config = {k: 'same' for k in ('photo_info_sha256','label_sha256',
                      'query_instruction','text_budgets','frame_preprocessing')}
            config['item_index_configs'] = [dict(item_ids_sha256='pool', item_count=101,
                item_info_sha256='metadata', item_input='title+image', image_size=512)]
            for root in (base, trained):
                root.mkdir()
                (root/'config.json').write_text(json.dumps(config))
                (root/'selected_photo_ids.json').write_text(json.dumps(['q0','q1']))
                for mode in ('vision','multimodal'):
                    (root/'results'/mode).mkdir(parents=True)
                    for q in ('q0','q1'):
                        hit = q=='q0' or root==trained
                        candidates = ['p']+[str(i) for i in range(99)] if hit else [str(i) for i in range(100)]
                        (root/'results'/mode/(q+'.json')).write_text(json.dumps(
                            dict(photo_id=q,positive_item_id='p',top_item_ids=candidates)))
            result=compare(base,trained,expected=2,baseline_recall=.5)
            self.assertEqual(result['metrics']['union@100']['net'],1)
            self.assertFalse(result['expansion_gate'])  # numerical gain alone is insufficient
            with self.assertRaises(ValueError):
                compare(base,trained,expected=2)  # wrong frozen reference
            (trained/'selected_photo_ids.json').write_text(json.dumps(['q1','q0']))
            with self.assertRaises(ValueError):
                compare(base,trained,expected=2,baseline_recall=.5)

    def test_paired_significance(self):
        result=paired([False]*10+[True]*2,[True]*10+[False]*2)
        self.assertEqual((result['gained'],result['lost'],result['net']),(10,2,8))
        self.assertAlmostEqual(result['paired_p_two_sided'],0.03857421875)
        self.assertEqual(paired([True],[True])['paired_p_two_sided'],1)


if __name__=='__main__':
    unittest.main()
