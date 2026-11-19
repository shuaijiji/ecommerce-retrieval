import copy
import unittest
from prepare_embedding_pilot import check_row, choose_negatives
from compare_embedding_recall import paired


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

    def test_paired_significance(self):
        result=paired([False]*10+[True]*2,[True]*10+[False]*2)
        self.assertEqual((result['gained'],result['lost'],result['net']),(10,2,8))
        self.assertAlmostEqual(result['paired_p_two_sided'],0.03857421875)
        self.assertEqual(paired([True],[True])['paired_p_two_sided'],1)


if __name__=='__main__':
    unittest.main()
