import unittest
from evaluate_cached_rank_fusion import fuse,choose_weight


class CachedRankFusionTest(unittest.TestCase):
    def test_coverage_and_zero_weight_order(self):
        original=['b','c','a']
        routes=[['a','b','c','outside'],['a','c','b']]
        self.assertEqual(fuse(original,routes,0),original)
        after=fuse(original,routes,1)
        self.assertEqual(after[0],'a')
        self.assertEqual(set(after),set(original))
        with self.assertRaises(ValueError):
            fuse(original,routes,1.1)

    def test_missing_route_has_zero_contribution(self):
        self.assertEqual(fuse(['a','b'],[['b'],['b']],1),['b','a'])

    def test_weight_selection_uses_only_provided_calibration_ids(self):
        variants={0:{'cal':['x','p'],'test':['p','x']},
                  .5:{'cal':['p','x'],'test':['x','p']},
                  1:{'cal':['p','x'],'test':['p','x']}}
        labels={'cal':'p','test':'p'}
        self.assertEqual(choose_weight(['cal'],variants,labels),.5)
        labels['test']='x'
        self.assertEqual(choose_weight(['cal'],variants,labels),.5)


if __name__=='__main__':
    unittest.main()
