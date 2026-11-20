import unittest
from run_embedding_index_8gpu import summarize


class EightGpuPerformanceTest(unittest.TestCase):
    def test_uses_overall_encoding_window_and_per_worker_utilization(self):
        records=[dict(encoded_items=100,start_unix=10,end_unix=20),
                 dict(encoded_items=100,start_unix=12,end_unix=22)]
        samples=[dict(timestamp=5,a=0,b=0),dict(timestamp=15,a=80,b=90),
                 dict(timestamp=21,a=0,b=70),dict(timestamp=30,a=0,b=0)]
        r=summarize(records,samples,['a','b'])
        self.assertAlmostEqual(r['items_per_second'],200/12)
        self.assertEqual(r['gpu_utilization_medians'],dict(a=80,b=80))

    def test_rejects_missing_samples_and_incomplete_workers(self):
        records=[dict(encoded_items=100,start_unix=10,end_unix=20)]
        with self.assertRaises(ValueError):
            summarize(records,[],['a'])
        with self.assertRaises(ValueError):
            summarize(records,[],['a','b'])


if __name__=='__main__':
    unittest.main()
