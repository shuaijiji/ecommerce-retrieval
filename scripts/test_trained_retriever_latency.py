import json
from pathlib import Path
import tempfile
import unittest
from rerank_pipeline import digest
from rerank_multimodal import INSTRUCTION
from run_trained_retriever_latency_pilot import load_fixed_route


class LatencyRouteProvenanceTest(unittest.TestCase):
    def fixture(self,root):
        model=root/'model'
        model.mkdir()
        (model/'model.safetensors').write_bytes(b'fake test weights')
        (model/'config.json').write_text('{}')
        data=root/'data'
        data.mkdir()
        for name in ('photo_info.jsonl','item_info.jsonl'):
            (data/name).write_text('{}\n')
        c10,c25=root/'10.jsonl',root/'25.jsonl'
        c10.write_text('{}\n')
        c25.write_text('{}\n')
        threshold=dict(cutoff=.87109375,tie_probability=.5,tie_seed=20261119)
        weights={'model.safetensors':digest(model/'model.safetensors')}
        cfg=root/'offline/experiment_config.json'
        cfg.parent.mkdir()
        cfg.write_text(json.dumps(dict(router_thresholds={'pilot100':threshold},model_weights=weights)))
        report=root/'offline/comparison.json'
        report.write_text(json.dumps(dict(primary_arm='pilot100',router_fixed_before_results=dict(policy='low_top1_score',budget=.25,seed=20261119),
            arms={'pilot100':dict(router={'threshold':threshold},test_top1_benefit_gate=True,
                                input_hashes={str(c10):digest(c10),str(c25):digest(c25)})})))
        worker=root/'offline/pilot100/worker_0/config.json'
        worker.parent.mkdir(parents=True)
        worker.write_text(json.dumps(dict(model_config_sha256=digest(model/'config.json'),max_length=10240,model_weights_sha256=weights,batch_size=4,image_size=512,instruction=INSTRUCTION,
            photo_sha256=digest(data/'photo_info.jsonl'),item_sha256=digest(data/'item_info.jsonl'),
            preprocessing_sha256=digest(Path(__file__).with_name('retrieval_common.py')))))
        return report,cfg,model,data,c10,c25

    def test_reads_frozen_threshold_and_rejects_changed_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d))
            threshold,_=load_fixed_route(*args)
            self.assertEqual(threshold['cutoff'],.87109375)
            args[-1].write_text('changed')
            with self.assertRaisesRegex(ValueError,'candidate input changed'):
                load_fixed_route(*args)

    def test_rejects_unregistered_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            args=self.fixture(Path(d))
            cfg=json.loads(args[1].read_text())
            cfg['router_thresholds']['pilot100']['cutoff']=.5
            args[1].write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'threshold mismatch'):
                load_fixed_route(*args)


if __name__=='__main__':
    unittest.main()
