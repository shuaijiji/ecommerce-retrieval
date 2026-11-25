import json
from pathlib import Path
import tempfile
import unittest
from reranker_text_dedup import dedup_segments,transform_photo
from run_reranker_text_ablation import freeze_candidates,summarize,validate_plan
from prepare_reranker_sft import compose_video_text
from rerank_pipeline import digest,write_rows,save


class TextAblationTest(unittest.TestCase):
    def test_exact_only_preserves_models_numbers_and_other_fields(self):
        value='型号 A1 100g|型号 A1 100g|型号 A1 500g|型号 a1 100g'
        self.assertEqual(dedup_segments(value),'型号 A1 100g|型号 A1 500g|型号 a1 100g')
        original=dict(photo_id='123',photo_caption='A|A',photo_frames=['frame.jpg'],
                      photo_ocr=value,photo_asr='保留\n保留\n末尾品牌')
        changed=transform_photo(original)
        self.assertEqual(changed['photo_caption'],original['photo_caption'])
        self.assertEqual(changed['photo_frames'],original['photo_frames'])
        self.assertEqual(changed['photo_asr'],'保留|末尾品牌')
        self.assertEqual(original['photo_ocr'],value)
        self.assertEqual(dedup_segments('数字100 100 无分隔重复重复'),'数字100 100 无分隔重复重复')
        self.assertEqual(dedup_segments('A|B'),'A|B')
        self.assertEqual(dedup_segments(None),None)

    def test_dedup_before_budget_restores_tail_without_removing_asr(self):
        photo=dict(photo_ocr=('直播优惠|'*200)+'汤尼猫1.8L',photo_asr='原始ASR 品牌2.0L')
        self.assertNotIn('汤尼猫',compose_video_text(photo))
        rendered=compose_video_text(transform_photo(photo))
        self.assertIn('汤尼猫1.8L',rendered)
        self.assertIn('原始ASR 品牌2.0L',rendered)

    def test_frozen_routing_uses_only_old_scores_and_preserves_order(self):
        top10={q:dict(candidates=[dict(item_id='a'),dict(item_id='b')]) for q in ('1','2')}
        top25={q:dict(candidates=[dict(item_id='c'),dict(item_id='a'),dict(item_id='b')]) for q in ('1','2')}
        examples={q:dict(top1_score=score,top10_ranking=['a','b'],incremental_ranking=['a','c','b'])
                  for q,score in [('1',.9),('2',.7)]}
        rows,decisions=freeze_candidates(top10,top25,examples,dict(cutoff=.8,tie_probability=0,tie_seed=1))
        self.assertEqual([c['item_id'] for c in rows[0]['candidates']],['a','b'])
        self.assertEqual([c['item_id'] for c in rows[1]['candidates']],['a','b','c'])
        self.assertEqual([r['expanded'] for r in decisions],[False,True])

    def test_report_paired_raw_vs_dedup_and_text_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'experiment'
            labels=root/'artifacts/train_v0/clean_labels.jsonl'
            write_rows([dict(photo_id=str(q),item_id='p') for q in range(4)],labels)
            ids=[str(q) for q in range(4)]
            save(dict(calibration=ids[:2],test=ids[2:]),out/'split.json')
            write_rows([dict(photo_id=q,raw='raw text',dedup='dedup text') for q in ids],out/'text_preview.jsonl')
            write_rows([dict(photo_id=q,predicted_top3=['p','a','b'],bucket='correct_top1') for q in ids],out/'cached_baseline_cases.jsonl')
            rows=[dict(photo_id=q,candidates=[dict(item_id=i) for i in ['p','a','b','c']]) for q in ids]
            for mode,correct in [('raw',{'0','1'}),('dedup',{'0','2','3'})]:
                write_rows(rows,out/'candidates'/(mode+'.jsonl'))
                shard=out/'shards'/mode/'worker_0.jsonl';write_rows(rows,shard)
                metadata=out/'data'/mode/'photo_info.jsonl';write_rows([dict(photo_id=q,photo_ocr=mode) for q in ids],metadata)
                save(dict(candidate_sha256=digest(shard),model_weights_sha256={'fake':'fixed'},
                          photo_sha256=digest(metadata)),out/mode/'worker_0/config.json')
                predictions=[]
                for q in ids:
                    order=['p','a','b','c'] if q in correct else ['a','b','c','p']
                    predictions.append(dict(photo_id=q,ranking=[dict(item_id=i,score=1-j*.1) for j,i in enumerate(order)],
                                            seconds=1,query_text=mode+' text'))
                write_rows(predictions,out/mode/'worker_0/predictions_4.jsonl')
            plan=dict(root=str(root),workers=1,count=4,objective='test',text_changed=4)
            save(plan,out/'plan.json')
            summarize(out,plan)
            report=json.loads((out/'comparison.json').read_text())
            paired=report['metrics']['all_2000']['top1']
            self.assertEqual((paired['gained'],paired['lost'],paired['net']),(2,1,1))
            self.assertFalse(report['automatic_promotion'])
            # Resume results with stale query text must not pass.
            path=out/'dedup/worker_0/predictions_4.jsonl'
            values=[json.loads(s) for s in path.read_text().splitlines()]
            values[0]['query_text']='stale'
            write_rows(values,path)
            with self.assertRaisesRegex(ValueError,'query text'):
                summarize(out,plan)

    def test_frozen_plan_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);source=out/'input';source.write_text('before')
            save(dict(output=str(out),inputs={str(source):digest(source)},generated={},data_links={},
                      model=str(out/'missing'),weights={}),out/'plan.json')
            source.write_text('after')
            with self.assertRaisesRegex(ValueError,'frozen experiment input changed'):
                validate_plan(out)


if __name__=='__main__':
    unittest.main()
