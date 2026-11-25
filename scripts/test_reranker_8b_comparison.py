import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from download_qwen_reranker_8b import model_files,validate_download,MODEL_ID,REVISION,RECEIPT,REQUIRED
from run_reranker_8b_comparison import check_protocols,metrics,command,report
from rerank_pipeline import digest,save,write_rows


class Reranker8BTest(unittest.TestCase):
    def test_pinned_model_completeness_and_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for name in REQUIRED:
                (folder/name).write_text('{}')
            save(dict(model_type='qwen3_vl',text_config=dict(hidden_size=4096,num_hidden_layers=36)),folder/'config.json')
            shards=[f'model-{i:05d}-of-00004.safetensors' for i in range(1,5)]
            save(dict(weight_map={f'weight{i}':s for i,s in enumerate(shards)}),folder/'model.safetensors.index.json')
            for name in shards:
                (folder/name).write_bytes(b'fixture, not real weights')
            save(dict(model_id=MODEL_ID,revision=REVISION,files={p.name:digest(p) for p in model_files(folder)}),folder/RECEIPT)
            self.assertEqual(validate_download(folder)['revision'],REVISION)
            (folder/'tokenizer.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'download receipt'):
                validate_download(folder)
            (folder/shards[0]).unlink()
            with self.assertRaisesRegex(ValueError,'weight shards'):
                model_files(folder)

    def test_only_intended_model_fields_may_change(self):
        base=dict(model='2B',model_config_sha256='2',model_weights_sha256={'2':'sha'},batch_size=4,photo_sha256='same')
        new={**base,'model':'8B','model_config_sha256':'8','model_weights_sha256':{'8':'sha'}}
        check_protocols(base,new)
        for field,value in [('photo_sha256','other'),('batch_size',2)]:
            with self.assertRaisesRegex(ValueError,'protocol differs'):
                check_protocols(base,{**new,field:value})

    def test_full_denominator_and_candidate_conditional_accuracy(self):
        raw={'1':dict(ranking=[dict(item_id=i) for i in ['a','p']]),
             '2':dict(ranking=[dict(item_id=i) for i in ['p','a']]),
             '3':dict(ranking=[dict(item_id=i) for i in ['a','b']])}
        new={'1':dict(ranking=[dict(item_id=i) for i in ['p','a']]),
             '2':dict(ranking=[dict(item_id=i) for i in ['a','p']]),'3':raw['3']}
        candidates={q:dict(candidates=row['ranking']) for q,row in raw.items()}
        result=metrics(list(raw),raw,new,dict.fromkeys(raw,'p'),candidates)
        self.assertAlmostEqual(result['candidate_recall'],2/3)
        self.assertAlmostEqual(result['top1']['trained'],1/3)
        self.assertEqual(result['challenger_accuracy_given_candidate_hit'],.5)
        self.assertEqual((result['top1']['gained'],result['top1']['lost']),(1,1))

    def test_worker_command_keeps_input_protocol(self):
        plan=dict(root='/root/project',data_root='/same/raw',model='/models/pinned8b')
        cmd=command(plan,Path('/candidates'),Path('/output'),100)
        self.assertEqual(cmd[cmd.index('--model')+1],plan['model'])
        self.assertEqual(cmd[cmd.index('--data-root')+1],plan['data_root'])
        self.assertEqual(cmd[cmd.index('--batch-size')+1],'4')
        self.assertEqual(cmd[cmd.index('--num-queries')+1],'100')

    def test_completed_report_compares_models_and_rejects_changed_query_text(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';out=root/'out'
            labels=root/'artifacts/train_v0/clean_labels.jsonl'
            ids=['1','2','3','4']
            write_rows([dict(photo_id=q,item_id='p') for q in ids],labels)
            save(dict(calibration=ids[:2],test=ids[2:]),out/'split.json')
            candidates=[dict(photo_id=q,candidates=[dict(item_id=i) for i in ['a','b','c','p']]) for q in ids]
            configs={}
            for name,directory,correct in [('raw',source,{'1','2'}),('base8b',out,{'1','3','4'})]:
                file=directory/'candidates'/(name+'.jsonl');write_rows(candidates,file)
                shard=directory/'shards'/name/'worker_0.jsonl';write_rows(candidates,shard)
                configs[name]=dict(candidate_sha256=digest(shard),model=name,model_config_sha256=name,
                    model_weights_sha256={name+'.safetensors':'hash'},photo_sha256='same',batch_size=4)
                save(configs[name],directory/name/'worker_0/config.json')
                rows=[]
                for q in ids:
                    order=['p','a','b','c'] if q in correct else ['a','b','c','p']
                    rows.append(dict(photo_id=q,ranking=[dict(item_id=i,score=1-j*.1) for j,i in enumerate(order)],
                        seconds=1,peak_memory_gib=18,query_text='same',frame_paths=['frame'],original_frames=1,encoded_frames=2))
                target=(directory/'raw/predictions_2000.jsonl') if name=='raw' else (directory/'base8b/worker_0/predictions_4.jsonl')
                write_rows(rows,target)
            plan=dict(root=str(root),source=str(source),workers=1,count=4,model_files={'base8b.safetensors':'hash'},
                      objective='test',confounding='size and training differ')
            save(plan,out/'plan.json')
            with contextlib.redirect_stdout(io.StringIO()):
                report(out,plan)
            result=json.loads((out/'comparison.json').read_text())
            self.assertEqual(result['metrics']['all_2000']['top1']['net'],1)
            self.assertFalse(result['automatic_promotion'])
            target=out/'base8b/worker_0/predictions_4.jsonl'
            rows=[json.loads(s) for s in target.read_text().splitlines()];rows[0]['query_text']='different'
            write_rows(rows,target)
            with self.assertRaisesRegex(ValueError,'query text/frame input differs'):
                report(out,plan)


if __name__=='__main__':
    unittest.main()
