import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reranker_8b_lora_utils import identify_group,select_groups,training_command,analyze_history
from run_reranker_8b_lora_pilot import checkpoint_summary,prepare_eval,export
from rerank_pipeline import digest,save,write_rows


class Reranker8BLoraTest(unittest.TestCase):
    def fixture_group(self):
        items=['p']+[str(i) for i in range(7)]
        row=dict(videos=[['/media/photo_frames/123/00.png']],positive_messages=[['positive']],
                 negative_messages=[['negative'] for _ in range(7)],positive_images=[['/media/item_images/p.png']],
                 negative_images=[[f'/media/item_images/{i}.png'] for i in range(7)])
        canonical={'123':dict(positive_item_id='p',negatives=[dict(item_id=i) for i in items[1:]])}
        return row,canonical

    def test_group_alignment_duplicates_and_development_exclusion(self):
        row,canonical=self.fixture_group()
        self.assertEqual(identify_group(row,canonical,{},set()),'123')
        with self.assertRaisesRegex(ValueError,'known duplicate'):
            identify_group(row,canonical,{'p':{'p','0'}},set())
        with self.assertRaisesRegex(ValueError,'development positive'):
            identify_group(row,canonical,{}, {'1'})
        row['negative_images'][0],row['negative_images'][1]=row['negative_images'][1],row['negative_images'][0]
        with self.assertRaisesRegex(ValueError,'identity/order'):
            identify_group(row,canonical,{},set())

    def test_stable_query_selection_and_smoke_starts_from_base(self):
        self.assertEqual(select_groups(dict.fromkeys(['1','2','3']),2),select_groups(dict.fromkeys(['3','2','1']),2))
        plan=dict(output='/pilot',model='/official8b')
        for phase,steps in [('smoke','5'),('train','50')]:
            cmd=training_command(plan,phase)
            self.assertEqual(cmd[cmd.index('--model')+1],'/official8b')
            self.assertEqual(cmd[cmd.index('--model_type')+1],'qwen3_vl_reranker')
            self.assertEqual(cmd[cmd.index('--max_steps')+1],steps)
            self.assertEqual(cmd[cmd.index('--strict')+1],'true')
            self.assertNotIn('--adapters',cmd)
            self.assertNotIn('--resume_from_checkpoint',cmd)

    def state(self):
        return dict(global_step=50,log_history=[
            dict(step=0,eval_loss=1.2,eval_mrr=.7,eval_ndcg=.8),dict(step=1,loss=1.4,grad_norm=2),
            dict(step=50,loss=1.1,grad_norm=1),dict(step=50,eval_loss=1,eval_mrr=.72,eval_ndcg=.8)])

    def test_functional_and_direction_gates_are_separate(self):
        state=self.state();report=analyze_history(state,50)
        self.assertTrue(report['functional_gate']);self.assertTrue(report['direction_gate'])
        state['log_history'][-1]['eval_mrr']=.69
        self.assertTrue(analyze_history(state,50)['functional_gate'])
        self.assertFalse(analyze_history(state,50)['direction_gate'])
        state['log_history'][-1].pop('eval_mrr');state['log_history'][-1].pop('eval_ndcg')
        self.assertTrue(analyze_history(state,50)['missing_ranking_metrics'])
        state['global_step']=49
        self.assertFalse(analyze_history(state,50)['functional_gate'])
        state=self.state()
        for row in state['log_history']:
            if 'grad_norm' in row:
                row['grad_norm']=0
        self.assertFalse(analyze_history(state,50)['functional_gate'])

    def test_nonfinite_loss_or_gradient_is_rejected(self):
        for key,value in [('loss',float('nan')),('grad_norm',float('inf'))]:
            state=self.state();state['log_history'][1][key]=value
            with self.assertRaisesRegex(ValueError,'non-finite'):
                analyze_history(state,50)

    def test_checkpoint_summary_binds_adapter_and_gpu_activity(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);checkpoint=out/'train/model/v0/checkpoint-50';checkpoint.mkdir(parents=True)
            save(self.state(),checkpoint/'trainer_state.json')
            save(dict(r=8,lora_alpha=32,base_model_name_or_path=str(out/'base')),checkpoint/'adapter_config.json')
            (checkpoint/'adapter_model.safetensors').write_bytes(b'test adapter')
            (out/'train/gpu_metrics.csv').write_text('timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]\nnow,1,99,20000,500\n')
            plan=dict(model=str(out/'base'),workers=1);save(plan,out/'plan.json')
            result=checkpoint_summary(out,plan,'train')
            self.assertTrue(result['functional_gate']);self.assertTrue(result['direction_gate'])
            self.assertEqual(result['nominal_train_query_exposures'],50)
            (out/'train/gpu_metrics.csv').write_text('timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]\nnow,1,0,10,100\n')
            self.assertFalse(checkpoint_summary(out,plan,'train')['functional_gate'])

    def test_export_restores_original_tokenizer_assets(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);base=out/'base';base.mkdir()
            config=dict(model_type='qwen3_vl',image_token_id=1,video_token_id=2,
                        text_config=dict(hidden_size=4096,num_hidden_layers=36,vocab_size=10))
            save(config,base/'config.json');(base/'tokenizer.json').write_text('original tokenizer')
            plan=dict(root=str(out),model=str(base),model_files={'config.json':digest(base/'config.json'),
                      'tokenizer.json':digest(base/'tokenizer.json')})
            save(plan,out/'plan.json')
            summary=dict(functional_gate=True,direction_gate=True,checkpoint='/adapter',adapter_sha256='hash')
            def fake_export(*args,**kwargs):
                target=out/'merged';target.mkdir()
                save(config,target/'config.json');(target/'model.safetensors').write_bytes(b'merged fixture')
                (target/'tokenizer.json').write_text('modified by exporter')
                self.assertEqual(kwargs['env']['NPROC_PER_NODE'],'1')
            with patch('run_reranker_8b_lora_pilot.checkpoint_summary',return_value=summary),\
                 patch('run_reranker_8b_lora_pilot.gpu_uuids',return_value=['uuid']),\
                 patch('run_reranker_8b_lora_pilot.subprocess.run',side_effect=fake_export),\
                 contextlib.redirect_stdout(io.StringIO()):
                export(out,plan,['1'])
            self.assertEqual((out/'merged/tokenizer.json').read_text(),'original tokenizer')
            self.assertIn('tokenizer.json',json.loads((out/'export_receipt.json').read_text())['preprocessing_assets_restored_from_base'])

    def test_eval_plan_replaces_model_without_mutating_completed_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'pilot';source=root/'base_comparison';checkpoint=out/'checkpoint-50'
            checkpoint.mkdir(parents=True);(checkpoint/'adapter_model.safetensors').write_bytes(b'adapter')
            merged=out/'merged';merged.mkdir();(merged/'model.safetensors').write_bytes(b'merged')
            save({},out/'plan.json')
            summary=dict(functional_gate=True,direction_gate=True,checkpoint=str(checkpoint),adapter_sha256=digest(checkpoint/'adapter_model.safetensors'))
            save(summary,out/'train/summary.json')
            save(dict(plan_sha256=digest(out/'plan.json'),adapter_sha256=summary['adapter_sha256'],
                      model_files={'model.safetensors':digest(merged/'model.safetensors')}),out/'export_receipt.json')
            rows=[dict(photo_id=str(i),candidates=[dict(item_id='p')]) for i in range(4)]
            write_rows(rows,source/'candidates/base8b.jsonl');write_rows(rows,source/'base8b/predictions_2000.jsonl')
            save(dict(calibration=['0','1'],test=['2','3']),source/'split.json')
            save({},source/'plan.json');save({},source/'comparison.json')
            original=dict(model='base8b',inputs={},generated={},workers=1)
            original_copy=copy.deepcopy(original)
            plan=dict(source_comparison=str(source))
            with patch('run_reranker_8b_lora_pilot.validate_comparison',return_value=original),contextlib.redirect_stdout(io.StringIO()):
                prepare_eval(out,plan,2)
            result=json.loads((out/'eval/plan.json').read_text())
            self.assertEqual(result['model'],str(merged));self.assertEqual(result['workers'],2)
            self.assertEqual(original,original_copy)
            self.assertIn(str(out/'export_receipt.json'),result['inputs'])


if __name__=='__main__':
    unittest.main()
