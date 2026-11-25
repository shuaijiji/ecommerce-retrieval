#!/usr/bin/env python3
"""Audited 400/100-query 8B LoRA Pilot, export and fixed-candidate evaluation."""
import argparse
import copy
import csv
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from compare_retriever_rerank import compare_cases
from download_qwen_reranker_8b import FOLDER,validate_download
from prepare_reranker_sft import load_duplicate_groups
from rerank_pipeline import digest,load_labels,load_rows,save,write_rows
from reranker_8b_lora_utils import SEED,STEPS,identify_group,select_groups,training_command,analyze_history
from run_reranker_8b_comparison import gpu_uuids,validate as validate_comparison,run as run_comparison,metrics,check_protocols


def runtime():
    import torch,transformers,swift,flash_attn,peft
    from swift.loss import loss_map
    if swift.__version__.split('+')[0]!='4.6.1':
        raise ValueError('use the established ms-swift 4.6.1 training environment; no automatic environment upgrade')
    loss=loss_map.get('listwise_reranker')
    if loss is None:
        raise ValueError('listwise_reranker is not registered in this training environment')
    module=inspect.getmodule(loss)
    if module is None or not getattr(module,'__file__',None):
        raise ValueError('cannot locate registered listwise loss source for provenance')
    path=Path(module.__file__)
    return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        transformers=transformers.__version__,swift=swift.__version__,flash_attn=flash_attn.__version__,
        peft=peft.__version__,loss_module=str(path),loss_module_sha256=digest(path))


def prepare(root,out,workers,port):
    if out.exists():
        raise ValueError('prepare requires a new output directory')
    print('Checking training runtime, completed 8B comparison and frozen source data...',flush=True)
    environment=runtime()
    base=root/'artifacts/train_v0';source=base/'reranker_8b_comparison_v1'
    source_plan=validate_comparison(source)
    if json.loads((source/'comparison.json').read_text())['plan_sha256']!=digest(source/'plan.json'):
        raise ValueError('8B comparison does not match its plan')
    model=root/'models'/FOLDER;receipt=validate_download(model)
    data=base/'reranker_sft_smoke_40k'
    manifest=json.loads((data/'manifest.json').read_text())
    if (manifest.get('complete') is not True or manifest.get('effective_pointwise_pairs')!=40000 or
        manifest.get('rows',{}).get('swift_train')!=4500 or manifest.get('rows',{}).get('swift_smoke_eval')!=500):
        raise ValueError('audited 4500/500 training dataset not ready')
    for name,expected in manifest['outputs'].items():
        if Path(name).name!=name or digest(data/name)!=expected:
            raise ValueError('audited dataset output changed')
    config=json.loads((data/'config.json').read_text())
    group_file=base/'proposed_item_duplicate_groups.jsonl'
    if digest(group_file)!=config['source_hashes']['duplicate_groups']:
        raise ValueError('known duplicate groups changed')
    groups=load_duplicate_groups(group_file)
    canonical={r['photo_id']:r for r in load_rows(data/'samples.jsonl')}
    _,labels=load_labels(base/'clean_labels.jsonl')
    for q,row in canonical.items():
        if labels.get(q)!=row['positive_item_id']:
            raise ValueError('canonical training positive differs from clean label')
    dev_ids={r['photo_id'] for r in load_rows(source/'candidates/base8b.jsonl')}
    forbidden=set()
    for q in dev_ids:
        item=labels[q];forbidden.add(item);forbidden.update(groups.get(item,{item}))
    def index_rows(path):
        indexed={}
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        for row in rows:
            q=identify_group(row,canonical,groups,forbidden)
            if q in indexed or q in dev_ids:
                raise ValueError('duplicate training query or development query leakage')
            indexed[q]=row
        return indexed
    train=index_rows(data/'swift_train.jsonl');monitor=index_rows(data/'swift_smoke_eval.jsonl')
    if len(train)!=4500 or len(monitor)!=500 or set(train)&set(monitor) or set(train)|set(monitor)!=set(canonical):
        raise ValueError('training/monitor coverage changed')
    train_ids=select_groups(train,400);monitor_ids=select_groups(monitor,100)
    print('Validating media paths for selected 400 training and 100 monitor queries...',flush=True)
    media_root=(data/'media_512_png').resolve()
    for row in [train[q] for q in train_ids]+[monitor[q] for q in monitor_ids]:
        paths=row['videos'][0]+[p for field in ('positive_images','negative_images') for images in row[field] for p in images]
        for name in paths:
            path=Path(name).resolve();path.relative_to(media_root)
            if not path.is_file():
                raise ValueError(f'missing normalized training media: {path}')
    out.mkdir(parents=True)
    for name,ids,rows in [('train',train_ids,train),('monitor',monitor_ids,monitor),
                          ('smoke_train',train_ids[:100],train),('smoke_monitor',monitor_ids[:20],monitor)]:
        write_rows([dict(rows[q],photo_id=q) for q in ids],out/'data'/(name+'.with_ids.jsonl'))
        # Swift receives the original audited schema; IDs are saved separately for inspection.
        path=out/'data'/(name+'.jsonl')
        path.write_text(''.join(json.dumps(rows[q],ensure_ascii=False)+'\n' for q in ids))
    inputs={str(data/name):digest(data/name) for name in ['manifest.json',*manifest['outputs']]}
    for path in (group_file,base/'clean_labels.jsonl',source/'plan.json',source/'comparison.json',
                 source/'candidates/base8b.jsonl',source/'base8b/predictions_2000.jsonl',model/'download_receipt.json'):
        inputs[str(path)]=digest(path)
    for name in ('run_reranker_8b_lora_pilot.py','reranker_8b_lora_utils.py','run_reranker_8b_comparison.py',
                 'download_qwen_reranker_8b.py','rerank_pipeline.py','prepare_reranker_sft.py'):
        path=root/'scripts'/name;inputs[str(path)]=digest(path)
    plan=dict(root=str(root),output=str(out),source_comparison=str(source),workers=workers,master_port=port,
        model=str(model),model_files=receipt['files'],runtime=environment,inputs=inputs,
        generated={str(p):digest(p) for p in (out/'data').glob('*.jsonl')},train_queries=400,monitor_queries=100,
        seed=SEED,steps=STEPS,query_group_size=8,global_batch=workers,nominal_train_query_exposures=STEPS*workers,
        train_query_ids=train_ids,monitor_query_ids=monitor_ids,source_model_revision=source_plan['revision'],
        development_positive_and_known_duplicate_exclusion=True,
        monitor_warning='query holdout within training side; not item-disjoint generalization evidence',
        automatic_expansion=False,official_test_used=False)
    save(plan,out/'plan.json')
    print(f'Prepared 400 train + 100 monitor; {workers} workers, 50 optimizer steps. No GPU training.',flush=True)


def validate(out,check_runtime=False):
    plan=json.loads((out/'plan.json').read_text())
    if plan['output']!=str(out):
        raise ValueError('output differs from plan')
    for name,expected in {**plan['inputs'],**plan['generated']}.items():
        if digest(name)!=expected:
            raise ValueError(f'frozen input changed: {name}')
    folder=Path(plan['model'])
    if {name:digest(folder/name) for name in plan['model_files']}!=plan['model_files']:
        raise ValueError('official 8B weights/tokenizer changed')
    if check_runtime and runtime()!=plan['runtime']:
        raise ValueError('training runtime differs from prepare')
    return plan


def checkpoint_summary(out,plan,phase):
    steps=5 if phase=='smoke' else STEPS
    matches=list((out/phase/'model').rglob(f'checkpoint-{steps}/trainer_state.json'))
    if len(matches)!=1:
        raise ValueError(f'expected one final checkpoint-{steps} trainer state, found {len(matches)}')
    state=json.loads(matches[0].read_text());checkpoint=matches[0].parent
    config=json.loads((checkpoint/'adapter_config.json').read_text())
    if (config.get('r')!=8 or config.get('lora_alpha')!=32 or
            Path(config['base_model_name_or_path']).resolve()!=Path(plan['model']).resolve()):
        raise ValueError('adapter differs from frozen base/LoRA configuration')
    if not (checkpoint/'adapter_model.safetensors').is_file():
        raise ValueError('final adapter weights missing')
    summary=analyze_history(state,steps)
    by_gpu={}
    with (out/phase/'gpu_metrics.csv').open() as handle:
        for row in csv.DictReader(handle):
            index=row['index'].strip();values=by_gpu.setdefault(index,dict(memory_peak_mib=0,util_peak_percent=0))
            values['memory_peak_mib']=max(values['memory_peak_mib'],float(row['memory.used [MiB]'].strip()))
            values['util_peak_percent']=max(values['util_peak_percent'],float(row['utilization.gpu [%]'].strip()))
    expected=1 if phase=='smoke' else plan['workers']
    active=(len(by_gpu)==expected and all(r['memory_peak_mib']>1024 and r['util_peak_percent']>0 for r in by_gpu.values()))
    summary.update(checkpoint=str(checkpoint),adapter_sha256=digest(checkpoint/'adapter_model.safetensors'),
        plan_sha256=digest(out/'plan.json'),gpu=by_gpu,gpu_activity_verified=active,
        nominal_train_query_exposures=steps*expected,monitor_is_generalization_set=False)
    summary['functional_gate']=summary['functional_gate'] and active
    save(summary,out/phase/'summary.json')
    return summary


def launch_phase(out,plan,phase,gpus):
    destination=out/phase
    if (destination/'model').exists():
        if not (destination/'summary.json').exists():
            raise ValueError(f'partial training exists under {destination}; preserve logs and use a new Pilot directory')
        summary=checkpoint_summary(out,plan,phase)
        if not summary['functional_gate']:
            raise ValueError('previous phase did not pass functional gate')
        return summary
    destination.mkdir(exist_ok=True)
    uuids=gpu_uuids(gpus)
    env=os.environ.copy()
    for key in ('WORLD_SIZE','RANK','LOCAL_RANK','LOCAL_WORLD_SIZE'):
        env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES=','.join(uuids),NPROC_PER_NODE=str(len(gpus)),MASTER_PORT=str(plan['master_port']),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',MAX_POSITIVE_SAMPLES='1',MAX_NEGATIVE_SAMPLES='7',
        LISTWISE_RERANKER_TEMPERATURE='1.0',LISTWISE_RERANKER_MIN_GROUP_SIZE='2',IMAGE_MAX_TOKEN_NUM='1024',
        VIDEO_MAX_TOKEN_NUM='128',FPS_MAX_FRAMES='16',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    if shutil.which('swift') is None:
        raise ValueError('swift CLI not found in training environment')
    with (destination/'gpu_metrics.csv').open('w') as gpu_log, (destination/'train.log').open('w') as log:
        gpu_log.write('timestamp,index,utilization.gpu [%],memory.used [MiB],power.draw [W]\n');gpu_log.flush()
        monitor=subprocess.Popen(['nvidia-smi','--id='+','.join(gpus),
            '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','--loop=1'],
            stdout=gpu_log,stderr=subprocess.DEVNULL)
        proc=None
        try:
            print(f'{phase}: {5 if phase=="smoke" else 50} steps on GPUs {gpus}; log {destination/"train.log"}',flush=True)
            proc=subprocess.Popen(training_command(plan,phase),cwd=plan['root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=proc.wait()
            if code:
                raise RuntimeError(f'{phase} training exited {code}; inspect {destination/"train.log"}')
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait()
            monitor.terminate();monitor.wait()
    return checkpoint_summary(out,plan,phase)


def train(out,plan,gpus):
    gpu_uuids(gpus)
    smoke=launch_phase(out,plan,'smoke',gpus[:1])
    if not smoke['functional_gate']:
        raise ValueError('single-GPU smoke functional gate failed; no multi-GPU training launched')
    # Main Pilot starts again from the official base; smoke adapter is not continued.
    summary=launch_phase(out,plan,'train',gpus)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if not summary['functional_gate']:
        raise ValueError('50-step functional gate failed')
    print('Training direction gate: '+str(summary['direction_gate'])+'; no automatic expansion.',flush=True)


def export(out,plan,gpus):
    summary=checkpoint_summary(out,plan,'train')
    if not summary['functional_gate'] or not summary['direction_gate']:
        raise ValueError('Pilot functional and monitor direction gates are required before export/evaluation')
    target=out/'merged';receipt=out/'export_receipt.json'
    if target.exists():
        if not receipt.exists():
            raise ValueError('partial merge exists; preserve it and inspect export.log')
        saved=json.loads(receipt.read_text())
        if saved['adapter_sha256']!=summary['adapter_sha256'] or any(digest(target/name)!=sha for name,sha in saved['model_files'].items()):
            raise ValueError('export or adapter changed')
        print('Verified existing merged Pilot model',flush=True);return
    uuid=gpu_uuids(gpus[:1])[0]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'NPROC_PER_NODE':'1'}
    for key in ('WORLD_SIZE','RANK','LOCAL_RANK','LOCAL_WORLD_SIZE'):
        env.pop(key,None)
    with (out/'export.log').open('w') as log:
        subprocess.run(['swift','export','--adapters',summary['checkpoint'],'--merge_lora','true','--output_dir',str(target)],
                       env=env,cwd=plan['root'],stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (target/'config.json').is_file() or not list(target.glob('*.safetensors')):
        raise ValueError('merged Pilot model incomplete')
    if (target/'adapter_config.json').exists() or (target/'adapter_model.safetensors').exists():
        raise ValueError('export still contains adapter files; expected a standalone merged model')
    original_config=json.loads((Path(plan['model'])/'config.json').read_text())
    merged_config=json.loads((target/'config.json').read_text())
    for key in ('model_type','image_token_id','video_token_id'):
        if original_config.get(key)!=merged_config.get(key):
            raise ValueError('merged model architecture or multimodal token IDs changed')
    for key in ('hidden_size','num_hidden_layers','vocab_size'):
        if original_config['text_config'][key]!=merged_config['text_config'][key]:
            raise ValueError('merged model text architecture/vocabulary changed')
    restored=[]
    for name in plan['model_files']:
        if name not in ('config.json','model.safetensors.index.json') and not name.endswith('.safetensors'):
            shutil.copy2(Path(plan['model'])/name,target/name);restored.append(name)
    files=sorted(p for p in target.iterdir() if p.is_file() and p.suffix in ('.json','.txt','.jinja','.safetensors'))
    save(dict(adapter_sha256=summary['adapter_sha256'],plan_sha256=digest(out/'plan.json'),
              preprocessing_assets_restored_from_base=restored,model_files={p.name:digest(p) for p in files}),receipt)
    print(f'Exported fixed step-50 checkpoint: {target}',flush=True)


def prepare_eval(out,plan,workers):
    export_receipt=json.loads((out/'export_receipt.json').read_text())
    summary=json.loads((out/'train/summary.json').read_text())
    if (not summary['functional_gate'] or not summary['direction_gate'] or
            summary['adapter_sha256']!=export_receipt['adapter_sha256'] or
            digest(Path(summary['checkpoint'])/'adapter_model.safetensors')!=export_receipt['adapter_sha256']):
        raise ValueError('export adapter differs from a passed fixed-step Pilot')
    if export_receipt['plan_sha256']!=digest(out/'plan.json'):
        raise ValueError('export differs from Pilot plan')
    merged=out/'merged'
    if any(digest(merged/name)!=sha for name,sha in export_receipt['model_files'].items()):
        raise ValueError('merged model files changed')
    source=Path(plan['source_comparison']);original=validate_comparison(source)
    destination=out/'eval'
    if destination.exists():
        raise ValueError('evaluation directory already exists')
    destination.mkdir()
    rows=load_rows(source/'candidates/base8b.jsonl');write_rows(rows,destination/'candidates/base8b.jsonl')
    for worker in range(workers):
        write_rows(rows[worker::workers],destination/'shards/base8b'/f'worker_{worker}.jsonl')
    save(json.loads((source/'split.json').read_text()),destination/'split.json')
    eval_plan=copy.deepcopy(original)
    eval_plan.update(output=str(destination),workers=workers,model=str(merged),model_files=export_receipt['model_files'],
        model_id='Qwen3-VL-Reranker-8B + fixed 50-step project LoRA Pilot',
        objective='50-step domain LoRA 8B vs completed fine-tuned 2B on fixed candidates',
        confounding='2B and 8B domain training exposure differs; clean adaptation comparison is base8B vs LoRA8B')
    for path in (source/'plan.json',source/'comparison.json',source/'base8b/predictions_2000.jsonl',
                 out/'plan.json',out/'train/summary.json',out/'export_receipt.json',Path(__file__)):
        eval_plan['inputs'][str(path)]=digest(path)
    eval_plan['generated']={str(p):digest(p) for p in destination.rglob('*') if p.is_file()}
    save(eval_plan,destination/'plan.json')
    print(f'Prepared fixed-candidate Pilot evaluation: {destination}',flush=True)


def adaptation_report(out,plan):
    source=Path(plan['source_comparison']);destination=out/'eval'
    _,labels=load_labels(Path(plan['root'])/'artifacts/train_v0/clean_labels.jsonl')
    new=compare_cases([('base8b',destination/'candidates/base8b.jsonl',destination/'base8b/predictions_2000.jsonl',
                        sorted((destination/'base8b').glob('worker_*/config.json')))],labels)
    old=compare_cases([('base8b',source/'candidates/base8b.jsonl',source/'base8b/predictions_2000.jsonl',
                        sorted((source/'base8b').glob('worker_*/config.json')))],labels)
    check_protocols(old['reranker_protocol'],new['reranker_protocol'])
    before={r['photo_id']:r for r in load_rows(source/'base8b/predictions_2000.jsonl')}
    after={r['photo_id']:r for r in load_rows(destination/'base8b/predictions_2000.jsonl')}
    candidates={r['photo_id']:r for r in load_rows(destination/'candidates/base8b.jsonl')}
    for q in before:
        if any(before[q][key]!=after[q][key] for key in ('query_text','frame_paths','original_frames','encoded_frames')):
            raise ValueError('base8B/Pilot8B query inputs differ')
    groups={**json.loads((destination/'split.json').read_text()),'all_2000':sorted(candidates)}
    result=dict(objective='official base8B vs same-base 50-step project LoRA8B',
        metrics={name:metrics(ids,before,after,labels,candidates) for name,ids in groups.items()},
        comparison_vs_trained2b=str(destination/'comparison.json'),development_set_reused=True,
        automatic_promotion=False,independent_confirmation_required=True,official_test_used=False)
    save(result,destination/'comparison_vs_base8b.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','train','report','export','prepare-eval','eval'))
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpus',default='1,2,3,4,5,6,7')
    p.add_argument('--master-port',type=int,default=29525)
    a=p.parse_args();root=a.root.resolve();out=(a.output_dir or root/'artifacts/train_v0/reranker_8b_lora_pilot_v1').resolve()
    gpus=a.gpus.split(',')
    if not 1<=len(gpus)<=8 or len(set(gpus))!=len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError('choose 1-8 distinct physical GPU indices')
    if not 1024<=a.master_port<=65535:
        raise ValueError('invalid training master port')
    if a.stage=='prepare':
        prepare(root,out,len(gpus),a.master_port);return
    with (out/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Verifying frozen Pilot inputs and base model...',flush=True)
        plan=validate(out,check_runtime=a.stage in ('train','export'))
        if a.stage=='train':
            if len(gpus)!=plan['workers']:
                raise ValueError('training worker count differs from plan')
            train(out,plan,gpus)
        elif a.stage=='report':
            print(json.dumps(checkpoint_summary(out,plan,'train'),ensure_ascii=False,indent=2))
        elif a.stage=='export':
            export(out,plan,gpus)
        elif a.stage=='prepare-eval':
            prepare_eval(out,plan,len(gpus))
        elif a.stage=='eval':
            eval_plan=validate_comparison(out/'eval')
            if len(gpus)!=eval_plan['workers']:
                raise ValueError('evaluation worker count differs')
            run_comparison(out/'eval',eval_plan,gpus)
            adaptation_report(out,plan)


if __name__=='__main__':
    main()
