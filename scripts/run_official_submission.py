#!/usr/bin/env python3
"""Frozen official 50k pipeline: prepare -> index -> pilot -> full -> export; no labels."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from official_submission_common import (metadata,media_paths,select_pilot,assert_plan_files,
    validate_retrieval,validate_result)
from rerank_pipeline import digest,save


def decode_image(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
    return path


def frozen_config(path,value):
    if path.exists() and json.loads(path.read_text())!=value:
        raise ValueError(f'configuration changed; use a new output directory: {path}')
    save(value,path)


def prepare(root,data,out,workers):
    photos=metadata(data/'photo_info.jsonl','photo_id')
    items=metadata(data/'item_info.jsonl','item_id')
    if len(photos)!=50000 or len(items)!=100000:
        raise ValueError('official run requires 50,000 unique videos and 100,000 unique candidate items')
    paths=media_paths(data,photos,items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for count,_ in enumerate(pool.map(decode_image,paths),1):
            if count%10000==0 or count==len(paths):
                print(f'media decoded {count}/{len(paths)}',flush=True)
    base=root/'artifacts/train_v0'
    report_path=base/'retriever_dynamic_top25_7gpu_v1/comparison.json'
    prereg_path=report_path.parent/'experiment_config.json'
    report=json.loads(report_path.read_text())
    prereg=json.loads(prereg_path.read_text())
    threshold=report['arms']['pilot100']['router']['threshold']
    if (report['primary_arm']!='pilot100' or not report['arms']['pilot100']['test_top1_benefit_gate']
            or threshold!=prereg['router_thresholds']['pilot100'] or
            report['router_fixed_before_results']!={'policy':'low_top1_score','budget':.25,'seed':20261119}):
        raise ValueError('pilot100 fixed route/benefit provenance differs')
    embed=base/'embedding_pilot_v1/merged'
    reranker=base/'reranker_listwise_train_v1_merged/checkpoint-500-merged'
    from retrieval_common import model_weight_hashes,QUERY_INSTRUCTION,TEXT_BUDGETS
    from rerank_multimodal import INSTRUCTION
    weights=model_weight_hashes(reranker)
    embedding_weights=model_weight_hashes(embed)
    embedding_eval_config=base/'embedding_pilot_eval_v1/retrieval/config.json'
    evaluated=json.loads(embedding_eval_config.read_text())
    if (evaluated['model_config_sha256']!=digest(embed/'config.json') or
            any(c.get('model_weight_hashes')!=embedding_weights for c in evaluated['item_index_configs'])):
        raise ValueError('embedding weights differ from the evaluated pilot100 model')
    if weights!=prereg['model_weights']:
        raise ValueError('reranker differs from completed development evaluation')
    if not json.loads((base/'embedding_pilot_v1/summary.json').read_text())['functional_gate']:
        raise ValueError('embedding pilot functional gate missing')
    offline_top10=base/'retriever_rerank_top10_7gpu_v1/experiment_config.json'
    offline_model=json.loads(offline_top10.read_text())
    if offline_model['model_weights']!=weights:
        raise ValueError('Top-10 reranker provenance mismatch')
    worker_configs=sorted(report_path.parent.glob('pilot100/worker_*/config.json'))
    if not worker_configs:
        raise ValueError('missing prior reranker protocol')
    reference=json.loads(worker_configs[0].read_text())
    import torch
    import transformers
    upstream=root/'Qwen3-VL-Embedding'
    if (reference['model_config_sha256']!=digest(reranker/'config.json') or reference['torch']!=torch.__version__ or reference['transformers']!=transformers.__version__ or
            reference['batch_size']!=4 or reference['image_size']!=512 or reference['max_length']!=10240 or
            reference['instruction']!=INSTRUCTION or reference['title_chars']!=500 or
            reference['text_budgets']!=list(map(list,TEXT_BUDGETS)) or
            reference['preprocessing_sha256']!=digest(root/'scripts/retrieval_common.py') or
            reference['upstream_sha256']!=digest(upstream/'src/models/qwen3_vl_reranker.py')):
        raise ValueError('formal scoring environment/protocol differs from development')
    hashes=[data/'photo_info.jsonl',data/'item_info.jsonl',embed/'config.json',reranker/'config.json',
            report_path,prereg_path,offline_top10,embedding_eval_config,base/'embedding_pilot_v1/summary.json',
            upstream/'src/models/qwen3_vl_embedding.py',upstream/'src/models/qwen3_vl_reranker.py']
    hashes += [root/'scripts'/name for name in ('run_official_submission.py','official_submission_common.py',
        'official_submission_worker.py','export_official_submission.py','retrieval_common.py',
        'build_item_index.py','evaluate_retrieval.py','rerank_pipeline.py','rerank_multimodal.py',
        'rerank_dynamic_top25_incremental.py')]
    plan=dict(schema_version=1,data_root=str(data),output_dir=str(out),workers=workers,
        photo_count=50000,item_count=100000,embedding_model=str(embed),reranker_model=str(reranker),
        embedding_model_weights=embedding_weights,reranker_model_weights=weights,
        upstream_repo=str(upstream),threshold=threshold,query_instruction=QUERY_INSTRUCTION,
        pilot_ids=select_pilot(photos),pilot_seed=20261123,torch_version=torch.__version__,
        transformers_version=transformers.__version__,file_hashes={str(p):digest(p) for p in hashes},
        labels_used=False,submission_top_k=3,model_selection_on_official_test=False)
    frozen_config(out/'plan.json',plan)
    save(dict(media_files_decoded=len(paths),photos=len(photos),items=len(items),
              source='organizer test_data.zip; confirmed by user extraction and published dataset specification',
              no_accuracy_computed=True),out/'data_audit.json')
    print('Frozen plan ready: '+str(out/'plan.json'),flush=True)


def run_gpu_group(commands,uuids,logdir,name):
    handles=[]
    children=[]
    monitor=None
    resource=None
    try:
        resource=(logdir/(name+'_gpu.csv')).open('w')
        monitor=subprocess.Popen(['nvidia-smi','--id='+','.join(uuids),
            '--query-gpu=timestamp,uuid,utilization.gpu,memory.used,power.draw','--format=csv','--loop=2'],
            stdout=resource,stderr=subprocess.DEVNULL)
        for worker,(command,uuid) in enumerate(zip(commands,uuids)):
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            handle=(logdir/f'{name}_worker_{worker}.log').open('a')
            handles.append(handle)
            children.append(subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT))
        print(f'{name} launched on {len(children)} GPUs; logs: {logdir}',flush=True)
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None,0) for child in children):
                raise RuntimeError(f'{name} worker failed; inspect per-worker logs')
            time.sleep(2)
        if any(child.returncode for child in children):
            raise RuntimeError(f'{name} worker failed')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if monitor:
            monitor.terminate()
            monitor.wait()
        for handle in handles:
            handle.close()
        if resource:
            resource.close()


def summarize(plan,ids):
    out=Path(plan['output_dir'])
    item_set=set(metadata(Path(plan['data_root'])/'item_info.jsonl','item_id'))
    retrieval_seconds=rerank_seconds=expanded=score_count=0
    peak=0
    unique_scores=set()
    for q in ids:
        source=out/'retrieval'/f'{q}.json'
        row=json.loads(source.read_text())
        validate_retrieval(row,q,item_set)
        result=json.loads((out/'predictions'/f'{q}.json').read_text())
        validate_result(result,row,plan['threshold'],digest(source))
        retrieval_seconds+=row['seconds']
        rerank_seconds+=result['seconds']
        expanded+=result['expanded']
        score_count+=len(result['ranking'])
        peak=max(peak,result['peak_memory_gib'])
        unique_scores.update(r['score'] for r in result['ranking'])
    if len(unique_scores)<2:
        raise ValueError('all prediction scores identical')
    n=len(ids)
    return dict(count=n,expanded_count=expanded,expanded_rate=expanded/n,
        mean_candidates=score_count/n,mean_retrieval_seconds=retrieval_seconds/n,
        mean_rerank_seconds=rerank_seconds/n,peak_memory_gib=peak,
        estimated_full_hours_excluding_initialization_and_export=(retrieval_seconds+rerank_seconds)/n*50000/plan['workers']/3600,
        estimate_warning='linear estimate from summed per-query times; actual multi-GPU wall time and I/O may differ',
        scores_finite_and_complete=True,accuracy_computed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','index','pilot','full','export'))
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--data-root',type=Path)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpus',default='1,2,3,4,5,6,7')
    a=p.parse_args()
    root=a.root.resolve()
    data=(a.data_root or root/'data/test/extracted/test_large_package').resolve()
    out=(a.output_dir or root/'artifacts/official_submission_v1').resolve()
    physical=a.gpus.split(',')
    if not 1<=len(physical)<=8 or len(set(physical))!=len(physical):
        raise ValueError('specify 1–8 distinct physical GPUs')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'launcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.stage=='prepare':
        prepare(root,data,out,len(physical))
        return
    plan_path=out/'plan.json'
    plan=json.loads(plan_path.read_text())
    if plan['workers']!=len(physical) or plan['data_root']!=str(data):
        raise ValueError('worker count or data path changed; use frozen plan parameters')
    assert_plan_files(plan)
    if a.stage=='export':
        subprocess.run(['python',str(root/'scripts/export_official_submission.py'),'--plan',str(plan_path)],check=True)
        return
    uuids=[]
    for gpu in physical:
        row=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'],text=True).strip().split(',')
        if int(row[1])>1024:
            raise RuntimeError(f'GPU {gpu} occupied; select idle cards without changing worker count')
        uuids.append(row[0].strip())
    logs=out/'logs'/time.strftime('%Y%m%d-%H%M%S')
    logs.mkdir(parents=True,exist_ok=True)
    if a.stage=='index':
        common=['python','-u',str(root/'scripts/build_item_index.py'),'--data-root',str(data),
                '--model',plan['embedding_model'],'--record-weight-hashes','--output-dir',str(out/'items')]
        gate_path=out/'index_performance_gate.json'
        prior=json.loads(gate_path.read_text()) if gate_path.exists() else None
        if prior is not None and (prior['plan_sha256']!=digest(plan_path) or not prior['gate']):
            raise ValueError('saved index performance gate does not match frozen plan')
        rates={}
        for n in ([] if prior else sorted({1,2,4,len(uuids)} & set(range(1,len(uuids)+1)))):
            commands=[]
            reports=[]
            for worker in range(n):
                report=logs/f'index_perf_{n}_{worker}.json'
                reports.append(report)
                commands.append(common+['--partitions',str(n),'--partition-index',str(worker),
                    '--max-new-shards','2','--benchmark-output',str(report)])
            run_gpu_group(commands,uuids[:n],logs,f'index_perf_{n}')
            records=[json.loads(p.read_text()) for p in reports]
            start=min(r['start_unix'] for r in records)
            end=max(r['end_unix'] for r in records)
            rates[str(n)]=sum(r['encoded_items'] for r in records)/(end-start)
            print(f'index performance {n} GPUs: {rates[str(n)]:.2f} items/s',flush=True)
        speedup=prior['speedup'] if prior else rates[str(len(uuids))]/rates['1']
        report=prior or dict(items_per_second=rates,speedup=speedup,gate=speedup>=len(uuids)/2,plan_sha256=digest(plan_path))
        save(report,logs/'index_performance.json')
        if not report['gate']:
            raise RuntimeError('index scaling below half-linear speedup; preserve shards and inspect resource logs')
        save(report,gate_path)
        commands=[common+['--partitions',str(len(uuids)),'--partition-index',str(i)] for i in range(len(uuids))]
        run_gpu_group(commands,uuids,logs,'index_full')
        subprocess.run(common,env={**os.environ,'CUDA_VISIBLE_DEVICES':uuids[0]},check=True)
        manifest=out/'items/manifest.json'
        if not json.loads(manifest.read_text())['complete']:
            raise ValueError('item index incomplete')
        save(dict(plan_sha256=digest(plan_path),manifest_sha256=digest(manifest),
                  config_sha256=digest(out/'items/config.json')),out/'index_complete.json')
        print('Official item index complete.',flush=True)
        return
    index_proof=json.loads((out/'index_complete.json').read_text())
    if (index_proof['plan_sha256']!=digest(plan_path) or
            index_proof['manifest_sha256']!=digest(out/'items/manifest.json') or
            index_proof['config_sha256']!=digest(out/'items/config.json')):
        raise ValueError('index provenance changed')
    if a.stage=='full':
        pilot=json.loads((out/'pilot_pass.json').read_text())
        if pilot['plan_sha256']!=digest(plan_path) or pilot['index_proof_sha256']!=digest(out/'index_complete.json'):
            raise ValueError('pilot gate configuration mismatch')
    started=time.perf_counter()
    for phase in ('retrieval','rerank'):
        commands=[['python','-u',str(root/'scripts/official_submission_worker.py'),
                    '--plan',str(plan_path),'--worker',str(i),'--phase',phase]+
                   (['--pilot'] if a.stage=='pilot' else []) for i in range(len(uuids))]
        run_gpu_group(commands,uuids,logs,a.stage+'_'+phase)
    ids=plan['pilot_ids'] if a.stage=='pilot' else sorted(metadata(data/'photo_info.jsonl','photo_id'))
    summary=summarize(plan,ids)
    summary['stage_wall_seconds']=time.perf_counter()-started
    save(summary,out/(a.stage+'_summary.json'))
    if a.stage=='pilot':
        save(dict(plan_sha256=digest(plan_path),index_proof_sha256=digest(out/'index_complete.json'),count=100),out/'pilot_pass.json')
    else:
        subprocess.run(['python',str(root/'scripts/export_official_submission.py'),'--plan',str(plan_path)],check=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
