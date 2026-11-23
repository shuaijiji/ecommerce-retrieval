#!/usr/bin/env python3
"""Same checkpoint-500 and per-route Top-10 across original/100-query/1k retrieval."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
from rerank_pipeline import digest, generate, load_rows, save, write_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--gpus',default='5,6,7',help='physical idle GPU indices; e.g. 1,5,6,7 or all eight if idle')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--top-k',type=int,choices=(10,25),default=10)
    p.add_argument('--arms',nargs='+',choices=('original','pilot100','train1k'))
    p.add_argument('--top10-root',type=Path,help='cached completed Top-10 comparison directory for Top-25 analysis')
    a=p.parse_args()
    root=a.root.resolve()
    default_output='retriever_dynamic_top25_v1' if a.top_k==25 else 'retriever_rerank_top10_v1'
    out=(a.output_dir or root/'artifacts/train_v0'/default_output).resolve()
    if a.top_k==25 and a.top10_root is None:
        raise ValueError('Top-25 requires --top10-root for cached Top-10 scores')
    gpu_ids=a.gpus.split(',')
    if not 1<=len(gpu_ids)<=8 or len(set(gpu_ids))!=len(gpu_ids):
        raise ValueError('use 1–8 distinct physical GPUs')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'launcher.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    saved_config=out/'experiment_config.json'
    if saved_config.exists() and json.loads(saved_config.read_text())['worker_count']!=len(gpu_ids):
        raise ValueError('worker count changed; use a fresh output directory')
    base=root/'artifacts/train_v0'
    sources={'original':base/'item_disjoint_eval_v1/retrieval',
             'pilot100':base/'embedding_pilot_eval_v1/retrieval',
             'train1k':base/'embedding_eval_1k_v1/retrieval'}
    model=base/'reranker_listwise_train_v1_merged/checkpoint-500-merged'
    ids=json.loads((sources['original']/'selected_photo_ids.json').read_text())
    if len(ids)!=2000 or len(set(ids))!=2000:
        raise ValueError('need exactly 2,000 frozen queries')
    ids=sorted(ids)
    requested=a.arms or (['pilot100','train1k'] if a.top_k==25 else list(sources))
    if len(set(requested))!=len(requested):
        raise ValueError('duplicate arms')
    if a.top_k==10 and requested!=list(sources):
        raise ValueError('Top-10 comparison requires all three original arms')
    sources={name:sources[name] for name in requested}
    source_hashes={}
    reference=None
    for name,source in sources.items():
        if sorted(json.loads((source/'selected_photo_ids.json').read_text()))!=ids:
            raise ValueError('retrieval query selections differ')
        config=json.loads((source/'config.json').read_text())
        common={k:config[k] for k in ('photo_info_sha256','label_sha256','query_instruction',
                                    'text_budgets','frame_preprocessing')}
        common['pools']=[{k:c[k] for k in ('item_ids_sha256','item_count','item_info_sha256','item_input','image_size')}
                         for c in config['item_index_configs']]
        if reference is not None and common!=reference:
            raise ValueError('retrieval evaluation protocols or item pools differ')
        reference=common
        candidate=out/'candidates'/(name+'.jsonl')
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            staged=Path(temporary)/'candidates.jsonl'
            generate(argparse.Namespace(retrieval_dir=source,top_k=a.top_k,expected_count=2000,output=staged))
            if candidate.exists() and candidate.read_bytes()!=staged.read_bytes():
                raise ValueError('candidate source changed; use a fresh output directory')
            write_rows(load_rows(staged),candidate)
            save(json.loads(Path(str(staged)+'.manifest.json').read_text()),str(candidate)+'.manifest.json')
        source_hashes[name]=dict(config=digest(source/'config.json'),candidate=digest(candidate),
                                manifest=digest(str(candidate)+'.manifest.json'))
        rows=load_rows(candidate)
        if sorted(r['photo_id'] for r in rows)!=ids:
            raise ValueError('retrieval result IDs differ from frozen query selection')
        for worker in range(len(gpu_ids)):
            subset=[r for i,r in enumerate(rows) if i%len(gpu_ids)==worker]
            write_rows(subset,out/'shards'/name/f'worker_{worker}.jsonl')
    config={'sources':source_hashes,'worker_count':len(gpu_ids),'batch_size':4,
            'reranker':str(model),'model_weights':{p.name:digest(p) for p in sorted(model.glob('*.safetensors'))},
            'query_ids_sha256':__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(),
            'labels_used_for_candidates':False,'top_k_per_route':a.top_k}
    if a.top_k==25:
        cached=a.top10_root.resolve()
        config['cached_top10_root']=str(cached)
        config['router_seed']=20261119
        config['router_budget']=.25
        from evaluate_retriever_dynamic_top25 import freeze_threshold
        config['router_thresholds']={name:freeze_threshold(
            cached/'candidates'/(name+'.jsonl'),cached/name/'predictions_2000.jsonl') for name in sources}
        config['cached_top10_hashes']={name:{
            'candidates':digest(cached/'candidates'/(name+'.jsonl')),
            'predictions':digest(cached/name/'predictions_2000.jsonl')} for name in sources}
    if not config['model_weights']:
        raise ValueError('checkpoint-500 merged model missing')
    cp=out/'experiment_config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:
        raise ValueError('experiment configuration changed; use a fresh output directory')
    save(config,cp)
    if a.prepare_only:
        print('Candidate preparation complete; no GPU inference launched.',flush=True)
        return
    uuids=[]
    for gpu in gpu_ids:
        row=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.used',
                                     '--format=csv,noheader,nounits'],text=True).strip().split(',')
        if int(row[1])>1024:
            raise RuntimeError(f'GPU {gpu} occupied ({row[1].strip()} MiB); select idle cards')
        uuids.append(row[0].strip())
    logs=out/'logs'
    logs.mkdir(exist_ok=True)
    def run_worker(worker):
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=uuids[worker],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        for name in sources:
            candidate=out/'shards'/name/f'worker_{worker}.jsonl'
            count=len(load_rows(candidate))
            target=out/name/f'worker_{worker}'
            command=['python','-u',str(root/'scripts/rerank_multimodal.py'),
                     '--candidates',str(candidate),'--output-dir',str(target),
                     '--data-root',str(root/'data/train/all_data_package'),
                     '--repo',str(root/'Qwen3-VL-Embedding'),'--model',str(model),'--batch-size','4']
            for stage,n in [('pilot',100),('full',count)]:
                print(f'{name} worker {worker} GPU {gpu_ids[worker]} {stage}: {n} queries',flush=True)
                with (logs/f'{name}_worker_{worker}_{stage}.log').open('a') as log:
                    subprocess.run(command+['--num-queries',str(n)],env=env,cwd=root,
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        list(pool.map(run_worker,range(len(gpu_ids))))
    for name in sources:
        rows=[]
        for worker in range(len(gpu_ids)):
            count=len(load_rows(out/'shards'/name/f'worker_{worker}.jsonl'))
            rows+=load_rows(out/name/f'worker_{worker}'/f'predictions_{count}.jsonl')
        if sorted(r['photo_id'] for r in rows)!=ids:
            raise ValueError('merged predictions have missing/duplicate query IDs')
        write_rows(sorted(rows,key=lambda r:r['photo_id']),out/name/'predictions_2000.jsonl')
    if a.top_k==25:
        analysis=['python',str(root/'scripts/evaluate_retriever_dynamic_top25.py'),
                  '--top10-root',str(a.top10_root.resolve()),'--top25-root',str(out),
                  '--arms',*requested,'--labels',str(base/'clean_labels.jsonl')]
    else:
        analysis=['python',str(root/'scripts/compare_retriever_rerank.py'),
                  '--output-root',str(out),'--labels',str(base/'clean_labels.jsonl')]
    subprocess.run(analysis,cwd=root,check=True)
    print('Complete: '+str(out/'comparison.json'),flush=True)


if __name__=='__main__':
    main()
