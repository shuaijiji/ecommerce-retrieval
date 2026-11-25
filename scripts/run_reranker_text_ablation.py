#!/usr/bin/env python3
"""Fixed-model/fixed-candidate raw vs exact OCR/ASR deduplication experiment."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from audit_retriever_error_budget import build_cases
from compare_embedding_recall import paired
from compare_retriever_rerank import compare_cases
from evaluate_incremental_top25_router import build_examples
from evaluate_dynamic_top25_router import split_ids
from prepare_reranker_sft import compose_video_text,load_metadata,TEXT_BUDGETS
from rerank_pipeline import digest,load_labels,load_rows,save,write_rows
from reranker_text_dedup import transform_photo

MODES=('raw','dedup')


def freeze_candidates(rows10,rows25,examples,threshold):
    from rerank_dynamic_top25_incremental import should_expand
    fixed=[]
    decisions=[]
    for q in sorted(examples):
        e=examples[q]
        expanded=should_expand(q,e['top1_score'],threshold['cutoff'],threshold['tie_probability'],threshold['tie_seed'])
        candidates=list(rows10[q]['candidates'])
        seen={c['item_id'] for c in candidates}
        if expanded:
            candidates += [c for c in rows25[q]['candidates'] if c['item_id'] not in seen]
        expected=set(e['incremental_ranking'] if expanded else e['top10_ranking'])
        if len(candidates)!=len(expected) or {c['item_id'] for c in candidates}!=expected:
            raise ValueError('frozen candidate selection differs')
        fixed.append(dict(photo_id=q,candidates=candidates))
        decisions.append(dict(photo_id=q,expanded=expanded,baseline_top1_score=e['top1_score']))
    return fixed,decisions


def prepare(root,out,workers):
    if out.exists():
        raise ValueError('prepare needs a new output directory')
    print('Checking frozen cached inputs, models and scoring protocol...',flush=True)
    base=root/'artifacts/train_v0'
    cached=base/'retriever_rerank_top10_7gpu_v1'
    expanded=base/'retriever_dynamic_top25_7gpu_v1'
    source=expanded/'comparison.json'
    report=json.loads(source.read_text())
    arm=report['arms']['pilot100']
    labels_path=base/'clean_labels.jsonl'
    if digest(labels_path)!=report['label_sha256']:
        raise ValueError('development labels changed')
    _,labels=load_labels(labels_path)
    threshold=arm['router']['threshold']
    if threshold!=json.loads((expanded/'experiment_config.json').read_text())['router_thresholds']['pilot100']:
        raise ValueError('frozen router changed')
    files=[cached/'candidates/pilot100.jsonl',cached/'pilot100/predictions_2000.jsonl',
           expanded/'candidates/pilot100.jsonl',expanded/'pilot100/predictions_2000.jsonl']
    for path in files:
        if arm['input_hashes'].get(str(path))!=digest(path):
            raise ValueError(f'cached source changed: {path}')
    protocols=[]
    for directory in (cached,expanded):
        protocols.append(compare_cases([('pilot100',directory/'candidates/pilot100.jsonl',
            directory/'pilot100/predictions_2000.jsonl',sorted((directory/'pilot100').glob('worker_*/config.json')))],labels)['reranker_protocol'])
    if protocols[0]!=protocols[1]:
        raise ValueError('source scoring protocols differ')
    protocol=protocols[0]
    data=root/'data/train/all_data_package'
    model=base/'reranker_listwise_train_v1_merged/checkpoint-500-merged'
    weights={p.name:digest(p) for p in sorted(model.glob('*.safetensors'))}
    if not weights or weights!=protocol['model_weights_sha256']:
        raise ValueError('checkpoint differs from completed baseline')
    checks={data/'photo_info.jsonl':protocol['photo_sha256'],data/'item_info.jsonl':protocol['item_sha256'],
            model/'config.json':protocol['model_config_sha256'],
            root/'scripts/rerank_multimodal.py':protocol['runner_sha256'],
            root/'scripts/retrieval_common.py':protocol['preprocessing_sha256'],
            root/'Qwen3-VL-Embedding/src/models/qwen3_vl_reranker.py':protocol['upstream_sha256']}
    for path,expected in checks.items():
        if digest(path)!=expected:
            raise ValueError(f'baseline file changed: {path}')
    if (protocol['batch_size']!=4 or protocol['image_size']!=512 or protocol['seed']!=20261105 or
            protocol['text_budgets']!=list(map(list,TEXT_BUDGETS))):
        raise ValueError('unsupported cached preprocessing protocol')
    indexed=[{r['photo_id']:r for r in load_rows(path)} for path in files]
    examples,_=build_examples(*indexed,labels)
    original_cases=build_cases(examples,base/'embedding_pilot_eval_v1/retrieval',labels,threshold)
    candidates,decisions=freeze_candidates(indexed[0],indexed[2],examples,threshold)
    print('Loading selected photo metadata and preparing raw/dedup overlays...',flush=True)
    photos=load_metadata(data/'photo_info.jsonl','photo_id',set(examples))
    transformed={q:transform_photo(photo) for q,photo in photos.items()}
    preview=[dict(photo_id=q,raw=compose_video_text(photos[q]),dedup=compose_video_text(transformed[q])) for q in sorted(photos)]
    if not any(row['raw']!=row['dedup'] for row in preview):
        raise ValueError('deduplication changed no rendered query text')
    out.mkdir(parents=True)
    for mode in MODES:
        overlay=out/'data'/mode
        overlay.mkdir(parents=True)
        # Only metadata is copied; both variants read the same original image bytes.
        for name in ('item_info.jsonl','photo_frames','item_images'):
            target=data/name
            if not target.exists():
                raise ValueError(f'missing original data: {target}')
            (overlay/name).symlink_to(target,target_is_directory=target.is_dir())
        values=photos if mode=='raw' else transformed
        write_rows([values[q] for q in sorted(values)],overlay/'photo_info.jsonl')
        write_rows(candidates,out/'candidates'/(mode+'.jsonl'))
        for worker in range(workers):
            write_rows(candidates[worker::workers],out/'shards'/mode/f'worker_{worker}.jsonl')
    write_rows(preview,out/'text_preview.jsonl')
    write_rows(decisions,out/'frozen_routing.jsonl')
    write_rows(original_cases,out/'cached_baseline_cases.jsonl')
    calibration,test=split_ids(examples,20261119)
    save(dict(calibration=calibration,test=test),out/'split.json')
    generated=[p for p in out.rglob('*') if p.is_file() and not p.is_symlink()]
    inputs={str(p):digest(p) for p in [*files,source,labels_path,*checks]}
    for name in ('run_reranker_text_ablation.py','reranker_text_dedup.py','rerank_pipeline.py',
                 'prepare_reranker_sft.py','compare_retriever_rerank.py','compare_embedding_recall.py',
                 'compare_reranker_predictions.py','audit_retriever_error_budget.py',
                 'evaluate_incremental_top25_router.py','evaluate_dynamic_top25_router.py',
                 'rerank_dynamic_top25_incremental.py'):
        path=root/'scripts'/name
        inputs[str(path)]=digest(path)
    plan=dict(objective='fixed scored candidates and frozen routing; raw vs exact OCR/ASR dedup',
        root=str(root),output=str(out),workers=workers,count=len(candidates),model=str(model),weights=weights,
        baseline_protocol=protocol,inputs=inputs,generated={str(p):digest(p) for p in generated},
        data_links={str(out/'data'/mode/name):str(data/name) for mode in MODES
                    for name in ('item_info.jsonl','photo_frames','item_images')},
        text_changed=sum(row['raw']!=row['dedup'] for row in preview),
        text_policy='exact segments delimited by pipe/newline; first occurrence retained; OCR/ASR only; before unchanged field budgets',
        development_set_reused=True,automatic_promotion=False,official_test_used=False)
    save(plan,out/'plan.json')
    print(f'Prepared {len(candidates)} queries, changed text {plan["text_changed"]}; no GPU inference.',flush=True)


def validate_plan(out):
    plan=json.loads((out/'plan.json').read_text())
    if Path(plan['output'])!=out:
        raise ValueError('plan output directory changed')
    for path,expected in {**plan['inputs'],**plan['generated']}.items():
        if digest(path)!=expected:
            raise ValueError(f'frozen experiment input changed: {path}')
    for path,target in plan['data_links'].items():
        if not Path(path).is_symlink() or Path(path).resolve()!=Path(target).resolve():
            raise ValueError('media/metadata overlay link changed')
    model=Path(plan['model'])
    if {p.name:digest(p) for p in sorted(model.glob('*.safetensors'))}!=plan['weights']:
        raise ValueError('model weights changed')
    return plan


def summarize(out,plan):
    _,labels=load_labels(Path(plan['root'])/'artifacts/train_v0/clean_labels.jsonl')
    split=json.loads((out/'split.json').read_text())
    candidates={r['photo_id']:r for r in load_rows(out/'candidates/raw.jsonl')}
    predictions={}
    protocols=[]
    for mode in MODES:
        merged=[]
        for worker in range(plan['workers']):
            count=len(load_rows(out/'shards'/mode/f'worker_{worker}.jsonl'))
            merged+=load_rows(out/mode/f'worker_{worker}'/f'predictions_{count}.jsonl')
        write_rows(sorted(merged,key=lambda r:r['photo_id']),out/mode/'predictions_2000.jsonl')
        checked=compare_cases([(mode,out/'candidates'/(mode+'.jsonl'),out/mode/'predictions_2000.jsonl',
                              sorted((out/mode).glob('worker_*/config.json')))],labels,expected=plan['count'])
        protocols.append(checked['reranker_protocol'])
        predictions[mode]={r['photo_id']:r for r in merged}
        preview={r['photo_id']:r[mode] for r in load_rows(out/'text_preview.jsonl')}
        if any(r['query_text']!=preview[q] for q,r in predictions[mode].items()):
            raise ValueError('worker query text differs from frozen text preview')
        if protocols[-1]['photo_sha256']!=digest(out/'data'/mode/'photo_info.jsonl'):
            raise ValueError('worker used different photo metadata')
    normalize=lambda p:{k:v for k,v in p.items() if k!='photo_sha256'}
    if normalize(protocols[0])!=normalize(protocols[1]):
        raise ValueError('A/B scoring protocol differs beyond intended photo text')
    cached={r['photo_id']:r for r in load_rows(out/'cached_baseline_cases.jsonl')}
    def flags(mode,ids,k):
        return [labels[q] in [r['item_id'] for r in predictions[mode][q]['ranking'][:k]] for q in ids]
    groups={**split,'all_2000':sorted(candidates)}
    metrics={}
    for name,ids in groups.items():
        metrics[name]={f'top{k}':paired(flags('raw',ids,k),flags('dedup',ids,k)) for k in (1,3)}
        metrics[name]['candidate_recall']=sum(labels[q] in {c['item_id'] for c in candidates[q]['candidates']} for q in ids)/len(ids)
        metrics[name]['cached_vs_raw']={f'top{k}':paired(
            [labels[q] in (cached[q]['predicted_top3'][:k]) for q in ids],flags('raw',ids,k)) for k in (1,3)}
    changed=[]
    for q in sorted(candidates):
        raw=predictions['raw'][q]['ranking'];dedup=predictions['dedup'][q]['ranking']
        changed.append(dict(photo_id=q,old_bucket=cached[q]['bucket'],truth=labels[q],
            raw_top3=[r['item_id'] for r in raw[:3]],dedup_top3=[r['item_id'] for r in dedup[:3]]))
    write_rows(changed,out/'paired_cases.jsonl')
    engineering={mode:dict(mean_seconds_per_query=sum(r['seconds'] for r in rows.values())/len(rows),
        peak_memory_gib=max(r.get('peak_memory_gib',0) for r in rows.values()))
        for mode,rows in predictions.items()}
    result=dict(objective=plan['objective'],metrics=metrics,text_changed=plan['text_changed'],engineering=engineering,
        plan_sha256=digest(out/'plan.json'),development_set_reused=True,automatic_promotion=False,
        independent_confirmation_required=True,official_test_used=False,
        interpretation='Primary effect is dedup vs freshly rerun raw on identical candidates; cached vs raw diagnoses batch/runtime drift. Reused development split, not a fresh confirmation.')
    save(result,out/'comparison.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','run','report'))
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpus',default='1,2,3,4,5,6,7')
    a=p.parse_args()
    root=a.root.resolve();out=(a.output_dir or root/'artifacts/train_v0/reranker_text_ablation_v1').resolve()
    gpus=a.gpus.split(',')
    if not 1<=len(gpus)<=8 or len(set(gpus))!=len(gpus) or not all(v.isdigit() for v in gpus):
        raise ValueError('choose 1-8 distinct physical GPU indices')
    if a.stage=='prepare':
        prepare(root,out,len(gpus));return
    with (out/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=validate_plan(out)
        if a.stage=='report':
            summarize(out,plan);return
        if plan['root']!=str(root) or len(gpus)!=plan['workers']:
            raise ValueError('root/worker count differs from prepared plan')
        uuids=[]
        for gpu in gpus:
            fields=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
            if int(fields[1])>1024:
                raise ValueError(f'GPU {gpu} occupied; select idle cards')
            uuids.append(fields[0].strip())
        logs=out/'logs';logs.mkdir(exist_ok=True)
        def worker(index):
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=uuids[index],OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            for mode in MODES:
                shard=out/'shards'/mode/f'worker_{index}.jsonl'
                count=len(load_rows(shard))
                command=[sys.executable,'-u',str(root/'scripts/rerank_multimodal.py'),
                    '--candidates',str(shard),'--output-dir',str(out/mode/f'worker_{index}'),
                    '--data-root',str(out/'data'/mode),'--repo',str(root/'Qwen3-VL-Embedding'),
                    '--model',plan['model'],'--batch-size','4','--image-size','512','--seed','20261105']
                for stage,n in [('pilot',100),('full',count)]:
                    print(f'{mode} worker {index} GPU {gpus[index]} {stage}: {n}',flush=True)
                    with (logs/f'{mode}_worker_{index}_{stage}.log').open('a') as log:
                        subprocess.run(command+['--num-queries',str(n)],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            list(pool.map(worker,range(len(gpus))))
        validate_plan(out)
        summarize(out,plan)


if __name__=='__main__':
    main()
