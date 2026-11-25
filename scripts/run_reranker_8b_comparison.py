#!/usr/bin/env python3
"""Fixed-candidate official 8B versus completed fine-tuned 2B raw baseline."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from compare_embedding_recall import paired
from compare_retriever_rerank import compare_cases
from download_qwen_reranker_8b import FOLDER,MODEL_ID,REVISION,RECEIPT,validate_download
from rerank_pipeline import digest,load_labels,load_rows,save,write_rows
from run_reranker_text_ablation import validate_plan as validate_source_plan

MODEL_KEYS={'model','model_config_sha256','model_weights_sha256'}


def check_protocols(baseline,challenger):
    if {k:v for k,v in baseline.items() if k not in MODEL_KEYS}!={k:v for k,v in challenger.items() if k not in MODEL_KEYS}:
        raise ValueError('scoring protocol differs beyond intended model replacement')


def checked_baseline(source,labels,expected=2000):
    return compare_cases([('raw',source/'candidates/raw.jsonl',source/'raw/predictions_2000.jsonl',
                           sorted((source/'raw').glob('worker_*/config.json')))],labels,expected=expected)


def prepare(root,out,workers):
    if out.exists():
        raise ValueError('prepare needs a new output directory')
    source=root/'artifacts/train_v0/reranker_text_ablation_v1'
    print('Verifying completed 2B raw baseline and pinned 8B files...',flush=True)
    previous=validate_source_plan(source)
    comparison=json.loads((source/'comparison.json').read_text())
    if comparison['plan_sha256']!=digest(source/'plan.json'):
        raise ValueError('completed baseline plan differs')
    model=root/'models'/FOLDER
    receipt=validate_download(model)
    labels_path=root/'artifacts/train_v0/clean_labels.jsonl'
    _,labels=load_labels(labels_path)
    baseline=checked_baseline(source,labels)
    # Completed raw baseline must reproduce the user-reported A/B reference.
    for k in (1,3):
        if abs(baseline['arms']['raw'][f'system_top{k}']-comparison['metrics']['all_2000'][f'top{k}']['baseline'])>1e-12:
            raise ValueError('raw baseline differs from completed comparison')
    rows=load_rows(source/'candidates/raw.jsonl')
    if len(rows)!=2000:
        raise ValueError('expected 2000 fixed development queries')
    split=json.loads((source/'split.json').read_text())
    if (len(split['calibration'])!=1000 or len(split['test'])!=1000 or
            len(set(split['calibration']+split['test']))!=2000 or
            set(split['calibration']+split['test'])!={r['photo_id'] for r in rows}):
        raise ValueError('development split changed')
    out.mkdir(parents=True)
    write_rows(rows,out/'candidates/base8b.jsonl')
    for worker in range(workers):
        write_rows(rows[worker::workers],out/'shards/base8b'/f'worker_{worker}.jsonl')
    save(split,out/'split.json')
    inputs={**previous['inputs'],**previous['generated']}
    for path in [source/'plan.json',source/'comparison.json',source/'raw/predictions_2000.jsonl',model/RECEIPT,
                 *sorted((source/'raw').glob('worker_*/config.json')),
                 *[p for p in Path(previous['model']).iterdir() if p.is_file() and p.suffix in ('.json','.txt','.jinja')]]:
        inputs[str(path)]=digest(path)
    for name in ('run_reranker_8b_comparison.py','download_qwen_reranker_8b.py'):
        path=root/'scripts'/name;inputs[str(path)]=digest(path)
    generated={str(p):digest(p) for p in out.rglob('*') if p.is_file()}
    save(dict(root=str(root),output=str(out),source=str(source),workers=workers,count=2000,
        model=str(model),model_id=MODEL_ID,revision=REVISION,model_files=receipt['files'],
        baseline_model=previous['model'],baseline_weights=previous['weights'],
        baseline_protocol=baseline['reranker_protocol'],data_root=str(source/'data/raw'),
        inputs=inputs,generated=generated,data_links=previous['data_links'],
        objective='official base 8B vs domain-fine-tuned 2B on identical scored candidates and original text',
        primary='Top1; report Top3 and latency; no model promotion on reused development set',
        confounding='model size AND domain fine-tuning differ; not a pure parameter-scaling experiment',
        development_set_reused=True,automatic_promotion=False,official_test_used=False),out/'plan.json')
    print(f'Prepared {len(rows)} fixed queries; {workers} workers. No GPU inference.',flush=True)


def validate(out):
    plan=json.loads((out/'plan.json').read_text())
    if str(out)!=plan['output']:
        raise ValueError('output differs from frozen plan')
    for path,expected in {**plan['inputs'],**plan['generated']}.items():
        if digest(path)!=expected:
            raise ValueError(f'frozen experiment input changed: {path}')
    for path,target in plan['data_links'].items():
        if not Path(path).is_symlink() or Path(path).resolve()!=Path(target).resolve():
            raise ValueError('source data overlay changed')
    folder=Path(plan['model'])
    if {name:digest(folder/name) for name in plan['model_files']}!=plan['model_files']:
        raise ValueError('8B model/tokenizer files changed')
    baseline=Path(plan['baseline_model'])
    if {p.name:digest(p) for p in baseline.glob('*.safetensors')}!=plan['baseline_weights']:
        raise ValueError('2B reference weights changed')
    return plan


def metrics(ids,baseline,challenger,labels,candidates):
    def flags(rows,k):
        return [labels[q] in [r['item_id'] for r in rows[q]['ranking'][:k]] for q in ids]
    hits=sum(labels[q] in {c['item_id'] for c in candidates[q]['candidates']} for q in ids)
    result={f'top{k}':paired(flags(baseline,k),flags(challenger,k)) for k in (1,3)}
    result.update(count=len(ids),candidate_recall=hits/len(ids),
                  baseline_accuracy_given_candidate_hit=sum(flags(baseline,1))/hits if hits else None,
                  challenger_accuracy_given_candidate_hit=sum(flags(challenger,1))/hits if hits else None)
    return result


def report(out,plan):
    source=Path(plan['source'])
    _,labels=load_labels(Path(plan['root'])/'artifacts/train_v0/clean_labels.jsonl')
    baseline_check=checked_baseline(source,labels,expected=plan['count'])
    predictions=[]
    for worker in range(plan['workers']):
        count=len(load_rows(out/'shards/base8b'/f'worker_{worker}.jsonl'))
        predictions+=load_rows(out/'base8b'/f'worker_{worker}'/f'predictions_{count}.jsonl')
    write_rows(sorted(predictions,key=lambda r:r['photo_id']),out/'base8b/predictions_2000.jsonl')
    challenger_check=compare_cases([('base8b',out/'candidates/base8b.jsonl',out/'base8b/predictions_2000.jsonl',
                                    sorted((out/'base8b').glob('worker_*/config.json')))],labels,expected=plan['count'])
    check_protocols(baseline_check['reranker_protocol'],challenger_check['reranker_protocol'])
    if challenger_check['reranker_protocol']['model_weights_sha256']!={name:sha for name,sha in plan['model_files'].items() if name.endswith('.safetensors')}:
        raise ValueError('worker model differs from pinned 8B weights')
    baseline={r['photo_id']:r for r in load_rows(source/'raw/predictions_2000.jsonl')}
    challenger={r['photo_id']:r for r in predictions}
    candidates={r['photo_id']:r for r in load_rows(out/'candidates/base8b.jsonl')}
    if set(baseline)!=set(challenger) or set(candidates)!=set(baseline):
        raise ValueError('query coverage differs')
    for q in baseline:
        for key in ('query_text','frame_paths','original_frames','encoded_frames'):
            if baseline[q][key]!=challenger[q][key]:
                raise ValueError('query text/frame input differs between models')
    groups={**json.loads((out/'split.json').read_text()),'all_2000':sorted(baseline)}
    results={name:metrics(ids,baseline,challenger,labels,candidates) for name,ids in groups.items()}
    cases=[dict(photo_id=q,truth=labels[q],baseline_top3=[r['item_id'] for r in baseline[q]['ranking'][:3]],
                challenger_top3=[r['item_id'] for r in challenger[q]['ranking'][:3]]) for q in sorted(baseline)]
    write_rows(cases,out/'paired_cases.jsonl')
    engineering={name:dict(mean_seconds_per_query=sum(r['seconds'] for r in values.values())/len(values),
                           peak_memory_gib=max(r['peak_memory_gib'] for r in values.values()))
                 for name,values in [('trained2b',baseline),('base8b',challenger)]}
    result=dict(objective=plan['objective'],metrics=results,engineering=engineering,
        confounding=plan['confounding'],plan_sha256=digest(out/'plan.json'),
        prediction_sha256=dict(trained2b=digest(source/'raw/predictions_2000.jsonl'),base8b=digest(out/'base8b/predictions_2000.jsonl')),
        development_set_reused=True,independent_confirmation_required=True,automatic_promotion=False,
        official_test_used=False,interpretation='Conditional reranker diagnostic on unchanged candidate coverage; no new recall or dynamic-route fitting.')
    save(result,out/'comparison.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


def gpu_uuids(gpus):
    uuids=[]
    for gpu in gpus:
        row=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if int(row[1])>1024:
            raise ValueError(f'GPU {gpu} occupied; select idle cards')
        uuids.append(row[0].strip())
    return uuids


def command(plan,candidates,destination,count):
    root=Path(plan['root'])
    return [sys.executable,'-u',str(root/'scripts/rerank_multimodal.py'),
        '--candidates',str(candidates),'--output-dir',str(destination),'--data-root',plan['data_root'],
        '--repo',str(root/'Qwen3-VL-Embedding'),'--model',plan['model'],'--batch-size','4',
        '--image-size','512','--seed','20261105','--num-queries',str(count)]


def run(out,plan,gpus):
    logs=out/'logs';logs.mkdir(exist_ok=True)
    uuids=gpu_uuids(gpus)
    def invoke(gpu,cmd,path):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        with path.open('a') as log:
            subprocess.run(cmd,env=env,cwd=plan['root'],stdout=log,stderr=subprocess.STDOUT,check=True)
    # Global engineering Pilot before any full worker inference; no accuracy selection.
    print(f'8B engineering pilot: 100 queries on physical GPU {gpus[0]}; {logs/"pilot.log"}',flush=True)
    invoke(uuids[0],command(plan,out/'candidates/base8b.jsonl',out/'pilot',100),logs/'pilot.log')
    gate=json.loads((out/'pilot/pilot_pass.json').read_text())
    config=json.loads((out/'pilot/config.json').read_text())
    if gate['count']!=100 or gate['config_sha256']!=digest(out/'pilot/config.json'):
        raise ValueError('invalid engineering Pilot receipt')
    check_protocols(plan['baseline_protocol'],{k:v for k,v in config.items() if k!='candidate_sha256'})
    save(dict(count=100,plan_sha256=digest(out/'plan.json'),config_sha256=digest(out/'pilot/config.json'),
              pilot_pass_sha256=digest(out/'pilot/pilot_pass.json'),accuracy_used_for_gate=False),out/'engineering_gate.json')
    uuids=gpu_uuids(gpus)
    def worker(index):
        shard=out/'shards/base8b'/f'worker_{index}.jsonl'
        count=len(load_rows(shard));destination=out/'base8b'/f'worker_{index}'
        for stage,n in [('pilot',100),('full',count)]:
            print(f'base8b worker {index} GPU {gpus[index]} {stage}: {n}',flush=True)
            invoke(uuids[index],command(plan,shard,destination,n),logs/f'worker_{index}_{stage}.log')
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker,range(len(gpus))))
    validate(out)
    report(out,plan)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','run','report'))
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--gpus',default='1,2,3,4,5,6,7')
    a=p.parse_args();root=a.root.resolve()
    out=(a.output_dir or root/'artifacts/train_v0/reranker_8b_comparison_v1').resolve()
    gpus=a.gpus.split(',')
    if not 1<=len(gpus)<=8 or len(set(gpus))!=len(gpus) or not all(g.isdigit() for g in gpus):
        raise ValueError('choose 1-8 distinct physical GPU indices')
    if a.stage=='prepare':
        prepare(root,out,len(gpus));return
    with (out/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('Verifying frozen model, baseline and input hashes...',flush=True)
        plan=validate(out)
        if a.stage=='report':
            report(out,plan);return
        if plan['root']!=str(root) or plan['workers']!=len(gpus):
            raise ValueError('root/worker count differs from plan')
        run(out,plan,gpus)


if __name__=='__main__':
    main()
