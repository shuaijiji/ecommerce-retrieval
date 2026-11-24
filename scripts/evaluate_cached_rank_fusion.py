#!/usr/bin/env python3
"""CPU diagnostic: fuse cached rerank order with retrieval ranks on unchanged candidates."""
import argparse
import json
from pathlib import Path
from compare_embedding_recall import paired
from evaluate_incremental_top25_router import build_examples
from evaluate_dynamic_top25_router import split_ids
from rerank_dynamic_top25_incremental import should_expand
from compare_retriever_rerank import compare_cases
from rerank_pipeline import digest,load_labels,load_rows,save,write_rows
from audit_retriever_error_budget import build_cases

WEIGHTS=(0.0,0.1,0.25,0.5,1.0)
RRF_K=60


def fuse(ranking,routes,weight):
    if not 0<=weight<=1 or not ranking or len(set(ranking))!=len(ranking):
        raise ValueError('invalid weight or candidate ranking')
    if len(routes)!=2 or any(len(ids)!=len(set(ids)) for ids in routes):
        raise ValueError('need two unique retrieval rankings')
    rank_maps=[{item:index+1 for index,item in enumerate(ids)} for ids in routes]
    positions={item:index+1 for index,item in enumerate(ranking)}
    def score(item):
        retrieval=sum(1/(RRF_K+r[item]) if item in r else 0 for r in rank_maps)/2
        return (1-weight)/(RRF_K+positions[item])+weight*retrieval
    # Preserve original order at exact ties, including the zero-weight baseline.
    return sorted(ranking,key=lambda item:(-score(item),positions[item],item))


def flags(ids,rankings,labels,k):
    return [labels[q] in rankings[q][:k] for q in ids]


def choose_weight(ids,variants,labels):
    return max(variants,key=lambda w:(sum(flags(ids,variants[w],labels,1)),
                                     sum(flags(ids,variants[w],labels,3)),-w))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    root=a.root.resolve()
    base=root/'artifacts/train_v0'
    out=(a.output_dir or base/'cached_rank_fusion_diagnostic_v1').resolve()
    out.mkdir(parents=True,exist_ok=False)
    cached=base/'retriever_rerank_top10_7gpu_v1'
    expanded=base/'retriever_dynamic_top25_7gpu_v1'
    report_path=expanded/'comparison.json'
    report=json.loads(report_path.read_text())
    arm=report['arms']['pilot100']
    labels_path=base/'clean_labels.jsonl'
    if digest(labels_path)!=report['label_sha256']:
        raise ValueError('labels changed')
    _,labels=load_labels(labels_path)
    threshold=arm['router']['threshold']
    prereg=json.loads((expanded/'experiment_config.json').read_text())
    if threshold!=prereg['router_thresholds']['pilot100']:
        raise ValueError('frozen router differs')
    files=[cached/'candidates/pilot100.jsonl',cached/'pilot100/predictions_2000.jsonl',
           expanded/'candidates/pilot100.jsonl',expanded/'pilot100/predictions_2000.jsonl']
    for path in files:
        if arm['input_hashes'].get(str(path))!=digest(path):
            raise ValueError(f'cached input changed: {path}')
    protocols=[]
    for directory in (cached,expanded):
        checked=compare_cases([('pilot100',directory/'candidates/pilot100.jsonl',
            directory/'pilot100/predictions_2000.jsonl',sorted((directory/'pilot100').glob('worker_*/config.json')))],labels)
        protocols.append(checked['reranker_protocol'])
    if protocols[0]!=protocols[1]:
        raise ValueError('reranker protocols differ')
    indexed=[{r['photo_id']:r for r in load_rows(path)} for path in files]
    examples,_=build_examples(*indexed,labels)
    retrieval=base/'embedding_pilot_eval_v1/retrieval'
    cases=build_cases(examples,retrieval,labels,threshold)
    calibration,test=split_ids(examples,20261119)
    variants={w:{} for w in WEIGHTS}
    for q,e in examples.items():
        expand=should_expand(q,e['top1_score'],threshold['cutoff'],threshold['tie_probability'],threshold['tie_seed'])
        original=e['incremental_ranking'] if expand else e['top10_ranking']
        routes=[json.loads((retrieval/'results'/mode/(q+'.json')).read_text())['top_item_ids']
                for mode in ('vision','multimodal')]
        for w in WEIGHTS:
            variants[w][q]=fuse(original,routes,w)
            if set(variants[w][q])!=set(original):
                raise AssertionError('candidate coverage changed')
    selected=choose_weight(calibration,variants,labels)
    calibration_results={str(w):{f'top{k}':sum(flags(calibration,v,labels,k))/len(calibration)
                                for k in (1,3)} for w,v in variants.items()}
    test_results={f'top{k}':paired(flags(test,variants[0],labels,k),flags(test,variants[selected],labels,k))
                  for k in (1,3)}
    for k in (1,3):
        if abs(test_results[f'top{k}']['baseline']-arm['test_dynamic_incremental'][f'system_top{k}'])>1e-12:
            raise ValueError('baseline does not reproduce frozen development report')
    saved={row['photo_id']:row for row in cases}
    changes=[]
    for q in test:
        before,after=variants[0][q],variants[selected][q]
        changes.append(dict(photo_id=q,truth=labels[q],old_top3=before[:3],new_top3=after[:3],
            old_error_bucket=saved[q]['bucket'],rescued=before[0]!=labels[q] and after[0]==labels[q],
            harmed=before[0]==labels[q] and after[0]!=labels[q]))
    write_rows(changes,out/'test_cases.jsonl')
    result=dict(objective='cached rank fusion diagnostic on unchanged scored candidates',
        weights=list(WEIGHTS),rrf_k=RRF_K,selected_weight=selected,
        selection='calibration Top1, then Top3, then smaller retrieval weight',
        calibration_count=len(calibration),test_count=len(test),calibration=calibration_results,test=test_results,
        frozen_router=threshold,source_report_sha256=digest(report_path),input_hashes={str(p):digest(p) for p in files},
        label_sha256=digest(labels_path),candidate_sets_unchanged=True,extra_gpu_inference=False,
        development_set_reused=True,independent_confirmation_required=True,automatic_promotion=False,
        limitation='Exploratory diagnostic after seeing these development outcomes; paired p is descriptive, not fresh confirmation. No official test labels or inference used.')
    save(result,out/'summary.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
