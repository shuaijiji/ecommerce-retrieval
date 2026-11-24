#!/usr/bin/env python3
"""CPU-only error budget from existing 2k retrieval and cached dynamic reranking."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from evaluate_incremental_top25_router import build_examples
from evaluate_dynamic_top25_router import split_ids
from rerank_pipeline import digest,load_labels,load_rows,save,write_rows
from rerank_dynamic_top25_incremental import should_expand
from compare_retriever_rerank import compare_cases

BUCKETS=('correct_top1','retrieval_miss_top100','truncated_between_top100_and_top25',
         'not_expanded_truth_only_in_top25_tail','rerank_truth_at_rank2_or3','rerank_truth_at_rank4_or_later')


def classify(truth,ids100,ids25,ranking):
    if ranking[0]==truth:
        return 'correct_top1'
    if truth not in ids100:
        return 'retrieval_miss_top100'
    if truth not in ids25:
        return 'truncated_between_top100_and_top25'
    if truth not in ranking:
        return 'not_expanded_truth_only_in_top25_tail'
    return 'rerank_truth_at_rank2_or3' if ranking.index(truth)<3 else 'rerank_truth_at_rank4_or_later'


def summarize_cases(cases):
    n=len(cases)
    if not n:
        raise ValueError('empty analysis split')
    counts=Counter(row['bucket'] for row in cases)
    if set(counts)-set(BUCKETS):
        raise ValueError('unknown error category')
    correct=counts['correct_top1']
    errors=n-correct
    return dict(count=n,top1=correct/n,top3=sum(row['truth_rank_final'] is not None and row['truth_rank_final']<=3 for row in cases)/n,
        error_count=errors,error_buckets={k:dict(count=counts[k],fraction_of_all=counts[k]/n,
            fraction_of_errors=counts[k]/errors if errors else 0) for k in BUCKETS if k!='correct_top1'},
        recall_top100=sum(row['hit_top100'] for row in cases)/n,
        recall_top25=sum(row['hit_top25'] for row in cases)/n,
        current_scored_candidate_recall=sum(row['truth_rank_final'] is not None for row in cases)/n,
        perfect_reranker_on_current_scored_candidates_upper_bound=sum(row['truth_rank_final'] is not None for row in cases)/n,
        expanded_count=sum(row['expanded'] for row in cases))


def build_cases(examples,retrieval_dir,labels,threshold):
    cases=[]
    for index,(q,example) in enumerate(examples.items(),1):
        truth=labels[q]
        routes={}
        ranks={}
        hashes={}
        for mode in ('vision','multimodal'):
            path=retrieval_dir/'results'/mode/(q+'.json')
            row=json.loads(path.read_text())
            ids=row['top_item_ids']
            scores=row['top_scores']
            if (row['photo_id']!=q or str(row['positive_item_id'])!=truth or len(ids)!=100 or
                    len(set(ids))!=100 or len(scores)!=100 or not all(math.isfinite(v) for v in scores) or
                    any(a<b for a,b in zip(scores,scores[1:]))):
                raise ValueError('retrieval coverage, labels, or scores invalid')
            routes[mode]=ids
            ranks[mode]=ids.index(truth)+1 if truth in ids else None
            hashes[mode]=digest(path)
        ids100=set(routes['vision'])|set(routes['multimodal'])
        ids25={v for ids in routes.values() for v in ids[:25]}
        ids10={v for ids in routes.values() for v in ids[:10]}
        if set(example['top10_ranking'])!=ids10 or set(example['full_top25_ranking'])!=ids25:
            raise ValueError('candidate truncation differs from saved retrieval')
        expand=should_expand(q,example['top1_score'],threshold['cutoff'],threshold['tie_probability'],threshold['tie_seed'])
        ranking=example['incremental_ranking'] if expand else example['top10_ranking']
        rank=ranking.index(truth)+1 if truth in ranking else None
        cases.append(dict(photo_id=q,truth=truth,predicted_top1=ranking[0],predicted_top3=ranking[:3],
            bucket=classify(truth,ids100,ids25,ranking),expanded=expand,top1_score=example['top1_score'],
            truth_rank_final=rank,truth_retrieval_ranks=ranks,hit_top100=truth in ids100,hit_top25=truth in ids25,
            hit_top10=truth in ids10,top10_top1=example['top10_ranking'][0],retrieval_hashes=hashes))
        if index%250==0 or index==len(examples):
            print(f'error budget reconstructed {index}/{len(examples)}',flush=True)
    return cases


def selected_cases(cases,limit=10,seed=20261124):
    selected=[]
    for bucket in BUCKETS[1:]:
        rows=[row for row in cases if row['bucket']==bucket]
        selected+=sorted(rows,key=lambda row:hashlib.sha256(f'{seed}:{bucket}:{row["photo_id"]}'.encode()).digest())[:limit]
    return selected


def enrich(selected,data):
    from prepare_reranker_sft import load_metadata,compose_video_text
    photos=load_metadata(data/'photo_info.jsonl','photo_id',{row['photo_id'] for row in selected})
    wanted={row['truth'] for row in selected}|{i for row in selected for i in row['predicted_top3']}
    items=load_metadata(data/'item_info.jsonl','item_id',wanted)
    output=[]
    for row in selected:
        photo=photos[row['photo_id']]
        details={i:dict(title=str(items[i].get('item_title') or '')[:500],image=items[i].get('item_image'))
                 for i in {row['truth'],*row['predicted_top3']}}
        output.append({**row,'review_status':'unreviewed; do not infer cause from missing text alone',
            'query_text':compose_video_text(photo),'frames':photo.get('photo_frames'),
            'has_ocr':bool(str(photo.get('photo_ocr') or '').strip()),
            'has_asr':bool(str(photo.get('photo_asr') or '').strip()),'items':details})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--sample-per-bucket',type=int,default=10)
    a=p.parse_args()
    if a.sample_per_bucket<1:
        raise ValueError('sample-per-bucket must be positive')
    root=a.root.resolve()
    base=root/'artifacts/train_v0'
    out=(a.output_dir or base/'error_budget_after_submission_v1').resolve()
    out.mkdir(parents=True,exist_ok=False)
    cached=base/'retriever_rerank_top10_7gpu_v1'
    expanded=base/'retriever_dynamic_top25_7gpu_v1'
    report=json.loads((expanded/'comparison.json').read_text())
    prereg=json.loads((expanded/'experiment_config.json').read_text())
    labels_path=base/'clean_labels.jsonl'
    if digest(labels_path)!=report['label_sha256']:
        raise ValueError('development labels changed')
    _,labels=load_labels(labels_path)
    summaries={}
    for name,folder in [('pilot100','embedding_pilot_eval_v1'),('train1k','embedding_eval_1k_v1')]:
        print(f'auditing {name}: cached provenance, retrieval and frozen routing',flush=True)
        arm=report['arms'][name]
        threshold=arm['router']['threshold']
        if threshold!=prereg['router_thresholds'][name]:
            raise ValueError('router provenance changed')
        files=[cached/'candidates'/(name+'.jsonl'),cached/name/'predictions_2000.jsonl',
               expanded/'candidates'/(name+'.jsonl'),expanded/name/'predictions_2000.jsonl']
        for path in files:
            if arm['input_hashes'].get(str(path))!=digest(path):
                raise ValueError('cached input changed from completed evaluation')
        protocols=[]
        for directory in (cached,expanded):
            checked=compare_cases([(name,directory/'candidates'/(name+'.jsonl'),directory/name/'predictions_2000.jsonl',
                            sorted((directory/name).glob('worker_*/config.json')))],labels)
            protocols.append(checked['reranker_protocol'])
        if protocols[0]!=protocols[1]:
            raise ValueError('cached and expanded reranker protocols differ')
        values=[{row['photo_id']:row for row in load_rows(path)} for path in files]
        examples,_=build_examples(*values,labels)
        cases=build_cases(examples,base/folder/'retrieval',labels,threshold)
        calibration,test=split_ids(examples,20261119)
        calset,testset=set(calibration),set(test)
        for row in cases:
            row['split']='calibration' if row['photo_id'] in calset else 'test'
        group={'all_2000':cases,'calibration':[r for r in cases if r['photo_id'] in calset],
               'test':[r for r in cases if r['photo_id'] in testset]}
        summaries[name]={key:summarize_cases(rows) for key,rows in group.items()}
        if (abs(summaries[name]['test']['top1']-arm['test_dynamic_incremental']['system_top1'])>1e-12 or
                abs(summaries[name]['test']['top3']-arm['test_dynamic_incremental']['system_top3'])>1e-12):
            raise ValueError('error budget does not reproduce completed dynamic evaluation')
        write_rows(cases,out/(name+'_cases.jsonl'))
        chosen=selected_cases(cases,a.sample_per_bucket)
        print(f'{name}: enriching {len(chosen)} review samples from training metadata',flush=True)
        write_rows(enrich(chosen,root/'data/train/all_data_package') if chosen else [],out/(name+'_review_samples.jsonl'))
    result=dict(arms=summaries,analysis_only=True,extra_gpu_inference=False,
        development_set_reused=True,independent_confirmation_required_for_future_model_changes=True,
        label_sha256=digest(labels_path),source_report_sha256=digest(expanded/'comparison.json'),
        interpretation='Mutually exclusive Top1 error buckets; distinguish retrieval coverage, truncation, skipped expansion, and scoring failures. No automatic training decision.')
    save(result,out/'summary.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
