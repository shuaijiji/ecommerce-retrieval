#!/usr/bin/env python3
"""Fixed 25% low-score router on calibration; report test Top1/Top3 for two retrievers."""
import argparse
import json
from pathlib import Path
from compare_retriever_rerank import compare_cases
from evaluate_incremental_top25_router import build_examples, evaluate
from evaluate_dynamic_top25_router import split_ids, fit_top1_score_threshold
from evaluate_incremental_top25_router import _validate_ranking
from rerank_pipeline import digest, load_labels, load_rows, save


def freeze_threshold(candidate_path, prediction_path, expected=2000):
    candidates={r['photo_id']:r for r in load_rows(candidate_path)}
    predictions={r['photo_id']:r for r in load_rows(prediction_path)}
    if len(candidates)!=expected or set(candidates)!=set(predictions):
        raise ValueError('cached Top-10 query coverage invalid')
    examples={}
    for q,row in predictions.items():
        ids={c['item_id'] for c in candidates[q]['candidates']}
        _validate_ranking(q,row['ranking'],ids,'cached Top-10')
        examples[q]={'top1_score':float(row['ranking'][0]['score'])}
    calibration,_=split_ids(examples,20261119)
    threshold=fit_top1_score_threshold(calibration,examples,.25,tie_seed=20261119)
    threshold['feature']='cached independent Top-10 top1 score'
    return threshold


def analyze_arm(name, cached, expanded, labels):
    reports=[]
    for directory in (cached,expanded):
        arm=directory/name
        cases=[(name,directory/'candidates'/(name+'.jsonl'),arm/'predictions_2000.jsonl',
                sorted(arm.glob('worker_*/config.json')))]
        reports.append(compare_cases(cases,labels))
    if reports[0]['reranker_protocol']!=reports[1]['reranker_protocol']:
        raise ValueError('cached Top-10 and Top-25 scoring protocols differ')
    def index(path):
        return {r['photo_id']:r for r in load_rows(path)}
    candidates10=cached/'candidates'/(name+'.jsonl')
    candidates25=expanded/'candidates'/(name+'.jsonl')
    prediction10=cached/name/'predictions_2000.jsonl'
    prediction25=expanded/name/'predictions_2000.jsonl'
    examples,differences=build_examples(index(candidates10),index(prediction10),
                                       index(candidates25),index(prediction25),labels)
    result=evaluate(examples,differences,seed=20261119,budget=.25)
    # Features only: fit does not inspect truth or expanded outcomes.
    result['router']['threshold']['feature']='cached independent Top-10 top1 score'
    dynamic=result['test_dynamic_incremental']
    full=result['test_all_incremental_top25']
    dynamic['retained_full_top25_net_gain']=(dynamic['net_correct_vs_top10']/full['net_correct_vs_top10']
                                           if full['net_correct_vs_top10']>0 else None)
    result['test_top1_benefit_gate']=(dynamic['net_correct_vs_top10']>0 and
                                       dynamic['paired_exact_two_sided_p']<.05)
    result['all_2000_top10']=reports[0]['arms'][name]
    result['all_2000_full_rerun_top25']=reports[1]['arms'][name]
    result['input_hashes']={str(path):digest(path) for path in
                            (candidates10,candidates25,prediction10,prediction25)}
    result['split_hashes']={k:__import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest()
                           for k,ids in zip(('calibration','test'),split_ids(examples,20261119))}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--top10-root',type=Path,required=True)
    p.add_argument('--top25-root',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=('original','pilot100','train1k'),default=['pilot100','train1k'])
    a=p.parse_args()
    cfg=json.loads((a.top25_root/'experiment_config.json').read_text())
    if (cfg['top_k_per_route']!=25 or cfg['router_budget']!=.25 or cfg['router_seed']!=20261119
            or set(cfg['sources'])!=set(a.arms)):
        raise ValueError('experiment is not the fixed 25% Top-25 protocol')
    if cfg['cached_top10_root']!=str(a.top10_root.resolve()):
        raise ValueError('cached Top-10 source directory changed')
    for name in a.arms:
        expected=cfg['cached_top10_hashes'][name]
        if (expected['candidates']!=digest(a.top10_root/'candidates'/(name+'.jsonl')) or
            expected['predictions']!=digest(a.top10_root/name/'predictions_2000.jsonl')):
            raise ValueError('cached Top-10 scores/candidates changed since registration')
    _,labels=load_labels(a.labels)
    if digest(a.labels)!=json.loads((a.top10_root/'comparison.json').read_text())['label_sha256']:
        raise ValueError('evaluation labels changed from Top-10 comparison')
    reports={name:analyze_arm(name,a.top10_root,a.top25_root,labels) for name in a.arms}
    for name,result in reports.items():
        if result['router']['threshold']!=cfg['router_thresholds'][name]:
            raise ValueError('router threshold differs from pre-GPU frozen configuration')
    if len({json.dumps(r['split_hashes'],sort_keys=True) for r in reports.values()})!=1:
        raise ValueError('calibration/test split differs across arms')
    result={'primary_arm':'pilot100','challenger_arm':'train1k','arms':reports,
            'router_fixed_before_results':{'policy':'low_top1_score','budget':.25,'seed':20261119},
            'test_results_used_for_current_router_fit':False,
            'label_sha256':digest(a.labels),'official_10k_used':False,
            'limitation':'offline cached-score emulation; full Top-25 supplies new-item scores; no end-to-end latency claim',
            'promotion':'report test Top1/Top3 and costs; no automatic promotion or test-driven retuning'}
    save(result,a.top25_root/'comparison.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
