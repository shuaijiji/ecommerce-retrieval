#!/usr/bin/env python3
"""Compare fixed-reranker Top1/Top3 across different retriever candidate sets."""
import argparse
import itertools
import json
from pathlib import Path
from compare_reranker_predictions import summarize
from compare_embedding_recall import paired
from rerank_pipeline import digest, load_labels, load_rows, save


def compare_cases(cases, labels, expected=2000):
    metrics, flags, reference, protocol = {}, {}, None, None
    for name, candidates_path, prediction_path, config_paths in cases:
        if name in metrics:
            raise ValueError('duplicate arm name')
        rows=load_rows(candidates_path)
        candidates={r['photo_id']:r for r in rows}
        ids=sorted(candidates)
        if len(ids)!=expected or (reference is not None and ids!=reference):
            raise ValueError('query coverage differs between arms')
        reference=ids
        for row in rows:
            values=[c['item_id'] for c in row['candidates']]
            if not values or len(values)!=len(set(values)):
                raise ValueError('empty or duplicate candidate IDs')
        if not config_paths:
            raise ValueError('missing reranker provenance')
        config_queries=[]
        for path in config_paths:
            config=json.loads(path.read_text())
            candidate_file=path.parent.parent.parent/'shards'/name/(path.parent.name+'.jsonl')
            if config.get('candidate_sha256')!=digest(candidate_file):
                raise ValueError('reranker shard candidate provenance mismatch')
            config_queries.extend(r['photo_id'] for r in load_rows(candidate_file))
            common={k:v for k,v in config.items() if k!='candidate_sha256'}
            if not common.get('model_weights_sha256'):
                raise ValueError('missing reranker weight hashes')
            if protocol is not None and common!=protocol:
                raise ValueError('reranker model or scoring protocol differs')
            protocol=common
        if sorted(config_queries)!=ids:
            raise ValueError('reranker shards do not exactly cover candidate queries')
        metrics[name], top1=summarize(prediction_path,candidates,labels)
        predictions=load_rows(prediction_path)
        top3={r['photo_id']:labels[r['photo_id']] in [v['item_id'] for v in r['ranking'][:3]] for r in predictions}
        metrics[name].update(system_top3=sum(top3.values())/expected,
                             correct_top3=sum(top3.values()),candidate_sha256=digest(candidates_path),
                             mean_candidates=sum(len(r['candidates']) for r in rows)/expected)
        flags[name]={'top1':top1,'top3':top3}
    comparisons={}
    for base,current in itertools.combinations(metrics,2):
        comparisons[current+'_vs_'+base]={
            k:paired([flags[base][k][q] for q in reference],
                     [flags[current][k][q] for q in reference]) for k in ('top1','top3')}
    return {'queries':expected,'arms':metrics,'paired_comparisons':comparisons,
            'reranker_protocol':protocol,'selection_rule':'Top1 primary; Top3 secondary; no automatic promotion',
            'official_10k_used':False,'dynamic_top25_used':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True)
    a=p.parse_args()
    cases=[]
    for name in ('original','pilot100','train1k'):
        arm=a.output_root/name
        cases.append((name,a.output_root/'candidates'/(name+'.jsonl'),arm/'predictions_2000.jsonl',
                      sorted(arm.glob('worker_*/config.json'))))
    _,labels=load_labels(a.labels)
    report=compare_cases(cases,labels)
    report['label_sha256']=digest(a.labels)
    save(report,a.output_root/'comparison.json')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
