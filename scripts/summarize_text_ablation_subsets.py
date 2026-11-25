#!/usr/bin/env python3
"""Read-only analysis of changed/unchanged text subsets from a completed A/B run."""
import argparse
import json
from pathlib import Path
from compare_embedding_recall import paired
from rerank_pipeline import digest,load_rows,save


def summarize_subset(ids,cases):
    result={'count':len(ids)}
    for k in (1,3):
        result[f'top{k}']=paired([cases[q]['truth'] in cases[q]['raw_top3'][:k] for q in ids],
                               [cases[q]['truth'] in cases[q]['dedup_top3'][:k] for q in ids]) if ids else None
    result['top3_order_changed']=sum(cases[q]['raw_top3']!=cases[q]['dedup_top3'] for q in ids)
    return result


def analyze(cases,preview,split,comparison):
    ids=set(cases)
    if ids!=set(preview):
        raise ValueError('case/text query coverage differs')
    calibration,test=split['calibration'],split['test']
    if (len(set(calibration+test))!=len(calibration)+len(test) or set(calibration+test)!=ids):
        raise ValueError('split coverage differs or overlaps')
    for row in cases.values():
        for field in ('raw_top3','dedup_top3'):
            if len(row[field])!=3 or len(set(row[field]))!=3:
                raise ValueError('invalid Top3 ranking')
    changed={q for q in ids if preview[q]['raw']!=preview[q]['dedup']}
    if len(changed)!=comparison['text_changed']:
        raise ValueError('changed-text count differs from completed experiment')
    groups={'all_2000':sorted(ids),'calibration':calibration,'test':test}
    result={}
    for name,queries in groups.items():
        whole=summarize_subset(queries,cases)
        for k in (1,3):
            expected=comparison['metrics'][name][f'top{k}']
            for field in ('baseline','trained','gained','lost','net','paired_p_two_sided'):
                if abs(whole[f'top{k}'][field]-expected[field])>1e-12:
                    raise ValueError('paired cases do not reproduce completed metrics')
        result[name]=dict(all=whole,
            changed=summarize_subset([q for q in queries if q in changed],cases),
            unchanged=summarize_subset([q for q in queries if q not in changed],cases))
    return dict(subsets=result,changed_count=len(changed),unchanged_count=len(ids-changed),
        unchanged_text_top1_prediction_changes=[q for q in sorted(ids-changed) if cases[q]['raw_top3'][0]!=cases[q]['dedup_top3'][0]],
        unchanged_text_top3_order_changes=[q for q in sorted(ids-changed) if cases[q]['raw_top3']!=cases[q]['dedup_top3']],
        interpretation='Separate treatment changes from prediction drift on unchanged text. Empty subsets have null metrics. Reused development set; no model promotion.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-root',type=Path,default=Path('/code/ecommerce-retrieval/artifacts/train_v0/reranker_text_ablation_v1'))
    a=p.parse_args();out=a.experiment_root.resolve()
    comparison=json.loads((out/'comparison.json').read_text())
    plan=json.loads((out/'plan.json').read_text())
    if digest(out/'plan.json')!=comparison['plan_sha256']:
        raise ValueError('plan differs from completed report')
    for name in ('text_preview.jsonl','split.json'):
        if plan['generated'].get(str(out/name))!=digest(out/name):
            raise ValueError('frozen text/split changed')
    cases={r['photo_id']:r for r in load_rows(out/'paired_cases.jsonl')}
    preview={r['photo_id']:r for r in load_rows(out/'text_preview.jsonl')}
    if len(cases)!=plan['count']:
        raise ValueError('incomplete cases')
    result=analyze(cases,preview,json.loads((out/'split.json').read_text()),comparison)
    result.update(source_hashes={name:digest(out/name) for name in
        ('plan.json','comparison.json','paired_cases.jsonl','text_preview.jsonl','split.json')},extra_gpu_inference=False)
    save(result,out/'changed_text_summary.json')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
