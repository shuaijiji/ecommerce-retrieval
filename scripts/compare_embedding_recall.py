#!/usr/bin/env python3
"""Paired recall comparison on the frozen item-disjoint query selection."""
import argparse
import json
import math
from pathlib import Path
from prepare_reranker_sft import write_json_atomic


def paired(before, after):
    gained = sum(not a and b for a, b in zip(before, after))
    lost = sum(a and not b for a, b in zip(before, after))
    n = gained + lost
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n) if n else 1.0
    return {'baseline': sum(before)/len(before), 'trained': sum(after)/len(after),
            'gained': gained, 'lost': lost, 'net': gained-lost, 'paired_p_two_sided': p}


def compare(base, trained, expected=2000):
    ids = json.loads((base/'selected_photo_ids.json').read_text())
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError('wrong baseline selection')
    if json.loads((trained/'selected_photo_ids.json').read_text()) != ids:
        raise ValueError('query selections differ')
    configs = [json.loads((d/'config.json').read_text()) for d in (base, trained)]
    for key in ('photo_info_sha256', 'label_sha256', 'query_instruction', 'text_budgets', 'frame_preprocessing'):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f'evaluation protocol changed: {key}')
    pools = [[(c['item_ids_sha256'], c['item_count'], c['item_info_sha256'], c['item_input'], c['image_size'])
              for c in cfg['item_index_configs']] for cfg in configs]
    if pools[0] != pools[1]:
        raise ValueError('item pools or item preprocessing differ')
    hits = [{(m,k): [] for m in ('vision','multimodal','union') for k in (10,25,50,100)} for _ in range(2)]
    for photo in ids:
        truth = None
        for idx, directory in enumerate((base, trained)):
            routes = {}
            for mode in ('vision', 'multimodal'):
                row = json.loads((directory/'results'/mode/(photo+'.json')).read_text())
                if str(row['photo_id']) != photo:
                    raise ValueError('result query mismatch')
                positive = str(row['positive_item_id'])
                if truth is not None and truth != positive:
                    raise ValueError('result labels differ')
                truth = positive
                routes[mode] = [str(x) for x in row['top_item_ids']]
                if len(routes[mode]) != 100 or len(set(routes[mode])) != 100:
                    raise ValueError('need complete unique Top-100')
            for k in (10,25,50,100):
                for mode in routes:
                    hits[idx][mode,k].append(truth in routes[mode][:k])
                hits[idx]['union',k].append(any(truth in v[:k] for v in routes.values()))
    metrics = {f'{m}@{k}': paired(hits[0][m,k], hits[1][m,k]) for m,k in hits[0]}
    main = metrics['union@100']
    if abs(main['baseline'] - 0.833) > 1e-9:
        raise ValueError('baseline Top-100 union is not the frozen 83.3%')
    return {'queries': expected, 'metrics': metrics,
            'expansion_gate': main['trained'] > .833 and main['paired_p_two_sided'] < .05,
            'gate_definition': 'union@100 > 83.3% and paired two-sided exact p < 0.05',
            'official_10k_used': False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','trained','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    result=compare(a.baseline,a.trained)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_json_atomic(result,a.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
