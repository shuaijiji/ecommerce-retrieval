"""Strict, label-free metadata/result checks shared by official inference and export."""
import hashlib
import json
import math
from pathlib import Path
from rerank_pipeline import digest
from rerank_dynamic_top25_incremental import should_expand


def identifier(value):
    if isinstance(value,bool) or not isinstance(value,(str,int)) or not str(value).isdigit():
        raise ValueError(f'invalid ID: {value!r}')
    return str(value)


def metadata(path,key):
    result={}
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row=json.loads(line)
            value=identifier(row[key])
            if value in result:
                raise ValueError(f'duplicate {key}: {value}')
            row[key]=value
            result[value]=row
    if not result:
        raise ValueError(f'empty metadata: {path}')
    return result


def media_paths(data,photos,items):
    paths=set()
    for row in photos.values():
        frames=row.get('photo_frames')
        if not isinstance(frames,list) or not 1<=len(frames)<=6:
            raise ValueError(f'invalid frames: {row["photo_id"]}')
        paths.update(frames)
    for row in items.values():
        if not isinstance(row.get('item_image'),str) or not row['item_image']:
            raise ValueError(f'missing item image: {row["item_id"]}')
        paths.add(row['item_image'])
    root=Path(data).resolve()
    for relative in paths:
        if not isinstance(relative,str):
            raise ValueError('media path must be string')
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'missing or out-of-root media: {relative}')
    return sorted(str((root/p).resolve()) for p in paths)


def select_pilot(ids,count=100,seed=20261123):
    return sorted(ids,key=lambda q:(hashlib.sha256(f'{seed}:{q}'.encode()).hexdigest(),q))[:count]


def validate_retrieval(row,photo_id,item_ids):
    if row['photo_id']!=photo_id:
        raise ValueError('retrieval photo mismatch')
    for mode in ('vision','multimodal'):
        values=row[mode]['item_ids']
        scores=row[mode]['scores']
        if len(values)!=25 or len(set(values))!=25 or set(values)-item_ids or len(scores)!=25:
            raise ValueError('invalid Top-25 retrieval')
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in scores) or any(a<b for a,b in zip(scores,scores[1:])):
            raise ValueError('invalid retrieval scores')


def candidate_ids(row,k):
    return sorted({v for mode in ('vision','multimodal') for v in row[mode]['item_ids'][:k]})


def validate_ranking(ranking,expected):
    ids=[r['item_id'] for r in ranking]
    if not ranking or len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('incomplete/duplicate candidate ranking')
    scores=[r['score'] for r in ranking]
    if not all(isinstance(s,(int,float)) and math.isfinite(s) and 0<=s<=1 for s in scores):
        raise ValueError('invalid reranker scores')
    if ranking!=sorted(ranking,key=lambda r:(-r['score'],r['item_id'])):
        raise ValueError('reranker scores or tie order invalid')


def validate_result(result,retrieval,threshold,source_sha):
    if result['photo_id']!=retrieval['photo_id'] or result['retrieval_sha256']!=source_sha:
        raise ValueError('result retrieval provenance mismatch')
    ids10=candidate_ids(retrieval,10)
    ids25=candidate_ids(retrieval,25)
    validate_ranking(result['top10_ranking'],ids10)
    score=result['top10_ranking'][0]['score']
    expand=should_expand(result['photo_id'],score,threshold['cutoff'],threshold['tie_probability'],threshold['tie_seed'])
    if result['expanded']!=expand:
        raise ValueError('saved routing decision differs from fixed threshold')
    validate_ranking(result['ranking'],ids25 if expand else ids10)
    final={r['item_id']:r['score'] for r in result['ranking']}
    if any(final[r['item_id']]!=r['score'] for r in result['top10_ranking']):
        raise ValueError('cached Top-10 scores changed in final ranking')
    if result['scored_new_items']!=(len(ids25)-len(ids10) if expand else 0):
        raise ValueError('incorrect new-candidate scoring count')
    if not math.isfinite(result['seconds']) or result['seconds']<0:
        raise ValueError('invalid inference time')


def assert_plan_files(plan):
    for path,expected in plan['file_hashes'].items():
        if digest(path)!=expected:
            raise ValueError(f'frozen input/code changed: {path}')
    for key in ('embedding_model','reranker_model'):
        folder=Path(plan[key])
        actual={p.name:digest(p) for p in sorted(folder.glob('*.safetensors'))}
        if not actual or actual!=plan[key+'_weights']:
            raise ValueError(f'frozen model weights changed: {folder}')
