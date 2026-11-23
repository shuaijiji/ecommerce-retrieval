#!/usr/bin/env python3
"""GPU worker for official retrieval and dynamic reranking; never opens labels."""
import argparse
import json
from pathlib import Path
import time
from official_submission_common import metadata,candidate_ids,validate_retrieval,validate_result
from rerank_pipeline import digest,save
from rerank_dynamic_top25_incremental import should_expand


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--worker',type=int,required=True)
    p.add_argument('--phase',choices=('retrieval','rerank'),required=True)
    p.add_argument('--pilot',action='store_true')
    a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    out=Path(plan['output_dir'])
    data=Path(plan['data_root'])
    photos=metadata(data/'photo_info.jsonl','photo_id')
    items=metadata(data/'item_info.jsonl','item_id')
    item_set=set(items)
    assigned=sorted(photos)[a.worker::plan['workers']]
    if a.pilot:
        assigned=[q for q in assigned if q in set(plan['pilot_ids'])]
    if not assigned:
        return
    import torch
    import transformers
    from retrieval_common import (load_embedder,encode_checked,load_video_frames,compose_video_text,
                                  QUERY_INSTRUCTION,letterbox)
    if torch.__version__!=plan['torch_version'] or transformers.__version__!=plan['transformers_version']:
        raise ValueError('worker environment differs from frozen plan')
    if a.phase=='retrieval':
        from evaluate_retrieval import load_item_index
        index=out/'items'
        ids,matrix,index_config=load_item_index(index)
        if set(ids)!=item_set or len(ids)!=len(item_set):
            raise ValueError('index item pool differs from official metadata')
        if index_config.get('model_weight_hashes')!=plan['embedding_model_weights']:
            raise ValueError('embedding/index weights differ')
        matrix=matrix.to('cuda',dtype=torch.float32)
        if not torch.isfinite(matrix).all():
            raise ValueError('nonfinite item embeddings')
        model=load_embedder(Path(plan['upstream_repo']),Path(plan['embedding_model']),512)
        for number,q in enumerate(assigned,1):
            path=out/'retrieval'/f'{q}.json'
            if path.exists():
                validate_retrieval(json.loads(path.read_text()),q,item_set)
                continue
            start=time.perf_counter()
            frames=load_video_frames(photos[q],data,512)
            if len(frames)>6:
                raise ValueError('unexpected frame count')
            text=compose_video_text(photos[q])
            row={'photo_id':q}
            for mode in ('vision','multimodal'):
                query={'instruction':QUERY_INSTRUCTION,'video':frames}
                if mode=='multimodal':
                    query['text']=text
                embedding=encode_checked(model,[query],'pixel_values_videos')
                scores,positions=torch.topk((embedding.float()@matrix.T).squeeze(0),k=25)
                row[mode]={'item_ids':[ids[i] for i in positions.tolist()],'scores':scores.tolist()}
            torch.cuda.synchronize()
            row['seconds']=time.perf_counter()-start
            validate_retrieval(row,q,item_set)
            save(row,path)
            print(f'retrieval worker={a.worker} {number}/{len(assigned)} photo={q} seconds={row["seconds"]:.3f}',flush=True)
        return
    import sys
    sys.path.insert(0,plan['upstream_repo'])
    from src.models.qwen3_vl_reranker import Qwen3VLReranker
    from rerank_multimodal import INSTRUCTION
    model=Qwen3VLReranker(model_name_or_path=plan['reranker_model'],dtype=torch.bfloat16,
                          attn_implementation='sdpa',max_length=10240,max_pixels=512**2,
                          total_pixels=6*512**2,max_frames=6)
    def score(values,text,frames):
        ranking=[]
        for start in range(0,len(values),4):
            chunk=values[start:start+4]
            pairs=[model.format_mm_instruction(query_text=text,query_video=frames,
                    doc_text=str(items[i].get('item_title') or '')[:500],
                    doc_image=letterbox(data/items[i]['item_image'],512),instruction=INSTRUCTION) for i in chunk]
            original=model.truncate_tokens_optimized
            def reject(tokens,max_length,special_tokens):
                if len(tokens)>max_length:
                    raise ValueError('token budget exceeded; no silent truncation')
                return tokens
            model.truncate_tokens_optimized=reject
            try:
                inputs=model.tokenize(pairs)
            finally:
                model.truncate_tokens_optimized=original
            for key,grid in [('pixel_values','image_grid_thw'),('pixel_values_videos','video_grid_thw')]:
                if inputs.get(key) is None or inputs[key].numel()==0 or len(inputs[grid])!=len(chunk):
                    raise ValueError('missing or partial visual preprocessing')
            inputs={k:v.to(model.device) if hasattr(v,'to') else v for k,v in inputs.items()}
            with torch.inference_mode():
                scores=model.compute_scores(inputs)
            if len(scores)!=len(chunk):
                raise ValueError('reranker batch length mismatch')
            ranking.extend(dict(item_id=i,score=float(s)) for i,s in zip(chunk,scores))
        ranking.sort(key=lambda r:(-r['score'],r['item_id']))
        from official_submission_common import validate_ranking
        validate_ranking(ranking,values)
        return ranking
    threshold=plan['threshold']
    for number,q in enumerate(assigned,1):
        retrieved=out/'retrieval'/f'{q}.json'
        row=json.loads(retrieved.read_text())
        validate_retrieval(row,q,item_set)
        source_sha=digest(retrieved)
        target=out/'predictions'/f'{q}.json'
        if target.exists():
            validate_result(json.loads(target.read_text()),row,threshold,source_sha)
            continue
        torch.cuda.synchronize()
        start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        frames=load_video_frames(photos[q],data,512)
        text=compose_video_text(photos[q])
        ids10,ids25=candidate_ids(row,10),candidate_ids(row,25)
        ranking10=score(ids10,text,frames)
        expand=should_expand(q,ranking10[0]['score'],threshold['cutoff'],threshold['tie_probability'],threshold['tie_seed'])
        new=[i for i in ids25 if i not in set(ids10)] if expand else []
        ranking=sorted(ranking10+(score(new,text,frames) if new else []),key=lambda r:(-r['score'],r['item_id']))
        torch.cuda.synchronize()
        result=dict(photo_id=q,ranking=ranking,top10_ranking=ranking10,expanded=expand,
                    scored_new_items=len(new),seconds=time.perf_counter()-start,
                    retrieval_sha256=source_sha,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
        validate_result(result,row,threshold,source_sha)
        save(result,target)
        print(f'rerank worker={a.worker} {number}/{len(assigned)} photo={q} expand={int(expand)} seconds={result["seconds"]:.3f}',flush=True)


if __name__=='__main__':
    main()
