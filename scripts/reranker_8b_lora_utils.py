"""CPU-only validation and summaries for a fixed 8B listwise LoRA Pilot."""
import hashlib
import json
import math
from pathlib import Path

SEED=20261125
STEPS=50


def identify_group(row,canonical,duplicate_groups,forbidden_dev_items):
    videos=row.get('videos')
    if not isinstance(videos,list) or len(videos)!=1 or not isinstance(videos[0],list) or not videos[0]:
        raise ValueError('expected one video frame list')
    ids={Path(p).parent.name for p in videos[0]}
    if len(ids)!=1:
        raise ValueError('video frames belong to different queries')
    q=next(iter(ids))
    if not q.isdigit() or q not in canonical:
        raise ValueError('unknown training query')
    if len(row.get('positive_messages',[]))!=1 or len(row.get('negative_messages',[]))!=7:
        raise ValueError('expected one positive and seven negative messages')
    images=row.get('positive_images',[])+row.get('negative_images',[])
    if len(images)!=8 or any(not isinstance(v,list) or len(v)!=1 for v in images):
        raise ValueError('expected one image per candidate')
    items=[Path(v[0]).stem for v in images]
    source=canonical[q]
    expected=[source['positive_item_id']]+[n['item_id'] for n in source['negatives']]
    if items!=expected or len(set(items))!=8:
        raise ValueError('candidate identity/order differs from canonical audited group')
    if set(items[1:]) & set(duplicate_groups.get(items[0],{items[0]})):
        raise ValueError('known duplicate positive in training negatives')
    if set(items)&forbidden_dev_items:
        raise ValueError('training/monitor group contains development positive or its known duplicate')
    return q


def select_groups(rows,count,seed=SEED):
    if not 1<=count<=len(rows):
        raise ValueError('invalid subset size')
    return sorted(rows,key=lambda q:hashlib.sha256(f'{seed}:{q}'.encode()).digest())[:count]


def training_command(plan,phase):
    if phase not in ('smoke','train'):
        raise ValueError('unknown training phase')
    out=Path(plan['output']);smoke=phase=='smoke'
    steps=5 if smoke else STEPS
    # Local checkpoint architecture is shared by chat, embedding and reranker registrations.
    # Pin Swift's reranker loader/template instead of relying on directory-name inference.
    return ['swift','sft','--model',plan['model'],'--model_type','qwen3_vl_reranker','--task_type','generative_reranker',
        '--loss_type','listwise_reranker','--label_names','labels','--tuner_type','lora',
        '--lora_rank','8','--lora_alpha','32','--learning_rate','1e-5','--target_modules','all-linear',
        '--dataset',str(out/'data'/('smoke_train.jsonl' if smoke else 'train.jsonl')),
        '--val_dataset',str(out/'data'/('smoke_monitor.jsonl' if smoke else 'monitor.jsonl')),
        '--split_dataset_ratio','0','--attn_impl','flash_attn','--padding_free','true',
        '--torch_dtype','bfloat16','--load_from_cache_file','false','--strict','true',
        '--truncation_strategy','delete','--eval_strategy','steps','--eval_on_start','true',
        '--save_strategy','steps','--save_steps',str(steps if smoke else 25),
        '--eval_steps',str(steps if smoke else 25),'--save_total_limit','2',
        '--logging_steps','1','--max_steps',str(steps),'--max_length','4096',
        '--per_device_train_batch_size','1','--per_device_eval_batch_size','1',
        '--gradient_accumulation_steps','1','--gradient_checkpointing','true',
        '--dataloader_num_workers','2','--dataset_num_proc','2','--warmup_ratio','0.05',
        '--seed',str(SEED),'--data_seed',str(SEED),'--report_to','none',
        '--output_dir',str(out/phase/'model')]


def analyze_history(state,expected_steps):
    history=state.get('log_history',[])
    losses=[r['loss'] for r in history if 'loss' in r]
    gradients=[r['grad_norm'] for r in history if 'grad_norm' in r]
    values=[];evaluations={}
    for row in history:
        for k,v in row.items():
            if k=='grad_norm' or 'loss' in k.lower() or 'mrr' in k.lower() or 'ndcg' in k.lower():
                if not isinstance(v,(int,float)) or not math.isfinite(v):
                    raise ValueError(f'non-finite or nonnumeric training metric: {k}={v}')
                values.append(v)
        if any(k.startswith('eval_') for k in row):
            step=row.get('step')
            if step is None:
                raise ValueError('evaluation step missing')
            evaluations.setdefault(step,{}).update({k:v for k,v in row.items() if k.startswith('eval_')})
    before=evaluations.get(0,{})
    after=evaluations.get(expected_steps,{})
    ranking_keys=sorted(k for k in before.keys()&after.keys() if 'mrr' in k.lower() or 'ndcg' in k.lower())
    functional=(state.get('global_step')==expected_steps and bool(losses) and bool(gradients) and
                'eval_loss' in before and 'eval_loss' in after and bool(values) and any(v>0 for v in gradients))
    direction=(functional and after['eval_loss']<before['eval_loss'] and
               bool(ranking_keys) and any(after[k]>before[k] for k in ranking_keys))
    return dict(completed_steps=state.get('global_step'),functional_gate=functional,direction_gate=direction,
        train_loss_first=losses[0] if losses else None,train_loss_last=losses[-1] if losses else None,
        monitor_before=before,monitor_after=after,ranking_metric_keys=ranking_keys,
        missing_ranking_metrics=not bool(ranking_keys),generalization_claim=False,
        gate_definition='functional: completed step, finite loss, finite nonzero gradient, initial/final eval; direction: eval_loss lower AND at least one recorded MRR/NDCG higher')
