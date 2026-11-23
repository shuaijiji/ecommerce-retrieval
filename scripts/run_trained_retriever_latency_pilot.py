#!/usr/bin/env python3
"""Run the fixed pilot100 route as real Top10 + conditional new-only scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from rerank_pipeline import digest, save


def load_fixed_route(report_path, experiment_path, model, data_root, top10, top25):
    report=json.loads(report_path.read_text())
    experiment=json.loads(experiment_path.read_text())
    if report['primary_arm']!='pilot100' or report['router_fixed_before_results']!={
            'policy':'low_top1_score','budget':.25,'seed':20261119}:
        raise ValueError('not the fixed pilot100 route')
    arm=report['arms']['pilot100']
    threshold=arm['router']['threshold']
    if not arm['test_top1_benefit_gate'] or threshold!=experiment['router_thresholds']['pilot100']:
        raise ValueError('offline benefit gate or pre-registered threshold mismatch')
    for path in (top10,top25):
        if arm['input_hashes'].get(str(path))!=digest(path):
            raise ValueError('candidate input changed from offline experiment')
    model_hashes={f.name:digest(f) for f in sorted(model.glob('*.safetensors'))}
    if not model_hashes or model_hashes!=experiment['model_weights']:
        raise ValueError('reranker weights differ from offline experiment')
    # Validate against an actual completed worker configuration.
    workers=sorted(experiment_path.parent.glob('pilot100/worker_*/config.json'))
    if not workers:
        raise ValueError('missing offline scoring provenance')
    configs=[json.loads(path.read_text()) for path in workers]
    model_config_hash=digest(model/'config.json')
    for cfg in configs:
        if (cfg['model_config_sha256']!=model_config_hash or cfg['max_length']!=10240
                or cfg['model_weights_sha256']!=model_hashes or cfg['batch_size']!=4 or cfg['image_size']!=512
                or cfg['instruction']!=__import__('rerank_multimodal').INSTRUCTION):
            raise ValueError('offline scoring protocol differs')
    for key,file in [('photo_sha256','photo_info.jsonl'),('item_sha256','item_info.jsonl')]:
        metadata_hash=digest(data_root/file)
        if any(c[key]!=metadata_hash for c in configs):
            raise ValueError('dataset metadata differs from offline experiment')
    if any(c['preprocessing_sha256']!=digest(Path(__file__).with_name('retrieval_common.py')) for c in configs):
        raise ValueError('preprocessing code differs from offline experiment')
    return threshold,configs[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    p.add_argument('--gpu',default='1',help='one idle physical GPU; serial latency measurement')
    a=p.parse_args()
    root=a.root.resolve()
    base=root/'artifacts/train_v0'
    offline=base/'retriever_dynamic_top25_7gpu_v1'
    top10=base/'retriever_rerank_top10_7gpu_v1/candidates/pilot100.jsonl'
    top25=offline/'candidates/pilot100.jsonl'
    model=base/'reranker_listwise_train_v1_merged/checkpoint-500-merged'
    data=root/'data/train/all_data_package'
    threshold,protocol=load_fixed_route(offline/'comparison.json',offline/'experiment_config.json',model,data,top10,top25)
    if digest(root/'artifacts/train_v0/clean_labels.jsonl')!=json.loads((offline/'comparison.json').read_text())['label_sha256']:
        raise ValueError('labels changed since offline experiment')
    if protocol['upstream_sha256']!=digest(root/'Qwen3-VL-Embedding/src/models/qwen3_vl_reranker.py'):
        raise ValueError('upstream reranker code differs')
    import torch
    import transformers
    if protocol['torch']!=torch.__version__ or protocol['transformers']!=transformers.__version__:
        raise ValueError('runtime package versions differ from offline scoring environment')
    row=subprocess.check_output(['nvidia-smi','--id='+a.gpu,'--query-gpu=uuid,memory.used',
                                  '--format=csv,noheader,nounits'],text=True).strip().split(',')
    if int(row[1])>1024:
        raise RuntimeError(f'GPU {a.gpu} occupied; select an idle card')
    output=base/'retriever_dynamic_latency_pilot100_v1'
    if output.exists():
        raise ValueError('latency output already exists; inspect existing results before rerunning')
    output.mkdir()
    save({'offline_report_sha256':digest(offline/'comparison.json'),
          'offline_experiment_sha256':digest(offline/'experiment_config.json'),
          'threshold':threshold,'physical_gpu':a.gpu,'gpu_uuid':row[0].strip(),
          'protocol':'100 fixed-seed queries; Top10 scoring then conditional new-only scoring; no threshold adjustment'},
         output/'source.json')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=row[0].strip(),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    command=['python','-u',str(root/'scripts/rerank_dynamic_top25_incremental.py'),
             '--top10-candidates',str(top10),'--top25-candidates',str(top25),
             '--labels',str(base/'clean_labels.jsonl'),'--data-root',str(data),
             '--repo',str(root/'Qwen3-VL-Embedding'),'--model',str(model),
             '--output-dir',str(output),'--num-queries','100','--batch-size','4',
             '--query-seed','20261105','--score-cutoff',str(threshold['cutoff']),
             '--tie-probability',str(threshold['tie_probability']),
             '--tie-seed',str(threshold['tie_seed'])]
    with (output/'run.log').open('w') as log:
        subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print((output/'pilot_summary.json').read_text(),flush=True)


if __name__=='__main__':
    main()
