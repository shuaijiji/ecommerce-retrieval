#!/usr/bin/env python3
"""Download a pinned official 8B reranker; record model and tokenizer file hashes."""
import argparse
import json
from pathlib import Path
from rerank_pipeline import digest,save

MODEL_ID='Qwen/Qwen3-VL-Reranker-8B'
REVISION='b212dc8c91a8164aef1ea2de9c1a867611e75c04'
FOLDER='Qwen3-VL-Reranker-8B-b212dc8'
RECEIPT='download_receipt.json'
REQUIRED=('config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json',
          'video_preprocessor_config.json','chat_template.jinja','model.safetensors.index.json')


def model_files(folder):
    for name in REQUIRED:
        if not (folder/name).is_file():
            raise ValueError(f'incomplete model download: {name}')
    config=json.loads((folder/'config.json').read_text())
    text=config.get('text_config',{})
    if config.get('model_type')!='qwen3_vl' or text.get('hidden_size')!=4096 or text.get('num_hidden_layers')!=36:
        raise ValueError('configuration differs from expected Qwen3-VL 8B architecture')
    index=json.loads((folder/'model.safetensors.index.json').read_text())
    shards=set(index['weight_map'].values())
    if len(shards)!=4 or any(Path(name).name!=name for name in shards):
        raise ValueError('expected four local weight shards')
    if {p.name for p in folder.glob('*.safetensors')}!=shards:
        raise ValueError('missing or extra weight shards')
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name!=RECEIPT and
                  p.suffix in ('.json','.txt','.jinja','.safetensors'))


def validate_download(folder):
    receipt=json.loads((folder/RECEIPT).read_text())
    if (receipt['model_id'],receipt['revision'])!=(MODEL_ID,REVISION):
        raise ValueError('model receipt has wrong repository/revision')
    actual={p.name:digest(p) for p in model_files(folder)}
    if actual!=receipt['files']:
        raise ValueError('model/tokenizer files differ from download receipt')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/code/ecommerce-retrieval'))
    a=p.parse_args();folder=a.root.resolve()/'models'/FOLDER
    if (folder/RECEIPT).exists():
        validate_download(folder)
        print(f'Pinned model already downloaded and hashes verified: {folder}',flush=True)
        return
    from huggingface_hub import snapshot_download
    print(f'Downloading {MODEL_ID}@{REVISION} to {folder}; no GPU inference.',flush=True)
    snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_dir=str(folder),max_workers=4,
                      allow_patterns=['*.json','*.txt','*.jinja','*.safetensors'])
    print('Checking model shards and hashing local files...',flush=True)
    files=model_files(folder)
    save(dict(model_id=MODEL_ID,revision=REVISION,files={p.name:digest(p) for p in files},
              source='huggingface_hub.snapshot_download with immutable revision; no custom model code downloaded'),folder/RECEIPT)
    print(f'Download complete: {folder}',flush=True)


if __name__=='__main__':
    main()
