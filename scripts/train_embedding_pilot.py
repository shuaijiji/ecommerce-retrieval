#!/usr/bin/env python3
"""Single-GPU LoRA contrastive pipeline pilot; no implicit in-batch negatives."""
import argparse
import json
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from retrieval_common import (load_embedder, load_video_frames, compose_video_text,
                              QUERY_INSTRUCTION, sha256_file, atomic_json_dump, model_weight_hashes)


def encode_trainable(embedder, inputs, visual_key):
    conversations = [embedder.format_model_input(**row) for row in inputs]
    processed = embedder._preprocess_inputs(conversations)
    visual = processed.get(visual_key)
    grid_key = 'video_grid_thw' if visual_key == 'pixel_values_videos' else 'image_grid_thw'
    grid = processed.get(grid_key)
    if visual is None or visual.numel() == 0 or grid is None or len(grid) != len(inputs):
        raise RuntimeError('missing or partial visual preprocessing; refusing NULL fallback')
    processed = {k: v.to(embedder.model.device) if hasattr(v, 'to') else v
                 for k, v in processed.items()}
    # The upstream wrapper's forward is no_grad. Call the underlying module directly.
    output = embedder.model(**processed)
    embeddings = F.normalize(embedder._pooling_last(
        output.last_hidden_state, processed['attention_mask']).float(), dim=-1)
    if embeddings.shape != (len(inputs), 2048) or not torch.isfinite(embeddings).all():
        raise RuntimeError('invalid embeddings')
    return embeddings


def objective(embedder, row, root, temperature):
    query = {'instruction': QUERY_INSTRUCTION,
             'video': load_video_frames(row['query'], root, 512),
             'text': compose_video_text(row['query'])}
    q = encode_trainable(embedder, [query], 'pixel_values_videos')
    # Item images and full titles use the same protocol as build_item_index.py.
    docs = [{'image': str(root / i['item_image']), 'text': i.get('item_title') or ''}
            for i in row['items']]
    d = encode_trainable(embedder, docs, 'pixel_values')
    scores = q @ d.T
    loss = F.cross_entropy(scores / temperature, torch.zeros(1, dtype=torch.long, device=q.device))
    margin = scores[0, 0] - scores[0, 1:].max()
    return loss, margin


def measure(embedder, rows, root, temperature):
    embedder.model.eval()
    losses, margins = [], []
    with torch.no_grad():
        for row in rows:
            loss, margin = objective(embedder, row, root, temperature)
            losses.append(float(loss))
            margins.append(float(margin))
    if not all(__import__('math').isfinite(v) for v in losses + margins):
        raise RuntimeError('nonfinite evaluation')
    return {'loss': sum(losses) / len(losses), 'hardest_negative_margin': sum(margins) / len(margins)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('repo', 'model', 'dataset', 'output-dir'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--seed', type=int, default=20261119)
    a = p.parse_args()
    if a.steps < 1 or a.lr <= 0 or a.temperature <= 0:
        raise ValueError('invalid hyperparameters')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('expose exactly one idle GPU via CUDA_VISIBLE_DEVICES=GPU-UUID')
    manifest = json.loads((a.dataset / 'manifest.json').read_text())
    samples = a.dataset / 'samples.jsonl'
    if not manifest['complete'] or sha256_file(samples) != manifest['samples_sha256']:
        raise ValueError('dataset manifest/hash mismatch')
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    if not 100 <= len(rows) <= 500:
        raise ValueError('require 100–500 training queries')
    root = Path(manifest['data_root'])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    free, total = torch.cuda.mem_get_info()
    if total - free > 1024 ** 3:
        raise RuntimeError('GPU already occupied; select an idle GPU')
    embedder = load_embedder(a.repo, a.model, 512)
    targets = [name for name, module in embedder.model.named_modules()
               if 'language_model' in name and name.endswith(('.q_proj', '.v_proj'))
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise RuntimeError('no language attention LoRA targets found')
    embedder.model = get_peft_model(embedder.model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=targets, bias='none'))
    embedder.model.config.use_cache = False
    embedder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = [p for p in embedder.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=a.lr)
    atomic_json_dump({'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                     'dataset_manifest_sha256': sha256_file(a.dataset / 'manifest.json'),
                     'upstream_source_sha256': sha256_file(a.repo / 'src/models/qwen3_vl_embedding.py'),
                     'torch_version': torch.__version__, 'targets': targets,
                     'base_model_weight_hashes': model_weight_hashes(a.model),
                     'trainable_parameters': sum(p.numel() for p in trainable),
                     'query_mode': 'multimodal', 'in_batch_negatives': False}, a.output_dir / 'config.json')
    before = measure(embedder, rows, root, a.temperature)
    atomic_json_dump(before, a.output_dir / 'before.json')
    order = list(range(len(rows)))
    random.shuffle(order)
    embedder.model.train()
    torch.cuda.reset_peak_memory_stats()
    with (a.output_dir / 'steps.jsonl').open('w') as log:
        for step in range(a.steps):
            if step and step % len(order) == 0:
                random.shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            loss, margin = objective(embedder, rows[order[step % len(order)]], root, a.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            if norm <= 0:
                raise RuntimeError('zero trainable gradient')
            optimizer.step()
            record = {'step': step + 1, 'loss': float(loss.detach()),
                      'margin': float(margin.detach()), 'gradient_norm': float(norm)}
            log.write(json.dumps(record) + '\n')
            log.flush()
            print(json.dumps(record), flush=True)
    after = measure(embedder, rows, root, a.temperature)
    passed = after['loss'] < before['loss'] and after['hardest_negative_margin'] > before['hardest_negative_margin']
    embedder.model.save_pretrained(a.output_dir / 'adapter')
    atomic_json_dump({'before': before, 'after': after, 'functional_gate': passed,
                     'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                     'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
                     'generalization_claim': False}, a.output_dir / 'summary.json')
    if not passed:
        raise RuntimeError('functional gate failed; retain artifacts, do not expand')
    merged = embedder.model.merge_and_unload()
    merged.save_pretrained(a.output_dir / 'merged')
    embedder.processor.save_pretrained(a.output_dir / 'merged')


if __name__ == '__main__':
    main()
