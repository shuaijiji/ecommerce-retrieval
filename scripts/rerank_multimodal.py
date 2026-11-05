#!/usr/bin/env python3
"""Pointwise Qwen3-VL-Reranker inference, without loading labels."""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

from rerank_pipeline import digest, load_rows, save, write_rows

INSTRUCTION = '判断候选商品是否为视频主要讲解的同一款商品。结合视频画面和文本与商品主图及标题，区分品牌、型号、外观和规格；仅同类别或配件相关不足以判定为同款。'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/code/ecommerce-retrieval/data/eval/test_small_package'))
    p.add_argument('--repo', type=Path, default=Path('/code/ecommerce-retrieval/Qwen3-VL-Embedding'))
    p.add_argument('--model', type=Path, default=Path('/code/ecommerce-retrieval/models/Qwen3-VL-Reranker-2B'))
    p.add_argument('--num-queries', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--image-size', type=int, default=512)
    args = p.parse_args()
    import torch
    from retrieval_common import compose_video_text, load_video_frames, letterbox, read_jsonl, TEXT_BUDGETS
    sys.path.insert(0, str(args.repo))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    rows = load_rows(args.candidates)
    random.Random(args.seed).shuffle(rows)
    if not 100 <= args.num_queries <= len(rows) or args.batch_size < 1:
        raise ValueError('require 100 <= num-queries <= available and positive batch size')
    rows = rows[:args.num_queries]
    config = {'candidate_sha256': digest(args.candidates), 'model': str(args.model.resolve()),
              'model_config_sha256': digest(args.model / 'config.json'),
              'model_weights_sha256': {f.name: digest(f) for f in sorted(args.model.glob('*.safetensors'))},
              'upstream_sha256': digest(args.repo / 'src/models/qwen3_vl_reranker.py'),
              'runner_sha256': digest(Path(__file__)), 'preprocessing_sha256': digest(Path(__file__).with_name('retrieval_common.py')),
              'photo_sha256': digest(args.data_root / 'photo_info.jsonl'),
              'item_sha256': digest(args.data_root / 'item_info.jsonl'),
              'batch_size': args.batch_size, 'seed': args.seed, 'image_size': args.image_size,
              'instruction': INSTRUCTION, 'max_length': 10240, 'text_budgets': list(map(list, TEXT_BUDGETS)),
              'title_chars': 500, 'torch': torch.__version__}
    if not config['model_weights_sha256']:
        raise ValueError('no local safetensors weights')
    import transformers
    config['transformers'] = transformers.__version__
    cp = args.output_dir / 'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('configuration changed; use a new output directory')
    save(config, cp)
    gate_path = args.output_dir / 'pilot_pass.json'
    if args.num_queries > 500:
        if not gate_path.exists():
            raise ValueError('run 100-500 query pilot before full inference')
        gate = json.loads(gate_path.read_text())
        if gate['config_sha256'] != digest(cp):
            raise ValueError('pilot configuration mismatch')
    photos = read_jsonl(args.data_root / 'photo_info.jsonl', 'photo_id')
    items = read_jsonl(args.data_root / 'item_info.jsonl', 'item_id')
    model = Qwen3VLReranker(model_name_or_path=str(args.model), dtype=torch.bfloat16,
                            attn_implementation='sdpa', max_length=10240,
                            max_pixels=args.image_size ** 2,
                            total_pixels=6 * args.image_size ** 2, max_frames=6)
    predictions = []
    for number, row in enumerate(rows, 1):
        pid = row['photo_id']
        if not pid.isdigit():
            raise ValueError('unsafe photo ID filename')
        dest = args.output_dir / 'results' / (pid + '.json')
        if dest.exists():
            out = json.loads(dest.read_text())
            if out['photo_id'] != pid or {c['item_id'] for c in out['ranking']} != {c['item_id'] for c in row['candidates']}:
                raise ValueError('invalid resume result')
            predictions.append(out)
            continue
        torch.cuda.synchronize()
        start = time.perf_counter()
        photo = photos[pid]
        frames = load_video_frames(photo, args.data_root, args.image_size)
        if len(frames) > 6:
            raise ValueError('more than 6 frames: explicit new protocol required')
        text = compose_video_text(photo)
        ranking, token_counts = [], []
        torch.cuda.reset_peak_memory_stats()
        for offset in range(0, len(row['candidates']), args.batch_size):
            chunk = row['candidates'][offset:offset + args.batch_size]
            pairs = []
            for c in chunk:
                item = items[c['item_id']]
                pairs.append(model.format_mm_instruction(
                    query_text=text, query_video=frames,
                    doc_text=str(item.get('item_title') or '')[:500],
                    doc_image=letterbox(args.data_root / item['item_image'], args.image_size),
                    instruction=INSTRUCTION))
            # Detect upstream truncation instead of silently losing candidate text.
            original_truncate = model.truncate_tokens_optimized
            def reject_truncation(tokens, max_length, special_tokens):
                if len(tokens) > max_length:
                    raise ValueError('token budget exceeded; refusing silent text truncation')
                return tokens
            model.truncate_tokens_optimized = reject_truncation
            try:
                inputs = model.tokenize(pairs)
            finally:
                model.truncate_tokens_optimized = original_truncate
            for key in ('pixel_values', 'pixel_values_videos'):
                if inputs.get(key) is None or inputs[key].numel() == 0:
                    raise ValueError(f'missing {key}; upstream visual fallback rejected')
            if len(inputs['image_grid_thw']) != len(chunk) or len(inputs['video_grid_thw']) != len(chunk):
                raise ValueError('image/video count does not match pair count')
            token_counts.extend(inputs['attention_mask'].sum(1).tolist())
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.inference_mode():
                scores = model.compute_scores(inputs)
            if len(scores) != len(chunk) or not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
                raise ValueError('invalid relevance scores')
            ranking.extend({'item_id': c['item_id'], 'score': float(s)} for c, s in zip(chunk, scores))
        ranking.sort(key=lambda c: (-c['score'], c['item_id']))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        out = {'photo_id': pid, 'ranking': ranking, 'seconds': seconds,
               'original_frames': len(photo['photo_frames']), 'encoded_frames': len(frames),
               'frame_paths': photo['photo_frames'], 'query_text': text, 'pair_token_counts': token_counts,
               'peak_memory_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
               'timing_scope': 'frame/item I/O + preprocessing + all pair scoring + sorting; excludes model load/result write'}
        save(out, dest)
        predictions.append(out)
        print(f'{number}/{len(rows)} photo={pid} candidates={len(ranking)} seconds={seconds:.3f}', flush=True)
    write_rows(predictions, args.output_dir / f'predictions_{args.num_queries}.jsonl')
    if args.num_queries <= 500:
        flat_scores = [c['score'] for r in predictions for c in r['ranking']]
        if max(flat_scores) == min(flat_scores):
            raise ValueError('all pilot relevance scores identical')
        gate = {'count': len(predictions), 'config_sha256': digest(cp),
                'mean_seconds_per_video': sum(r['seconds'] for r in predictions) / len(predictions),
                'estimated_10000_serial_hours': sum(r['seconds'] for r in predictions) / len(predictions) * 10000 / 3600,
                'checks': 'both visual tensors/counts, finite scores, no truncation, full ranking; no accuracy tuning'}
        save(gate, gate_path)
        print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
