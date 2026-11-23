#!/usr/bin/env python3
"""Latency Pilot for cached Top-10 scoring plus conditional Top-25 expansion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

from rerank_pipeline import digest, load_labels, load_rows, save, write_rows
from rerank_multimodal import INSTRUCTION


def hash_fraction(photo_id: str, seed: int) -> float:
    value = int(hashlib.sha256(f"{seed}:router-tie:{photo_id}".encode()).hexdigest(), 16)
    return value / (1 << 256)


def should_expand(photo_id: str, score: float, cutoff: float, tie_probability: float, tie_seed: int) -> bool:
    if score < cutoff:
        return True
    if score > cutoff:
        return False
    return hash_fraction(photo_id, tie_seed) < tie_probability


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def exact_p_value(rescued: int, harmed: int) -> float:
    discordant = rescued + harmed
    if not discordant:
        return 1.0
    smaller = min(rescued, harmed)
    return min(
        1.0,
        2 * sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2 ** discordant),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top10-candidates", type=Path, required=True)
    parser.add_argument("--top25-candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/code/ecommerce-retrieval/data/train/all_data_package")
    )
    parser.add_argument(
        "--repo", type=Path, default=Path("/code/ecommerce-retrieval/Qwen3-VL-Embedding")
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--query-seed", type=int, default=20261105)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--score-cutoff", type=float, default=.85546875)
    parser.add_argument("--tie-probability", type=float, default=.5)
    parser.add_argument("--tie-seed", type=int, default=20261119)
    args = parser.parse_args()
    if not 100 <= args.num_queries <= 500:
        raise ValueError("latency Pilot requires 100 <= num-queries <= 500")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if not 0 <= args.tie_probability <= 1:
        raise ValueError("tie probability must be between zero and one")

    import torch
    from retrieval_common import (
        TEXT_BUDGETS,
        compose_video_text,
        letterbox,
        load_video_frames,
        read_jsonl,
    )

    sys.path.insert(0, str(args.repo))
    from src.models.qwen3_vl_reranker import Qwen3VLReranker

    rows10 = load_rows(args.top10_candidates)
    rows25 = {row["photo_id"]: row for row in load_rows(args.top25_candidates)}
    if {row["photo_id"] for row in rows10} != set(rows25):
        raise ValueError("Top-10 and Top-25 query sets differ")
    random.Random(args.query_seed).shuffle(rows10)
    rows10 = rows10[: args.num_queries]
    _, labels = load_labels(args.labels)
    if {row["photo_id"] for row in rows10} - set(labels):
        raise ValueError("selected queries missing labels")

    import transformers

    config = {
        "objective": "dynamic_top25_incremental_latency_pilot",
        "top10_candidate_sha256": digest(args.top10_candidates),
        "top25_candidate_sha256": digest(args.top25_candidates),
        "label_sha256": digest(args.labels),
        "labels_used_for_routing": False,
        "model": str(args.model.resolve()),
        "model_config_sha256": digest(args.model / "config.json"),
        "model_weights_sha256": {
            path.name: digest(path) for path in sorted(args.model.glob("*.safetensors"))
        },
        "upstream_sha256": digest(args.repo / "src/models/qwen3_vl_reranker.py"),
        "runner_sha256": digest(Path(__file__)),
        "preprocessing_sha256": digest(Path(__file__).with_name("retrieval_common.py")),
        "photo_sha256": digest(args.data_root / "photo_info.jsonl"),
        "item_sha256": digest(args.data_root / "item_info.jsonl"),
        "batch_size": args.batch_size,
        "query_seed": args.query_seed,
        "num_queries": args.num_queries,
        "image_size": args.image_size,
        "score_cutoff": args.score_cutoff,
        "tie_probability": args.tie_probability,
        "tie_seed": args.tie_seed,
        "instruction": INSTRUCTION,
        "max_length": 10240,
        "text_budgets": list(map(list, TEXT_BUDGETS)),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if not config["model_weights_sha256"]:
        raise ValueError("no local safetensors weights")
    config_path = args.output_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("configuration changed; use a new output directory")
    save(config, config_path)

    photos = read_jsonl(args.data_root / "photo_info.jsonl", "photo_id")
    items = read_jsonl(args.data_root / "item_info.jsonl", "item_id")
    model = Qwen3VLReranker(
        model_name_or_path=str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_length=10240,
        max_pixels=args.image_size ** 2,
        total_pixels=6 * args.image_size ** 2,
        max_frames=6,
    )

    def score_candidates(candidates: list[dict], text: str, frames: list) -> tuple[list[dict], list[int]]:
        scored, token_counts = [], []
        for offset in range(0, len(candidates), args.batch_size):
            chunk = candidates[offset : offset + args.batch_size]
            pairs = []
            for candidate in chunk:
                item = items[candidate["item_id"]]
                pairs.append(
                    model.format_mm_instruction(
                        query_text=text,
                        query_video=frames,
                        doc_text=str(item.get("item_title") or "")[:500],
                        doc_image=letterbox(args.data_root / item["item_image"], args.image_size),
                        instruction=INSTRUCTION,
                    )
                )
            original_truncate = model.truncate_tokens_optimized

            def reject_truncation(tokens, max_length, special_tokens):
                if len(tokens) > max_length:
                    raise ValueError("token budget exceeded; refusing silent text truncation")
                return tokens

            model.truncate_tokens_optimized = reject_truncation
            try:
                inputs = model.tokenize(pairs)
            finally:
                model.truncate_tokens_optimized = original_truncate
            for key in ("pixel_values", "pixel_values_videos"):
                if inputs.get(key) is None or inputs[key].numel() == 0:
                    raise ValueError(f"missing {key}; upstream visual fallback rejected")
            if len(inputs["image_grid_thw"]) != len(chunk) or len(inputs["video_grid_thw"]) != len(chunk):
                raise ValueError("image/video count does not match pair count")
            token_counts.extend(inputs["attention_mask"].sum(1).tolist())
            inputs = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                scores = model.compute_scores(inputs)
            if len(scores) != len(chunk) or not all(
                math.isfinite(score) and 0 <= score <= 1 for score in scores
            ):
                raise ValueError("invalid relevance scores")
            scored.extend(
                {"item_id": candidate["item_id"], "score": float(score)}
                for candidate, score in zip(chunk, scores)
            )
        scored.sort(key=lambda row: (-row["score"], row["item_id"]))
        return scored, token_counts

    predictions = []
    for number, row10 in enumerate(rows10, 1):
        photo_id = row10["photo_id"]
        if not photo_id.isdigit():
            raise ValueError("unsafe photo ID filename")
        row25 = rows25[photo_id]
        ids10 = {candidate["item_id"] for candidate in row10["candidates"]}
        ids25 = {candidate["item_id"] for candidate in row25["candidates"]}
        if not ids10 < ids25:
            raise ValueError(f"Top-10 must be a strict subset of Top-25: {photo_id}")
        destination = args.output_dir / "results" / f"{photo_id}.json"
        if destination.exists():
            result = json.loads(destination.read_text())
            if result["photo_id"] != photo_id:
                raise ValueError("invalid resumed photo ID")
            predictions.append(result)
            continue

        torch.cuda.synchronize()
        total_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        photo = photos[photo_id]
        frames = load_video_frames(photo, args.data_root, args.image_size)
        if len(frames) > 6:
            raise ValueError("more than 6 frames: explicit new protocol required")
        text = compose_video_text(photo)
        ranking10, tokens10 = score_candidates(row10["candidates"], text, frames)
        torch.cuda.synchronize()
        top10_seconds = time.perf_counter() - total_start
        expand = should_expand(
            photo_id,
            ranking10[0]["score"],
            args.score_cutoff,
            args.tie_probability,
            args.tie_seed,
        )

        incremental_start = time.perf_counter()
        new_candidates = [
            candidate for candidate in row25["candidates"] if candidate["item_id"] not in ids10
        ]
        if expand:
            new_ranking, new_tokens = score_candidates(new_candidates, text, frames)
            ranking = ranking10 + new_ranking
            ranking.sort(key=lambda row: (-row["score"], row["item_id"]))
        else:
            new_ranking, new_tokens, ranking = [], [], ranking10
        torch.cuda.synchronize()
        incremental_seconds = time.perf_counter() - incremental_start if expand else 0.0
        total_seconds = time.perf_counter() - total_start
        truth = labels[photo_id]
        result = {
            "photo_id": photo_id,
            "ranking": ranking,
            "expanded": expand,
            "top10_top1_score": ranking10[0]["score"],
            "top10_top1_item_id": ranking10[0]["item_id"],
            "top10_correct": ranking10[0]["item_id"] == truth,
            "final_correct": ranking[0]["item_id"] == truth,
            "top10_top3_correct": truth in [r["item_id"] for r in ranking10[:3]],
            "final_top3_correct": truth in [r["item_id"] for r in ranking[:3]],
            "top10_candidate_count": len(ids10),
            "new_candidate_count": len(new_candidates) if expand else 0,
            "final_candidate_count": len(ranking),
            "top10_seconds": top10_seconds,
            "incremental_seconds": incremental_seconds,
            "seconds": total_seconds,
            "top10_pair_token_counts": tokens10,
            "new_pair_token_counts": new_tokens,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "timing_scope": "single video load + cached Top-10 scoring + conditional new-candidate scoring",
        }
        save(result, destination)
        predictions.append(result)
        print(
            f"{number}/{len(rows10)} photo={photo_id} expand={int(expand)} "
            f"candidates={len(ranking)} top10={top10_seconds:.3f}s "
            f"incremental={incremental_seconds:.3f}s total={total_seconds:.3f}s",
            flush=True,
        )

    write_rows(predictions, args.output_dir / f"predictions_{args.num_queries}.jsonl")
    total_times = [row["seconds"] for row in predictions]
    top10_times = [row["top10_seconds"] for row in predictions]
    incremental_times = [row["incremental_seconds"] for row in predictions]
    expanded_rows = [row for row in predictions if row["expanded"]]
    expanded_incremental_times = [row["incremental_seconds"] for row in expanded_rows]
    rescued = sum(not row["top10_correct"] and row["final_correct"] for row in predictions)
    harmed = sum(row["top10_correct"] and not row["final_correct"] for row in predictions)
    summary = {
        "objective": "dynamic_top25_incremental_latency_pilot",
        "count": len(predictions),
        "expanded_count": len(expanded_rows),
        "expanded_rate": len(expanded_rows) / len(predictions),
        "top10_correct": sum(row["top10_correct"] for row in predictions),
        "top10_system_top1": sum(row["top10_correct"] for row in predictions) / len(predictions),
        "final_correct": sum(row["final_correct"] for row in predictions),
        "final_system_top1": sum(row["final_correct"] for row in predictions) / len(predictions),
        "top10_system_top3": sum(row["top10_top3_correct"] for row in predictions) / len(predictions),
        "final_system_top3": sum(row["final_top3_correct"] for row in predictions) / len(predictions),
        "top3_rescued": sum(not row["top10_top3_correct"] and row["final_top3_correct"] for row in predictions),
        "top3_harmed": sum(row["top10_top3_correct"] and not row["final_top3_correct"] for row in predictions),
        "rescued_from_top10": rescued,
        "harmed_vs_top10": harmed,
        "paired_exact_two_sided_p": exact_p_value(rescued, harmed),
        "mean_final_candidates": sum(row["final_candidate_count"] for row in predictions) / len(predictions),
        "mean_top10_seconds": sum(top10_times) / len(top10_times),
        "p95_top10_seconds": percentile(top10_times, .95),
        "mean_incremental_seconds_all_queries": sum(incremental_times) / len(incremental_times),
        "mean_incremental_seconds_expanded": (
            sum(expanded_incremental_times) / len(expanded_incremental_times)
            if expanded_rows
            else 0.0
        ),
        "p95_incremental_seconds_expanded": (
            percentile(expanded_incremental_times, .95) if expanded_incremental_times else 0.0
        ),
        "mean_total_seconds": sum(total_times) / len(total_times),
        "p50_total_seconds": percentile(total_times, .50),
        "p95_total_seconds": percentile(total_times, .95),
        "p99_total_seconds": percentile(total_times, .99),
        "latency_overhead_vs_top10_mean": sum(total_times) / sum(top10_times) - 1,
        "peak_memory_gib": max(row["peak_memory_gib"] for row in predictions),
        "timing_excludes": ["embedding query encoding", "item retrieval", "model loading"],
        "accuracy_warning": "100-query engineering sample, not a new accuracy selection set",
        "config_sha256": digest(config_path),
        "prediction_sha256": digest(args.output_dir / f"predictions_{args.num_queries}.jsonl"),
    }
    save(summary, args.output_dir / "pilot_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
