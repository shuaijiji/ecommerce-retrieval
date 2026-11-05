#!/usr/bin/env python3
"""Evaluate fixed video queries against the complete item index."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from retrieval_common import (
    QUERY_INSTRUCTION,
    TEXT_BUDGETS,
    atomic_json_dump,
    compose_video_text,
    encode_checked,
    load_embedder,
    load_video_frames,
    read_jsonl,
    sha256_file,
)


MODES = ("text", "vision", "multimodal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/code/ecommerce-retrieval/Qwen3-VL-Embedding"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/code/ecommerce-retrieval/models/Qwen3-VL-Embedding-2B"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/code/ecommerce-retrieval/data/eval/test_small_package"),
    )
    parser.add_argument(
        "--item-index",
        type=Path,
        default=Path("/code/ecommerce-retrieval/artifacts/zero_shot_v0/items"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/code/ecommerce-retrieval/artifacts/zero_shot_v0/eval100"),
    )
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return parser.parse_args()


def load_item_index(index_dir: Path) -> tuple[list[str], torch.Tensor, dict]:
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("item index manifest is not complete")
    config = json.loads((index_dir / manifest["config"]).read_text(encoding="utf-8"))
    ids: list[str] = []
    matrices: list[torch.Tensor] = []
    for name in manifest["shards"]:
        payload = torch.load(index_dir / name, map_location="cpu", weights_only=True)
        ids.extend(str(value) for value in payload["item_ids"])
        matrices.append(payload["embeddings"])
    matrix = torch.cat(matrices, dim=0)
    if len(ids) != config["item_count"] or matrix.shape != (len(ids), 2048):
        raise RuntimeError("assembled item index does not match its config")
    return ids, matrix, config


def choose_photo_ids(
    labels: list[dict],
    photos: dict[str, dict],
    count: int,
    seed: int,
) -> list[str]:
    if count > len(labels):
        raise ValueError("num-queries exceeds label count")
    rng = random.Random(seed)
    by_frame_count: dict[int, list[str]] = defaultdict(list)
    all_ids: list[str] = []
    for label in labels:
        photo_id = str(label["photo_id"])
        all_ids.append(photo_id)
        by_frame_count[len(photos[photo_id].get("photo_frames") or [])].append(photo_id)

    # Cover every observed frame count once, then fill randomly. This is a
    # functional pilot rather than the final unbiased 10k evaluation.
    if count < len(by_frame_count):
        raise ValueError(
            f"num-queries must be at least {len(by_frame_count)} "
            "to cover every observed frame count"
        )
    selected = [rng.choice(values) for _, values in sorted(by_frame_count.items())]
    selected_set = set(selected)
    remaining = [photo_id for photo_id in all_ids if photo_id not in selected_set]
    selected.extend(rng.sample(remaining, count - len(selected)))
    rng.shuffle(selected)
    return selected


def result_path(output_dir: Path, mode: str, photo_id: str) -> Path:
    return output_dir / "results" / mode / f"{photo_id}.json"


def compute_summary(
    output_dir: Path,
    selected_ids: list[str],
    modes: list[str],
) -> dict:
    summary: dict[str, dict] = {}
    for mode in modes:
        rows = []
        for photo_id in selected_ids:
            path = result_path(output_dir, mode, photo_id)
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        if not rows:
            continue
        summary[mode] = {
            "completed": len(rows),
            "top1": sum(row["rank"] <= 1 for row in rows) / len(rows),
            "recall_at_10": sum(row["rank"] <= 10 for row in rows) / len(rows),
            "recall_at_50": sum(row["rank"] <= 50 for row in rows) / len(rows),
            "recall_at_100": sum(row["rank"] <= 100 for row in rows) / len(rows),
            "mrr": sum(1.0 / row["rank"] for row in rows) / len(rows),
            "mean_query_seconds": sum(row["query_seconds"] for row in rows) / len(rows),
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    photos = read_jsonl(args.data_root / "photo_info.jsonl", "photo_id")
    assert isinstance(photos, dict)
    with (args.data_root / "label.json").open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    label_by_photo = {str(row["photo_id"]): str(row["item_id"]) for row in labels}

    index_config = json.loads(
        (args.item_index / "config.json").read_text(encoding="utf-8")
    )
    model_config_sha256 = sha256_file(args.model / "config.json")
    if model_config_sha256 != index_config["model_config_sha256"]:
        raise RuntimeError(
            "query model differs from the model used to build the item index"
        )
    run_config = {
        "schema_version": 1,
        "kind": "qwen3-vl-full-pool-pilot",
        "model_path": str(args.model),
        "model_config_sha256": model_config_sha256,
        "photo_info_sha256": sha256_file(args.data_root / "photo_info.jsonl"),
        "label_sha256": sha256_file(args.data_root / "label.json"),
        "item_index_config": index_config,
        "num_queries": args.num_queries,
        "seed": args.seed,
        "top_k": args.top_k,
        "query_instruction": QUERY_INSTRUCTION,
        "text_budgets": [list(value) for value in TEXT_BUDGETS],
        "frame_preprocessing": (
            f"aspect-preserving {index_config['image_size']}-square letterbox; "
            "pad odd T"
        ),
    }
    run_config_path = args.output_dir / "config.json"
    if run_config_path.exists():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise RuntimeError(
                "existing evaluation configuration differs; use a new output directory"
            )
    else:
        atomic_json_dump(run_config, run_config_path)

    selection_path = args.output_dir / "selected_photo_ids.json"
    if selection_path.exists():
        selected_ids = json.loads(selection_path.read_text(encoding="utf-8"))
        if len(selected_ids) != args.num_queries:
            raise RuntimeError(
                "existing query selection has a different size; use a new output directory"
            )
    else:
        selected_ids = choose_photo_ids(labels, photos, args.num_queries, args.seed)
        atomic_json_dump(selected_ids, selection_path)

    print("loading complete item index...")
    item_ids, item_matrix, index_config = load_item_index(args.item_index)
    item_to_position = {item_id: position for position, item_id in enumerate(item_ids)}
    if len(item_to_position) != len(item_ids):
        raise RuntimeError("item index contains duplicate IDs")
    image_size = int(index_config["image_size"])
    item_matrix = item_matrix.to(device="cuda", dtype=torch.float32)
    print("item matrix:", tuple(item_matrix.shape), item_matrix.device)

    print("loading query model...")
    model = load_embedder(args.repo, args.model, image_size)
    top_k = min(args.top_k, len(item_ids))

    for mode in args.modes:
        print(f"\n=== mode: {mode} ===")
        for number, photo_id in enumerate(selected_ids, 1):
            path = result_path(args.output_dir, mode, photo_id)
            if path.exists():
                print(f"resume {mode} {number}/{len(selected_ids)}: {photo_id}")
                continue

            photo = photos[photo_id]
            text = compose_video_text(photo)
            original_frame_count = len(photo.get("photo_frames") or [])
            frames = None
            if mode in {"vision", "multimodal"}:
                frames = load_video_frames(photo, args.data_root, image_size)

            query_input = {"instruction": QUERY_INSTRUCTION}
            required_visual_key = None
            if mode in {"text", "multimodal"}:
                query_input["text"] = text
            if mode in {"vision", "multimodal"}:
                query_input["video"] = frames
                required_visual_key = "pixel_values_videos"

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            query = encode_checked(model, [query_input], required_visual_key)
            torch.cuda.synchronize()
            query_seconds = time.perf_counter() - started

            scores = (query.float() @ item_matrix.T).squeeze(0)
            positive_id = label_by_photo[photo_id]
            positive_position = item_to_position[positive_id]
            positive_score = scores[positive_position]
            rank = int((scores > positive_score).sum().item()) + 1
            top_scores, top_positions = torch.topk(scores, k=top_k)

            result = {
                "mode": mode,
                "photo_id": photo_id,
                "positive_item_id": positive_id,
                "positive_score": float(positive_score.item()),
                "rank": rank,
                "query_seconds": query_seconds,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "text_chars": len(text),
                "original_frame_count": original_frame_count,
                "encoded_frame_count": len(frames) if frames is not None else 0,
                "top_item_ids": [item_ids[index] for index in top_positions.tolist()],
                "top_scores": [float(value) for value in top_scores.tolist()],
            }
            atomic_json_dump(result, path)
            print(
                f"{number}/{len(selected_ids)} photo={photo_id} "
                f"rank={rank} time={query_seconds:.3f}s"
            )

        summary = compute_summary(args.output_dir, selected_ids, args.modes)
        atomic_json_dump(summary, args.output_dir / "summary.json")
        print(json.dumps(summary.get(mode, {}), ensure_ascii=False, indent=2))

    print("\nevaluation complete:", args.output_dir)


if __name__ == "__main__":
    main()
