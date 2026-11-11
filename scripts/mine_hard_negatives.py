#!/usr/bin/env python3
"""Mine resumable vision/multimodal hard negatives from the frozen train index."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import torch

from retrieval_common import (
    QUERY_INSTRUCTION,
    TEXT_BUDGETS,
    atomic_json_dump,
    compose_video_text,
    encode_checked,
    load_embedder,
    load_video_frames,
    sha256_file,
)


MODES = ("vision", "multimodal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--photo-ids", type=Path, required=True)
    parser.add_argument("--item-index", type=Path, required=True)
    parser.add_argument("--duplicate-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        help="use only the first N photo IDs (pilot only; use a fresh output directory)",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="validate every expected result shard and publish manifest without a model",
    )
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values:
        raise ValueError(f"no IDs in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate IDs in {path}")
    return values


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            yield line_number, row


def load_rows_for_ids(path: Path, key: str, wanted: set[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line_number, row in iter_jsonl(path):
        value = str(row[key])
        if value not in wanted:
            continue
        if value in rows:
            raise ValueError(f"duplicate {key}={value} at {path}:{line_number}")
        rows[value] = row
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"{len(missing)} requested {key}s missing from {path}: {sorted(missing)[:5]}")
    return rows


def load_positive_items(path: Path, wanted: set[str]) -> dict[str, str]:
    positives: dict[str, str] = {}
    for line_number, row in iter_jsonl(path):
        photo_id = str(row["photo_id"])
        if photo_id not in wanted:
            continue
        item_id = str(row["item_id"])
        existing = positives.get(photo_id)
        if existing is not None and existing != item_id:
            raise ValueError(f"conflicting label for photo {photo_id} at {path}:{line_number}")
        positives[photo_id] = item_id
    missing = wanted - positives.keys()
    if missing:
        raise ValueError(f"{len(missing)} requested photos missing labels: {sorted(missing)[:5]}")
    return positives


def load_duplicate_groups(path: Path) -> tuple[dict[str, tuple[str, frozenset[str]]], int]:
    item_to_group: dict[str, tuple[str, frozenset[str]]] = {}
    group_count = 0
    for line_number, row in iter_jsonl(path):
        group_id = str(row.get("proposed_group_id") or "")
        members = row.get("members")
        if not group_id or not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"invalid duplicate group at {path}:{line_number}")
        item_ids = frozenset(str(member["item_id"]) for member in members)
        if len(item_ids) != len(members):
            raise ValueError(f"duplicate member in group {group_id}")
        overlap = item_ids & item_to_group.keys()
        if overlap:
            raise ValueError(f"items occur in multiple duplicate groups: {sorted(overlap)[:5]}")
        for item_id in item_ids:
            item_to_group[item_id] = (group_id, item_ids)
        group_count += 1
    if not group_count:
        raise ValueError(f"no duplicate groups in {path}")
    return item_to_group, group_count


def load_item_index(index_dir: Path) -> tuple[list[str], torch.Tensor, dict, dict]:
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("item index manifest is not complete")
    config = json.loads((index_dir / manifest["config"]).read_text(encoding="utf-8"))
    ids: list[str] = []
    matrices: list[torch.Tensor] = []
    for name in manifest["shards"]:
        payload = torch.load(index_dir / name, map_location="cpu", weights_only=True)
        shard_ids = [str(value) for value in payload["item_ids"]]
        embeddings = payload["embeddings"]
        if tuple(embeddings.shape) != (len(shard_ids), 2048):
            raise RuntimeError(f"invalid item-index shard shape: {name}")
        ids.extend(shard_ids)
        matrices.append(embeddings)
    matrix = torch.cat(matrices, dim=0)
    if len(ids) != config["item_count"] or matrix.shape != (len(ids), 2048):
        raise RuntimeError("assembled item index does not match its config")
    if len(ids) != len(set(ids)):
        raise RuntimeError("item index contains duplicate IDs")
    return ids, matrix, config, manifest


def shard_specs(photo_ids: list[str], shard_size: int) -> list[tuple[int, int, list[str], str]]:
    specs = []
    for start in range(0, len(photo_ids), shard_size):
        end = min(start + shard_size, len(photo_ids))
        specs.append((start // shard_size, start, photo_ids[start:end], f"part-{start:06d}-{end:06d}.jsonl"))
    return specs


def validate_result_shard(
    path: Path,
    expected_photo_ids: list[str],
    top_k: int,
    item_to_group: dict[str, tuple[str, frozenset[str]]],
) -> bool:
    if not path.is_file():
        return False
    seen: list[str] = []
    for line_number, row in iter_jsonl(path):
        photo_id = str(row.get("photo_id"))
        seen.append(photo_id)
        positive = str(row.get("positive_item_id"))
        group = item_to_group.get(positive)
        forbidden = set(group[1]) if group else {positive}
        forbidden.add(positive)
        for mode in MODES:
            result = row.get(mode)
            if not isinstance(result, dict):
                raise ValueError(f"missing {mode} result at {path}:{line_number}")
            item_ids = [str(value) for value in result.get("item_ids", [])]
            scores = result.get("scores", [])
            raw_ranks = result.get("raw_ranks", [])
            if len(item_ids) != top_k or len(scores) != top_k or len(raw_ranks) != top_k:
                raise ValueError(f"wrong {mode} Top-K length at {path}:{line_number}")
            if len(item_ids) != len(set(item_ids)) or forbidden & set(item_ids):
                raise ValueError(f"invalid {mode} negatives at {path}:{line_number}")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in scores
            ):
                raise ValueError(f"invalid {mode} scores at {path}:{line_number}")
            if not all(isinstance(value, int) and value > 0 for value in raw_ranks):
                raise ValueError(f"invalid {mode} ranks at {path}:{line_number}")
            if raw_ranks != sorted(raw_ranks) or len(raw_ranks) != len(set(raw_ranks)):
                raise ValueError(f"unordered {mode} ranks at {path}:{line_number}")
    if seen != expected_photo_ids:
        raise ValueError(f"photo ID/order mismatch in {path}")
    return True


def atomic_jsonl_dump(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def select_allowed_topk(
    scores: torch.Tensor,
    item_ids: list[str],
    item_to_position: dict[str, int],
    forbidden_item_ids: set[str],
    top_k: int,
) -> tuple[list[str], list[float], list[int]]:
    forbidden_positions = {
        item_to_position[item_id]
        for item_id in forbidden_item_ids
        if item_id in item_to_position
    }
    if top_k > scores.numel() - len(forbidden_positions):
        raise ValueError("Top-K exceeds the number of allowed indexed items")
    search_k = min(scores.numel(), top_k + len(forbidden_positions))
    values, positions = torch.topk(scores, k=search_k)
    if not torch.isfinite(values).all():
        raise RuntimeError("non-finite hard-negative score selected")
    selected_ids: list[str] = []
    selected_scores: list[float] = []
    selected_ranks: list[int] = []
    for raw_rank, (value, position) in enumerate(
        zip(values.tolist(), positions.tolist()), 1
    ):
        if position in forbidden_positions:
            continue
        selected_ids.append(item_ids[position])
        selected_scores.append(float(value))
        selected_ranks.append(raw_rank)
        if len(selected_ids) == top_k:
            break
    if len(selected_ids) != top_k:
        raise RuntimeError("failed to select the requested number of allowed negatives")
    if forbidden_item_ids & set(selected_ids):
        raise RuntimeError("forbidden item selected as a hard negative")
    return selected_ids, selected_scores, selected_ranks


def build_run_config(args: argparse.Namespace, photo_count: int, group_count: int) -> dict:
    index_manifest_path = args.item_index / "manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    index_config_path = args.item_index / index_manifest["config"]
    index_config = json.loads(index_config_path.read_text(encoding="utf-8"))
    model_hash = sha256_file(args.model / "config.json")
    if model_hash != index_config["model_config_sha256"]:
        raise RuntimeError("query model differs from the model used for the item index")
    return {
        "schema_version": 1,
        "kind": "frozen-retriever-train-hard-negative-candidates",
        "model_path": str(args.model),
        "model_config_sha256": model_hash,
        "data_root": str(args.data_root),
        "photo_info_sha256": sha256_file(args.data_root / "photo_info.jsonl"),
        "labels_sha256": sha256_file(args.labels),
        "photo_ids_sha256": sha256_file(args.photo_ids),
        "duplicate_groups_sha256": sha256_file(args.duplicate_groups),
        "duplicate_group_count": group_count,
        "item_index_config_sha256": sha256_file(index_config_path),
        "item_index_manifest_sha256": sha256_file(index_manifest_path),
        "photo_count": photo_count,
        "top_k_per_mode": args.top_k,
        "modes": list(MODES),
        "shard_size": args.shard_size,
        "query_instruction": QUERY_INSTRUCTION,
        "text_budgets": [list(value) for value in TEXT_BUDGETS],
        "frame_preprocessing": (
            f"aspect-preserving {index_config['image_size']}-square letterbox; pad odd T"
        ),
        "negative_exclusions": [
            "positive_item",
            "all members of the positive item's proposed duplicate group",
        ],
        "duplicate_group_status": "proposed_not_confirmed_conservative_exclusion",
    }


def finalize_results(
    output_dir: Path,
    config_path: Path,
    specs: list[tuple[int, int, list[str], str]],
    top_k: int,
    item_to_group: dict[str, tuple[str, frozenset[str]]],
) -> None:
    for _, _, expected, name in specs:
        if not validate_result_shard(
            output_dir / "shards" / name,
            expected,
            top_k,
            item_to_group,
        ):
            raise RuntimeError(f"missing result shard: {name}")
    atomic_json_dump(
        {
            "config": config_path.name,
            "shards": [name for _, _, _, name in specs],
            "complete": True,
        },
        output_dir / "manifest.json",
    )


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.shard_size < 1:
        raise ValueError("top-k and shard-size must be positive")
    if args.partitions < 1 or not 0 <= args.partition_index < args.partitions:
        raise ValueError("require partitions >= 1 and 0 <= partition-index < partitions")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    photo_ids = read_ids(args.photo_ids)
    if args.limit is not None:
        photo_ids = photo_ids[: args.limit]
    specs = shard_specs(photo_ids, args.shard_size)
    item_to_group, group_count = load_duplicate_groups(args.duplicate_groups)
    run_config = build_run_config(args, len(photo_ids), group_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != run_config:
            raise RuntimeError("existing run config differs; use a new output directory")
    else:
        atomic_json_dump(run_config, config_path)

    if args.finalize_only:
        finalize_results(
            args.output_dir,
            config_path,
            specs,
            args.top_k,
            item_to_group,
        )
        print(f"hard-negative candidates complete: {args.output_dir}")
        return

    assigned = [spec for spec in specs if spec[0] % args.partitions == args.partition_index]
    missing = [
        spec
        for spec in assigned
        if not validate_result_shard(
            args.output_dir / "shards" / spec[3],
            spec[2],
            args.top_k,
            item_to_group,
        )
    ]
    if not missing:
        print(f"partition {args.partition_index}/{args.partitions} has no missing shards")
        if args.partitions == 1:
            finalize_results(
                args.output_dir,
                config_path,
                specs,
                args.top_k,
                item_to_group,
            )
            print(f"hard-negative candidates complete: {args.output_dir}")
        return

    assigned_photo_ids = {photo_id for _, _, ids, _ in missing for photo_id in ids}
    print(f"loading metadata for {len(assigned_photo_ids)} assigned photos...", flush=True)
    photos = load_rows_for_ids(args.data_root / "photo_info.jsonl", "photo_id", assigned_photo_ids)
    positives = load_positive_items(args.labels, assigned_photo_ids)

    print("loading complete training item index...", flush=True)
    item_ids, item_matrix, index_config, _ = load_item_index(args.item_index)
    item_to_position = {item_id: position for position, item_id in enumerate(item_ids)}
    positive_items = {positives[photo_id] for photo_id in assigned_photo_ids}
    missing_positive_items = positive_items - item_to_position.keys()
    if missing_positive_items:
        raise RuntimeError(
            f"{len(missing_positive_items)} positive items missing from train index: "
            f"{sorted(missing_positive_items)[:5]}"
        )
    image_size = int(index_config["image_size"])
    item_matrix = item_matrix.to(device="cuda", dtype=torch.float32)
    print(f"item matrix: {tuple(item_matrix.shape)} {item_matrix.device}", flush=True)

    print("loading frozen query model...", flush=True)
    model = load_embedder(args.repo, args.model, image_size)
    worker_started = time.perf_counter()
    completed_photos = 0

    for _, _, shard_photo_ids, name in missing:
        shard_started = time.perf_counter()
        rows: list[dict] = []
        for photo_id in shard_photo_ids:
            photo = photos[photo_id]
            positive_item_id = positives[photo_id]
            group = item_to_group.get(positive_item_id)
            group_id = group[0] if group else None
            forbidden = set(group[1]) if group else {positive_item_id}
            forbidden.add(positive_item_id)
            frames = load_video_frames(photo, args.data_root, image_size)
            text = compose_video_text(photo)
            mode_results: dict[str, dict] = {}
            for mode in MODES:
                query_input = {"instruction": QUERY_INSTRUCTION, "video": frames}
                if mode == "multimodal":
                    query_input["text"] = text
                started = time.perf_counter()
                query = encode_checked(model, [query_input], "pixel_values_videos")
                scores = (query.float() @ item_matrix.T).squeeze(0)
                selected_ids, selected_scores, selected_ranks = select_allowed_topk(
                    scores, item_ids, item_to_position, forbidden, args.top_k
                )
                torch.cuda.synchronize()
                mode_results[mode] = {
                    "item_ids": selected_ids,
                    "scores": selected_scores,
                    "raw_ranks": selected_ranks,
                    "query_and_search_seconds": time.perf_counter() - started,
                }
            rows.append(
                {
                    "photo_id": photo_id,
                    "positive_item_id": positive_item_id,
                    "proposed_duplicate_group_id": group_id,
                    "forbidden_indexed_item_count": sum(
                        item_id in item_to_position for item_id in forbidden
                    ),
                    "original_frame_count": len(photo.get("photo_frames") or []),
                    "encoded_frame_count": len(frames),
                    "text_chars": len(text),
                    **mode_results,
                }
            )
            completed_photos += 1
            if completed_photos % 25 == 0:
                elapsed = time.perf_counter() - worker_started
                print(
                    f"partition {args.partition_index}: {completed_photos}/{len(assigned_photo_ids)} "
                    f"photos ({completed_photos / elapsed:.2f} photos/s)",
                    flush=True,
                )
        shard_path = args.output_dir / "shards" / name
        atomic_jsonl_dump(rows, shard_path)
        print(
            f"saved {name} in {time.perf_counter() - shard_started:.1f}s",
            flush=True,
        )

    print(
        f"partition {args.partition_index}/{args.partitions} complete: "
        f"{completed_photos} photos",
        flush=True,
    )
    if args.partitions == 1:
        finalize_results(
            args.output_dir,
            config_path,
            specs,
            args.top_k,
            item_to_group,
        )
        print(f"hard-negative candidates complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
