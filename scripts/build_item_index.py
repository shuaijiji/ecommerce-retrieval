#!/usr/bin/env python3
"""Build a resumable, sharded item index, optionally with multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import transformers

from retrieval_common import (
    atomic_json_dump,
    encode_checked,
    load_embedder,
    read_jsonl,
    sha256_file,
)


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
        "--output-dir",
        type=Path,
        default=Path("/code/ecommerce-retrieval/artifacts/zero_shot_v0/items"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--item-ids",
        type=Path,
        help="optional newline-delimited allowlist, e.g. train_item_ids.txt",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=1,
        help="number of disjoint shard workers",
    )
    parser.add_argument(
        "--partition-index",
        type=int,
        default=0,
        help="zero-based shard worker index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="encode only the first N filtered items (smoke tests only)",
    )
    return parser.parse_args()


def read_id_allowlist(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [value for value in ids if value]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {path}")
    return ids


def validate_existing_shard(path: Path, expected_ids: list[str]) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    ids = [str(value) for value in payload["item_ids"]]
    embeddings = payload["embeddings"]
    if ids != expected_ids:
        raise RuntimeError(f"ID mismatch in existing shard {path}")
    if tuple(embeddings.shape) != (len(ids), 2048):
        raise RuntimeError(f"shape mismatch in existing shard {path}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError(f"non-finite values in existing shard {path}")
    return True


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.shard_size < args.batch_size:
        raise ValueError("require 1 <= batch-size <= shard-size")
    if args.partitions < 1 or not 0 <= args.partition_index < args.partitions:
        raise ValueError("require partitions >= 1 and 0 <= partition-index < partitions")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    item_path = args.data_root / "item_info.jsonl"
    items = read_jsonl(item_path)
    assert isinstance(items, list)
    item_ids_sha256 = None
    if args.item_ids is not None:
        allowed_ids = read_id_allowlist(args.item_ids)
        allowed = set(allowed_ids)
        available = {str(row["item_id"]) for row in items}
        missing = allowed - available
        if missing:
            examples = sorted(missing)[:10]
            raise ValueError(f"{len(missing)} allowlisted items missing: {examples}")
        items = [row for row in items if str(row["item_id"]) in allowed]
        if len(items) != len(allowed):
            raise RuntimeError("filtered item count does not match the allowlist")
        item_ids_sha256 = sha256_file(args.item_ids)
    if args.limit is not None:
        items = items[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "schema_version": 1,
        "kind": "qwen3-vl-item-index",
        "model_path": str(args.model),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "item_info_sha256": sha256_file(item_path),
        "item_ids_sha256": item_ids_sha256,
        "item_ids_source": str(args.item_ids) if args.item_ids else None,
        "item_count": len(items),
        "embedding_dimension": 2048,
        "image_size": args.image_size,
        "max_length": 4096,
        "dtype": "bfloat16",
        "item_input": "title+image",
        "item_instruction": "Represent the user's input.",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "shard_size": args.shard_size,
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "existing index configuration differs; use a new output directory"
            )
    else:
        atomic_json_dump(config, config_path)

    print(f"items: {len(items)}")
    shard_paths: list[str] = []
    shard_specs: list[tuple[int, int, int, list[dict], list[str], Path]] = []

    for shard_number, shard_start in enumerate(range(0, len(items), args.shard_size)):
        shard_end = min(shard_start + args.shard_size, len(items))
        shard_items = items[shard_start:shard_end]
        shard_ids = [str(row["item_id"]) for row in shard_items]
        shard_path = args.output_dir / f"part-{shard_start:06d}-{shard_end:06d}.pt"
        shard_paths.append(shard_path.name)
        shard_specs.append((shard_number, shard_start, shard_end, shard_items, shard_ids, shard_path))

    completed = {
        shard_path
        for _, _, _, _, shard_ids, shard_path in shard_specs
        if validate_existing_shard(shard_path, shard_ids)
    }
    if len(completed) == len(shard_specs):
        atomic_json_dump(
            {
                "config": config_path.name,
                "shards": shard_paths,
                "complete": True,
            },
            args.output_dir / "manifest.json",
        )
        print("all item shards are already valid:", args.output_dir)
        return

    assigned_specs = [
        spec for spec in shard_specs if spec[0] % args.partitions == args.partition_index
    ]
    missing_assigned = [spec for spec in assigned_specs if spec[-1] not in completed]
    if not missing_assigned:
        print(
            f"partition {args.partition_index}/{args.partitions} has no missing shards; "
            "other partitions may still be running"
        )
        return

    print(
        f"partition {args.partition_index}/{args.partitions}: "
        f"{len(missing_assigned)} missing of {len(assigned_specs)} assigned shards"
    )
    print("loading model...")
    model = load_embedder(args.repo, args.model, args.image_size)
    run_start = time.perf_counter()
    newly_encoded = 0

    for _, shard_start, shard_end, shard_items, shard_ids, shard_path in missing_assigned:

        shard_embeddings: list[torch.Tensor] = []
        shard_started = time.perf_counter()
        for begin in range(0, len(shard_items), args.batch_size):
            batch = shard_items[begin : begin + args.batch_size]
            batch_inputs = [
                {
                    "text": row.get("item_title") or "",
                    "image": str(args.data_root / row["item_image"]),
                }
                for row in batch
            ]
            embeddings = encode_checked(model, batch_inputs, "pixel_values")
            shard_embeddings.append(embeddings.to(device="cpu", dtype=torch.bfloat16))
            newly_encoded += len(batch)
            elapsed = time.perf_counter() - run_start
            rate = newly_encoded / elapsed if elapsed else 0
            print(
                f"encoded {shard_start + begin + len(batch)}/{len(items)} "
                f"({rate:.2f} items/s)"
            )

        embeddings = torch.cat(shard_embeddings, dim=0)
        temporary = shard_path.with_name(f"{shard_path.name}.tmp.{os.getpid()}")
        torch.save(
            {"item_ids": shard_ids, "embeddings": embeddings},
            temporary,
        )
        os.replace(temporary, shard_path)
        print(
            f"saved {shard_path.name} in "
            f"{time.perf_counter() - shard_started:.1f}s"
        )

    completed_after = {
        shard_path
        for _, _, _, _, shard_ids, shard_path in shard_specs
        if validate_existing_shard(shard_path, shard_ids)
    }
    if len(completed_after) == len(shard_specs):
        atomic_json_dump(
            {
                "config": config_path.name,
                "shards": shard_paths,
                "complete": True,
            },
            args.output_dir / "manifest.json",
        )
        print("item index complete:", args.output_dir)
    else:
        print(
            f"partition complete; index progress "
            f"{len(completed_after)}/{len(shard_specs)} shards"
        )


if __name__ == "__main__":
    main()
