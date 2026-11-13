#!/usr/bin/env python3
"""Sample hard negatives and build a reproducible multimodal reranker SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

INSTRUCTION = (
    "判断候选商品是否为视频主要讲解的同一款商品。结合视频画面和文本与商品主图及标题，"
    "区分品牌、型号、外观和规格；仅同类别或配件相关不足以判定为同款。"
)
TEXT_BUDGETS = (
    ("photo_caption", "视频描述", 500),
    ("photo_title", "视频标题", 150),
    ("photo_text", "视频文本", 200),
    ("photo_ocr", "画面文字", 700),
    ("photo_asr", "语音文本", 500),
)
BUCKETS = (
    ("vision", "top_1_10", 1, 10),
    ("multimodal", "top_1_10", 1, 10),
    ("vision", "middle_11_50", 11, 50),
    ("multimodal", "middle_11_50", 11, 50),
    ("vision", "tail_51_100", 51, 10**9),
    ("multimodal", "tail_51_100", 51, 10**9),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--photo-ids", type=Path, required=True)
    parser.add_argument("--train-item-ids", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--duplicate-groups", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-queries", type=int, default=5000)
    parser.add_argument("--eval-queries", type=int, default=500)
    parser.add_argument("--negatives-per-query", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--media-workers", type=int, default=32)
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="write sampling manifests only; intended for synthetic tests",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(seed: int, *parts: object) -> bytes:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).digest()


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"empty or duplicate IDs in {path}")
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


def select_ids(values: list[str], count: int, seed: int) -> list[str]:
    if not 0 < count <= len(values):
        raise ValueError("num-queries must be positive and not exceed available IDs")
    # Keep only the smallest keyed hashes rather than sorting almost one million tuples.
    heap: list[tuple[int, str]] = []
    for value in values:
        key = int.from_bytes(stable_digest(seed, "query", value), "big")
        entry = (-key, value)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [value for _, value in sorted((-key, value) for key, value in heap)]


def parse_shard_bounds(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"part-(\d+)-(\d+)\.jsonl", name)
    if match is None:
        raise ValueError(f"invalid candidate shard name: {name}")
    return int(match.group(1)), int(match.group(2))


def load_selected_candidate_rows(
    candidate_dir: Path,
    photo_ids: list[str],
    selected_ids: list[str],
) -> dict[str, dict]:
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("hard-negative candidate manifest is incomplete")
    wanted = set(selected_ids)
    rows: dict[str, dict] = {}
    for name in manifest["shards"]:
        start, end = parse_shard_bounds(name)
        expected = photo_ids[start:end]
        if not wanted.intersection(expected):
            continue
        path = candidate_dir / "shards" / name
        line_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for offset, line in enumerate(handle):
                line_count += 1
                photo_id = expected[offset]
                if photo_id not in wanted:
                    continue
                row = json.loads(line)
                if str(row.get("photo_id")) != photo_id:
                    raise ValueError(f"photo/order mismatch in {path}:{offset + 1}")
                rows[photo_id] = row
        if line_count != len(expected):
            raise ValueError(f"line count mismatch in {path}")
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"{len(missing)} selected photos missing candidates")
    return rows


def load_duplicate_groups(path: Path) -> dict[str, frozenset[str]]:
    mapping: dict[str, frozenset[str]] = {}
    for line_number, row in iter_jsonl(path):
        members = row.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(f"invalid duplicate group at {path}:{line_number}")
        item_ids = frozenset(str(member["item_id"]) for member in members)
        overlap = item_ids & mapping.keys()
        if overlap:
            raise ValueError(f"duplicate-group overlap: {sorted(overlap)[:5]}")
        for item_id in item_ids:
            mapping[item_id] = item_ids
    return mapping


def mode_candidates(row: dict, mode: str) -> list[dict]:
    result = row[mode]
    item_ids = [str(value) for value in result["item_ids"]]
    scores = result["scores"]
    ranks = result["raw_ranks"]
    if not (len(item_ids) == len(scores) == len(ranks) == 100):
        raise ValueError(f"expected 100 {mode} candidates for {row['photo_id']}")
    output = []
    for item_id, score, rank in zip(item_ids, scores, ranks):
        if not isinstance(rank, int) or rank < 1 or not math.isfinite(float(score)):
            raise ValueError(f"invalid {mode} candidate for {row['photo_id']}")
        output.append(
            {
                "item_id": item_id,
                "source_mode": mode,
                "raw_rank": rank,
                "score": float(score),
            }
        )
    return output


def choose_from_bucket(
    candidates: list[dict],
    lower: int,
    upper: int,
    used: set[str],
    seed: int,
    photo_id: str,
    bucket: str,
) -> dict | None:
    eligible = [
        candidate
        for candidate in candidates
        if lower <= candidate["raw_rank"] <= upper and candidate["item_id"] not in used
    ]
    if not eligible:
        return None
    if lower == 1:
        chosen = min(eligible, key=lambda row: (row["raw_rank"], row["item_id"]))
    else:
        chosen = min(
            eligible,
            key=lambda row: stable_digest(seed, photo_id, bucket, row["item_id"]),
        )
    return {**chosen, "source_bucket": bucket}


def choose_random_item(
    train_item_ids: list[str],
    forbidden: set[str],
    used: set[str],
    seed: int,
    photo_id: str,
) -> str:
    for attempt in range(10000):
        index = int.from_bytes(
            stable_digest(seed, photo_id, "random", attempt)[:8], "big"
        ) % len(train_item_ids)
        item_id = train_item_ids[index]
        if item_id not in forbidden and item_id not in used:
            return item_id
    raise RuntimeError(f"unable to choose a random negative for {photo_id}")


def sample_negatives(
    row: dict,
    train_item_ids: list[str],
    duplicate_groups: dict[str, frozenset[str]],
    count: int,
    seed: int,
) -> list[dict]:
    photo_id = str(row["photo_id"])
    positive = str(row["positive_item_id"])
    forbidden = set(duplicate_groups.get(positive, frozenset({positive})))
    forbidden.add(positive)
    modes = {mode: mode_candidates(row, mode) for mode in ("vision", "multimodal")}
    for candidates in modes.values():
        if forbidden & {candidate["item_id"] for candidate in candidates}:
            raise ValueError(f"forbidden candidate survived mining for {photo_id}")

    selected: list[dict] = []
    used: set[str] = set()
    for mode, bucket, lower, upper in BUCKETS:
        if len(selected) >= max(0, count - 1):
            break
        chosen = choose_from_bucket(
            modes[mode], lower, upper, used, seed, photo_id, f"{mode}:{bucket}"
        )
        if chosen is not None:
            selected.append(chosen)
            used.add(chosen["item_id"])

    combined: dict[str, dict] = {}
    for mode in ("vision", "multimodal"):
        for candidate in modes[mode]:
            existing = combined.get(candidate["item_id"])
            if existing is None or candidate["raw_rank"] < existing["raw_rank"]:
                combined[candidate["item_id"]] = candidate
    fallback = sorted(
        (candidate for candidate in combined.values() if candidate["item_id"] not in used),
        key=lambda row: (row["raw_rank"], row["item_id"]),
    )
    for candidate in fallback:
        if len(selected) >= max(0, count - 1):
            break
        selected.append({**candidate, "source_bucket": "fallback_ranked"})
        used.add(candidate["item_id"])

    if len(selected) < count:
        random_item = choose_random_item(
            train_item_ids, forbidden, used, seed, photo_id
        )
        selected.append(
            {
                "item_id": random_item,
                "source_mode": "random",
                "source_bucket": "random_train_item",
                "raw_rank": None,
                "score": None,
            }
        )
        used.add(random_item)

    if len(selected) < count:
        for candidate in fallback:
            if candidate["item_id"] in used:
                continue
            selected.append({**candidate, "source_bucket": "fallback_ranked"})
            used.add(candidate["item_id"])
            if len(selected) == count:
                break
    selected = selected[:count]
    if len(selected) != count or len(used) != count:
        raise RuntimeError(f"failed to sample {count} unique negatives for {photo_id}")
    if forbidden & used:
        raise RuntimeError(f"false negative selected for {photo_id}")
    return selected


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compose_video_text(photo: dict) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for field, label, limit in TEXT_BUDGETS:
        value = clean_text(photo.get(field))
        if not value or value in seen:
            continue
        seen.add(value)
        parts.append(f"{label}：{value[:limit]}")
    return "\n".join(parts)


def load_metadata(path: Path, key: str, wanted: set[str]) -> dict[str, dict]:
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
        raise ValueError(f"{len(missing)} requested {key}s missing from {path}")
    return rows


def normalize_image(task: tuple[str, str, int]) -> str:
    from PIL import Image

    source_string, destination_string, size = task
    source, destination = Path(source_string), Path(destination_string)
    if destination.is_file():
        try:
            with Image.open(destination) as image:
                if image.size == (size, size) and image.mode == "RGB":
                    return destination_string
        except OSError:
            pass
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.png")
    canvas.save(temporary, format="PNG", optimize=False)
    os.replace(temporary, destination)
    return destination_string


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def write_json_atomic(value: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def make_swift_row(
    canonical: dict,
    photo: dict,
    items: dict[str, dict],
    media_root: Path,
    image_size: int,
) -> dict:
    photo_id = canonical["photo_id"]
    frame_paths = [
        str((media_root / "photo_frames" / photo_id / f"{index:02d}.png").resolve())
        for index, _ in enumerate(photo.get("photo_frames") or [])
    ]
    if not frame_paths:
        raise ValueError(f"photo {photo_id} has no frames")
    if len(frame_paths) % 2:
        frame_paths.append(frame_paths[-1])
    positive = canonical["positive_item_id"]
    negative_ids = [row["item_id"] for row in canonical["negatives"]]
    video_text = compose_video_text(photo)
    query_content = "<video>"
    if video_text:
        query_content += "\n" + video_text

    def document_message(item_id: str) -> list[dict]:
        title = clean_text(items[item_id].get("item_title"))[:500]
        return [{"role": "assistant", "content": "<image>\n" + title}]

    def item_image(item_id: str) -> str:
        return str((media_root / "item_images" / f"{item_id}.png").resolve())

    return {
        "messages": [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": query_content},
        ],
        "videos": [frame_paths],
        "positive_messages": [document_message(positive)],
        "positive_images": [[item_image(positive)]],
        "negative_messages": [document_message(item_id) for item_id in negative_ids],
        "negative_images": [[item_image(item_id)] for item_id in negative_ids],
        "chat_template_kwargs": {"max_pixels": image_size * image_size},
    }


def main() -> None:
    args = parse_args()
    if args.negatives_per_query < 2:
        raise ValueError("require at least two negatives per query")
    if not 0 <= args.eval_queries < args.num_queries:
        raise ValueError("eval-queries must be nonnegative and smaller than num-queries")
    if args.media_workers < 1 or args.image_size < 32:
        raise ValueError("invalid media-workers or image-size")

    photo_ids = read_ids(args.photo_ids)
    train_item_ids = read_ids(args.train_item_ids)
    selected_ids = select_ids(photo_ids, args.num_queries, args.seed)
    candidate_rows = load_selected_candidate_rows(
        args.candidates, photo_ids, selected_ids
    )
    duplicate_groups = load_duplicate_groups(args.duplicate_groups)

    canonical_rows = []
    needed_items: set[str] = set()
    for photo_id in selected_ids:
        row = candidate_rows[photo_id]
        negatives = sample_negatives(
            row,
            train_item_ids,
            duplicate_groups,
            args.negatives_per_query,
            args.seed,
        )
        positive = str(row["positive_item_id"])
        needed_items.add(positive)
        needed_items.update(negative["item_id"] for negative in negatives)
        canonical_rows.append(
            {
                "photo_id": photo_id,
                "positive_item_id": positive,
                "negatives": negatives,
            }
        )

    photos = load_metadata(
        args.data_root / "photo_info.jsonl", "photo_id", set(selected_ids)
    )
    items = load_metadata(
        args.data_root / "item_info.jsonl", "item_id", needed_items
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = args.output_dir / "samples.jsonl"
    write_jsonl(canonical_rows, canonical_path)

    eval_ids = set(
        select_ids(selected_ids, args.eval_queries, args.seed + 1)
        if args.eval_queries
        else []
    )
    split = {
        row["photo_id"]: ("smoke_eval" if row["photo_id"] in eval_ids else "train")
        for row in canonical_rows
    }

    if not args.skip_media:
        media_root = args.output_dir / "media_512_png"
        tasks: dict[str, tuple[str, str, int]] = {}
        for photo_id in selected_ids:
            for index, relative in enumerate(photos[photo_id].get("photo_frames") or []):
                destination = media_root / "photo_frames" / photo_id / f"{index:02d}.png"
                tasks[str(destination)] = (
                    str(args.data_root / relative),
                    str(destination),
                    args.image_size,
                )
        for item_id in needed_items:
            destination = media_root / "item_images" / f"{item_id}.png"
            tasks[str(destination)] = (
                str(args.data_root / items[item_id]["item_image"]),
                str(destination),
                args.image_size,
            )
        with ProcessPoolExecutor(max_workers=args.media_workers) as executor:
            for number, _ in enumerate(
                executor.map(normalize_image, tasks.values(), chunksize=32), 1
            ):
                if number % 1000 == 0 or number == len(tasks):
                    print(f"normalized media {number}/{len(tasks)}", flush=True)

        swift_rows = [
            (row, make_swift_row(row, photos[row["photo_id"]], items, media_root, args.image_size))
            for row in canonical_rows
        ]
        write_jsonl(
            (swift for row, swift in swift_rows if split[row["photo_id"]] == "train"),
            args.output_dir / "swift_train.jsonl",
        )
        write_jsonl(
            (swift for row, swift in swift_rows if split[row["photo_id"]] == "smoke_eval"),
            args.output_dir / "swift_smoke_eval.jsonl",
        )
    else:
        tasks = {}

    bucket_counts: dict[str, int] = {}
    for row in canonical_rows:
        for negative in row["negatives"]:
            key = f"{negative['source_mode']}:{negative['source_bucket']}"
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
    source_manifest = json.loads(
        (args.candidates / "manifest.json").read_text(encoding="utf-8")
    )
    config = {
        "schema_version": 1,
        "kind": "qwen3-vl-reranker-hard-negative-sft-smoke",
        "seed": args.seed,
        "num_queries": args.num_queries,
        "train_queries": args.num_queries - args.eval_queries,
        "smoke_eval_queries": args.eval_queries,
        "negatives_per_query": args.negatives_per_query,
        "effective_pointwise_pairs": args.num_queries * (1 + args.negatives_per_query),
        "image_size": args.image_size,
        "media_format": "512-square letterbox RGB lossless PNG" if not args.skip_media else None,
        "odd_video_frames": "repeat final normalized frame path",
        "instruction": INSTRUCTION,
        "sampling_buckets": [list(value) for value in BUCKETS],
        "bucket_counts": bucket_counts,
        "random_negative_policy": "stable hash into train_item_ids with positive duplicate-group exclusion",
        "smoke_eval_warning": "query holdout for pipeline monitoring only; not item-disjoint model selection",
        "source_hashes": {
            "script": sha256_file(Path(__file__)),
            "photo_ids": sha256_file(args.photo_ids),
            "train_item_ids": sha256_file(args.train_item_ids),
            "candidate_config": sha256_file(args.candidates / source_manifest["config"]),
            "candidate_manifest": sha256_file(args.candidates / "manifest.json"),
            "duplicate_groups": sha256_file(args.duplicate_groups),
            "photo_info": sha256_file(args.data_root / "photo_info.jsonl"),
            "item_info": sha256_file(args.data_root / "item_info.jsonl"),
            "samples": sha256_file(canonical_path),
        },
        "media_files": len(tasks),
    }
    if not args.skip_media:
        config["source_hashes"]["swift_train"] = sha256_file(
            args.output_dir / "swift_train.jsonl"
        )
        config["source_hashes"]["swift_smoke_eval"] = sha256_file(
            args.output_dir / "swift_smoke_eval.jsonl"
        )
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(
                f"existing config differs; use a new output directory: {config_path}"
            )
    write_json_atomic(config, config_path)

    output_hashes = {
        "config.json": sha256_file(config_path),
        "samples.jsonl": sha256_file(canonical_path),
    }
    if not args.skip_media:
        output_hashes.update(
            {
                "swift_train.jsonl": sha256_file(args.output_dir / "swift_train.jsonl"),
                "swift_smoke_eval.jsonl": sha256_file(
                    args.output_dir / "swift_smoke_eval.jsonl"
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "config": "config.json",
        "outputs": output_hashes,
        "rows": {
            "samples": args.num_queries,
            "swift_train": 0 if args.skip_media else args.num_queries - args.eval_queries,
            "swift_smoke_eval": 0 if args.skip_media else args.eval_queries,
        },
        "effective_pointwise_pairs": args.num_queries
        * (1 + args.negatives_per_query),
        "media_files": len(tasks),
    }
    write_json_atomic(manifest, args.output_dir / "manifest.json")
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    print(f"reranker SFT dataset ready: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
