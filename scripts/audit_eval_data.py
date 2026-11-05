#!/usr/bin/env python3
"""Audit evaluation metadata, references, and image paths."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TEXT_FIELDS = (
    "photo_caption",
    "photo_title",
    "photo_text",
    "photo_ocr",
    "photo_asr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/code/ecommerce-retrieval/data/eval/test_small_package"
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> tuple[list[dict], list[tuple[int, str]]]:
    rows: list[dict] = []
    errors: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    errors.append((line_number, "not a JSON object"))
                else:
                    rows.append(value)
            except Exception as exc:  # report all malformed source rows
                errors.append((line_number, str(exc)))
    return rows, errors


def main() -> None:
    root = parse_args().data_root
    items, item_errors = read_jsonl(root / "item_info.jsonl")
    photos, photo_errors = read_jsonl(root / "photo_info.jsonl")

    with (root / "label.json").open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list):
        raise TypeError("label.json must contain a JSON array")

    item_ids = [str(row.get("item_id")) for row in items]
    photo_ids = [str(row.get("photo_id")) for row in photos]
    item_set = set(item_ids)
    photo_set = set(photo_ids)
    label_pairs = [
        (str(row.get("photo_id")), str(row.get("item_id")))
        for row in labels
    ]

    missing_item_images = [
        (row.get("item_id"), row.get("item_image"))
        for row in items
        if not row.get("item_image")
        or not (root / row["item_image"]).is_file()
    ]

    missing_frames: list[tuple[object, object]] = []
    frame_counts: list[int] = []
    for row in photos:
        frames = row.get("photo_frames") or []
        frame_counts.append(len(frames))
        for frame in frames:
            if not (root / frame).is_file():
                missing_frames.append((row.get("photo_id"), frame))

    label_photo_counts = Counter(photo_id for photo_id, _ in label_pairs)

    print("=== Counts ===")
    print("items:", len(items))
    print("photos:", len(photos))
    print("labels:", len(labels))
    print("\n=== JSON errors ===")
    print("item_info.jsonl:", len(item_errors), item_errors[:5])
    print("photo_info.jsonl:", len(photo_errors), photo_errors[:5])
    print("\n=== ID checks ===")
    print("duplicate item_id:", len(item_ids) - len(item_set))
    print("duplicate photo_id:", len(photo_ids) - len(photo_set))
    print("duplicate label pair:", len(label_pairs) - len(set(label_pairs)))
    print(
        "photos with multiple labels:",
        sum(count > 1 for count in label_photo_counts.values()),
    )
    print("\n=== Label references ===")
    print(
        "missing photo reference:",
        sum(photo_id not in photo_set for photo_id, _ in label_pairs),
    )
    print(
        "missing item reference:",
        sum(item_id not in item_set for _, item_id in label_pairs),
    )
    print("photos without labels:", len(photo_set - set(label_photo_counts)))
    print("\n=== Image paths ===")
    print("missing item images:", len(missing_item_images))
    print("missing frame images:", len(missing_frames))
    print("\n=== Frame count distribution ===")
    print(dict(sorted(Counter(frame_counts).items())))
    print("\n=== Non-empty video text ===")
    for field in TEXT_FIELDS:
        count = sum(
            bool(str(row.get(field)).strip())
            for row in photos
            if row.get(field) is not None
        )
        ratio = count / len(photos) if photos else 0
        print(f"{field}: {count}/{len(photos)} ({ratio:.2%})")


if __name__ == "__main__":
    main()
