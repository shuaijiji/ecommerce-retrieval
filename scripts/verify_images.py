#!/usr/bin/env python3
"""Decode every evaluation image with Pillow and report failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/code/ecommerce-retrieval/data/eval/test_small_package"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root
    paths: list[Path] = []

    with (root / "item_info.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            paths.append(root / row["item_image"])

    with (root / "photo_info.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            paths.extend(root / frame for frame in row.get("photo_frames") or [])

    failures: list[tuple[str, str]] = []
    formats: Counter[str] = Counter()
    for index, path in enumerate(paths, 1):
        try:
            with Image.open(path) as image:
                image.verify()
                formats[str(image.format)] += 1
        except Exception as exc:  # keep checking to produce a complete report
            failures.append((str(path), repr(exc)))

        if args.progress_every and index % args.progress_every == 0:
            print(f"checked {index}/{len(paths)}, failures {len(failures)}")

    print("total images:", len(paths))
    print("formats:", dict(formats))
    print("decode failures:", len(failures))
    print("failure examples:", failures[:20])

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
