#!/usr/bin/env python3
"""Select a deterministic query subset shared by GRPO and listwise Pilot arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(source: Path, count: int, seed: int) -> list[str]:
    rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if count < 1 or count > len(rows):
        raise ValueError(f"count must be in [1, {len(rows)}]")
    keyed = []
    for index, line in enumerate(rows):
        row = json.loads(line)
        required = {
            "messages",
            "videos",
            "positive_messages",
            "positive_images",
            "negative_messages",
            "negative_images",
        }
        if not required.issubset(row):
            raise ValueError(f"source row {index + 1} is not a reranker group")
        if len(row["positive_messages"]) != 1 or len(row["negative_messages"]) != 7:
            raise ValueError(f"source row {index + 1} is not a 1-positive/7-negative group")
        key = hashlib.sha256(f"{seed}:{index}:".encode() + line.encode()).digest()
        keyed.append((key, index, line))
    return [line for _, _, line in sorted(keyed)[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Pilot data or manifest already exists")

    selected = select_rows(args.source, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    result = {
        "objective": "score_policy_grpo_paired_pilot_data",
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(args.output),
        "query_groups": len(selected),
        "positive_per_group": 1,
        "negative_per_group": 7,
        "seed": args.seed,
        "shared_by_arms": ["score_policy_grpo", "continued_listwise"],
    }
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

