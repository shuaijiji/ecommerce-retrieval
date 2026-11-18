#!/usr/bin/env python3
"""Aggregate per-step score-policy rollout statistics written by the plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("no rollout statistics found")
    groups = sum(int(row["groups"]) for row in rows)
    active = sum(int(row["active_groups"]) for row in rows)

    def weighted_mean(key: str) -> float:
        return sum(float(row[key]) * int(row["groups"]) for row in rows) / groups

    return {
        "objective": "categorical_score_policy_grpo_training_pilot",
        "records": len(rows),
        "groups": groups,
        "active_groups": active,
        "active_group_rate": active / groups,
        "mean_reward": weighted_mean("mean_reward"),
        "mean_absolute_advantage": weighted_mean("mean_absolute_advantage"),
        "mean_policy_entropy": weighted_mean("mean_policy_entropy"),
        "mean_positive_policy_probability": weighted_mean(
            "mean_positive_policy_probability"
        ),
        "all_losses_finite": all(
            float("-inf") < float(row["loss"]) < float("inf") for row in rows
        ),
        "preregistered_active_rate_threshold": 0.50,
        "passes_training_signal_gate": active / groups >= 0.50,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

