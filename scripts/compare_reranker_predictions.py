#!/usr/bin/env python3
"""Compare reranker predictions on one frozen candidate set with paired counts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rerank_pipeline import digest, load_labels, load_rows, save


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def summarize(
    predictions_path: Path,
    candidates: dict[str, dict],
    labels: dict[str, str],
) -> tuple[dict, dict[str, bool]]:
    predictions = load_rows(predictions_path)
    if {row["photo_id"] for row in predictions} != set(candidates):
        raise ValueError(f"prediction query set differs from candidates: {predictions_path}")
    correct_by_photo: dict[str, bool] = {}
    correct = hits = 0
    reciprocal_rank_sum = ndcg_sum = seconds = 0.0
    for prediction in predictions:
        photo_id = prediction["photo_id"]
        truth = labels[photo_id]
        candidate_ids = {row["item_id"] for row in candidates[photo_id]["candidates"]}
        ranking = prediction["ranking"]
        if {row["item_id"] for row in ranking} != candidate_ids or len(ranking) != len(candidate_ids):
            raise ValueError(f"invalid ranking for {photo_id} in {predictions_path}")
        scores = [float(row["score"]) for row in ranking]
        if not all(math.isfinite(score) for score in scores) or any(a < b for a, b in zip(scores, scores[1:])):
            raise ValueError(f"invalid score order for {photo_id} in {predictions_path}")
        hit = truth in candidate_ids
        ok = ranking[0]["item_id"] == truth
        rank = next((index for index, row in enumerate(ranking, 1) if row["item_id"] == truth), None)
        hits += int(hit)
        correct += int(ok)
        correct_by_photo[photo_id] = ok
        if rank is not None:
            reciprocal_rank_sum += 1 / rank
            ndcg_sum += 1 / math.log2(rank + 1)
        value = float(prediction["seconds"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid latency for {photo_id} in {predictions_path}")
        seconds += value
    count = len(predictions)
    return {
        "count": count,
        "candidate_hits": hits,
        "candidate_recall": hits / count,
        "correct": correct,
        "system_top1": correct / count,
        "candidate_rerank_accuracy": correct / hits if hits else None,
        "system_mrr": reciprocal_rank_sum / count,
        "system_ndcg": ndcg_sum / count,
        "mean_seconds_per_video": seconds / count,
        "prediction_sha256": digest(predictions_path),
    }, correct_by_photo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--prediction", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _ in args.prediction]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError("provide at least two predictions with unique names; first is the baseline")
    candidates = {row["photo_id"]: row for row in load_rows(args.candidates)}
    _, labels = load_labels(args.labels)
    missing_labels = set(candidates) - set(labels)
    if missing_labels:
        raise ValueError(f"candidate queries missing labels: {sorted(missing_labels)[:10]}")

    metrics: dict[str, dict] = {}
    correctness: dict[str, dict[str, bool]] = {}
    paths: dict[str, str] = {}
    for name, path in args.prediction:
        metrics[name], correctness[name] = summarize(path, candidates, labels)
        paths[name] = str(path)

    baseline = names[0]
    paired = {}
    for name in names[1:]:
        base = correctness[baseline]
        current = correctness[name]
        rescued = sum(not base[photo_id] and current[photo_id] for photo_id in candidates)
        harmed = sum(base[photo_id] and not current[photo_id] for photo_id in candidates)
        paired[name] = {
            "baseline": baseline,
            "rescued_from_baseline": rescued,
            "harmed_vs_baseline": harmed,
            "net_correct_vs_baseline": rescued - harmed,
            "top1_delta": metrics[name]["system_top1"] - metrics[baseline]["system_top1"],
            "mrr_delta": metrics[name]["system_mrr"] - metrics[baseline]["system_mrr"],
            "ndcg_delta": metrics[name]["system_ndcg"] - metrics[baseline]["system_ndcg"],
        }
    result = {
        "candidate_sha256": digest(args.candidates),
        "label_sha256": digest(args.labels),
        "prediction_paths": paths,
        "models": metrics,
        "paired_vs_baseline": paired,
    }
    save(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
