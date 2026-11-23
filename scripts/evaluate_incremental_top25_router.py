#!/usr/bin/env python3
"""Emulate cached Top-10 scoring plus Top-25 new-candidate-only scoring.

Router features and scores for existing candidates come from an independent
Top-10 inference run.  Scores for only the newly introduced candidates come
from the saved Top-25 run.  The calibration half fits one fixed score cutoff;
the test half is evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluate_dynamic_top25_router import (
    fit_top1_score_threshold,
    split_ids,
    threshold_selected_ids,
)
from rerank_pipeline import digest, load_labels, load_rows, save


def _index(path: Path) -> dict[str, dict]:
    return {row["photo_id"]: row for row in load_rows(path)}


def _candidate_ids(row: dict) -> set[str]:
    return {candidate["item_id"] for candidate in row["candidates"]}


def _validate_ranking(photo_id: str, ranking: list[dict], expected: set[str], name: str) -> None:
    if len(ranking) != len(expected) or {row["item_id"] for row in ranking} != expected:
        raise ValueError(f"invalid {name} ranking: {photo_id}")
    scores = [float(row["score"]) for row in ranking]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError(f"non-finite {name} score: {photo_id}")
    if any(left < right for left, right in zip(scores, scores[1:])):
        raise ValueError(f"unsorted {name} ranking: {photo_id}")


def _rank(rows: list[dict]) -> list[dict]:
    result = [{"item_id": row["item_id"], "score": float(row["score"])} for row in rows]
    result.sort(key=lambda row: (-row["score"], row["item_id"]))
    return result


def build_examples(
    top10_candidates: dict[str, dict],
    top10_predictions: dict[str, dict],
    top25_candidates: dict[str, dict],
    top25_predictions: dict[str, dict],
    labels: dict[str, str],
) -> tuple[dict[str, dict], list[float]]:
    query_ids = set(top10_candidates)
    if not query_ids:
        raise ValueError("no queries")
    for name, values in (
        ("Top-10 predictions", top10_predictions),
        ("Top-25 candidates", top25_candidates),
        ("Top-25 predictions", top25_predictions),
    ):
        if set(values) != query_ids:
            raise ValueError(f"{name} query set differs from Top-10 candidates")
    missing_labels = query_ids - set(labels)
    if missing_labels:
        raise ValueError(f"queries missing labels: {sorted(missing_labels)[:10]}")

    examples = {}
    score_differences = []
    for photo_id in sorted(query_ids):
        ids10 = _candidate_ids(top10_candidates[photo_id])
        ids25 = _candidate_ids(top25_candidates[photo_id])
        if not ids10 < ids25:
            raise ValueError(f"Top-10 must be a strict subset of Top-25: {photo_id}")
        ranking10 = top10_predictions[photo_id]["ranking"]
        ranking25 = top25_predictions[photo_id]["ranking"]
        _validate_ranking(photo_id, ranking10, ids10, "Top-10")
        _validate_ranking(photo_id, ranking25, ids25, "Top-25")

        scores10 = {row["item_id"]: float(row["score"]) for row in ranking10}
        scores25 = {row["item_id"]: float(row["score"]) for row in ranking25}
        score_differences.extend(abs(scores10[item_id] - scores25[item_id]) for item_id in ids10)
        new_rows = [row for row in ranking25 if row["item_id"] not in ids10]
        incremental = _rank(list(ranking10) + new_rows)
        truth = labels[photo_id]
        examples[photo_id] = {
            "photo_id": photo_id,
            "top1_score": float(ranking10[0]["score"]),
            "top10_candidate_count": len(ids10),
            "top25_candidate_count": len(ids25),
            "top10_ranking": [row["item_id"] for row in ranking10],
            "incremental_ranking": [row["item_id"] for row in incremental],
            "full_top25_ranking": [row["item_id"] for row in ranking25],
            "truth": truth,
        }
    return examples, score_differences


def _exact_p_value(rescued: int, harmed: int) -> float:
    discordant = rescued + harmed
    if not discordant:
        return 1.0
    smaller = min(rescued, harmed)
    return min(
        1.0,
        2 * sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2 ** discordant),
    )


def summarize(
    ids: list[str], examples: dict[str, dict], expanded: set[str], expanded_mode: str
) -> dict:
    if expanded_mode not in {"incremental", "full_top25"}:
        raise ValueError(f"unknown expanded mode: {expanded_mode}")
    correct = hits = rescued = harmed = 0
    top3_correct = top3_rescued = top3_harmed = 0
    reciprocal_rank_sum = ndcg_sum = candidate_sum = 0.0
    for photo_id in ids:
        example = examples[photo_id]
        base = example["top10_ranking"]
        use_expanded = photo_id in expanded
        if use_expanded:
            ranking = example[
                "incremental_ranking" if expanded_mode == "incremental" else "full_top25_ranking"
            ]
            candidate_sum += example["top25_candidate_count"]
        else:
            ranking = base
            candidate_sum += example["top10_candidate_count"]
        truth = example["truth"]
        base_ok = base[0] == truth
        ok = ranking[0] == truth
        hit = truth in ranking
        rank = ranking.index(truth) + 1 if hit else None
        base_top3 = truth in base[:3]
        current_top3 = truth in ranking[:3]
        top3_correct += current_top3
        top3_rescued += not base_top3 and current_top3
        top3_harmed += base_top3 and not current_top3
        correct += ok
        hits += hit
        rescued += use_expanded and not base_ok and ok
        harmed += use_expanded and base_ok and not ok
        if rank is not None:
            reciprocal_rank_sum += 1 / rank
            ndcg_sum += 1 / math.log2(rank + 1)
    count = len(ids)
    base_correct = sum(examples[photo_id]["top10_ranking"][0] == examples[photo_id]["truth"] for photo_id in ids)
    return {
        "count": count,
        "expanded_count": len(expanded),
        "expanded_rate": len(expanded) / count,
        "candidate_hits": hits,
        "candidate_recall": hits / count,
        "correct": correct,
        "system_top1": correct / count,
        "correct_top3": top3_correct,
        "system_top3": top3_correct / count,
        "top3_rescued_from_top10": top3_rescued,
        "top3_harmed_vs_top10": top3_harmed,
        "top3_net_vs_top10": top3_rescued - top3_harmed,
        "top3_paired_exact_two_sided_p": _exact_p_value(top3_rescued, top3_harmed),
        "system_mrr": reciprocal_rank_sum / count,
        "system_ndcg": ndcg_sum / count,
        "net_correct_vs_top10": correct - base_correct,
        "top1_delta_vs_top10": (correct - base_correct) / count,
        "rescued_from_top10": rescued,
        "harmed_vs_top10": harmed,
        "paired_exact_two_sided_p": _exact_p_value(rescued, harmed),
        "mean_candidates": candidate_sum / count,
    }


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    count = len(ordered)
    return {
        "count": count,
        "mean": sum(ordered) / count,
        "p50": ordered[math.ceil(count * .50) - 1],
        "p95": ordered[math.ceil(count * .95) - 1],
        "p99": ordered[math.ceil(count * .99) - 1],
        "max": ordered[-1],
        "count_gt_1e-6": sum(value > 1e-6 for value in ordered),
        "count_gt_1e-4": sum(value > 1e-4 for value in ordered),
    }


def evaluate(examples: dict[str, dict], differences: list[float], seed: int, budget: float) -> dict:
    calibration_ids, test_ids = split_ids(examples, seed)
    threshold = fit_top1_score_threshold(calibration_ids, examples, budget, tie_seed=seed)
    calibration_expanded = threshold_selected_ids(calibration_ids, examples, threshold)
    test_expanded = threshold_selected_ids(test_ids, examples, threshold)
    return {
        "objective": "cached_top10_plus_new_top25_candidates_offline_emulation",
        "split": {
            "method": "sha256(seed:photo_id), first half calibration, second half test",
            "seed": seed,
            "calibration_count": len(calibration_ids),
            "test_count": len(test_ids),
        },
        "router": {
            "policy": "low_top1_score",
            "requested_budget": budget,
            "threshold_fitted_on": "independent Top-10 calibration predictions only",
            "threshold": threshold,
        },
        "calibration_dynamic_incremental": summarize(
            calibration_ids, examples, calibration_expanded, "incremental"
        ),
        "test_top10": summarize(test_ids, examples, set(), "incremental"),
        "test_dynamic_incremental": summarize(
            test_ids, examples, test_expanded, "incremental"
        ),
        "test_all_incremental_top25": summarize(
            test_ids, examples, set(test_ids), "incremental"
        ),
        "test_all_full_rerun_top25": summarize(
            test_ids, examples, set(test_ids), "full_top25"
        ),
        "common_top10_score_abs_diff_between_runs": _distribution(differences),
        "limitations": [
            "new-candidate scores are reused from the saved full Top-25 run",
            "this validates cached-score ranking semantics, not runtime latency",
            "a real incremental inference Pilot is still required",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top10-candidates", type=Path, required=True)
    parser.add_argument("--top10-predictions", type=Path, required=True)
    parser.add_argument("--top25-candidates", type=Path, required=True)
    parser.add_argument("--top25-predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--budget", type=float, default=.25)
    args = parser.parse_args()

    top10_candidates = _index(args.top10_candidates)
    top10_predictions = _index(args.top10_predictions)
    top25_candidates = _index(args.top25_candidates)
    top25_predictions = _index(args.top25_predictions)
    _, labels = load_labels(args.labels)
    examples, differences = build_examples(
        top10_candidates, top10_predictions, top25_candidates, top25_predictions, labels
    )
    result = evaluate(examples, differences, args.seed, args.budget)
    result["inputs"] = {
        name: {"path": str(path), "sha256": digest(path)}
        for name, path in (
            ("top10_candidates", args.top10_candidates),
            ("top10_predictions", args.top10_predictions),
            ("top25_candidates", args.top25_candidates),
            ("top25_predictions", args.top25_predictions),
            ("labels", args.labels),
        )
    }
    save(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
