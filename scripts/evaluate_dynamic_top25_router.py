#!/usr/bin/env python3
"""Evaluate a budgeted Top-10 to Top-25 router without test-label leakage.

The Top-10 baseline is reconstructed by restricting the Top-25 model ranking to
the original Top-10 candidate set.  This keeps the baseline and expanded arm in
the same inference run and removes run/batch drift from the routing comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from rerank_pipeline import digest, load_labels, load_rows, save


DEFAULT_BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
POLICY_ORDER = (
    "low_margin",
    "low_top1_score",
    "route_disagreement_then_low_margin",
    "large_union_then_low_margin",
)


def _index(path: Path) -> dict[str, dict]:
    return {row["photo_id"]: row for row in load_rows(path)}


def _candidate_ids(row: dict) -> set[str]:
    return {candidate["item_id"] for candidate in row["candidates"]}


def _multimodal_top1(row: dict) -> str:
    values = [
        candidate["item_id"]
        for candidate in row["candidates"]
        if candidate.get("multimodal_rank") == 1
    ]
    if len(values) != 1:
        raise ValueError(f"expected exactly one multimodal rank-1 item: {row['photo_id']}")
    return values[0]


def prepare_examples(
    top10_candidates: dict[str, dict],
    top25_candidates: dict[str, dict],
    top25_predictions: dict[str, dict],
    labels: dict[str, str],
) -> dict[str, dict]:
    query_ids = set(top10_candidates)
    if not query_ids:
        raise ValueError("no queries")
    for name, values in (
        ("top25 candidates", top25_candidates),
        ("top25 predictions", top25_predictions),
    ):
        if set(values) != query_ids:
            raise ValueError(f"{name} query set differs from Top-10 candidates")
    missing_labels = query_ids - set(labels)
    if missing_labels:
        raise ValueError(f"queries missing labels: {sorted(missing_labels)[:10]}")

    examples = {}
    for photo_id in sorted(query_ids):
        row10 = top10_candidates[photo_id]
        row25 = top25_candidates[photo_id]
        ids10 = _candidate_ids(row10)
        ids25 = _candidate_ids(row25)
        if not ids10 < ids25:
            raise ValueError(f"Top-10 candidates must be a strict subset of Top-25: {photo_id}")

        ranking = top25_predictions[photo_id]["ranking"]
        if len(ranking) != len(ids25) or {row["item_id"] for row in ranking} != ids25:
            raise ValueError(f"invalid Top-25 ranking: {photo_id}")
        scores = [float(row["score"]) for row in ranking]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"non-finite score: {photo_id}")
        if any(left < right for left, right in zip(scores, scores[1:])):
            raise ValueError(f"unsorted Top-25 ranking: {photo_id}")

        restricted = [row for row in ranking if row["item_id"] in ids10]
        if len(restricted) != len(ids10) or len(restricted) < 2:
            raise ValueError(f"invalid same-run Top-10 restriction: {photo_id}")
        top1_score = float(restricted[0]["score"])
        margin = top1_score - float(restricted[1]["score"])
        if margin < 0:
            raise ValueError(f"negative Top-10 margin: {photo_id}")
        truth = labels[photo_id]
        examples[photo_id] = {
            "photo_id": photo_id,
            "top1_score": top1_score,
            "margin": margin,
            "route_disagreement": row10["vision_top1"] != _multimodal_top1(row10),
            "top10_candidate_count": len(ids10),
            "top25_candidate_count": len(ids25),
            "top10_correct": restricted[0]["item_id"] == truth,
            "top25_correct": ranking[0]["item_id"] == truth,
        }
    return examples


def split_ids(examples: dict[str, dict], seed: int) -> tuple[list[str], list[str]]:
    ordered = sorted(
        examples,
        key=lambda photo_id: (
            hashlib.sha256(f"{seed}:{photo_id}".encode()).hexdigest(),
            photo_id,
        ),
    )
    midpoint = len(ordered) // 2
    if midpoint == 0 or midpoint == len(ordered):
        raise ValueError("need at least two queries for calibration/test split")
    return ordered[:midpoint], ordered[midpoint:]


def policy_key(name: str, example: dict) -> tuple:
    photo_id = example["photo_id"]
    if name == "low_margin":
        return (example["margin"], photo_id)
    if name == "low_top1_score":
        return (example["top1_score"], photo_id)
    if name == "route_disagreement_then_low_margin":
        return (not example["route_disagreement"], example["margin"], photo_id)
    if name == "large_union_then_low_margin":
        return (-example["top10_candidate_count"], example["margin"], photo_id)
    raise ValueError(f"unknown policy: {name}")


def selected_ids(
    ids: list[str], examples: dict[str, dict], policy: str, budget: float
) -> set[str]:
    count = min(len(ids), max(0, round(len(ids) * budget)))
    ordered = sorted(ids, key=lambda photo_id: policy_key(policy, examples[photo_id]))
    return set(ordered[:count])


def _hash_fraction(photo_id: str, seed: int) -> float:
    value = int(hashlib.sha256(f"{seed}:router-tie:{photo_id}".encode()).hexdigest(), 16)
    return value / (1 << 256)


def fit_top1_score_threshold(
    ids: list[str], examples: dict[str, dict], budget: float, tie_seed: int
) -> dict:
    """Fit a deployable scalar cutoff using calibration features only."""
    if not 0 < budget <= 1:
        raise ValueError("deployment budget must be in (0, 1]")
    target = min(len(ids), max(1, round(len(ids) * budget)))
    ordered_scores = sorted(float(examples[photo_id]["top1_score"]) for photo_id in ids)
    cutoff = ordered_scores[target - 1]
    below_count = sum(examples[photo_id]["top1_score"] < cutoff for photo_id in ids)
    equal_count = sum(examples[photo_id]["top1_score"] == cutoff for photo_id in ids)
    tie_probability = (target - below_count) / equal_count
    return {
        "feature": "same-run restricted Top-10 top1 score",
        "operator": "expand if score < cutoff; at equality use stable hash sampling",
        "cutoff": cutoff,
        "tie_probability": tie_probability,
        "tie_seed": tie_seed,
        "calibration_target_count": target,
        "calibration_below_cutoff_count": below_count,
        "calibration_equal_cutoff_count": equal_count,
    }


def threshold_selected_ids(
    ids: list[str], examples: dict[str, dict], threshold: dict
) -> set[str]:
    cutoff = threshold["cutoff"]
    tie_probability = threshold["tie_probability"]
    tie_seed = threshold["tie_seed"]
    selected = set()
    for photo_id in ids:
        score = examples[photo_id]["top1_score"]
        if score < cutoff or (
            score == cutoff and _hash_fraction(photo_id, tie_seed) < tie_probability
        ):
            selected.add(photo_id)
    return selected


def summarize_arm(
    ids: list[str], examples: dict[str, dict], expanded: set[str]
) -> dict:
    base_correct = sum(examples[photo_id]["top10_correct"] for photo_id in ids)
    full_correct = sum(examples[photo_id]["top25_correct"] for photo_id in ids)
    correct = rescued = harmed = candidate_sum = 0
    for photo_id in ids:
        example = examples[photo_id]
        use_top25 = photo_id in expanded
        ok = example["top25_correct"] if use_top25 else example["top10_correct"]
        correct += ok
        rescued += use_top25 and not example["top10_correct"] and example["top25_correct"]
        harmed += use_top25 and example["top10_correct"] and not example["top25_correct"]
        candidate_sum += (
            example["top25_candidate_count"]
            if use_top25
            else example["top10_candidate_count"]
        )
    full_gain = full_correct - base_correct
    gain = correct - base_correct
    return {
        "count": len(ids),
        "expanded_count": len(expanded),
        "expanded_rate": len(expanded) / len(ids),
        "correct": correct,
        "system_top1": correct / len(ids),
        "top1_delta_vs_same_run_top10": gain / len(ids),
        "net_correct_vs_same_run_top10": gain,
        "rescued_from_top10": rescued,
        "harmed_vs_top10": harmed,
        "mean_candidates": candidate_sum / len(ids),
        "full_top25_gain": full_gain,
        "retained_full_top25_gain": gain / full_gain if full_gain > 0 else None,
    }


def evaluate(
    examples: dict[str, dict],
    seed: int,
    budgets: tuple[float, ...],
    deployment_budget: float = 0.25,
) -> dict:
    calibration_ids, test_ids = split_ids(examples, seed)
    curves = []
    for budget in budgets:
        if not 0 <= budget <= 1:
            raise ValueError("budgets must be between zero and one")
        candidates = []
        for policy in POLICY_ORDER:
            expanded = selected_ids(calibration_ids, examples, policy, budget)
            summary = summarize_arm(calibration_ids, examples, expanded)
            candidates.append((summary["correct"], -POLICY_ORDER.index(policy), policy, summary))
        _, _, chosen_policy, calibration = max(candidates)
        test_expanded = selected_ids(test_ids, examples, chosen_policy, budget)
        curves.append({
            "requested_budget": budget,
            "selected_policy_on_calibration": chosen_policy,
            "calibration": calibration,
            "test": summarize_arm(test_ids, examples, test_expanded),
        })

    no_expansion = summarize_arm(test_ids, examples, set())
    full_expansion = summarize_arm(test_ids, examples, set(test_ids))
    oracle = {
        "correct": sum(
            examples[photo_id]["top10_correct"] or examples[photo_id]["top25_correct"]
            for photo_id in test_ids
        )
    }
    oracle["system_top1"] = oracle["correct"] / len(test_ids)
    threshold = fit_top1_score_threshold(
        calibration_ids, examples, deployment_budget, tie_seed=seed
    )
    calibration_threshold_ids = threshold_selected_ids(calibration_ids, examples, threshold)
    test_threshold_ids = threshold_selected_ids(test_ids, examples, threshold)
    return {
        "objective": "budgeted_dynamic_top25_router_offline_evaluation",
        "split": {
            "method": "sha256(seed:photo_id), first half calibration, second half test",
            "seed": seed,
            "calibration_count": len(calibration_ids),
            "test_count": len(test_ids),
        },
        "leakage_control": {
            "router_features": [
                "same-run restricted Top-10 top1 score",
                "same-run restricted Top-10 top1-top2 margin",
                "vision/multimodal route disagreement",
                "Top-10 union candidate count",
            ],
            "top25_outcome_used_as_router_feature": False,
            "test_labels_used_for_policy_selection": False,
            "policy_candidates": list(POLICY_ORDER),
        },
        "test_same_run_top10": no_expansion,
        "test_full_top25": full_expansion,
        "test_oracle_top10_or_top25": oracle,
        "budget_curve": curves,
        "deployment_candidate": {
            "policy": "low_top1_score",
            "requested_budget": deployment_budget,
            "threshold_fitted_on": "calibration only",
            "threshold": threshold,
            "calibration_fixed_threshold": summarize_arm(
                calibration_ids, examples, calibration_threshold_ids
            ),
            "test_fixed_threshold": summarize_arm(
                test_ids, examples, test_threshold_ids
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top10-candidates", type=Path, required=True)
    parser.add_argument("--top25-candidates", type=Path, required=True)
    parser.add_argument("--top25-predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--deployment-budget",
        type=float,
        default=0.25,
        help="calibration-only budget used to export a fixed low-top1 threshold",
    )
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--budget", type=float, action="append")
    args = parser.parse_args()

    top10 = _index(args.top10_candidates)
    top25 = _index(args.top25_candidates)
    predictions = _index(args.top25_predictions)
    _, labels = load_labels(args.labels)
    examples = prepare_examples(top10, top25, predictions, labels)
    budgets = tuple(args.budget) if args.budget else DEFAULT_BUDGETS
    result = evaluate(examples, args.seed, budgets, args.deployment_budget)
    result["inputs"] = {
        "top10_candidates": str(args.top10_candidates),
        "top10_candidates_sha256": digest(args.top10_candidates),
        "top25_candidates": str(args.top25_candidates),
        "top25_candidates_sha256": digest(args.top25_candidates),
        "top25_predictions": str(args.top25_predictions),
        "top25_predictions_sha256": digest(args.top25_predictions),
        "labels": str(args.labels),
        "labels_sha256": digest(args.labels),
    }
    save(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
