#!/usr/bin/env python3
"""Audit whether reranker scores provide usable categorical GRPO rewards.

This does not train a model. It converts each candidate's saved yes/no
probability back to a logit, forms a categorical policy over the frozen
candidate set, and estimates how often K sampled actions would have non-zero
within-group reward variance.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional

from rerank_pipeline import digest, load_labels, load_rows, save


def parse_csv_numbers(raw: str, cast, name: str) -> list:
    try:
        values = [cast(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {raw}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return values


def probability_to_logit(value: float, epsilon: float = 1e-6) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"score must be finite and in [0, 1], got {value}")
    value = min(max(value, epsilon), 1 - epsilon)
    return math.log(value) - math.log1p(-value)


def softmax(values: list[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def quantile(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def simulate_group(
    positive_probability: float,
    group_size: int,
    replicates: int,
    rng: random.Random,
) -> tuple[int, float, float]:
    active = 0
    reward_sum = 0.0
    absolute_advantage_sum = 0.0
    sample_count = group_size * replicates
    for _ in range(replicates):
        positives = sum(rng.random() < positive_probability for _ in range(group_size))
        reward_sum += positives
        if positives in {0, group_size}:
            continue
        active += 1
        reward_mean = positives / group_size
        # torch.std defaults to the sample standard deviation (correction=1).
        reward_std = math.sqrt(positives * (group_size - positives) / (group_size * (group_size - 1)))
        positive_advantage = (1 - reward_mean) / reward_std
        negative_advantage = -reward_mean / reward_std
        absolute_advantage_sum += (
            positives * abs(positive_advantage)
            + (group_size - positives) * abs(negative_advantage)
        )
    return active, reward_sum / sample_count, absolute_advantage_sum / sample_count


def audit(
    candidates_path: Path,
    predictions_path: Path,
    labels_path: Path,
    group_sizes: list[int],
    temperatures: list[float],
    replicates: int,
    seed: int,
) -> dict:
    if any(size < 2 for size in group_sizes):
        raise ValueError("all group sizes must be at least 2")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    candidates = {row["photo_id"]: row for row in load_rows(candidates_path)}
    predictions = {row["photo_id"]: row for row in load_rows(predictions_path)}
    _, labels = load_labels(labels_path)
    if set(predictions) != set(candidates):
        raise ValueError("prediction query set differs from candidates")
    missing_labels = set(candidates) - set(labels)
    if missing_labels:
        raise ValueError(f"candidate queries missing labels: {sorted(missing_labels)[:10]}")

    prepared: list[tuple[str, str, list[str], list[float]]] = []
    candidate_hits = 0
    for photo_id, candidate_row in candidates.items():
        truth = labels[photo_id]
        candidate_ids = [row["item_id"] for row in candidate_row["candidates"]]
        ranking = predictions[photo_id]["ranking"]
        ranking_ids = [row["item_id"] for row in ranking]
        if len(ranking_ids) != len(set(ranking_ids)) or set(ranking_ids) != set(candidate_ids):
            raise ValueError(f"prediction candidates differ for {photo_id}")
        score_by_item = {row["item_id"]: float(row["score"]) for row in ranking}
        logits = [probability_to_logit(score_by_item[item_id]) for item_id in candidate_ids]
        candidate_hits += int(truth in candidate_ids)
        prepared.append((photo_id, truth, candidate_ids, logits))

    configurations = []
    for temperature in temperatures:
        probabilities = []
        entropies = []
        nll_values = []
        for _, truth, candidate_ids, logits in prepared:
            policy = softmax(logits, temperature)
            entropies.append(-sum(p * math.log(max(p, 1e-300)) for p in policy))
            if truth in candidate_ids:
                probability = policy[candidate_ids.index(truth)]
                probabilities.append(probability)
                nll_values.append(-math.log(max(probability, 1e-300)))

        for group_size in group_sizes:
            expected_zero_rates = [
                probability**group_size + (1 - probability)**group_size
                for probability in probabilities
            ]
            rng = random.Random(f"{seed}:{temperature}:{group_size}")
            active = 0
            simulated_reward_means = []
            simulated_abs_advantages = []
            for probability in probabilities:
                query_active, reward_mean, abs_advantage = simulate_group(
                    probability, group_size, replicates, rng
                )
                active += query_active
                simulated_reward_means.append(reward_mean)
                simulated_abs_advantages.append(abs_advantage)
            total_groups = len(probabilities) * replicates
            configurations.append({
                "temperature": temperature,
                "group_size": group_size,
                "queries_with_positive": len(probabilities),
                "mean_positive_policy_probability": mean(probabilities),
                "p10_positive_policy_probability": quantile(probabilities, 0.10),
                "p50_positive_policy_probability": quantile(probabilities, 0.50),
                "p90_positive_policy_probability": quantile(probabilities, 0.90),
                "mean_policy_entropy": mean(entropies),
                "mean_listwise_nll": mean(nll_values),
                "expected_zero_variance_group_rate": mean(expected_zero_rates),
                "expected_active_group_rate": (
                    1 - mean(expected_zero_rates) if expected_zero_rates else None
                ),
                "simulated_active_group_rate": active / total_groups if total_groups else None,
                "simulated_mean_reward": mean(simulated_reward_means),
                "simulated_mean_absolute_advantage": mean(simulated_abs_advantages),
            })

    return {
        "objective": "categorical_score_policy_grpo_rollout_variance_audit",
        "note": (
            "Diagnostic only: saved sigmoid yes/no scores are converted back to logits, "
            "then normalized across each frozen candidate set."
        ),
        "count": len(prepared),
        "candidate_hits": candidate_hits,
        "candidate_recall": candidate_hits / len(prepared) if prepared else None,
        "candidate_misses_have_zero_reward_variance": True,
        "replicates_per_query": replicates,
        "seed": seed,
        "candidate_sha256": digest(candidates_path),
        "prediction_sha256": digest(predictions_path),
        "label_sha256": digest(labels_path),
        "configurations": configurations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--group-sizes", default="4,8,16")
    parser.add_argument("--temperatures", default="0.5,1.0,2.0")
    parser.add_argument("--replicates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.candidates,
        args.predictions,
        args.labels,
        parse_csv_numbers(args.group_sizes, int, "group sizes"),
        parse_csv_numbers(args.temperatures, float, "temperatures"),
        args.replicates,
        args.seed,
    )
    save(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
