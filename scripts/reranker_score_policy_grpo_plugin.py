#!/usr/bin/env python3
"""Register a categorical score-policy GRPO loss for ms-swift rerankers.

This is a contextual-bandit adaptation, not token-level GRPO.  For every
positive-first reranker group, it samples candidate indices from the softmax
of the reranker logits and uses exact item correctness as a binary reward.
The sampled rewards are standardized inside the group before applying the
on-policy score-function objective.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from swift.loss import BaseLoss, loss_map


def _positive_first_groups(labels: torch.Tensor, expected_size: int) -> list[tuple[int, int]]:
    labels = labels.reshape(-1)
    positive_indices = torch.nonzero(labels == 1, as_tuple=False).reshape(-1).tolist()
    if not positive_indices:
        raise ValueError("score-policy GRPO batch contains no positive sample")
    if positive_indices[0] != 0:
        raise ValueError("score-policy GRPO requires each batch to start with a positive sample")

    boundaries = positive_indices + [labels.numel()]
    groups = []
    for start, end in zip(boundaries, boundaries[1:]):
        size = end - start
        if size != expected_size:
            raise ValueError(
                f"score-policy GRPO expected group size {expected_size}, got {size}"
            )
        group_labels = labels[start:end]
        if group_labels[0].item() != 1 or torch.count_nonzero(group_labels).item() != 1:
            raise ValueError("each score-policy GRPO group must contain one positive at index 0")
        groups.append((start, end))
    return groups


def score_policy_group_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    group_size: int = 8,
    rollout_samples: int = 8,
    temperature: float = 1.0,
    std_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return the group-relative policy loss and detached rollout statistics."""
    if group_size < 2 or rollout_samples < 2:
        raise ValueError("group_size and rollout_samples must both be at least 2")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    logits = logits.reshape(-1)
    labels = labels.reshape(-1)
    if logits.numel() != labels.numel():
        raise ValueError("logit and label counts differ")
    groups = _positive_first_groups(labels, group_size)

    losses = []
    active_groups = 0
    reward_sum = 0.0
    absolute_advantage_sum = 0.0
    entropy_sum = 0.0
    positive_probability_sum = 0.0
    for start, end in groups:
        log_policy = torch.log_softmax(logits[start:end] / temperature, dim=0)
        policy = log_policy.exp()
        actions = torch.multinomial(
            policy.detach(), num_samples=rollout_samples, replacement=True
        )
        rewards = (actions == 0).to(logits.dtype)
        reward_std = rewards.std(correction=1)
        if reward_std.item() <= std_epsilon:
            advantages = torch.zeros_like(rewards)
        else:
            active_groups += 1
            advantages = (rewards - rewards.mean()) / reward_std

        # For one on-policy update, this has the same local gradient as the
        # unclipped GRPO importance-ratio objective at ratio == 1.  Rollouts
        # are deliberately not reused across optimizer steps in this Pilot.
        losses.append(-(log_policy[actions] * advantages.detach()).mean())
        reward_sum += rewards.detach().sum().item()
        absolute_advantage_sum += advantages.detach().abs().sum().item()
        entropy_sum += (-(policy.detach() * log_policy.detach()).sum()).item()
        positive_probability_sum += policy.detach()[0].item()

    loss = torch.stack(losses).mean()
    sample_count = len(groups) * rollout_samples
    stats: dict[str, object] = {
        "groups": len(groups),
        "active_groups": active_groups,
        "active_group_rate": active_groups / len(groups),
        "mean_reward": reward_sum / sample_count,
        "mean_absolute_advantage": absolute_advantage_sum / sample_count,
        "mean_policy_entropy": entropy_sum / len(groups),
        "mean_positive_policy_probability": positive_probability_sum / len(groups),
        "loss": loss.detach().item(),
    }
    return loss, stats


class ScorePolicyGRPOLoss(BaseLoss):
    """ms-swift loss wrapper for the score-policy Pilot."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.group_size = int(os.environ.get("SCORE_POLICY_GRPO_GROUP_SIZE", "8"))
        self.rollout_samples = int(
            os.environ.get("SCORE_POLICY_GRPO_ROLLOUT_SAMPLES", "8")
        )
        self.temperature = float(
            os.environ.get("SCORE_POLICY_GRPO_TEMPERATURE", "1.0")
        )
        self.stats_path = os.environ.get("SCORE_POLICY_GRPO_STATS_PATH")

    def _write_stats(self, stats: dict[str, object]) -> None:
        if not self.stats_path:
            return
        rank = os.environ.get("RANK", "0")
        path = Path(f"{self.stats_path}.rank{rank}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": int(getattr(self.trainer.state, "global_step", 0)),
            "rank": int(rank),
            **stats,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        loss, stats = score_policy_group_loss(
            outputs.logits.squeeze(-1),
            labels,
            group_size=self.group_size,
            rollout_samples=self.rollout_samples,
            temperature=self.temperature,
        )
        if self.trainer.model.training:
            self._write_stats(stats)
        return loss


loss_map["score_policy_grpo"] = ScorePolicyGRPOLoss

