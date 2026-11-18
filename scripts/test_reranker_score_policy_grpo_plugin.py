import unittest
from unittest.mock import patch

import torch

from reranker_score_policy_grpo_plugin import score_policy_group_loss


class ScorePolicyGRPOLossTest(unittest.TestCase):
    def test_active_group_has_finite_gradient(self):
        logits = torch.zeros(8, requires_grad=True)
        labels = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
        actions = torch.tensor([0, 1, 0, 2, 0, 3, 4, 5])
        with patch("torch.multinomial", return_value=actions):
            loss, stats = score_policy_group_loss(logits, labels)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(stats["groups"], 1)
        self.assertEqual(stats["active_groups"], 1)
        self.assertGreater(stats["mean_absolute_advantage"], 0)
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1:].mean().item(), 0)

    def test_zero_variance_group_has_zero_loss_and_gradient(self):
        logits = torch.tensor([50.0, -50.0, -50.0, -50.0], requires_grad=True)
        labels = torch.tensor([1, 0, 0, 0])
        actions = torch.zeros(8, dtype=torch.long)
        with patch("torch.multinomial", return_value=actions):
            loss, stats = score_policy_group_loss(
                logits, labels, group_size=4, rollout_samples=8
            )
        loss.backward()
        self.assertEqual(stats["active_groups"], 0)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_rejects_malformed_positive_first_group(self):
        with self.assertRaisesRegex(ValueError, "start with a positive"):
            score_policy_group_loss(
                torch.zeros(4),
                torch.tensor([0, 1, 0, 0]),
                group_size=4,
            )


if __name__ == "__main__":
    unittest.main()

