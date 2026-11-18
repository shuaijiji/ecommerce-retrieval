import json
import tempfile
import unittest
from pathlib import Path

from prepare_score_policy_grpo_pilot import select_rows


def row(value: int) -> str:
    return json.dumps({
        "messages": [{"role": "user", "content": str(value)}],
        "videos": [["frame.png"]],
        "positive_messages": [[{"role": "assistant", "content": "positive"}]],
        "positive_images": [["positive.png"]],
        "negative_messages": [
            [{"role": "assistant", "content": f"negative-{index}"}]
            for index in range(7)
        ],
        "negative_images": [[f"negative-{index}.png"] for index in range(7)],
    })


class PrepareScorePolicyPilotTest(unittest.TestCase):
    def test_selection_is_deterministic_and_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            source.write_text("\n".join(row(index) for index in range(10)) + "\n")
            first = select_rows(source, 4, 9)
            second = select_rows(source, 4, 9)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertEqual(len(set(first)), 4)

    def test_rejects_wrong_group_size(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            malformed = json.loads(row(1))
            malformed["negative_messages"].pop()
            source.write_text(json.dumps(malformed) + "\n")
            with self.assertRaisesRegex(ValueError, "1-positive/7-negative"):
                select_rows(source, 1, 9)


if __name__ == "__main__":
    unittest.main()

