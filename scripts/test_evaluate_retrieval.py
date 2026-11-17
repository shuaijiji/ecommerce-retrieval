"""CPU-only checks for fixed-query selection and concatenated item indexes."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from evaluate_retrieval import choose_photo_ids, load_item_indexes, load_labels


@unittest.skipIf(torch is None, "torch is required for item-index tests")
class EvaluateRetrievalTest(unittest.TestCase):
    def make_index(self, root: Path, name: str, ids: list[str], marker: str = "same") -> Path:
        index = root / name
        index.mkdir()
        torch.save(
            {"item_ids": ids, "embeddings": torch.zeros((len(ids), 2048), dtype=torch.bfloat16)},
            index / "part.pt",
        )
        config = {
            "item_count": len(ids),
            "model_config_sha256": marker,
            "embedding_dimension": 2048,
            "image_size": 512,
            "dtype": "bfloat16",
            "item_input": "title+image",
        }
        (index / "config.json").write_text(json.dumps(config))
        (index / "manifest.json").write_text(json.dumps({
            "complete": True, "config": "config.json", "shards": ["part.pt"]
        }))
        return index

    def test_concatenate_disjoint_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self.make_index(root, "train", ["a", "b"])
            dev = self.make_index(root, "dev", ["c"])
            ids, matrix, configs = load_item_indexes([train, dev])
            self.assertEqual(ids, ["a", "b", "c"])
            self.assertEqual(tuple(matrix.shape), (3, 2048))
            self.assertEqual(len(configs), 2)
            duplicate = self.make_index(root, "duplicate", ["b"])
            with self.assertRaises(RuntimeError):
                load_item_indexes([train, duplicate])
            mismatch = self.make_index(root, "mismatch", ["d"], marker="different")
            with self.assertRaises(RuntimeError):
                load_item_indexes([train, mismatch])

    def test_jsonl_labels_and_uniform_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            path.write_text('\n'.join(json.dumps({"photo_id": str(i), "item_id": f"i{i}"}) for i in range(10)) + '\n')
            labels = load_labels(path)
            photos = {str(i): {"photo_frames": ["x"]} for i in range(10)}
            first = choose_photo_ids(labels, photos, 4, 7, "uniform")
            second = choose_photo_ids(labels, photos, 4, 7, "uniform")
            self.assertEqual(first, second)
            self.assertEqual(len(first), len(set(first)))


if __name__ == "__main__":
    unittest.main()
