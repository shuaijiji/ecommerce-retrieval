#!/usr/bin/env python3
"""Run a checked Qwen3-VL-Embedding smoke test on one labeled video."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


TEXT_FIELDS = (
    "photo_caption",
    "photo_title",
    "photo_text",
    "photo_ocr",
    "photo_asr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/code/ecommerce-retrieval/Qwen3-VL-Embedding"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/code/ecommerce-retrieval/models/Qwen3-VL-Embedding-2B"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/code/ecommerce-retrieval/data/eval/test_small_package"
        ),
    )
    parser.add_argument("--photo-id")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-text-chars", type=int, default=4000)
    return parser.parse_args()


def read_jsonl(path: Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row[key])] = row
    return rows


def letterbox(path: Path, size: int) -> Image.Image:
    """Preserve aspect ratio and pad frames to one stackable size."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        canvas.paste(
            image,
            ((size - image.width) // 2, (size - image.height) // 2),
        )
        return canvas


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    items = read_jsonl(args.data_root / "item_info.jsonl", "item_id")
    photos = read_jsonl(args.data_root / "photo_info.jsonl", "photo_id")
    with (args.data_root / "label.json").open("r", encoding="utf-8") as handle:
        labels = json.load(handle)

    if args.photo_id:
        label = next(
            row for row in labels if str(row["photo_id"]) == args.photo_id
        )
    else:
        label = labels[0]

    photo_id = str(label["photo_id"])
    item_id = str(label["item_id"])
    photo = photos[photo_id]
    item = items[item_id]
    frame_paths = [args.data_root / path for path in photo["photo_frames"]]
    frames = [letterbox(path, args.image_size) for path in frame_paths]
    video_text = "\n".join(
        str(photo[field]).strip()
        for field in TEXT_FIELDS
        if photo.get(field) and str(photo[field]).strip()
    )[: args.max_text_chars]

    print("photo_id:", photo_id)
    print("item_id:", item_id)
    print("item title:", item["item_title"])
    print("original frame sizes:")
    for path in frame_paths:
        with Image.open(path) as image:
            print(" ", path.name, image.size)
    print("normalized frame sizes:", [image.size for image in frames])

    model = Qwen3VLEmbedder(
        model_name_or_path=str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_length=4096,
        max_pixels=args.image_size * args.image_size,
        total_pixels=6 * args.image_size * args.image_size,
        max_frames=6,
    )

    conversation = [
        model.format_model_input(
            text=video_text,
            video=frames,
            instruction=(
                "给定电商视频的画面和文本，检索视频主要讲解的商品。"
            ),
        )
    ]
    processed = model._preprocess_inputs(conversation)
    video_pixels = processed.get("pixel_values_videos")
    if video_pixels is None or video_pixels.numel() == 0:
        raise RuntimeError("video preprocessing produced no pixel_values_videos")

    print("pixel_values_videos:", tuple(video_pixels.shape))
    print("video_grid_thw:", processed["video_grid_thw"].tolist())
    processed = {
        key: value.to(model.model.device) if hasattr(value, "to") else value
        for key, value in processed.items()
    }

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.forward(processed)
        video_embedding = model._pooling_last(
            output["last_hidden_state"], output["attention_mask"]
        )
        video_embedding = F.normalize(video_embedding, p=2, dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    item_embedding = model.process(
        [
            {
                "text": item["item_title"],
                "image": str(args.data_root / item["item_image"]),
            }
        ]
    )
    similarity = (video_embedding.float() @ item_embedding.float().T).item()

    print("embedding shape:", tuple(video_embedding.shape))
    print("finite:", torch.isfinite(video_embedding).all().item())
    print("L2 norm:", video_embedding.float().norm(dim=-1).tolist())
    print("video inference seconds:", round(elapsed, 3))
    print(
        "peak memory GiB:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    )
    print("positive similarity:", round(similarity, 6))


if __name__ == "__main__":
    main()
