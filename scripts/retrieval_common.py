"""Shared, checked preprocessing and encoding helpers for retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


QUERY_INSTRUCTION = "给定电商视频的画面和文本，检索视频主要讲解的商品。"
TEXT_BUDGETS = (
    ("photo_caption", "视频描述", 500),
    ("photo_title", "视频标题", 150),
    ("photo_text", "视频文本", 200),
    ("photo_ocr", "画面文字", 700),
    ("photo_asr", "语音文本", 500),
)


def read_jsonl(path: Path, key: str | None = None) -> list[dict] | dict[str, dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"expected object at {path}:{line_number}")
            rows.append(row)
    if key is None:
        return rows
    return {str(row[key]): row for row in rows}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A PID-qualified temporary name keeps concurrent index workers from
    # replacing one another's temporary file.  os.replace still makes the
    # final publication atomic.
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compose_video_text(photo: dict) -> str:
    """Use per-field budgets so long OCR cannot eliminate ASR."""
    parts: list[str] = []
    seen: set[str] = set()
    for field, label, limit in TEXT_BUDGETS:
        text = clean_text(photo.get(field))
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(f"{label}：{text[:limit]}")
    return "\n".join(parts)


def letterbox(path: Path, size: int) -> Image.Image:
    """Preserve aspect ratio and pad an image to a fixed square."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), (127, 127, 127))
        canvas.paste(
            image,
            ((size - image.width) // 2, (size - image.height) // 2),
        )
        return canvas


def load_video_frames(photo: dict, data_root: Path, image_size: int) -> list[Image.Image]:
    frames = [
        letterbox(data_root / relative_path, image_size)
        for relative_path in photo.get("photo_frames") or []
    ]
    if not frames:
        raise ValueError(f"photo {photo.get('photo_id')} has no frames")
    # Qwen3-VL uses temporal patches of two frames. Explicit padding keeps
    # behavior deterministic for the many 1-, 3-, and 5-frame examples.
    if len(frames) % 2:
        frames.append(frames[-1].copy())
    return frames


def load_embedder(
    repo: Path,
    model_path: Path,
    image_size: int,
    max_length: int = 4096,
):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    return Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_length=max_length,
        max_pixels=image_size * image_size,
        total_pixels=6 * image_size * image_size,
        max_frames=6,
    )


def encode_checked(
    model,
    inputs: list[dict],
    required_visual_key: str | None,
) -> torch.Tensor:
    """Encode inputs while rejecting the upstream silent NULL fallback."""
    conversations = [
        model.format_model_input(
            text=row.get("text"),
            image=row.get("image"),
            video=row.get("video"),
            instruction=row.get("instruction"),
        )
        for row in inputs
    ]
    processed = model._preprocess_inputs(conversations)
    if required_visual_key is not None:
        visual = processed.get(required_visual_key)
        if visual is None or visual.numel() == 0:
            raise RuntimeError(
                f"preprocessing produced no {required_visual_key}; "
                "refusing a silent NULL embedding"
            )

    processed = {
        key: value.to(model.model.device) if hasattr(value, "to") else value
        for key, value in processed.items()
    }
    with torch.inference_mode():
        output = model.forward(processed)
        embeddings = model._pooling_last(
            output["last_hidden_state"], output["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    if embeddings.shape[0] != len(inputs):
        raise RuntimeError(
            f"expected {len(inputs)} embeddings, got {embeddings.shape[0]}"
        )
    if embeddings.shape[1] != 2048:
        raise RuntimeError(f"unexpected embedding dimension {embeddings.shape[1]}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("embedding contains NaN or Inf")
    return embeddings
