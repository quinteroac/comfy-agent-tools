"""Image artifact helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any

from PIL import Image

from .config import DEFAULT_SEED_COLOR


def create_seed_image(width: int, height: int) -> Image.Image:
    """Create the neutral image used to drive text-to-image through Qwen Edit."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    return Image.new("RGB", (width, height), DEFAULT_SEED_COLOR)


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load an image from disk as an RGB PIL image."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def tensor_to_pil(image_tensor: Any) -> Image.Image:
    """Convert a ComfyUI BHWC tensor to the first PIL image in the batch."""
    import numpy as np

    tensor = image_tensor
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        array = tensor.numpy()
    else:
        array = np.asarray(tensor)

    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected BHWC/HWC image tensor, got shape {array.shape!r}")

    array = (array.clip(0.0, 1.0) * 255.0).round().astype("uint8")
    if array.shape[-1] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)


def save_images(images: list[Image.Image], out_dir: str | Path, *, prefix: str) -> list[Path]:
    """Save PIL images as PNG files and return their paths."""
    if not images:
        raise ValueError("no images were produced")

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    paths: list[Path] = []
    for index, image in enumerate(images, start=1):
        path = output_dir / f"{prefix}-{stamp}-{suffix}-{index:02d}.png"
        image.save(path, format="PNG")
        paths.append(path)
    return paths
