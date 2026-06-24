"""NVIDIA RTX image upscaling helpers.

Reuses the NVIDIA RTX Video Super Resolution effect (``nvvfx.VideoSuperRes``)
to upscale a single still image. The effect operates per-frame, so a still image
is processed as a one-frame batch. Target-size resolution and quality presets
are shared with the video RTX upscaler for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from comfy_agent_tools.videogen.rtx_upscale import (
    RTX_QUALITY_LEVELS,
    RTX_RESOLUTION_PRESETS,
    DEFAULT_RTX_QUALITY,
    DEFAULT_RTX_RESOLUTION,
    RTXUpscaleConfig,
    target_size as _video_target_size,
)

# Public aliases used by the CLI layer.
RTX_IMAGE_QUALITY_LEVELS = RTX_QUALITY_LEVELS
RTX_IMAGE_RESOLUTION_PRESETS = RTX_RESOLUTION_PRESETS
DEFAULT_RTX_IMAGE_QUALITY = DEFAULT_RTX_QUALITY
DEFAULT_RTX_IMAGE_RESOLUTION = DEFAULT_RTX_RESOLUTION


@dataclass(frozen=True)
class RTXImageUpscaleConfig:
    """Runtime configuration for RTX image upscaling."""

    resolution: str | None = DEFAULT_RTX_RESOLUTION
    width: int | None = None
    height: int | None = None
    scale: float | None = None
    quality: str = DEFAULT_RTX_QUALITY


def target_size(config: RTXImageUpscaleConfig, *, input_width: int, input_height: int) -> tuple[tuple[int, int], str]:
    """Resolve the requested output dimensions for an image."""
    return _video_target_size(
        RTXUpscaleConfig(
            resolution=config.resolution,
            width=config.width,
            height=config.height,
            scale=config.scale,
            quality=config.quality,
        ),
        input_width=input_width,
        input_height=input_height,
    )


def run_rtx_upscale_image(
    *,
    image: Image.Image,
    config: RTXImageUpscaleConfig,
) -> dict[str, Any]:
    """Upscale a single image with NVIDIA RTX Video Super Resolution."""
    if config.quality not in RTX_QUALITY_LEVELS:
        raise ValueError(f"unsupported RTX quality: {config.quality}")

    input_width, input_height = image.size
    (output_width, output_height), target = target_size(
        config,
        input_width=input_width,
        input_height=input_height,
    )

    try:
        import nvvfx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("nvidia-vfx is required for RTX image upscaling") from exc

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RTX image upscaling requires a CUDA-capable NVIDIA RTX GPU")

    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }

    rgb = image.convert("RGB")
    array_in = np.asarray(rgb, dtype=np.float32) / 255.0
    frame_tensor = torch.from_numpy(array_in).cuda().permute(2, 0, 1).contiguous()

    with nvvfx.VideoSuperRes(quality_mapping[config.quality]) as sr:
        sr.output_width = output_width
        sr.output_height = output_height
        sr.load()
        dlpack_out = sr.run(frame_tensor).image
        out_tensor = torch.from_dlpack(dlpack_out).movedim(0, -1).detach().cpu().clamp(0.0, 1.0)
        out_array = (out_tensor.numpy() * 255.0).round().astype(np.uint8)
        upscaled = Image.fromarray(out_array, mode="RGB")

    return {
        "image": upscaled,
        "input_width": input_width,
        "input_height": input_height,
        "target": target,
        "target_width": output_width,
        "target_height": output_height,
        "quality": config.quality,
    }
