"""NVIDIA RTX Video Super Resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image


RTX_QUALITY_LEVELS = ("LOW", "MEDIUM", "HIGH", "ULTRA")
RTX_RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "8k": (7680, 4320),
}

DEFAULT_RTX_RESOLUTION = "1080p"
DEFAULT_RTX_QUALITY = "ULTRA"


@dataclass(frozen=True)
class RTXUpscaleConfig:
    """Runtime configuration for RTX video upscaling."""

    resolution: str | None = DEFAULT_RTX_RESOLUTION
    width: int | None = None
    height: int | None = None
    scale: float | None = None
    quality: str = DEFAULT_RTX_QUALITY
    chunk_size: int = 8


def target_size(config: RTXUpscaleConfig, *, input_width: int, input_height: int) -> tuple[tuple[int, int], str]:
    """Resolve the requested output dimensions."""
    explicit_width = config.width is not None
    explicit_height = config.height is not None
    if explicit_width != explicit_height:
        raise ValueError("--width and --height must be provided together")
    if explicit_width and config.resolution is not None:
        raise ValueError("--resolution cannot be combined with --width/--height")

    if config.scale is not None:
        if explicit_width or config.resolution is not None:
            raise ValueError("--scale cannot be combined with --resolution or --width/--height")
        if config.scale < 1.0 or config.scale > 4.0:
            raise ValueError("--scale must be between 1.0 and 4.0")
        return _round_to_multiple(input_width * config.scale, input_height * config.scale), "scale"

    if explicit_width and explicit_height:
        return _round_to_multiple(int(config.width), int(config.height)), "custom"

    resolution = config.resolution or DEFAULT_RTX_RESOLUTION
    if resolution not in RTX_RESOLUTION_PRESETS:
        raise ValueError(f"unsupported RTX target resolution: {resolution}")
    max_width, max_height = _oriented_preset_bounds(resolution, input_width=input_width, input_height=input_height)
    return _fit_to_bounds(input_width, input_height, max_width, max_height), resolution


def run_rtx_upscale_video(*, video: Path, config: RTXUpscaleConfig) -> dict[str, Any]:
    """Upscale an input video with NVIDIA RTX Video Super Resolution."""
    if config.quality not in RTX_QUALITY_LEVELS:
        raise ValueError(f"unsupported RTX quality: {config.quality}")
    if config.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")

    frames, fps = load_video_frames(video)
    if not frames:
        raise ValueError("input video did not contain any decodable frames")

    input_width, input_height = frames[0].size
    (output_width, output_height), target = target_size(
        config,
        input_width=input_width,
        input_height=input_height,
    )
    upscaled = _run_nvvfx(frames, output_width, output_height, config)
    return {
        "frames": upscaled,
        "fps": fps,
        "input_width": input_width,
        "input_height": input_height,
        "target": target,
        "target_width": output_width,
        "target_height": output_height,
        "quality": config.quality,
    }


def load_video_frames(path: Path) -> tuple[list[Image.Image], int]:
    """Decode a video into RGB PIL frames and return frames plus rounded fps."""
    import av

    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams.video), None)
        if stream is None:
            raise ValueError("input video does not contain a video stream")
        fps = _stream_fps(stream)
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
    return frames, fps


def _run_nvvfx(frames: list[Image.Image], output_width: int, output_height: int, config: RTXUpscaleConfig) -> list[Image.Image]:
    try:
        import nvvfx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("nvidia-vfx is required for RTX Video Super Resolution") from exc

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("RTX Video Super Resolution requires a CUDA-capable NVIDIA RTX GPU")

    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    selected_quality = quality_mapping[config.quality]
    upscaled: list[Image.Image] = []

    with nvvfx.VideoSuperRes(selected_quality) as sr:
        sr.output_width = output_width
        sr.output_height = output_height
        sr.load()

        for start in range(0, len(frames), config.chunk_size):
            chunk = frames[start : start + config.chunk_size]
            batch = np.stack([np.asarray(frame, dtype=np.float32) / 255.0 for frame in chunk], axis=0)
            batch_cuda = torch.from_numpy(batch).cuda().permute(0, 3, 1, 2).contiguous()
            for frame_tensor in batch_cuda:
                dlpack_out = sr.run(frame_tensor).image
                out_tensor = torch.from_dlpack(dlpack_out).movedim(0, -1).detach().cpu().clamp(0.0, 1.0)
                array = (out_tensor.numpy() * 255.0).round().astype(np.uint8)
                upscaled.append(Image.fromarray(array, mode="RGB"))

    return upscaled


def _stream_fps(stream: Any) -> int:
    rate = stream.average_rate or stream.base_rate
    if isinstance(rate, Fraction) and rate.denominator:
        return max(1, int(round(float(rate))))
    if rate:
        return max(1, int(round(float(rate))))
    return 24


def _round_to_multiple(width: float, height: float) -> tuple[int, int]:
    return max(8, round(int(width) / 8) * 8), max(8, round(int(height) / 8) * 8)


def _oriented_preset_bounds(resolution: str, *, input_width: int, input_height: int) -> tuple[int, int]:
    width, height = RTX_RESOLUTION_PRESETS[resolution]
    if input_height > input_width:
        return height, width
    return width, height


def _fit_to_bounds(input_width: int, input_height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / input_width, max_height / input_height)
    return _round_to_multiple(input_width * scale, input_height * scale)
