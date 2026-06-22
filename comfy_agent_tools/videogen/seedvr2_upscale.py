"""SeedVR2 video upscaling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


SEEDVR2_REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
SEEDVR2_RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "1440p": (1440, 2560),
    "4k": (2160, 3840),
}

DEFAULT_SEEDVR2_RESOLUTION = "1080p"
DEFAULT_SEEDVR2_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
DEFAULT_SEEDVR2_BATCH_SIZE = 5
DEFAULT_SEEDVR2_COLOR_CORRECTION = "lab"
DEFAULT_SEEDVR2_VIDEO_BACKEND = "ffmpeg"
SEEDVR2_COLOR_CORRECTION_MODES = ("lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none")
SEEDVR2_VIDEO_BACKENDS = ("opencv", "ffmpeg")


@dataclass(frozen=True)
class SeedVR2UpscaleConfig:
    """Runtime configuration for SeedVR2 video upscaling."""

    resolution: str = DEFAULT_SEEDVR2_RESOLUTION
    max_edge: int | None = None
    model: str = DEFAULT_SEEDVR2_MODEL
    model_dir: Path | None = None
    batch_size: int = DEFAULT_SEEDVR2_BATCH_SIZE
    chunk_size: int = 0
    temporal_overlap: int = 0
    cuda_device: str | None = None
    blocks_to_swap: int = 0
    color_correction: str = DEFAULT_SEEDVR2_COLOR_CORRECTION
    video_backend: str = DEFAULT_SEEDVR2_VIDEO_BACKEND
    upstream_dir: Path | None = None


def preset_values(resolution: str) -> tuple[int, int]:
    """Return SeedVR2 short edge and max edge for a named preset."""
    if resolution not in SEEDVR2_RESOLUTION_PRESETS:
        raise ValueError(f"unsupported SeedVR2 target resolution: {resolution}")
    return SEEDVR2_RESOLUTION_PRESETS[resolution]


def run_seedvr2_upscale_video(*, video: Path, output: Path, config: SeedVR2UpscaleConfig, verbose: bool = False) -> dict[str, Any]:
    """Upscale an input video using the pinned SeedVR2 upstream CLI."""
    video = video.resolve()
    output = output.resolve()
    if config.batch_size <= 0 or (config.batch_size - 1) % 4 != 0:
        raise ValueError("--batch-size must follow SeedVR2's 4n+1 pattern: 1, 5, 9, 13, ...")
    if config.chunk_size < 0:
        raise ValueError("--chunk-size must be 0 or greater")
    if config.temporal_overlap < 0:
        raise ValueError("--temporal-overlap must be 0 or greater")
    if config.blocks_to_swap < 0:
        raise ValueError("--blocks-to-swap must be 0 or greater")
    if config.color_correction not in SEEDVR2_COLOR_CORRECTION_MODES:
        raise ValueError(f"unsupported SeedVR2 color correction: {config.color_correction}")
    if config.video_backend not in SEEDVR2_VIDEO_BACKENDS:
        raise ValueError(f"unsupported SeedVR2 video backend: {config.video_backend}")

    short_edge, default_max_edge = preset_values(config.resolution)
    max_edge = config.max_edge if config.max_edge is not None else default_max_edge
    if max_edge < short_edge:
        raise ValueError("--max-edge must be greater than or equal to the target short edge")

    upstream_dir = config.upstream_dir or ensure_seedvr2_upstream()
    script = upstream_dir / "inference_cli.py"
    if not script.is_file():
        raise FileNotFoundError(f"SeedVR2 inference CLI not found: {script}")

    command = [
        sys.executable,
        str(script),
        str(video),
        "--output",
        str(output),
        "--output_format",
        "mp4",
        "--resolution",
        str(short_edge),
        "--max_resolution",
        str(max_edge),
        "--dit_model",
        config.model,
        "--batch_size",
        str(config.batch_size),
        "--chunk_size",
        str(config.chunk_size),
        "--temporal_overlap",
        str(config.temporal_overlap),
        "--color_correction",
        config.color_correction,
        "--video_backend",
        config.video_backend,
    ]
    if config.video_backend == "ffmpeg":
        command.append("--uniform_batch_size")
    if config.model_dir is not None:
        command.extend(["--model_dir", str(config.model_dir)])
    if config.cuda_device is not None:
        command.extend(["--cuda_device", config.cuda_device])
    if config.blocks_to_swap:
        command.extend(["--blocks_to_swap", str(config.blocks_to_swap), "--dit_offload_device", "cpu"])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{upstream_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.PIPE
    completed = subprocess.run(command, cwd=str(upstream_dir), env=env, stdout=stdout, stderr=stderr, text=True, check=False)
    if completed.returncode != 0:
        message = (completed.stderr or "").strip() if not verbose else ""
        raise RuntimeError(f"SeedVR2 upscale failed with exit code {completed.returncode}: {message}")
    if not output.is_file():
        raise FileNotFoundError(f"SeedVR2 did not produce output video: {output}")

    return {
        "artifact": output,
        "target": config.resolution,
        "short_edge": short_edge,
        "max_edge": max_edge,
        "model": config.model,
        "model_dir": str(config.model_dir) if config.model_dir is not None else None,
        "batch_size": config.batch_size,
        "chunk_size": config.chunk_size,
        "temporal_overlap": config.temporal_overlap,
        "color_correction": config.color_correction,
        "video_backend": config.video_backend,
        "command": command,
        "upstream_dir": str(upstream_dir),
        "upstream_commit": SEEDVR2_COMMIT,
    }


def ensure_seedvr2_upstream() -> Path:
    """Clone or update the pinned SeedVR2 upstream repository in the user cache."""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "comfy-agent-tools" / "seedvr2"
    repo_dir = cache_dir / SEEDVR2_COMMIT
    if (repo_dir / "inference_cli.py").is_file():
        return repo_dir
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch the SeedVR2 upstream repository")

    cache_dir.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    subprocess.run(["git", "clone", "--no-checkout", SEEDVR2_REPO_URL, str(repo_dir)], check=True)
    subprocess.run(["git", "checkout", SEEDVR2_COMMIT], cwd=str(repo_dir), check=True)
    return repo_dir
