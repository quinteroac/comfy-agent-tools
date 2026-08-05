"""MiniMax H3 chained-shot planning and MP4 stitching."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import tempfile

from comfy_agent_tools.videogen.artifacts import extract_last_video_frame, make_video_path, stitch_videos
from comfy_agent_tools.videogen.minimax import MiniMaxH3Config, run_i2v, run_r2v, run_t2v


DEFAULT_MINIMAX_SHOT_DURATION = 10.0
DEFAULT_MINIMAX_FPS = 24


def parse_shot_script(script: str) -> list[str]:
    """Parse JSON prompts or one prompt per ``---`` separator."""
    text = (script or "").strip()
    if not text:
        raise ValueError("multishot prompt script must not be empty")
    if text.startswith("{") or text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"multishot JSON script is invalid: {exc}") from exc
        prompts = data.get("prompts", []) if isinstance(data, dict) else data
        if isinstance(prompts, list):
            shots = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        else:
            shots = []
    else:
        shots = [part.strip() for part in re.split(r"(?m)^---\s*$", text) if part.strip()]
    if not shots:
        raise ValueError("multishot script contains no prompts")
    return shots


def plan_shots(
    script: str,
    *,
    duration: float,
    shot_duration: float = DEFAULT_MINIMAX_SHOT_DURATION,
    shot_count: int | None = None,
    fps: int = DEFAULT_MINIMAX_FPS,
) -> tuple[list[str], int]:
    """Return prompts and per-shot frame count for a target duration."""
    if duration <= 0:
        raise ValueError("duration must be greater than zero")
    if shot_duration <= 0:
        raise ValueError("shot_duration must be greater than zero")
    prompts = parse_shot_script(script)
    count = shot_count if shot_count and shot_count > 0 else math.ceil(duration / shot_duration)
    if count <= 0:
        raise ValueError("shot_count must be greater than zero")
    if len(prompts) > count:
        prompts = prompts[:count]
    while len(prompts) < count:
        prompts.append(prompts[-1])
    return prompts, max(5, round(shot_duration * fps))


def run_multishot_t2v(
    *,
    script: str,
    duration: float,
    config: MiniMaxH3Config,
    out_dir: Path,
    shot_duration: float = DEFAULT_MINIMAX_SHOT_DURATION,
    shot_count: int | None = None,
    frames_per_shot: int | None = None,
    start_images: list[Path] | None = None,
    first_mode: str = "auto",
) -> dict[str, object]:
    """Generate chained H3 shots, carrying each final frame into the next."""
    prompts, planned_frames = plan_shots(
        script,
        duration=duration,
        shot_duration=shot_duration,
        shot_count=shot_count,
    )
    frames_per_shot = frames_per_shot or planned_frames
    if frames_per_shot < 5:
        raise ValueError("frames_per_shot must be at least 5")
    initial_images = list(start_images or [])
    if first_mode not in {"auto", "t2v", "i2v", "r2v"}:
        raise ValueError("first_mode must be auto, t2v, i2v, or r2v")
    if first_mode == "auto":
        first_mode = "r2v" if len(initial_images) > 1 else "i2v" if initial_images else "t2v"
    if first_mode == "t2v" and initial_images:
        raise ValueError("first_mode=t2v does not accept start images")
    if first_mode in {"i2v", "r2v"} and not initial_images:
        raise ValueError(f"first_mode={first_mode} requires at least one start image")
    if first_mode == "i2v" and len(initial_images) != 1:
        raise ValueError("first_mode=i2v requires exactly one start image")
    with tempfile.TemporaryDirectory(prefix="minimax-h3-multishot-") as temp_dir:
        temp_root = Path(temp_dir)
        clips: list[Path] = []
        previous_frame: Path | None = None
        for index, prompt in enumerate(prompts):
            shot_dir = temp_root / f"shot-{index + 1}"
            shot_config = MiniMaxH3Config(
                **{
                    **config.__dict__,
                    "length": frames_per_shot,
                    "seed": config.seed + index,
                }
            )
            if previous_frame is None and first_mode == "t2v":
                result = run_t2v(prompt=prompt, config=shot_config, out_dir=shot_dir)
            elif previous_frame is None and first_mode == "i2v":
                result = run_i2v(image=initial_images[0], prompt=prompt, config=shot_config, out_dir=shot_dir)
            elif previous_frame is None and first_mode == "r2v":
                result = run_r2v(images=initial_images, prompt=prompt, config=shot_config, out_dir=shot_dir)
            else:
                result = run_i2v(image=previous_frame, prompt=prompt, config=shot_config, out_dir=shot_dir)
            clip = Path(result["artifact"])
            clips.append(clip)
            previous_frame = temp_root / f"shot-{index + 1}-last.png"
            extract_last_video_frame(clip, previous_frame)

        artifact = make_video_path(out_dir, prefix="comfy-videogen-minimax-h3-multishot-t2v")
        stitch_videos(clips, artifact, fps=DEFAULT_MINIMAX_FPS, target_duration=duration)
    return {
        "artifact": artifact,
        "shots": len(prompts),
        "prompts": prompts,
        "frames_per_shot": frames_per_shot,
        "duration": duration,
        "shot_duration": shot_duration,
        "first_mode": first_mode,
        "start_images": [str(path) for path in initial_images],
    }
