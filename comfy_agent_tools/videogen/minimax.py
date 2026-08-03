"""Local MiniMax H3 reference-to-video generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comfy_agent_tools.videogen.artifacts import make_video_path

DEFAULT_MINIMAX_WIDTH = 1344
DEFAULT_MINIMAX_HEIGHT = 768
DEFAULT_MINIMAX_LENGTH = 124
DEFAULT_MINIMAX_STEPS = 20
DEFAULT_MINIMAX_SEED = 0
DEFAULT_MINIMAX_REF_IMAGE_SIZE = "match"
DEFAULT_MINIMAX_FPS = 24


@dataclass(frozen=True)
class MiniMaxH3Config:
    """Runtime options for the local MiniMax H3 pipeline."""

    models_dir: Path
    width: int = DEFAULT_MINIMAX_WIDTH
    height: int = DEFAULT_MINIMAX_HEIGHT
    length: int = DEFAULT_MINIMAX_LENGTH
    steps: int = DEFAULT_MINIMAX_STEPS
    seed: int = DEFAULT_MINIMAX_SEED
    ref_image_size: str = DEFAULT_MINIMAX_REF_IMAGE_SIZE
    unet: Path | None = None
    text_encoder: Path | None = None
    audio_vae: Path | None = None
    video_vae: Path | None = None

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("MiniMax H3 width and height must be positive")
        if self.length <= 0 or self.steps <= 0:
            raise ValueError("MiniMax H3 length and steps must be positive")
        if self.ref_image_size not in {"match", "max"}:
            raise ValueError("MiniMax H3 ref_image_size must be 'match' or 'max'")

    def model_path(self, value: Path | None, default: str) -> Path:
        path = value if value is not None else Path(default)
        return path if path.is_absolute() else self.models_dir / path


def run_r2v(*, images: list[Path], prompt: str, config: MiniMaxH3Config, out_dir: Path) -> dict[str, Any]:
    """Generate a MiniMax H3 video with native synchronized audio."""
    config.validate()
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not images:
        raise ValueError("at least one reference image is required")
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reference image not found: {missing[0]}")

    try:
        from comfy_diffusion.image import load_image
        from comfy_diffusion.pipelines.video.minimax import h3
    except Exception as exc:
        raise RuntimeError("comfy-diffusion 2.6.0 with MiniMax H3 support is required") from exc

    reference_images = [load_image(path)[0] for path in images]
    artifact = make_video_path(out_dir, prefix="comfy-videogen-minimax-h3-r2v")
    # comfy-diffusion treats explicit override paths as already-resolved paths;
    # profile paths are intentionally relative to the configured models root.
    def resolved_override(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else config.models_dir / path

    result = h3.run(
        prompt,
        reference_images,
        models_dir=config.models_dir,
        width=config.width,
        height=config.height,
        length=config.length,
        steps=config.steps,
        seed=config.seed,
        ref_image_size=config.ref_image_size,
        unet_filename=resolved_override(config.unet),
        text_encoder_filename=resolved_override(config.text_encoder),
        audio_vae_filename=resolved_override(config.audio_vae),
        video_vae_filename=resolved_override(config.video_vae),
        output_path=artifact,
    )
    return {"artifact": artifact, "result": result}
