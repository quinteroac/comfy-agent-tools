"""Ideogram 4 local text-to-image helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image

from comfy_agent_tools.imagegen.artifacts import tensor_to_pil


DEFAULT_IDEOGRAM4_UNET = Path("diffusion_models/ideogram4_fp8_scaled.safetensors")
DEFAULT_IDEOGRAM4_UNCOND_UNET = Path("diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors")
DEFAULT_IDEOGRAM4_CLIP = Path("text_encoders/qwen3vl_8b_fp8_scaled.safetensors")
DEFAULT_IDEOGRAM4_VAE = Path("vae/flux2-vae.safetensors")
DEFAULT_IDEOGRAM4_WIDTH = 1024
DEFAULT_IDEOGRAM4_HEIGHT = 1024
DEFAULT_IDEOGRAM4_STEPS = 20
DEFAULT_IDEOGRAM4_CFG = 7.0
DEFAULT_IDEOGRAM4_CFG_OVERRIDE_VALUE = 3.0
DEFAULT_IDEOGRAM4_CFG_OVERRIDE_START = 0.7
DEFAULT_IDEOGRAM4_CFG_OVERRIDE_END = 1.0
DEFAULT_IDEOGRAM4_SEED = 0
DEFAULT_IDEOGRAM4_MU = 0.0
DEFAULT_IDEOGRAM4_STD = 1.75
DEFAULT_IDEOGRAM4_SAMPLER = "euler"


@dataclass(frozen=True)
class Ideogram4Config:
    """Runtime configuration for local Ideogram 4 generation."""

    models_dir: Path
    unet: Path = DEFAULT_IDEOGRAM4_UNET
    uncond_unet: Path = DEFAULT_IDEOGRAM4_UNCOND_UNET
    clip: Path = DEFAULT_IDEOGRAM4_CLIP
    vae: Path = DEFAULT_IDEOGRAM4_VAE
    width: int = DEFAULT_IDEOGRAM4_WIDTH
    height: int = DEFAULT_IDEOGRAM4_HEIGHT
    steps: int = DEFAULT_IDEOGRAM4_STEPS
    cfg: float = DEFAULT_IDEOGRAM4_CFG
    cfg_override_value: float | None = DEFAULT_IDEOGRAM4_CFG_OVERRIDE_VALUE
    cfg_override_start: float = DEFAULT_IDEOGRAM4_CFG_OVERRIDE_START
    cfg_override_end: float = DEFAULT_IDEOGRAM4_CFG_OVERRIDE_END
    seed: int = DEFAULT_IDEOGRAM4_SEED
    mu: float = DEFAULT_IDEOGRAM4_MU
    std: float = DEFAULT_IDEOGRAM4_STD
    sampler: str = DEFAULT_IDEOGRAM4_SAMPLER

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        return path if path.is_absolute() else self.models_dir / path


def run_ideogram4_t2i(*, prompt: str, config: Ideogram4Config) -> list[Image.Image]:
    """Run local Ideogram 4 text-to-image generation."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from comfy_diffusion.pipelines.image.ideogram4.t2i import run

    images = run(
        models_dir=config.models_dir,
        prompt=prompt,
        width=config.width,
        height=config.height,
        steps=config.steps,
        cfg=config.cfg,
        cfg_override_value=config.cfg_override_value,
        cfg_override_start=config.cfg_override_start,
        cfg_override_end=config.cfg_override_end,
        seed=config.seed,
        mu=config.mu,
        std=config.std,
        sampler_name=config.sampler,
        unet_filename=config.unet,
        uncond_unet_filename=config.uncond_unet,
        clip_filename=config.clip,
        vae_filename=config.vae,
    )
    return [_to_pil(image) for image in images]


def build_prompt(
    *,
    high_level_description: str | None,
    style_aesthetics: str | None,
    style_lighting: str | None,
    style_medium: str | None,
    style_photo: str | None,
    style_art_style: str | None,
    style_colors: list[str],
    background: str,
    objects: list[str],
    texts: list[str],
) -> str:
    """Build an Ideogram 4 JSON prompt string from CLI bbox pieces."""
    if not high_level_description or not high_level_description.strip():
        raise ValueError("ideogram4-generate requires --prompt")
    if not background.strip():
        raise ValueError("builder mode requires --background")
    if not objects and not texts:
        raise ValueError("builder mode requires at least one --object or --text")

    caption: dict[str, Any] = {
        "high_level_description": high_level_description.strip(),
    }
    style = _style_description(
        aesthetics=style_aesthetics,
        lighting=style_lighting,
        medium=style_medium,
        photo=style_photo,
        art_style=style_art_style,
        colors=style_colors,
    )
    if style:
        caption["style_description"] = style

    elements: list[dict[str, Any]] = []
    elements.extend(_parse_object(value) for value in objects)
    elements.extend(_parse_text(value) for value in texts)
    caption["compositional_deconstruction"] = {
        "background": background.strip(),
        "elements": elements,
    }
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def load_prompt_from_args(args: Any) -> str:
    """Build the effective Ideogram 4 JSON prompt from CLI flags."""
    prompt = getattr(args, "prompt", "")
    if str(prompt).lstrip().startswith("{"):
        raise ValueError("ideogram4-generate does not accept raw JSON; pass structured CLI flags instead")
    return build_prompt(
        high_level_description=str(prompt),
        style_aesthetics=args.style_aesthetics,
        style_lighting=args.style_lighting,
        style_medium=args.style_medium,
        style_photo=args.style_photo,
        style_art_style=args.style_art_style,
        style_colors=list(args.style_color or []),
        background=args.background or "",
        objects=list(args.object or []),
        texts=list(args.text or []),
    )


def _style_description(
    *,
    aesthetics: str | None,
    lighting: str | None,
    medium: str | None,
    photo: str | None,
    art_style: str | None,
    colors: list[str],
) -> dict[str, Any]:
    if bool(photo) == bool(art_style):
        raise ValueError("builder style requires exactly one of --style-photo or --style-art-style")
    if not aesthetics or not lighting or not medium:
        raise ValueError("builder style requires --style-aesthetics, --style-lighting, and --style-medium")

    style: dict[str, Any] = {
        "aesthetics": aesthetics.strip(),
        "lighting": lighting.strip(),
    }
    if photo:
        style["photo"] = photo.strip()
        style["medium"] = medium.strip()
    else:
        style["medium"] = medium.strip()
        style["art_style"] = str(art_style).strip()
    if colors:
        style["color_palette"] = [_normalize_hex_color(value) for value in colors]
    return style


def _parse_object(value: str) -> dict[str, Any]:
    parts = value.split("|", 1)
    if len(parts) != 2:
        raise ValueError("--object must use BBOX|DESCRIPTION")
    bbox = _parse_bbox(parts[0])
    desc = parts[1].strip()
    if not desc:
        raise ValueError("--object description must not be empty")
    return {"type": "obj", "bbox": bbox, "desc": desc}


def _parse_text(value: str) -> dict[str, Any]:
    parts = value.split("|", 2)
    if len(parts) != 3:
        raise ValueError("--text must use BBOX|TEXT|DESCRIPTION")
    bbox = _parse_bbox(parts[0])
    text = parts[1].strip()
    desc = parts[2].strip()
    if not text:
        raise ValueError("--text literal must not be empty")
    if not desc:
        raise ValueError("--text description must not be empty")
    return {"type": "text", "bbox": bbox, "text": text, "desc": desc}


def _parse_bbox(value: str) -> list[int]:
    try:
        bbox = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("bbox values must be integers") from exc
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four integers: y_min,x_min,y_max,x_max")
    y_min, x_min, y_max, x_max = bbox
    if any(coord < 0 or coord > 1000 for coord in bbox):
        raise ValueError("bbox coordinates must be between 0 and 1000")
    if y_min >= y_max or x_min >= x_max:
        raise ValueError("bbox must satisfy y_min < y_max and x_min < x_max")
    return bbox


def _normalize_hex_color(value: str) -> str:
    color = value.strip()
    if len(color) != 7 or not color.startswith("#"):
        raise ValueError("style colors must use #RRGGBB format")
    try:
        int(color[1:], 16)
    except ValueError as exc:
        raise ValueError("style colors must use #RRGGBB format") from exc
    return color.upper()


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return tensor_to_pil(image)
