"""Krea2 Turbo text-to-image execution helpers."""

from __future__ import annotations

from PIL import Image

from .krea2_config import Krea2Config
from .runtime import require_comfy_runtime


def run_krea2_t2i(
    *,
    prompt: str,
    width: int,
    height: int,
    config: Krea2Config,
) -> list[Image.Image]:
    """Run Krea2 Turbo text-to-image with the configured local model paths."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    require_comfy_runtime()

    from comfy_diffusion.pipelines.image.krea2.turbo import run as run_krea2

    images = run_krea2(
        models_dir=config.models_dir,
        prompt=prompt,
        width=width,
        height=height,
        steps=config.steps,
        cfg=config.cfg,
        sampler_name=config.sampler,
        scheduler=config.scheduler,
        denoise=config.denoise,
        seed=config.seed,
        rebalance_multiplier=config.rebalance_multiplier,
        unet_filename=str(config.resolve_model_path(config.unet)),
        clip_filename=str(config.resolve_model_path(config.clip)),
        vae_filename=str(config.resolve_model_path(config.vae)),
    )
    # The Krea2 pipeline returns decoded images via comfy_diffusion.vae.vae_decode,
    # which already yields PIL.Image instances.
    return [image if isinstance(image, Image.Image) else _to_pil(image) for image in images]


def _to_pil(image: object) -> Image.Image:
    """Convert a ComfyUI BHWC float tensor to a PIL image as a fallback."""
    from .artifacts import tensor_to_pil

    return tensor_to_pil(image)
