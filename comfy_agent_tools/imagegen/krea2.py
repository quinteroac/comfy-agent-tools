"""Krea2 Turbo text-to-image execution helpers."""

from __future__ import annotations

from PIL import Image

from comfy_agent_tools.loras import apply_extra_loras
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

    from comfy_diffusion.conditioning import (
        KREA2_REBALANCE_DEFAULT_WEIGHTS,
        conditioning_zero_out,
        encode_prompt,
        rebalance_krea2_conditioning,
    )
    from comfy_diffusion.latent import empty_latent_image
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    resolved_extra_loras = config.resolved_extra_loras
    mm = ModelManager(config.models_dir)
    model = mm.load_unet(config.resolve_model_path(config.unet))
    clip = mm.load_clip(config.resolve_model_path(config.clip), clip_type="krea2")
    vae = mm.load_vae(config.resolve_model_path(config.vae))
    model, clip = apply_extra_loras(model, clip, resolved_extra_loras)

    positive = encode_prompt(clip, prompt)
    positive = rebalance_krea2_conditioning(
        positive,
        multiplier=config.rebalance_multiplier,
        per_layer_weights=KREA2_REBALANCE_DEFAULT_WEIGHTS,
    )
    negative = conditioning_zero_out(positive)
    latent = empty_latent_image(width, height, batch_size=1)
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        config.steps,
        config.cfg,
        config.sampler,
        config.scheduler,
        config.seed,
        denoise=config.denoise,
    )
    images = [vae_decode(vae, latent_out)]
    # The Krea2 pipeline returns decoded images via comfy_diffusion.vae.vae_decode,
    # which already yields PIL.Image instances.
    return [image if isinstance(image, Image.Image) else _to_pil(image) for image in images]


def _to_pil(image: object) -> Image.Image:
    """Convert a ComfyUI BHWC float tensor to a PIL image as a fallback."""
    from .artifacts import tensor_to_pil

    return tensor_to_pil(image)
