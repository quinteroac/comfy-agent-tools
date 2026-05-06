"""Qwen Image Edit 2511 execution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .config import ImagegenConfig
from .runtime import require_comfy_runtime
from comfy_agent_tools.loras import apply_extra_loras

_AURA_FLOW_SHIFT = 3.1
_CFG_NORM_STRENGTH = 1.0
_LORA_STRENGTH = 1.0
_SAMPLER = "euler"
_SCHEDULER = "simple"


def run_qwen_edit(
    *,
    prompt: str,
    image: Image.Image,
    config: ImagegenConfig,
    image2: Image.Image | None = None,
    image3: Image.Image | None = None,
) -> list[Image.Image]:
    """Run Qwen Image Edit 2511 with configurable local model paths."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    require_comfy_runtime()

    from comfy_diffusion.conditioning import (
        apply_flux_kontext_multi_reference,
        encode_qwen_image_edit_plus,
    )
    from comfy_diffusion.image import flux_kontext_image_scale, image_to_tensor
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager, model_sampling_aura_flow
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode, vae_encode_tensor
    from comfy_diffusion.video import apply_cfg_norm

    mm = ModelManager(config.models_dir)

    model = mm.load_unet(config.resolve_model_path(config.unet))
    clip = mm.load_clip(config.resolve_model_path(config.clip), clip_type="qwen_image")
    vae = mm.load_vae(config.resolve_model_path(config.vae))

    model = model_sampling_aura_flow(model, shift=_AURA_FLOW_SHIFT)
    model = apply_cfg_norm(model, strength=_CFG_NORM_STRENGTH)
    model, _clip = apply_lora(
        model,
        clip,
        config.resolve_model_path(config.lora),
        _LORA_STRENGTH,
        0.0,
    )
    model, clip = apply_extra_loras(model, clip, config.resolved_extra_loras)

    image_tensor = image_to_tensor(image)
    image2_tensor = image_to_tensor(image2) if image2 is not None else None
    image3_tensor = image_to_tensor(image3) if image3 is not None else None

    scaled_image = flux_kontext_image_scale(image_tensor)
    negative = encode_qwen_image_edit_plus(
        clip, vae, scaled_image, image2_tensor, image3_tensor, prompt=""
    )
    positive = encode_qwen_image_edit_plus(
        clip, vae, scaled_image, image2_tensor, image3_tensor, prompt=prompt
    )

    negative = apply_flux_kontext_multi_reference(negative, "index_timestep_zero")
    positive = apply_flux_kontext_multi_reference(positive, "index_timestep_zero")

    latent = vae_encode_tensor(vae, scaled_image)
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        config.steps,
        config.cfg,
        _SAMPLER,
        _SCHEDULER,
        config.seed,
    )
    return [vae_decode(vae, latent_out)]


def path_arg(value: str | Path) -> Path:
    """Normalize CLI path strings."""
    return value if isinstance(value, Path) else Path(value)
