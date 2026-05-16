"""Anima text-to-image execution helpers."""

from __future__ import annotations

from PIL import Image

from .config import ImagegenConfig
from .runtime import require_comfy_runtime
from comfy_agent_tools.loras import apply_extra_loras

_LORA_STRENGTH = 1.0


def run_anima_t2i(
    *,
    prompt: str,
    width: int,
    height: int,
    config: ImagegenConfig,
) -> list[Image.Image]:
    """Run Anima text-to-image with the configured local model paths."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    require_comfy_runtime()

    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.latent import empty_latent_image
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode

    mm = ModelManager(config.models_dir)

    model = mm.load_unet(config.resolve_model_path(config.unet))
    clip = mm.load_clip(config.resolve_model_path(config.clip), clip_type="stable_diffusion")
    vae = mm.load_vae(config.resolve_model_path(config.vae))

    if config.lora is not None:
        from comfy_diffusion.lora import apply_lora

        model, _clip = apply_lora(
            model,
            clip,
            config.resolve_model_path(config.lora),
            _LORA_STRENGTH,
            0.0,
        )
    model, clip = apply_extra_loras(model, clip, config.resolved_extra_loras)

    positive, negative = encode_prompt(clip, prompt, "")
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
    )
    return [vae_decode(vae, latent_out)]
