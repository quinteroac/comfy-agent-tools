"""FLUX.2 Klein image generation and editing helpers."""

from __future__ import annotations

from PIL import Image

from .config import ImagegenConfig
from .runtime import require_comfy_runtime
from comfy_agent_tools.loras import apply_extra_loras

_FLUX_MAX_SHIFT = 1.15
_FLUX_MIN_SHIFT = 0.5
_LORA_STRENGTH = 1.0


def run_flux_klein_t2i(
    *,
    prompt: str,
    width: int,
    height: int,
    config: ImagegenConfig,
) -> list[Image.Image]:
    """Run FLUX.2 Klein 9B text-to-image with the configured SNOFS LoRA."""
    return _run_flux_klein(prompt=prompt, width=width, height=height, config=config)


def run_flux_klein_edit(
    *,
    prompt: str,
    image: Image.Image,
    config: ImagegenConfig,
) -> list[Image.Image]:
    """Run FLUX.2 Klein 9B distilled single-reference image editing."""
    width, height = image.size
    return _run_flux_klein(
        prompt=prompt,
        width=width,
        height=height,
        config=config,
        reference_image=image,
    )


def _run_flux_klein(
    *,
    prompt: str,
    width: int,
    height: int,
    config: ImagegenConfig,
    reference_image: Image.Image | None = None,
) -> list[Image.Image]:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("FLUX.2 Klein width and height must be divisible by 16")

    require_comfy_runtime()

    from comfy_diffusion.conditioning import reference_latent
    from comfy_diffusion.latent import empty_flux2_latent_image
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager, model_sampling_flux
    from comfy_diffusion.sampling import flux2_scheduler, get_sampler, sample_custom_simple
    from comfy_diffusion.vae import vae_decode

    mm = ModelManager(config.models_dir)

    model = mm.load_unet(config.resolve_model_path(config.unet))
    clip = mm.load_clip(config.resolve_model_path(config.clip), clip_type="flux2")
    vae = mm.load_vae(config.resolve_model_path(config.vae))
    model, clip = apply_lora(
        model,
        clip,
        config.resolve_model_path(config.lora),
        _LORA_STRENGTH,
        0.0,
    )
    model, clip = apply_extra_loras(model, clip, config.resolved_extra_loras)

    if reference_image is not None:
        from comfy_diffusion.conditioning import conditioning_zero_out, encode_prompt
        from comfy_diffusion.sampling import cfg_guider, random_noise, sample_custom
        from comfy_diffusion.vae import vae_encode

        positive, _ = encode_prompt(clip, prompt, "")
        negative = conditioning_zero_out(positive)
        ref_latent = vae_encode(vae, reference_image)
        positive = reference_latent(positive, ref_latent)
        negative = reference_latent(negative, ref_latent)
        latent = empty_flux2_latent_image(width, height, batch_size=1)
        sigmas = flux2_scheduler(config.steps, width, height)
        sampler = get_sampler(config.sampler)
        guider = cfg_guider(model, positive, negative, config.cfg)
        latent_out, _denoised = sample_custom(random_noise(config.seed), guider, sampler, sigmas, latent)
    else:
        model = model_sampling_flux(model, _FLUX_MAX_SHIFT, _FLUX_MIN_SHIFT, width, height)
        positive = _encode_flux2_prompt(clip, prompt, guidance=config.cfg)
        negative = _encode_flux2_prompt(clip, "", guidance=config.cfg)
        latent = empty_flux2_latent_image(width, height, batch_size=1)
        sigmas = flux2_scheduler(config.steps, width, height)
        sampler = get_sampler(config.sampler)
        latent_out = sample_custom_simple(
            model,
            True,
            config.seed,
            config.cfg,
            positive,
            negative,
            sampler,
            sigmas,
            latent,
        )
    return [vae_decode(vae, latent_out)]


def _encode_flux2_prompt(clip: object, text: str, *, guidance: float) -> object:
    """Encode text for FLUX.2 Klein's Qwen3 text encoder layout."""
    normalized = " " if text == "" else text
    tokens = clip.tokenize(normalized)
    if "qwen3_8b" not in tokens:
        keys = ", ".join(sorted(tokens))
        raise ValueError(f"expected FLUX.2 Klein qwen3_8b tokens, got: {keys}")
    return clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance})
