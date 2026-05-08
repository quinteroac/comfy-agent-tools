"""LTX 2.3 pipeline wrappers for comfy-videogen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_agent_tools.loras import apply_extra_loras
from .config import VideogenConfig

_SIGMAS_PASS1 = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_SIGMAS_PASS2 = "0.85, 0.7250, 0.4219, 0.0"


def run_t2v(*, prompt: str, config: VideogenConfig) -> dict[str, Any]:
    """Run LTX 2.3 text-to-video with the configured 10Eros checkpoint."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    from comfy_diffusion.pipelines.video.ltx.ltx23 import t2v

    return t2v.run(
        models_dir=config.models_dir,
        prompt=prompt,
        negative_prompt=config.negative_prompt,
        width=config.width,
        height=config.height,
        length=config.length,
        fps=config.fps,
        cfg=config.cfg,
        seed=config.seed,
        distilled_lora_strength=config.distilled_lora_strength,
        te_lora_strength=config.te_lora_strength,
        unet_filename=str(config.resolve_model_path(config.checkpoint)),
        text_encoder_filename=str(config.resolve_model_path(config.text_encoder)),
        distilled_lora_filename=str(config.resolve_model_path(config.distilled_lora)),
        te_lora_filename=str(config.resolve_model_path(config.te_lora)),
        upscaler_filename=str(config.resolve_model_path(config.upscaler)),
    )


def run_i2v(*, image: Path, prompt: str, config: VideogenConfig) -> dict[str, Any]:
    """Run LTX 2.3 image-to-video with the configured 10Eros checkpoint."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    from comfy_diffusion.pipelines.video.ltx.ltx23 import i2v

    return i2v.run(
        models_dir=config.models_dir,
        image=image,
        prompt=prompt,
        negative_prompt=config.negative_prompt,
        width=config.width,
        height=config.height,
        length=config.length,
        fps=config.fps,
        cfg=config.cfg,
        seed=config.seed,
        distilled_lora_strength=config.distilled_lora_strength,
        te_lora_strength=config.te_lora_strength,
        unet_filename=str(config.resolve_model_path(config.checkpoint)),
        text_encoder_filename=str(config.resolve_model_path(config.text_encoder)),
        distilled_lora_filename=str(config.resolve_model_path(config.distilled_lora)),
        te_lora_filename=str(config.resolve_model_path(config.te_lora)),
        upscaler_filename=str(config.resolve_model_path(config.upscaler)),
    )


def run_ia2av(*, image: Path, audio: Path, prompt: str, config: VideogenConfig) -> dict[str, Any]:
    """Run LTX 2.3 image+audio-to-video with the configured 10Eros checkpoint."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    if not audio.is_file():
        raise FileNotFoundError(f"input audio not found: {audio}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    from comfy_diffusion.pipelines.video.ltx.ltx23 import ia2v

    return ia2v.run(
        models_dir=config.models_dir,
        image=image,
        audio_path=audio,
        prompt=prompt,
        negative_prompt=config.negative_prompt,
        width=config.width,
        height=config.height,
        length=config.length,
        fps=config.fps,
        cfg=config.cfg,
        seed=config.seed,
        audio_start_time=config.audio_start_time,
        audio_duration=config.audio_duration,
        distilled_lora_strength=config.distilled_lora_strength,
        te_lora_strength=config.te_lora_strength,
        unet_filename=str(config.resolve_model_path(config.checkpoint)),
        text_encoder_filename=str(config.resolve_model_path(config.text_encoder)),
        distilled_lora_filename=str(config.resolve_model_path(config.distilled_lora)),
        te_lora_filename=str(config.resolve_model_path(config.te_lora)),
        upscaler_filename=str(config.resolve_model_path(config.upscaler)),
    )


def run_motion_track(*, image: Path, control_video: Path, prompt: str, config: VideogenConfig) -> dict[str, Any]:
    """Run LTX 2.3 image-to-video guided by a motion-track IC-LoRA control video."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    if not control_video.is_file():
        raise FileNotFoundError(f"control video not found: {control_video}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    ic_loader, add_ic_guide = _require_ic_lora_helpers()

    from comfy_diffusion.audio import (
        ltxv_audio_vae_decode,
        ltxv_concat_av_latent,
        ltxv_empty_latent_audio,
        ltxv_separate_av_latent,
    )
    from comfy_diffusion.conditioning import (
        encode_prompt,
        ltxv_conditioning,
        ltxv_crop_guides,
        ltxv_img_to_video,
    )
    from comfy_diffusion.image import load_image
    from comfy_diffusion.latent import ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import cfg_guider, get_sampler, manual_sigmas, random_noise, sample_custom
    from comfy_diffusion.vae import vae_decode_batch_tiled
    from comfy_diffusion.video import load_video

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    mm = ModelManager(config.models_dir)
    checkpoint_path = config.resolve_model_path(config.checkpoint)
    text_encoder_path = config.resolve_model_path(config.text_encoder)
    ic_lora_path = config.resolve_model_path(config.ic_lora)

    ckpt = mm.load_checkpoint_from_path(checkpoint_path)
    model = ckpt.model
    vae = ckpt.vae
    if vae is None:
        raise RuntimeError("checkpoint did not provide a video VAE")

    audio_vae = mm.load_ltxv_audio_vae(checkpoint_path)
    clip = mm.load_ltxav_text_encoder(text_encoder_path, checkpoint_path)
    upscaler_model = mm.load_latent_upscale_model(config.resolve_model_path(config.upscaler))

    model, _ = apply_lora(
        model,
        clip,
        config.resolve_model_path(config.distilled_lora),
        config.distilled_lora_strength,
        0.0,
    )
    _, clip = apply_lora(
        model,
        clip,
        config.resolve_model_path(config.te_lora),
        0.0,
        config.te_lora_strength,
    )
    model, clip = apply_extra_loras(model, clip, config.resolved_extra_loras)
    model, latent_downscale_factor = ic_loader(model, ic_lora_path, config.attention_strength)

    image_tensor, _ = load_image(image)
    control_tensor = load_video(control_video)

    base_positive, base_negative = encode_prompt(clip, prompt, config.negative_prompt)
    base_positive, base_negative = ltxv_conditioning(
        base_positive,
        base_negative,
        frame_rate=config.fps,
    )
    positive1, negative1, video_latent = ltxv_img_to_video(
        base_positive,
        base_negative,
        image_tensor,
        vae,
        width=config.width,
        height=config.height,
        length=config.length,
        strength=1.0,
    )
    positive1, negative1, video_latent = add_ic_guide(
        positive1,
        negative1,
        vae,
        video_latent,
        control_tensor,
        strength=config.attention_strength,
        latent_downscale_factor=latent_downscale_factor,
    )
    positive1, negative1, video_latent = ltxv_crop_guides(positive1, negative1, video_latent)

    audio_latent = ltxv_empty_latent_audio(
        audio_vae,
        frames_number=config.length,
        frame_rate=config.fps,
    )
    av_latent1 = ltxv_concat_av_latent(video_latent, audio_latent)

    guider1 = cfg_guider(model, positive1, negative1, config.cfg)
    noise1 = random_noise(config.seed)
    sigmas1 = manual_sigmas(_SIGMAS_PASS1)
    sampler1 = get_sampler("euler_ancestral_cfg_pp")
    _, denoised1 = sample_custom(noise1, guider1, sampler1, sigmas1, av_latent1)

    video_latent1, audio_latent1 = ltxv_separate_av_latent(denoised1)
    video_upscaled = ltxv_latent_upsample(video_latent1, upscaler_model, vae)
    positive2, negative2, video_upscaled = add_ic_guide(
        base_positive,
        base_negative,
        vae,
        video_upscaled,
        control_tensor,
        strength=config.attention_strength,
        latent_downscale_factor=latent_downscale_factor,
    )
    positive2, negative2, video_upscaled = ltxv_crop_guides(positive2, negative2, video_upscaled)
    av_latent2 = ltxv_concat_av_latent(video_upscaled, audio_latent1)

    guider2 = cfg_guider(model, positive2, negative2, config.cfg)
    noise2 = random_noise(config.seed)
    sigmas2 = manual_sigmas(_SIGMAS_PASS2)
    sampler2 = get_sampler("euler_cfg_pp")
    _, denoised2 = sample_custom(noise2, guider2, sampler2, sigmas2, av_latent2)

    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised2)
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)
    return {"frames": frames, "audio": audio}


def _require_ic_lora_helpers() -> tuple[Any, Any]:
    """Load comfy-diffusion IC-LoRA helpers, or raise a clean missing dependency."""
    try:
        from comfy_diffusion.conditioning import ltx_add_video_ic_lora_guide
        from comfy_diffusion.lora import apply_ic_lora_model_only
    except ImportError as exc:
        raise ModuleNotFoundError(
            "installed comfy-diffusion does not expose IC-LoRA helpers; install comfy-diffusion v2.2.0 or newer "
            "for LTXICLoRALoaderModelOnly and LTXAddVideoICLoRAGuide wrappers"
        ) from exc
    return apply_ic_lora_model_only, ltx_add_video_ic_lora_guide


def run_flf2v(
    *,
    first_image: Path,
    last_image: Path,
    prompt: str,
    config: VideogenConfig,
    first_frame_strength: float = 0.7,
    last_frame_strength: float = 0.7,
) -> dict[str, Any]:
    """Run an experimental first/last-frame pipeline adapted for the 10Eros checkpoint."""
    if not first_image.is_file():
        raise FileNotFoundError(f"first image not found: {first_image}")
    if not last_image.is_file():
        raise FileNotFoundError(f"last image not found: {last_image}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from comfy_diffusion.audio import (
        ltxv_audio_vae_decode,
        ltxv_concat_av_latent,
        ltxv_empty_latent_audio,
        ltxv_separate_av_latent,
    )
    from comfy_diffusion.conditioning import (
        encode_prompt,
        ltxv_add_guide,
        ltxv_conditioning,
        ltxv_crop_guides,
    )
    from comfy_diffusion.image import load_image, ltxv_preprocess
    from comfy_diffusion.latent import ltxv_empty_latent_video, ltxv_latent_upsample
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import cfg_guider, get_sampler, manual_sigmas, random_noise, sample_custom
    from comfy_diffusion.vae import vae_decode_batch_tiled

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    mm = ModelManager(config.models_dir)
    checkpoint_path = config.resolve_model_path(config.checkpoint)
    text_encoder_path = config.resolve_model_path(config.text_encoder)

    ckpt = mm.load_checkpoint_from_path(checkpoint_path)
    model = ckpt.model
    vae = ckpt.vae
    if vae is None:
        raise RuntimeError("checkpoint did not provide a video VAE")

    audio_vae = mm.load_ltxv_audio_vae(checkpoint_path)
    clip = mm.load_ltxav_text_encoder(text_encoder_path, checkpoint_path)
    upscaler_model = mm.load_latent_upscale_model(config.resolve_model_path(config.upscaler))

    model, _ = apply_lora(
        model,
        clip,
        config.resolve_model_path(config.distilled_lora),
        config.distilled_lora_strength,
        0.0,
    )
    _, clip = apply_lora(
        model,
        clip,
        config.resolve_model_path(config.te_lora),
        0.0,
        config.te_lora_strength,
    )
    model, clip = apply_extra_loras(model, clip, config.resolved_extra_loras)

    first_tensor, _ = load_image(first_image)
    last_tensor, _ = load_image(last_image)
    first_preprocessed = ltxv_preprocess(first_tensor, config.width, config.height)
    last_preprocessed = ltxv_preprocess(last_tensor, config.width, config.height)

    base_positive, base_negative = encode_prompt(clip, prompt, config.negative_prompt)
    base_positive, base_negative = ltxv_conditioning(
        base_positive,
        base_negative,
        frame_rate=config.fps,
    )

    def _apply_frame_guides(video_latent: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        positive, negative, guided = ltxv_add_guide(
            base_positive,
            base_negative,
            vae,
            video_latent,
            first_preprocessed,
            frame_idx=0,
            strength=first_frame_strength,
        )
        positive, negative, guided = ltxv_add_guide(
            positive,
            negative,
            vae,
            guided,
            last_preprocessed,
            frame_idx=-1,
            strength=last_frame_strength,
        )
        return ltxv_crop_guides(positive, negative, guided)

    video_latent = ltxv_empty_latent_video(
        width=config.width,
        height=config.height,
        length=config.length,
    )
    positive1, negative1, video_latent = _apply_frame_guides(video_latent)

    audio_latent = ltxv_empty_latent_audio(
        audio_vae,
        frames_number=config.length,
        frame_rate=config.fps,
    )
    av_latent1 = ltxv_concat_av_latent(video_latent, audio_latent)

    guider1 = cfg_guider(model, positive1, negative1, config.cfg)
    noise1 = random_noise(config.seed)
    sigmas1 = manual_sigmas(_SIGMAS_PASS1)
    sampler1 = get_sampler("euler_ancestral_cfg_pp")
    _, denoised1 = sample_custom(noise1, guider1, sampler1, sigmas1, av_latent1)

    video_latent1, audio_latent1 = ltxv_separate_av_latent(denoised1)
    video_upscaled = ltxv_latent_upsample(video_latent1, upscaler_model, vae)
    positive2, negative2, video_upscaled = _apply_frame_guides(video_upscaled)
    av_latent2 = ltxv_concat_av_latent(video_upscaled, audio_latent1)

    guider2 = cfg_guider(model, positive2, negative2, config.cfg)
    noise2 = random_noise(config.seed)
    sigmas2 = manual_sigmas(_SIGMAS_PASS2)
    sampler2 = get_sampler("euler_cfg_pp")
    _, denoised2 = sample_custom(noise2, guider2, sampler2, sigmas2, av_latent2)

    video_latent_out, audio_latent_out = ltxv_separate_av_latent(denoised2)
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)
    return {"frames": frames, "audio": audio}
