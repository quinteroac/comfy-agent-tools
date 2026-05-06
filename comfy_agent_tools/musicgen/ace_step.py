"""ACE-Step 1.5 pipeline wrappers for comfy-musicgen."""

from __future__ import annotations

from typing import Any

from comfy_agent_tools.loras import apply_extra_loras
from .config import MusicgenConfig


def run_ace_step(*, prompt: str, lyrics: str, config: MusicgenConfig) -> dict[str, Any]:
    """Run ACE-Step 1.5 split base with configurable local model paths."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if config.duration <= 0:
        raise ValueError("duration must be greater than 0")
    if config.bpm <= 0:
        raise ValueError("bpm must be greater than 0")
    if config.steps <= 0:
        raise ValueError("steps must be greater than 0")
    if config.cfg <= 0:
        raise ValueError("cfg must be greater than 0")

    _require_model_file(config, config.unet, "UNet")
    _require_model_file(config, config.clip_0_6b, "0.6B text encoder")
    _require_model_file(config, config.clip_1_7b, "1.7B text encoder")
    _require_model_file(config, config.vae, "VAE")

    from comfy_diffusion.audio import encode_ace_step_15_audio, empty_ace_step_15_latent_audio, vae_decode_audio
    from comfy_diffusion.conditioning import conditioning_zero_out
    from comfy_diffusion.models import ModelManager, model_sampling_aura_flow
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    mm = ModelManager(config.models_dir)
    model = mm.load_unet(str(config.resolve_model_path(config.unet)))
    clip = mm.load_clip(
        str(config.resolve_model_path(config.clip_0_6b)),
        str(config.resolve_model_path(config.clip_1_7b)),
        clip_type="ace",
    )
    vae = mm.load_vae(str(config.resolve_model_path(config.vae)))

    model = model_sampling_aura_flow(model, shift=3)
    model, clip = apply_extra_loras(model, clip, config.resolved_extra_loras)

    latent = empty_ace_step_15_latent_audio(config.duration, batch_size=1)
    positive = encode_ace_step_15_audio(
        clip,
        prompt,
        lyrics=lyrics,
        seed=config.seed,
        bpm=config.bpm,
        duration=config.duration,
        timesignature=config.time_signature,
        language=config.language,
        keyscale=config.keyscale,
    )
    negative = conditioning_zero_out(positive)
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
    waveform = vae_decode_audio(vae, latent_out)
    sample_rate = getattr(vae, "audio_sample_rate", 44100)
    return {"audio": {"waveform": waveform, "sample_rate": sample_rate}}


def _require_model_file(config: MusicgenConfig, path: object, label: str) -> None:
    resolved = config.resolve_model_path(path)  # type: ignore[arg-type]
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} model file not found: {resolved}")
