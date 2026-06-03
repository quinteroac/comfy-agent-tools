"""WAN 2.2 pipeline wrappers for comfy-videogen."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_WAN22_UNET_HIGH = Path("diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_UNET_LOW = Path("diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_S2V_UNET = Path("diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_TEXT_ENCODER = Path("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")
DEFAULT_WAN22_AUDIO_ENCODER = Path("audio_encoders/wav2vec2_large_english_fp16.safetensors")
DEFAULT_WAN22_VAE = Path("vae/wan_2.1_vae.safetensors")
DEFAULT_WAN22_WIDTH = 640
DEFAULT_WAN22_HEIGHT = 640
DEFAULT_WAN22_LENGTH = 81
DEFAULT_WAN22_S2V_LENGTH = 77
DEFAULT_WAN22_S2V_CHUNK_LENGTH = 77
DEFAULT_WAN22_FPS = 16
DEFAULT_WAN22_STEPS = 20
DEFAULT_WAN22_HIGH_STEPS = DEFAULT_WAN22_STEPS // 2
DEFAULT_WAN22_LOW_STEPS = DEFAULT_WAN22_STEPS - DEFAULT_WAN22_HIGH_STEPS
DEFAULT_WAN22_I2V_CFG = 3.5
DEFAULT_WAN22_FLF2V_CFG = 4.0
DEFAULT_WAN22_S2V_CFG = 6.0
DEFAULT_WAN22_S2V_SAMPLER = "uni_pc"
DEFAULT_WAN22_S2V_SCHEDULER = "simple"
DEFAULT_WAN22_S2V_SHIFT = 8.0
DEFAULT_WAN22_S2V_LORA_STRENGTH = 1.0
DEFAULT_WAN22_NEGATIVE_PROMPT = (
    "overexposed, static, blurry details, subtitles, watermark, low quality, jpeg artifacts, "
    "bad anatomy, malformed hands, malformed face, distorted limbs, messy background"
)


@dataclass(frozen=True)
class Wan22Config:
    """Runtime configuration for WAN 2.2 video commands."""

    models_dir: Path
    unet_high: Path = DEFAULT_WAN22_UNET_HIGH
    unet_low: Path = DEFAULT_WAN22_UNET_LOW
    text_encoder: Path = DEFAULT_WAN22_TEXT_ENCODER
    vae: Path = DEFAULT_WAN22_VAE
    width: int = DEFAULT_WAN22_WIDTH
    height: int = DEFAULT_WAN22_HEIGHT
    length: int = DEFAULT_WAN22_LENGTH
    fps: int = DEFAULT_WAN22_FPS
    steps: int = DEFAULT_WAN22_STEPS
    high_steps: int = DEFAULT_WAN22_HIGH_STEPS
    low_steps: int = DEFAULT_WAN22_LOW_STEPS
    cfg: float = DEFAULT_WAN22_I2V_CFG
    seed: int = 0
    negative_prompt: str = DEFAULT_WAN22_NEGATIVE_PROMPT

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path

    @property
    def split_step(self) -> int:
        """Step index where sampling switches from high-noise to low-noise."""
        return self.high_steps


@dataclass(frozen=True)
class Wan22S2VConfig:
    """Runtime configuration for WAN 2.2 sound-to-video commands."""

    models_dir: Path
    unet: Path = DEFAULT_WAN22_S2V_UNET
    text_encoder: Path = DEFAULT_WAN22_TEXT_ENCODER
    audio_encoder: Path = DEFAULT_WAN22_AUDIO_ENCODER
    vae: Path = DEFAULT_WAN22_VAE
    lora: Path | None = None
    width: int = DEFAULT_WAN22_WIDTH
    height: int = DEFAULT_WAN22_HEIGHT
    length: int = DEFAULT_WAN22_S2V_LENGTH
    chunk_length: int = DEFAULT_WAN22_S2V_CHUNK_LENGTH
    fps: int = DEFAULT_WAN22_FPS
    steps: int = DEFAULT_WAN22_STEPS
    cfg: float = DEFAULT_WAN22_S2V_CFG
    seed: int = 0
    negative_prompt: str = DEFAULT_WAN22_NEGATIVE_PROMPT
    sampler: str = DEFAULT_WAN22_S2V_SAMPLER
    scheduler: str = DEFAULT_WAN22_S2V_SCHEDULER
    shift: float = DEFAULT_WAN22_S2V_SHIFT
    lora_strength: float = DEFAULT_WAN22_S2V_LORA_STRENGTH
    audio_start_time: float = 0.0
    audio_duration: float | None = None

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path

    def validate(self) -> None:
        """Validate S2V runtime settings before loading large models."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be greater than 0")
        if self.length <= 0:
            raise ValueError("length must be greater than 0")
        if self.chunk_length < 73:
            raise ValueError("Wan 2.2 S2V chunk_length must be at least 73 frames")
        if self.fps <= 0:
            raise ValueError("fps must be greater than 0")
        if self.steps <= 0:
            raise ValueError("steps must be greater than 0")
        if self.cfg <= 0:
            raise ValueError("cfg must be greater than 0")
        if self.audio_start_time < 0:
            raise ValueError("audio_start_time must be greater than or equal to 0")
        if self.audio_duration is not None and self.audio_duration <= 0:
            raise ValueError("audio_duration must be greater than 0")


def run_i2v(*, image: Path, prompt: str, config: Wan22Config) -> dict[str, Any]:
    """Run WAN 2.2 image-to-video with high-noise first, low-noise second."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from comfy_diffusion.conditioning import encode_prompt, wan_image_to_video
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.models import ModelManager, model_sampling_sd3
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample_advanced
    from comfy_diffusion.vae import vae_decode_batch

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    mm = ModelManager(config.models_dir)
    model_high = mm.load_unet(config.resolve_model_path(config.unet_high))
    model_low = mm.load_unet(config.resolve_model_path(config.unet_low))
    clip = mm.load_clip(config.resolve_model_path(config.text_encoder), clip_type="wan")
    vae = mm.load_vae(config.resolve_model_path(config.vae))

    model_high = model_sampling_sd3(model_high, shift=5.0)
    model_low = model_sampling_sd3(model_low, shift=5.0)

    positive, negative = encode_prompt(clip, prompt, config.negative_prompt)

    with Image.open(image) as loaded:
        start_image = image_to_tensor(loaded.convert("RGB"))

    positive, negative, latent = wan_image_to_video(
        positive,
        negative,
        vae,
        width=config.width,
        height=config.height,
        length=config.length,
        start_image=start_image,
    )

    latent = sample_advanced(
        model_high,
        positive,
        negative,
        latent,
        steps=config.steps,
        cfg=config.cfg,
        sampler_name="euler",
        scheduler="simple",
        noise_seed=config.seed,
        add_noise=True,
        start_at_step=0,
        end_at_step=config.split_step,
        return_with_leftover_noise=True,
    )
    latent = sample_advanced(
        model_low,
        positive,
        negative,
        latent,
        steps=config.steps,
        cfg=config.cfg,
        sampler_name="euler",
        scheduler="simple",
        noise_seed=config.seed,
        add_noise=False,
        start_at_step=config.split_step,
        end_at_step=config.steps,
        return_with_leftover_noise=False,
    )
    frames = vae_decode_batch(vae, latent)
    return {"frames": frames}


def run_s2v(*, image: Path, audio: Path, prompt: str, config: Wan22S2VConfig) -> dict[str, Any]:
    """Run WAN 2.2 sound-to-video from a reference image and input audio."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    if not audio.is_file():
        raise FileNotFoundError(f"input audio not found: {audio}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    config.validate()

    from comfy_diffusion.audio import audio_encoder_encode, load_audio
    from comfy_diffusion.conditioning import encode_prompt, wan_sound_image_to_video, wan_sound_image_to_video_extend
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.latent import latent_concat
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager, model_sampling_sd3
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode_batch

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    mm = ModelManager(config.models_dir)
    model = mm.load_unet(config.resolve_model_path(config.unet))
    if config.lora is not None:
        model, _clip = apply_lora(
            model,
            None,
            config.resolve_model_path(config.lora),
            config.lora_strength,
            0.0,
        )
    clip = mm.load_clip(config.resolve_model_path(config.text_encoder), clip_type="wan")
    audio_encoder = mm.load_audio_encoder(config.resolve_model_path(config.audio_encoder))
    vae = mm.load_vae(config.resolve_model_path(config.vae))

    model = model_sampling_sd3(model, shift=config.shift)
    base_positive, base_negative = encode_prompt(clip, prompt, config.negative_prompt)
    audio_duration = config.audio_duration if config.audio_duration is not None else config.length / config.fps
    audio_payload = load_audio(audio, start_time=config.audio_start_time, duration=audio_duration)
    audio_encoder_output = audio_encoder_encode(audio_encoder, audio_payload)

    with Image.open(image) as loaded:
        ref_image = image_to_tensor(loaded.convert("RGB"))

    chunks = max(1, math.ceil(config.length / config.chunk_length))
    sampled_latent: dict[str, Any] | None = None
    for chunk_index in range(chunks):
        if sampled_latent is None:
            positive, negative, latent = wan_sound_image_to_video(
                base_positive,
                base_negative,
                vae,
                width=config.width,
                height=config.height,
                length=config.chunk_length,
                batch_size=1,
                audio_encoder_output=audio_encoder_output,
                ref_image=ref_image,
            )
        else:
            positive, negative, latent = wan_sound_image_to_video_extend(
                base_positive,
                base_negative,
                vae,
                config.chunk_length,
                sampled_latent,
                audio_encoder_output=audio_encoder_output,
                ref_image=ref_image,
            )

        denoised = sample(
            model,
            positive,
            negative,
            latent,
            steps=config.steps,
            cfg=config.cfg,
            sampler_name=config.sampler,
            scheduler=config.scheduler,
            seed=config.seed + chunk_index,
        )
        sampled_latent = denoised if sampled_latent is None else latent_concat(sampled_latent, denoised, dim="t")

    frames = vae_decode_batch(vae, sampled_latent)
    return {"frames": frames[: config.length], "audio": audio_payload}


def run_flf2v(*, first_image: Path, last_image: Path, prompt: str, config: Wan22Config) -> dict[str, Any]:
    """Run WAN 2.2 first/last-frame-to-video."""
    if not first_image.is_file():
        raise FileNotFoundError(f"first image not found: {first_image}")
    if not last_image.is_file():
        raise FileNotFoundError(f"last image not found: {last_image}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from comfy_diffusion.conditioning import encode_prompt, wan_first_last_frame_to_video
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.models import ModelManager, model_sampling_sd3
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample_advanced
    from comfy_diffusion.vae import vae_decode_batch

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    mm = ModelManager(config.models_dir)
    model_high = mm.load_unet(config.resolve_model_path(config.unet_high))
    model_low = mm.load_unet(config.resolve_model_path(config.unet_low))
    clip = mm.load_clip(config.resolve_model_path(config.text_encoder), clip_type="wan")
    vae = mm.load_vae(config.resolve_model_path(config.vae))

    model_high = model_sampling_sd3(model_high, shift=8.0)
    model_low = model_sampling_sd3(model_low, shift=8.0)

    positive, negative = encode_prompt(clip, prompt, config.negative_prompt)

    with Image.open(first_image) as first_loaded, Image.open(last_image) as last_loaded:
        start_image = image_to_tensor(first_loaded.convert("RGB"))
        end_image = image_to_tensor(last_loaded.convert("RGB"))

    positive, negative, latent = wan_first_last_frame_to_video(
        positive,
        negative,
        vae,
        width=config.width,
        height=config.height,
        length=config.length,
        start_image=start_image,
        end_image=end_image,
    )
    latent = sample_advanced(
        model_high,
        positive,
        negative,
        latent,
        steps=config.steps,
        cfg=config.cfg,
        sampler_name="euler",
        scheduler="simple",
        noise_seed=config.seed,
        add_noise=True,
        start_at_step=0,
        end_at_step=config.split_step,
        return_with_leftover_noise=True,
    )
    latent = sample_advanced(
        model_low,
        positive,
        negative,
        latent,
        steps=config.steps,
        cfg=config.cfg,
        sampler_name="euler",
        scheduler="simple",
        noise_seed=config.seed,
        add_noise=False,
        start_at_step=config.split_step,
        end_at_step=config.steps,
        return_with_leftover_noise=False,
    )
    frames = vae_decode_batch(vae, latent)
    return {"frames": frames}
