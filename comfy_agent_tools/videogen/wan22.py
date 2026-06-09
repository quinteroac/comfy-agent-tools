"""WAN 2.2 pipeline wrappers for comfy-videogen."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from comfy_agent_tools.loras import ExtraLora, apply_extra_loras, apply_extra_loras_to_models


DEFAULT_WAN22_UNET_HIGH = Path("diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_UNET_LOW = Path("diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_T2V_UNET_HIGH = Path("diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_T2V_UNET_LOW = Path("diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors")
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
DEFAULT_WAN22_T2V_CFG = 3.5
DEFAULT_WAN22_FLF2V_CFG = 4.0
DEFAULT_WAN22_S2V_CFG = 6.0
DEFAULT_WAN22_S2V_SAMPLER = "uni_pc"
DEFAULT_WAN22_S2V_SCHEDULER = "simple"
DEFAULT_WAN22_S2V_SHIFT = 8.0
DEFAULT_WAN22_S2V_LORA_STRENGTH = 1.0
DEFAULT_WAN22_VIDEO_AUDIO_CHUNK_OVERLAP = 4
DEFAULT_WAN22_VIDEO_AUDIO_PROMPT = "Audio-reactive motion, expressive movement, coherent video."
DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_PROMPT = "Speaking. Talking. Expressive lip movement."
DEFAULT_WAN22_VIDEO_AUDIO_STEPS = 4
DEFAULT_WAN22_VIDEO_AUDIO_DENOISE = 0.35
DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_STEPS = 4
DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_DENOISE = 0.45
DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_STEPS = 2
DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_DENOISE = 0.25
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
    extra_loras: list[ExtraLora] | None = None
    extra_loras_high: list[ExtraLora] | None = None
    extra_loras_low: list[ExtraLora] | None = None

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path

    @property
    def split_step(self) -> int:
        """Step index where sampling switches from high-noise to low-noise."""
        return self.high_steps

    @property
    def resolved_extra_loras(self) -> list[ExtraLora]:
        """Return validated extra LoRAs applied to both UNets, or an empty list."""
        from comfy_agent_tools.loras import resolve_extra_loras

        return resolve_extra_loras(self.models_dir, self.extra_loras or [])

    @property
    def resolved_extra_loras_high(self) -> list[ExtraLora]:
        """Return validated extra LoRAs applied only to the high-noise UNet."""
        from comfy_agent_tools.loras import resolve_extra_loras

        return resolve_extra_loras(self.models_dir, self.extra_loras_high or [])

    @property
    def resolved_extra_loras_low(self) -> list[ExtraLora]:
        """Return validated extra LoRAs applied only to the low-noise UNet."""
        from comfy_agent_tools.loras import resolve_extra_loras

        return resolve_extra_loras(self.models_dir, self.extra_loras_low or [])

    def validate_extra_loras(self) -> None:
        """Validate all configured extra LoRA groups."""
        _ = self.resolved_extra_loras
        _ = self.resolved_extra_loras_high
        _ = self.resolved_extra_loras_low


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


@dataclass(frozen=True)
class Wan22VideoAudioConfig:
    """Runtime configuration for WAN 2.2 video+audio commands."""

    models_dir: Path
    unet: Path = DEFAULT_WAN22_S2V_UNET
    text_encoder: Path = DEFAULT_WAN22_TEXT_ENCODER
    audio_encoder: Path = DEFAULT_WAN22_AUDIO_ENCODER
    vae: Path = DEFAULT_WAN22_VAE
    chunk_length: int = DEFAULT_WAN22_S2V_CHUNK_LENGTH
    chunk_overlap: int = DEFAULT_WAN22_VIDEO_AUDIO_CHUNK_OVERLAP
    fps: int = DEFAULT_WAN22_FPS
    steps: int = DEFAULT_WAN22_VIDEO_AUDIO_STEPS
    denoise: float = DEFAULT_WAN22_VIDEO_AUDIO_DENOISE
    lipsync_steps: int = DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_STEPS
    lipsync_denoise: float = DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_DENOISE
    lipsync_second_steps: int = DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_STEPS
    lipsync_second_denoise: float = DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_DENOISE
    cfg: float = 1.0
    seed: int = 0
    sampler: str = "euler"
    scheduler: str = "simple"
    shift: float = 10.0
    negative_prompt: str = ""
    audio_start_time: float = 0.0

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path

    def validate(self) -> None:
        """Validate video+audio runtime settings before loading large models."""
        if self.fps != DEFAULT_WAN22_FPS:
            raise ValueError("Wan 2.2 video+audio v1 supports only 16 fps")
        if self.chunk_length < 73:
            raise ValueError("Wan 2.2 video+audio chunk_length must be at least 73 frames")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if self.chunk_overlap >= self.chunk_length:
            raise ValueError("chunk_overlap must be smaller than chunk_length")
        for field_name in ("steps", "lipsync_steps", "lipsync_second_steps"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be greater than 0")
        for field_name in ("denoise", "lipsync_denoise", "lipsync_second_denoise"):
            value = getattr(self, field_name)
            if value <= 0 or value > 1:
                raise ValueError(f"{field_name} must be greater than 0 and less than or equal to 1")
        if self.cfg <= 0:
            raise ValueError("cfg must be greater than 0")
        if self.audio_start_time < 0:
            raise ValueError("audio_start_time must be greater than or equal to 0")


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

    model_high, model_low, clip = _apply_extra_loras(model_high, model_low, clip, config)
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


def run_t2v(*, prompt: str, config: Wan22Config) -> dict[str, Any]:
    """Run WAN 2.2 text-to-video with high-noise first, low-noise second."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from comfy_diffusion.conditioning import encode_prompt, wan_image_to_video
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

    model_high, model_low, clip = _apply_extra_loras(model_high, model_low, clip, config)
    model_high = model_sampling_sd3(model_high, shift=5.0)
    model_low = model_sampling_sd3(model_low, shift=5.0)

    positive, negative = encode_prompt(clip, prompt, config.negative_prompt)
    positive, negative, latent = wan_image_to_video(
        positive,
        negative,
        vae,
        width=config.width,
        height=config.height,
        length=config.length,
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


def run_video_audio(
    *,
    video: Path,
    audio: Path,
    mode: str,
    prompt: str,
    config: Wan22VideoAudioConfig,
    mask_video: Path | None = None,
    mask_image: Path | None = None,
) -> dict[str, Any]:
    """Run WAN 2.2 audio-conditioned video-to-video or masked lipsync."""
    if not video.is_file():
        raise FileNotFoundError(f"input video not found: {video}")
    if not audio.is_file():
        raise FileNotFoundError(f"input audio not found: {audio}")
    if mode not in {"audio-driven", "lipsync"}:
        raise ValueError("mode must be one of: audio-driven, lipsync")
    if mode == "lipsync":
        if mask_video is None and mask_image is None:
            raise ValueError("lipsync mode requires --mask-video or --mask-image")
        if mask_video is not None and mask_image is not None:
            raise ValueError("pass only one of --mask-video or --mask-image")
        if mask_video is not None and not mask_video.is_file():
            raise FileNotFoundError(f"mask video not found: {mask_video}")
        if mask_image is not None and not mask_image.is_file():
            raise FileNotFoundError(f"mask image not found: {mask_image}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    config.validate()

    from comfy_diffusion.audio import audio_encoder_encode, load_audio
    from comfy_diffusion.conditioning import conditioning_zero_out, encode_prompt, wan_sound_image_to_video
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.models import ModelManager, model_sampling_sd3
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode_batch, vae_encode_tensor
    from comfy_diffusion.video import get_video_metadata, load_video

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    metadata = get_video_metadata(video)
    video_fps = float(metadata.get("fps", 0.0))
    if abs(video_fps - config.fps) > 0.01:
        raise ValueError("Wan 2.2 video+audio v1 supports only 16 fps input video")

    source_frames = load_video(video)
    total_frames = _frame_count(source_frames)
    if total_frames <= 0:
        raise ValueError("input video contains no frames")

    source_pil_frames = _frames_to_pil_list(source_frames)
    if len(source_pil_frames) != total_frames:
        raise ValueError("loaded video frame count is inconsistent")

    mask_frames = (
        _load_mask_frames(mask_video=mask_video, mask_image=mask_image, length=total_frames)
        if mode == "lipsync"
        else None
    )

    mm = ModelManager(config.models_dir)
    model = mm.load_unet(config.resolve_model_path(config.unet))
    clip = mm.load_clip(config.resolve_model_path(config.text_encoder), clip_type="wan")
    audio_encoder = mm.load_audio_encoder(config.resolve_model_path(config.audio_encoder))
    vae = mm.load_vae(config.resolve_model_path(config.vae))
    model = model_sampling_sd3(model, shift=config.shift)

    base_positive, base_negative = encode_prompt(clip, prompt, config.negative_prompt)
    if not config.negative_prompt.strip():
        base_negative = conditioning_zero_out(base_positive)

    output_frames: list[Image.Image] = []
    chunks: list[dict[str, int | float]] = []
    final_audio = load_audio(
        audio,
        start_time=config.audio_start_time,
        duration=total_frames / config.fps,
    )

    for chunk_index, start_frame, actual_length in _chunk_ranges(
        total_frames,
        chunk_length=config.chunk_length,
        chunk_overlap=config.chunk_overlap,
    ):
        chunk_tensor = _slice_frame_batch(source_frames, start_frame, actual_length, config.chunk_length)
        chunk_source_pil = _pad_pil_frames(
            source_pil_frames[start_frame : start_frame + actual_length],
            config.chunk_length,
        )
        chunk_audio = load_audio(
            audio,
            start_time=config.audio_start_time + start_frame / config.fps,
            duration=config.chunk_length / config.fps,
        )
        audio_encoder_output = audio_encoder_encode(audio_encoder, chunk_audio)

        if mode == "audio-driven":
            generated = _run_video_audio_pass(
                model=model,
                vae=vae,
                positive=base_positive,
                negative=base_negative,
                source_tensor=chunk_tensor,
                ref_image=image_to_tensor(chunk_source_pil[0].convert("RGB")),
                audio_encoder_output=audio_encoder_output,
                width=chunk_source_pil[0].width,
                height=chunk_source_pil[0].height,
                length=config.chunk_length,
                steps=config.steps,
                cfg=config.cfg,
                sampler=config.sampler,
                scheduler=config.scheduler,
                seed=config.seed + chunk_index,
                denoise=config.denoise,
                wan_sound_image_to_video=wan_sound_image_to_video,
                vae_encode_tensor=vae_encode_tensor,
                sample=sample,
                vae_decode_batch=vae_decode_batch,
            )
            chunk_output = generated[:actual_length]
            chunks.append(
                {
                    "index": chunk_index,
                    "start_frame": start_frame,
                    "frames": actual_length,
                    "steps": config.steps,
                    "denoise": config.denoise,
                }
            )
        else:
            assert mask_frames is not None
            chunk_masks = _pad_pil_frames(
                mask_frames[start_frame : start_frame + actual_length],
                config.chunk_length,
            )
            first_pass = _run_video_audio_pass(
                model=model,
                vae=vae,
                positive=base_positive,
                negative=base_negative,
                source_tensor=chunk_tensor,
                ref_image=image_to_tensor(chunk_source_pil[0].convert("RGB")),
                audio_encoder_output=audio_encoder_output,
                width=chunk_source_pil[0].width,
                height=chunk_source_pil[0].height,
                length=config.chunk_length,
                steps=config.lipsync_steps,
                cfg=config.cfg,
                sampler=config.sampler,
                scheduler=config.scheduler,
                seed=config.seed + chunk_index,
                denoise=config.lipsync_denoise,
                wan_sound_image_to_video=wan_sound_image_to_video,
                vae_encode_tensor=vae_encode_tensor,
                sample=sample,
                vae_decode_batch=vae_decode_batch,
            )
            composited = _composite_with_masks(chunk_source_pil, first_pass, chunk_masks)
            second_source_tensor = _pil_frames_to_tensor(composited)
            second_pass = _run_video_audio_pass(
                model=model,
                vae=vae,
                positive=base_positive,
                negative=base_negative,
                source_tensor=second_source_tensor,
                ref_image=image_to_tensor(composited[0].convert("RGB")),
                audio_encoder_output=audio_encoder_output,
                width=composited[0].width,
                height=composited[0].height,
                length=config.chunk_length,
                steps=config.lipsync_second_steps,
                cfg=config.cfg,
                sampler=config.sampler,
                scheduler=config.scheduler,
                seed=config.seed + chunk_index + 10_000,
                denoise=config.lipsync_second_denoise,
                wan_sound_image_to_video=wan_sound_image_to_video,
                vae_encode_tensor=vae_encode_tensor,
                sample=sample,
                vae_decode_batch=vae_decode_batch,
            )
            chunk_output = _composite_with_masks(chunk_source_pil, second_pass, chunk_masks)[:actual_length]
            chunks.append(
                {
                    "index": chunk_index,
                    "start_frame": start_frame,
                    "frames": actual_length,
                    "steps": config.lipsync_steps,
                    "denoise": config.lipsync_denoise,
                    "second_steps": config.lipsync_second_steps,
                    "second_denoise": config.lipsync_second_denoise,
                }
            )

        output_frames = _merge_chunk_frames(output_frames, chunk_output, config.chunk_overlap)

    return {
        "frames": output_frames[:total_frames],
        "audio": final_audio,
        "chunks": chunks,
        "source_fps": video_fps,
    }


def _run_video_audio_pass(
    *,
    model: Any,
    vae: Any,
    positive: Any,
    negative: Any,
    source_tensor: Any,
    ref_image: Any,
    audio_encoder_output: Any,
    width: int,
    height: int,
    length: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    denoise: float,
    wan_sound_image_to_video: Any,
    vae_encode_tensor: Any,
    sample: Any,
    vae_decode_batch: Any,
) -> list[Image.Image]:
    pass_positive, pass_negative, _latent = wan_sound_image_to_video(
        positive,
        negative,
        vae,
        width=width,
        height=height,
        length=length,
        batch_size=1,
        audio_encoder_output=audio_encoder_output,
        ref_image=ref_image,
    )
    source_latent = vae_encode_tensor(vae, source_tensor)
    sampled = sample(
        model,
        pass_positive,
        pass_negative,
        source_latent,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        seed=seed,
        denoise=denoise,
    )
    return vae_decode_batch(vae, sampled)


def _chunk_ranges(
    total_frames: int,
    *,
    chunk_length: int,
    chunk_overlap: int,
) -> list[tuple[int, int, int]]:
    stride = chunk_length - chunk_overlap
    chunks: list[tuple[int, int, int]] = []
    start = 0
    index = 0
    while start < total_frames:
        actual_length = min(chunk_length, total_frames - start)
        chunks.append((index, start, actual_length))
        if start + actual_length >= total_frames:
            break
        start += stride
        index += 1
    return chunks


def _frame_count(frames: Any) -> int:
    shape = getattr(frames, "shape", None)
    if shape is not None and len(shape) >= 1:
        return int(shape[0])
    return len(frames)


def _slice_frame_batch(frames: Any, start: int, actual_length: int, target_length: int) -> Any:
    sliced = frames[start : start + actual_length]
    if actual_length >= target_length:
        return sliced
    missing = target_length - actual_length
    if hasattr(sliced, "detach") and hasattr(sliced, "shape"):
        import torch

        if sliced.shape[0] == 0:
            raise ValueError("cannot pad empty frame chunk")
        padding = sliced[-1:].repeat((missing, 1, 1, 1))
        return torch.cat([sliced, padding], dim=0)
    padded = list(sliced)
    if not padded:
        raise ValueError("cannot pad empty frame chunk")
    padded.extend([padded[-1]] * missing)
    return _pil_frames_to_tensor(_frames_to_pil_list(padded))


def _frames_to_pil_list(frames: Any) -> list[Image.Image]:
    if isinstance(frames, list):
        result = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                result.append(frame.convert("RGB"))
            else:
                result.append(_array_to_pil(frame))
        return result
    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    return [_array_to_pil(frame) for frame in frames]


def _array_to_pil(frame: Any) -> Image.Image:
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError("frames must be HWC images")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] == 4:
        array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max()) if array.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _pil_frames_to_tensor(frames: list[Image.Image]) -> Any:
    import numpy as np
    import torch

    arrays = [np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0 for frame in frames]
    return torch.from_numpy(np.stack(arrays, axis=0))


def _pad_pil_frames(frames: list[Image.Image], target_length: int) -> list[Image.Image]:
    if not frames:
        raise ValueError("cannot pad an empty frame list")
    if len(frames) >= target_length:
        return frames[:target_length]
    return [*frames, *([frames[-1]] * (target_length - len(frames)))]


def _load_mask_frames(
    *,
    mask_video: Path | None,
    mask_image: Path | None,
    length: int,
) -> list[Image.Image]:
    if mask_image is not None:
        with Image.open(mask_image) as loaded:
            mask = ImageOps.grayscale(loaded)
        return [mask.copy() for _ in range(length)]
    if mask_video is None:
        raise ValueError("mask_video or mask_image is required")
    from comfy_diffusion.video import load_video

    masks = [ImageOps.grayscale(frame) for frame in _frames_to_pil_list(load_video(mask_video))]
    if not masks:
        raise ValueError("mask video contains no frames")
    if len(masks) < length:
        masks.extend([masks[-1]] * (length - len(masks)))
    return masks[:length]


def _composite_with_masks(
    base_frames: list[Image.Image],
    generated_frames: list[Image.Image],
    mask_frames: list[Image.Image],
) -> list[Image.Image]:
    result: list[Image.Image] = []
    count = min(len(base_frames), len(generated_frames), len(mask_frames))
    for index in range(count):
        base = base_frames[index].convert("RGB")
        generated = generated_frames[index].convert("RGB")
        if generated.size != base.size:
            generated = generated.resize(base.size, Image.Resampling.LANCZOS)
        mask = ImageOps.grayscale(mask_frames[index])
        if mask.size != base.size:
            mask = mask.resize(base.size, Image.Resampling.BILINEAR)
        result.append(Image.composite(generated, base, mask))
    return result


def _merge_chunk_frames(
    existing: list[Image.Image],
    new_frames: list[Image.Image],
    chunk_overlap: int,
) -> list[Image.Image]:
    if not existing or chunk_overlap <= 0:
        return [*existing, *new_frames]
    overlap = min(chunk_overlap, len(existing), len(new_frames))
    merged = existing[:-overlap]
    for index in range(overlap):
        alpha = (index + 1) / (overlap + 1)
        merged.append(Image.blend(existing[-overlap + index], new_frames[index], alpha))
    merged.extend(new_frames[overlap:])
    return merged


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

    model_high, model_low, clip = _apply_extra_loras(model_high, model_low, clip, config)
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


def _apply_extra_loras(model_high: Any, model_low: Any, clip: Any, config: Wan22Config) -> tuple[Any, Any, Any]:
    """Apply WAN I2V/FLF2V LoRAs to both, high-only, and low-only targets."""
    (model_high, model_low), clip = apply_extra_loras_to_models(
        [model_high, model_low],
        clip,
        config.resolved_extra_loras,
    )
    model_high, clip = apply_extra_loras(model_high, clip, config.resolved_extra_loras_high)
    model_low, clip = apply_extra_loras(model_low, clip, config.resolved_extra_loras_low)
    return model_high, model_low, clip
