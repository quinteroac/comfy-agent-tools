"""WAN 2.2 pipeline wrappers for comfy-videogen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_WAN22_UNET_HIGH = Path("diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_UNET_LOW = Path("diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
DEFAULT_WAN22_TEXT_ENCODER = Path("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")
DEFAULT_WAN22_VAE = Path("vae/wan_2.1_vae.safetensors")
DEFAULT_WAN22_WIDTH = 640
DEFAULT_WAN22_HEIGHT = 640
DEFAULT_WAN22_LENGTH = 81
DEFAULT_WAN22_FPS = 16
DEFAULT_WAN22_STEPS = 20
DEFAULT_WAN22_I2V_CFG = 3.5
DEFAULT_WAN22_FLF2V_CFG = 4.0
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
    cfg: float = DEFAULT_WAN22_I2V_CFG
    seed: int = 0
    negative_prompt: str = DEFAULT_WAN22_NEGATIVE_PROMPT

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path


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

    split_step = config.steps // 2
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
        end_at_step=split_step,
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
        start_at_step=split_step,
        end_at_step=config.steps,
        return_with_leftover_noise=False,
    )
    frames = vae_decode_batch(vae, latent)
    return {"frames": frames}


def run_flf2v(*, first_image: Path, last_image: Path, prompt: str, config: Wan22Config) -> dict[str, Any]:
    """Run WAN 2.2 first/last-frame-to-video."""
    if not first_image.is_file():
        raise FileNotFoundError(f"first image not found: {first_image}")
    if not last_image.is_file():
        raise FileNotFoundError(f"last image not found: {last_image}")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    from comfy_diffusion.pipelines.video.wan.wan22 import flf2v

    with Image.open(first_image) as first_loaded, Image.open(last_image) as last_loaded:
        frames = flf2v.run(
            first_loaded.convert("RGB"),
            last_loaded.convert("RGB"),
            prompt,
            negative_prompt=config.negative_prompt,
            width=config.width,
            height=config.height,
            length=config.length,
            models_dir=config.models_dir,
            seed=config.seed,
            steps=config.steps,
            cfg=config.cfg,
            unet_high_filename=str(config.resolve_model_path(config.unet_high)),
            unet_low_filename=str(config.resolve_model_path(config.unet_low)),
            text_encoder_filename=str(config.resolve_model_path(config.text_encoder)),
            vae_filename=str(config.resolve_model_path(config.vae)),
        )
    return {"frames": frames}
