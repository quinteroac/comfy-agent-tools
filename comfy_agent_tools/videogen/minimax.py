"""Local MiniMax H3 reference-to-video generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from comfy_agent_tools.videogen.artifacts import make_video_path

DEFAULT_MINIMAX_WIDTH = 1344
DEFAULT_MINIMAX_HEIGHT = 768
DEFAULT_MINIMAX_LENGTH = 124
DEFAULT_MINIMAX_STEPS = 20
DEFAULT_MINIMAX_SEED = 0
DEFAULT_MINIMAX_REF_IMAGE_SIZE = "match"
DEFAULT_MINIMAX_FPS = 24
DEFAULT_MINIMAX_FL2VA_UNET = Path("diffusion_models/minimax/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
DEFAULT_MINIMAX_REF2VA_UNET = Path("diffusion_models/minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors")


@dataclass(frozen=True)
class MiniMaxH3Config:
    """Runtime options for the local MiniMax H3 pipeline."""

    models_dir: Path
    width: int = DEFAULT_MINIMAX_WIDTH
    height: int = DEFAULT_MINIMAX_HEIGHT
    length: int = DEFAULT_MINIMAX_LENGTH
    steps: int = DEFAULT_MINIMAX_STEPS
    seed: int = DEFAULT_MINIMAX_SEED
    ref_image_size: str = DEFAULT_MINIMAX_REF_IMAGE_SIZE
    unet: Path = DEFAULT_MINIMAX_REF2VA_UNET
    fl2va_unet: Path = DEFAULT_MINIMAX_FL2VA_UNET
    text_encoder: Path | None = None
    audio_vae: Path | None = None
    video_vae: Path | None = None

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("MiniMax H3 width and height must be positive")
        if self.length <= 0 or self.steps <= 0:
            raise ValueError("MiniMax H3 length and steps must be positive")
        if self.ref_image_size not in {"match", "max"}:
            raise ValueError("MiniMax H3 ref_image_size must be 'match' or 'max'")

    def model_path(self, value: Path | None, default: str) -> Path:
        path = value if value is not None else Path(default)
        return path if path.is_absolute() else self.models_dir / path


def run_r2v(*, images: list[Path], prompt: str, config: MiniMaxH3Config, out_dir: Path) -> dict[str, Any]:
    """Generate a MiniMax H3 video with native synchronized audio."""
    config.validate()
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not images:
        raise ValueError("at least one reference image is required")
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reference image not found: {missing[0]}")

    try:
        from comfy_diffusion.image import load_image
        from comfy_diffusion.pipelines.video.minimax import h3
    except Exception as exc:
        raise RuntimeError("comfy-diffusion 2.6.0 with MiniMax H3 support is required") from exc

    reference_images = [load_image(path)[0] for path in images]
    artifact = make_video_path(out_dir, prefix="comfy-videogen-minimax-h3-r2v")
    # comfy-diffusion treats explicit override paths as already-resolved paths;
    # profile paths are intentionally relative to the configured models root.
    def resolved_override(path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else config.models_dir / path

    result = h3.run(
        prompt,
        reference_images,
        models_dir=config.models_dir,
        width=config.width,
        height=config.height,
        length=config.length,
        steps=config.steps,
        seed=config.seed,
        ref_image_size=config.ref_image_size,
        unet_filename=resolved_override(config.unet),
        text_encoder_filename=resolved_override(config.text_encoder),
        audio_vae_filename=resolved_override(config.audio_vae),
        video_vae_filename=resolved_override(config.video_vae),
        output_path=artifact,
    )
    return {"artifact": artifact, "result": result}


def run_t2v(*, prompt: str, config: MiniMaxH3Config, out_dir: Path) -> dict[str, Any]:
    """Generate MiniMax H3 text-to-video with synchronized audio."""
    return _run_fl2va(prompt=prompt, first_image=None, config=config, out_dir=out_dir, mode="t2v")


def run_i2v(*, image: Path, prompt: str, config: MiniMaxH3Config, out_dir: Path) -> dict[str, Any]:
    """Generate MiniMax H3 image-to-video with synchronized audio."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    return _run_fl2va(prompt=prompt, first_image=image, config=config, out_dir=out_dir, mode="i2v")


def _run_fl2va(
    *,
    prompt: str,
    first_image: Path | None,
    config: MiniMaxH3Config,
    out_dir: Path,
    mode: str,
) -> dict[str, Any]:
    config.validate()
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    try:
        from comfy_diffusion._runtime import ensure_comfyui_on_path
        from comfy_diffusion.audio import vae_decode_audio
        from comfy_diffusion.image import load_image
        from comfy_diffusion.models import ModelManager
        from comfy_diffusion.runtime import check_runtime
        from comfy_diffusion.sampling import basic_guider, basic_scheduler, get_sampler, random_noise, sample_custom
        from comfy_diffusion.vae import vae_decode_batch
        from comfy_diffusion.video import save_video_with_audio

        runtime = check_runtime()
        if runtime.get("error"):
            raise RuntimeError(f"ComfyUI runtime not available: {runtime['error']}")
        comfy_root = ensure_comfyui_on_path()
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "runtime not available" in str(exc):
            raise
        raise RuntimeError("comfy-diffusion 2.6.0 with MiniMax H3 T2V/I2V support is required") from exc

    models_root = Path(config.models_dir)
    manager = ModelManager(models_root)
    model = manager.load_unet(_resolve_path(models_root, config.fl2va_unet))
    clip = manager.load_clip(
        _resolve_path(models_root, config.text_encoder or Path("text_encoders/minimax/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")),
        clip_type="minimax",
    )
    video_vae = manager.load_vae(
        _resolve_path(models_root, config.video_vae or Path("vae/minimax/minimax_h3_video_vae_fp16.safetensors"))
    )
    audio_vae = manager.load_vae(
        _resolve_path(models_root, config.audio_vae or Path("vae/minimax/minimax_h3_audio_vae_fp32.safetensors"))
    )

    first_tensor = load_image(first_image)[0] if first_image is not None else None
    node_result = MiniMaxH3ImageToVideo.execute(
        clip=clip,
        vae=video_vae,
        prompt=prompt,
        width=config.width,
        height=config.height,
        length=config.length,
        first_frame=first_tensor,
    )
    positive, latent = getattr(node_result, "result", node_result)
    guider = basic_guider(model, positive)
    sampled = sample_custom(
        random_noise(config.seed),
        guider,
        get_sampler("res_multistep"),
        basic_scheduler(model, "simple", config.steps),
        latent_image=latent,
    )[1]
    audio_samples = sampled["samples"].unbind()[-1]
    audio = vae_decode_audio(audio_vae, {"samples": audio_samples})
    if audio.ndim == 3 and audio.shape[1] > 8 and audio.shape[-1] <= 8:
        audio = audio.movedim(-1, 1)
    frames = vae_decode_batch(video_vae, {"samples": sampled["samples"]})
    artifact = make_video_path(out_dir, prefix=f"comfy-videogen-minimax-h3-{mode}")
    save_video_with_audio(
        frames,
        {"waveform": audio, "sample_rate": int(getattr(audio_vae, "audio_sample_rate", 32000))},
        artifact,
        fps=DEFAULT_MINIMAX_FPS,
    )
    return {"artifact": artifact, "frames": frames}


def _resolve_path(models_dir: Path, path: Path) -> Path:
    """Resolve profile-relative model paths for comfy-diffusion overrides."""
    return path if path.is_absolute() else models_dir / path
