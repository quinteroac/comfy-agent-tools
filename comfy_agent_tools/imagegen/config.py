"""Defaults for comfy-imagegen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from comfy_agent_tools.loras import ExtraLora


DEFAULT_MODELS_DIR = Path("/mnt/models/comfyui")
DEFAULT_UNET = Path("diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors")
DEFAULT_CLIP = Path("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors")
DEFAULT_VAE = Path("vae/qwen_image_vae.safetensors")
DEFAULT_LORA = Path("loras/qwen-image-edit/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
DEFAULT_UPSCALER = Path("upscale_models/4x-ClearRealityV1.pth")

DEFAULT_STEPS = 4
DEFAULT_CFG = 3.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_SEED = 0
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_OUT = Path("outputs")
DEFAULT_SEED_COLOR = (128, 128, 128)


@dataclass(frozen=True)
class ImagegenConfig:
    """Runtime configuration for comfy-imagegen commands."""

    models_dir: Path = DEFAULT_MODELS_DIR
    unet: Path = DEFAULT_UNET
    clip: Path = DEFAULT_CLIP
    vae: Path = DEFAULT_VAE
    lora: Path | None = DEFAULT_LORA
    upscaler: Path = DEFAULT_UPSCALER
    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    seed: int = DEFAULT_SEED
    extra_loras: list[ExtraLora] | None = None

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path

    @property
    def resolved_extra_loras(self) -> list[ExtraLora]:
        """Return validated extra LoRAs, or an empty list."""
        from comfy_agent_tools.loras import resolve_extra_loras

        return resolve_extra_loras(self.models_dir, self.extra_loras or [])
