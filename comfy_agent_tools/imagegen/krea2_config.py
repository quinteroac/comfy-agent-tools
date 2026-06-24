"""Krea2 Turbo runtime configuration for comfy-imagegen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS_DIR = Path("/mnt/models/comfyui")
DEFAULT_UNET = Path("diffusion_models/krea2_turbo_fp8_scaled.safetensors")
DEFAULT_CLIP = Path("text_encoders/qwen3vl_4b_fp8_scaled.safetensors")
DEFAULT_VAE = Path("vae/qwen_image_vae.safetensors")

DEFAULT_STEPS = 8
DEFAULT_CFG = 1.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_SEED = 0
DEFAULT_DENOISE = 1.0
DEFAULT_REBALANCE_MULTIPLIER = 4.0
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Public aliases used by the CLI layer.
DEFAULT_KREA2_UNET = DEFAULT_UNET
DEFAULT_KREA2_CLIP = DEFAULT_CLIP
DEFAULT_KREA2_VAE = DEFAULT_VAE
DEFAULT_KREA2_STEPS = DEFAULT_STEPS
DEFAULT_KREA2_CFG = DEFAULT_CFG
DEFAULT_KREA2_SAMPLER = DEFAULT_SAMPLER
DEFAULT_KREA2_SCHEDULER = DEFAULT_SCHEDULER
DEFAULT_KREA2_SEED = DEFAULT_SEED
DEFAULT_KREA2_DENOISE = DEFAULT_DENOISE
DEFAULT_KREA2_REBALANCE_MULTIPLIER = DEFAULT_REBALANCE_MULTIPLIER
DEFAULT_KREA2_WIDTH = DEFAULT_WIDTH
DEFAULT_KREA2_HEIGHT = DEFAULT_HEIGHT


@dataclass(frozen=True)
class Krea2Config:
    """Runtime configuration for the Krea2 Turbo image generation command."""

    models_dir: Path = DEFAULT_MODELS_DIR
    unet: Path = DEFAULT_UNET
    clip: Path = DEFAULT_CLIP
    vae: Path = DEFAULT_VAE
    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
    seed: int = DEFAULT_SEED
    denoise: float = DEFAULT_DENOISE
    rebalance_multiplier: float = DEFAULT_REBALANCE_MULTIPLIER

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path
