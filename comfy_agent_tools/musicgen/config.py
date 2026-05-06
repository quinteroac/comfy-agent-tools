"""Defaults for comfy-musicgen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from comfy_agent_tools.loras import ExtraLora


DEFAULT_MODELS_DIR = Path("/mnt/models/comfyui")
DEFAULT_UNET = Path("diffusion_models/acestep_v1.5_base.safetensors")
DEFAULT_CLIP_0_6B = Path("text_encoders/qwen_0.6b_ace15.safetensors")
DEFAULT_CLIP_1_7B = Path("text_encoders/qwen_1.7b_ace15.safetensors")
DEFAULT_VAE = Path("vae/ace_1.5_vae.safetensors")

DEFAULT_OUT = Path("outputs")
DEFAULT_DURATION = 120.0
DEFAULT_BPM = 120
DEFAULT_TIME_SIGNATURE = "4"
DEFAULT_LANGUAGE = "en"
DEFAULT_KEYSCALE = "C major"
DEFAULT_SEED = 0
DEFAULT_STEPS = 32
DEFAULT_CFG = 7.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"


@dataclass(frozen=True)
class MusicgenConfig:
    """Runtime configuration for comfy-musicgen commands."""

    models_dir: Path = DEFAULT_MODELS_DIR
    unet: Path = DEFAULT_UNET
    clip_0_6b: Path = DEFAULT_CLIP_0_6B
    clip_1_7b: Path = DEFAULT_CLIP_1_7B
    vae: Path = DEFAULT_VAE
    duration: float = DEFAULT_DURATION
    bpm: int = DEFAULT_BPM
    time_signature: str = DEFAULT_TIME_SIGNATURE
    language: str = DEFAULT_LANGUAGE
    keyscale: str = DEFAULT_KEYSCALE
    seed: int = DEFAULT_SEED
    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
    sampler: str = DEFAULT_SAMPLER
    scheduler: str = DEFAULT_SCHEDULER
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
