"""Defaults for comfy-videogen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from comfy_agent_tools.loras import ExtraLora


DEFAULT_MODELS_DIR = Path("/mnt/models/comfyui")
DEFAULT_CHECKPOINT = Path("checkpoints/10Eros_v1-fp8mixed_learned.safetensors")
DEFAULT_TEXT_ENCODER = Path("text_encoders/gemma_3_12B_it_fp4_mixed.safetensors")
DEFAULT_DISTILLED_LORA = Path("loras/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors")
DEFAULT_TE_LORA = Path("loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors")
DEFAULT_UPSCALER = Path("latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")

DEFAULT_OUT = Path("outputs")
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 320
DEFAULT_LENGTH = 49
DEFAULT_FPS = 24
DEFAULT_CFG = 1.0
DEFAULT_SEED = 0
DEFAULT_AUDIO_START_TIME = 0.0
DEFAULT_DISTILLED_LORA_STRENGTH = 0.5
DEFAULT_TE_LORA_STRENGTH = 1.0
DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


@dataclass(frozen=True)
class VideogenConfig:
    """Runtime configuration for comfy-videogen commands."""

    models_dir: Path = DEFAULT_MODELS_DIR
    checkpoint: Path = DEFAULT_CHECKPOINT
    text_encoder: Path = DEFAULT_TEXT_ENCODER
    distilled_lora: Path = DEFAULT_DISTILLED_LORA
    te_lora: Path = DEFAULT_TE_LORA
    upscaler: Path = DEFAULT_UPSCALER
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    length: int = DEFAULT_LENGTH
    fps: int = DEFAULT_FPS
    cfg: float = DEFAULT_CFG
    seed: int = DEFAULT_SEED
    audio_start_time: float = DEFAULT_AUDIO_START_TIME
    audio_duration: float | None = None
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    distilled_lora_strength: float = DEFAULT_DISTILLED_LORA_STRENGTH
    te_lora_strength: float = DEFAULT_TE_LORA_STRENGTH
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
