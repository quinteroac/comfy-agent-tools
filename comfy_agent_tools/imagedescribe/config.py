"""Defaults and runtime config for comfy-imagedescribe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS_DIR = Path("/mnt/models/comfyui")
DEFAULT_LLM = Path("LLM/Qwen-VL/Qwen3-VL-2B-Instruct")

DEFAULT_OUT = Path("outputs")
DEFAULT_SEED = 0
DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 64
DEFAULT_TOP_P = 0.95
DEFAULT_MIN_P = 0.05
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_DO_SAMPLE = True


@dataclass(frozen=True)
class ImagedescribeConfig:
    """Runtime configuration for comfy-imagedescribe commands."""

    models_dir: Path = DEFAULT_MODELS_DIR
    llm: Path = DEFAULT_LLM
    seed: int = DEFAULT_SEED
    max_length: int = DEFAULT_MAX_LENGTH
    do_sample: bool = DEFAULT_DO_SAMPLE
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    min_p: float = DEFAULT_MIN_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

    def resolve_model_path(self, path: Path) -> Path:
        """Resolve a model path relative to models_dir when needed."""
        if path.is_absolute():
            return path
        return self.models_dir / path
