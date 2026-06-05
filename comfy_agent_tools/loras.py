"""Shared helpers for ad hoc LoRA application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtraLora:
    """A LoRA requested for one command invocation."""

    path: Path
    strength_model: float = 1.0
    strength_clip: float = 0.0

    def resolved(self, models_dir: Path) -> "ExtraLora":
        """Return this LoRA with its path resolved against models_dir."""
        path = self.path if self.path.is_absolute() else models_dir / self.path
        return ExtraLora(path=path, strength_model=self.strength_model, strength_clip=self.strength_clip)

    def to_json(self, models_dir: Path) -> dict[str, Any]:
        """Return stable JSON metadata for this LoRA."""
        resolved = self.resolved(models_dir)
        return {
            "path": str(resolved.path),
            "strength_model": resolved.strength_model,
            "strength_clip": resolved.strength_clip,
        }


def parse_extra_lora(value: str) -> ExtraLora:
    """Parse PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]."""
    parts = value.split(":")
    if not parts or not parts[0]:
        raise ValueError("extra LoRA path must not be empty")
    if len(parts) > 3:
        raise ValueError(f"invalid extra LoRA spec: {value}")

    try:
        strength_model = float(parts[1]) if len(parts) >= 2 and parts[1] else 1.0
        strength_clip = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0
    except ValueError as exc:
        raise ValueError(f"invalid extra LoRA strength in spec: {value}") from exc

    return ExtraLora(path=Path(parts[0]), strength_model=strength_model, strength_clip=strength_clip)


def resolve_extra_loras(models_dir: Path, loras: list[ExtraLora]) -> list[ExtraLora]:
    """Resolve and validate LoRA files."""
    resolved = [lora.resolved(models_dir) for lora in loras]
    for lora in resolved:
        if not lora.path.is_file():
            raise FileNotFoundError(f"extra LoRA file not found: {lora.path}")
    return resolved


def extra_loras_json(models_dir: Path, loras: list[ExtraLora]) -> list[dict[str, Any]]:
    """Serialize extra LoRAs for command JSON output."""
    return [lora.to_json(models_dir) for lora in loras]


def apply_extra_loras(model: Any, clip: Any, loras: list[ExtraLora]) -> tuple[Any, Any]:
    """Apply extra LoRAs in order to a model/clip pair."""
    from comfy_diffusion.lora import apply_lora

    for lora in loras:
        model, clip = apply_lora(
            model,
            clip,
            lora.path,
            lora.strength_model,
            lora.strength_clip,
        )
    return model, clip


def apply_extra_loras_to_models(models: list[Any], clip: Any, loras: list[ExtraLora]) -> tuple[list[Any], Any]:
    """Apply extra LoRAs to multiple models, patching the shared clip once."""
    from comfy_diffusion.lora import apply_lora

    patched_models = list(models)
    if not patched_models:
        return patched_models, clip

    for lora in loras:
        patched_models[0], clip = apply_lora(
            patched_models[0],
            clip,
            lora.path,
            lora.strength_model,
            lora.strength_clip,
        )
        for index in range(1, len(patched_models)):
            patched_models[index], _ = apply_lora(
                patched_models[index],
                None,
                lora.path,
                lora.strength_model,
                0.0,
            )
    return patched_models, clip
