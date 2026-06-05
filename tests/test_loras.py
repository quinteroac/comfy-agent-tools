from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from comfy_agent_tools.loras import (
    ExtraLora,
    apply_extra_loras_to_models,
    extra_loras_json,
    parse_extra_lora,
    resolve_extra_loras,
)


def test_parse_extra_lora_path_only() -> None:
    lora = parse_extra_lora("loras/anima/realism.safetensors")

    assert lora.path == Path("loras/anima/realism.safetensors")
    assert lora.strength_model == 1.0
    assert lora.strength_clip == 0.0


def test_parse_extra_lora_model_strength() -> None:
    lora = parse_extra_lora("loras/anima/realism.safetensors:0.8")

    assert lora.path == Path("loras/anima/realism.safetensors")
    assert lora.strength_model == 0.8
    assert lora.strength_clip == 0.0


def test_parse_extra_lora_model_and_clip_strength() -> None:
    lora = parse_extra_lora("loras/anima/realism.safetensors:0.8:0.2")

    assert lora.path == Path("loras/anima/realism.safetensors")
    assert lora.strength_model == 0.8
    assert lora.strength_clip == 0.2


def test_parse_extra_lora_rejects_invalid_strength() -> None:
    with pytest.raises(ValueError, match="invalid extra LoRA strength"):
        parse_extra_lora("loras/anima/realism.safetensors:nope")


def test_resolve_extra_loras_validates_files(tmp_path: Path) -> None:
    lora_path = tmp_path / "loras" / "anima" / "realism.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")

    resolved = resolve_extra_loras(tmp_path, [parse_extra_lora("loras/anima/realism.safetensors:0.7")])

    assert resolved[0].path == lora_path
    assert extra_loras_json(tmp_path, resolved) == [
        {
            "path": str(lora_path),
            "strength_model": 0.7,
            "strength_clip": 0.0,
        }
    ]


def test_resolve_extra_loras_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="extra LoRA file not found"):
        resolve_extra_loras(tmp_path, [parse_extra_lora("loras/anima/missing.safetensors")])


def test_apply_extra_loras_to_models_patches_shared_clip_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object, Path, float, float]] = []
    lora_module = types.ModuleType("comfy_diffusion.lora")

    def apply_lora(
        model: str,
        clip: object,
        path: Path,
        strength_model: float,
        strength_clip: float,
    ) -> tuple[str, object]:
        calls.append((model, clip, path, strength_model, strength_clip))
        patched_clip = f"{clip}-patched" if clip is not None else None
        return f"{model}-patched", patched_clip

    lora_module.apply_lora = apply_lora
    monkeypatch.setitem(sys.modules, "comfy_diffusion", types.ModuleType("comfy_diffusion"))
    monkeypatch.setitem(sys.modules, "comfy_diffusion.lora", lora_module)

    models, clip = apply_extra_loras_to_models(
        ["high", "low"],
        "clip",
        [ExtraLora(Path("loras/wan22/style.safetensors"), strength_model=0.7, strength_clip=0.2)],
    )

    assert models == ["high-patched", "low-patched"]
    assert clip == "clip-patched"
    assert calls == [
        ("high", "clip", Path("loras/wan22/style.safetensors"), 0.7, 0.2),
        ("low", None, Path("loras/wan22/style.safetensors"), 0.7, 0.0),
    ]
