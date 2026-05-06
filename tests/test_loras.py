from __future__ import annotations

from pathlib import Path

import pytest

from comfy_agent_tools.loras import extra_loras_json, parse_extra_lora, resolve_extra_loras


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
