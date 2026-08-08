from __future__ import annotations

import sys
import types
from pathlib import Path

from comfy_agent_tools.videogen import minimax


def test_minimax_turbo_loader_resolves_nested_lora_and_strength(tmp_path: Path, monkeypatch) -> None:
    lora = tmp_path / "loras" / "minimax" / "minimax_h3_turbo_4step_ckpt850.safetensors"
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b"placeholder")

    calls: dict[str, object] = {}

    class Loader:
        def apply_lora(self, model, name, strength):
            calls.update(model=model, name=name, strength=strength)
            return ("patched-model",)

    node = types.SimpleNamespace(MiniMaxH3TurboLoRA=Loader)
    folder_paths = types.SimpleNamespace(
        add_model_folder_path=lambda family, path: calls.update(folder=(family, path))
    )
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(minimax, "_load_turbo_node", lambda _config: node)

    config = minimax.MiniMaxH3Config(
        models_dir=tmp_path,
        turbo_lora=lora.relative_to(tmp_path),
        turbo_lora_strength=0.85,
    )

    assert minimax._apply_turbo_lora("base-model", config) == "patched-model"
    assert calls == {
        "model": "base-model",
        "name": "minimax/minimax_h3_turbo_4step_ckpt850.safetensors",
        "strength": 0.85,
        "folder": ("loras", str(tmp_path / "loras")),
    }
