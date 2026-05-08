from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from comfy_agent_tools.cli import models


def test_models_list_uses_builtin_fallback(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(["list"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["config_source"] == "builtin"
    assert "anima" in payload["architectures"]
    assert "ltx23" in payload["architectures"]
    assert payload["profiles"]["anima-preview3-turbo"]["architecture"] == "anima"
    assert payload["profiles"]["ltx23-10eros"]["architecture"] == "ltx23"
    assert payload["profiles"]["ltx23-motion-track"]["supports"] == ["videogen.motion-track"]


def test_models_init_and_show(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(["init"])

    assert rc == 0
    assert (tmp_path / ".comfy-agent-tools.json").is_file()
    payload = json.loads(capsys.readouterr().out)
    assert payload["created"] is True

    rc = models.main(["show"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_source"] == "local"
    assert payload["models_dir"] == "/mnt/models/comfyui"
    assert payload["resolved_defaults"]["imagegen.generate"]["profile"] == "anima-preview3-turbo"
    assert payload["resolved_defaults"]["videogen.t2v"]["profile"] == "ltx23-10eros"


def test_models_set_models_dir(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(["set-models-dir", "/tmp/models"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models_dir"] == "/tmp/models"

    data = json.loads((tmp_path / ".comfy-agent-tools.json").read_text(encoding="utf-8"))
    assert data["models_dir"] == "/tmp/models"


def test_models_add_ltx_profile_and_set_default(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(
        [
            "add-profile",
            "my-ltx23-finetune",
            "--extends",
            "ltx23-10eros",
            "--checkpoint",
            "checkpoints/my_ltx23_finetune.safetensors",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "my-ltx23-finetune"

    rc = models.main(["set-default", "videogen.t2v", "my-ltx23-finetune"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["capability"] == "videogen.t2v"

    data = json.loads((tmp_path / ".comfy-agent-tools.json").read_text(encoding="utf-8"))
    assert data["profiles"]["my-ltx23-finetune"]["architecture"] == "ltx23"
    assert data["defaults"]["videogen.t2v"] == "my-ltx23-finetune"


def test_models_add_ltx_motion_profile_with_ic_lora(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(
        [
            "add-profile",
            "my-motion-track",
            "--extends",
            "ltx23-motion-track",
            "--ic-lora",
            "loras/ltx23/custom-motion-track.safetensors",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "my-motion-track"

    rc = models.main(["set-default", "videogen.motion-track", "my-motion-track"])

    assert rc == 0
    data = json.loads((tmp_path / ".comfy-agent-tools.json").read_text(encoding="utf-8"))
    assert data["profiles"]["my-motion-track"]["models"]["ic_lora"] == "loras/ltx23/custom-motion-track.safetensors"
    assert data["defaults"]["videogen.motion-track"] == "my-motion-track"


def test_models_set_default_rejects_unsupported_capability(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(["set-default", "imagegen.edit", "clear-reality"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_type"] == "unsupported_capability"


def test_models_add_profile_rejects_architecture_mismatch(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(
        [
            "add-profile",
            "bad-ltx",
            "--extends",
            "ltx23-10eros",
            "--architecture",
            "qwen-image-edit",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "architecture_mismatch"


def test_models_validate_profile_reports_missing_file(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(
        [
            "add-profile",
            "missing-ltx",
            "--extends",
            "ltx23-10eros",
            "--models-dir",
            str(tmp_path),
            "--checkpoint",
            "checkpoints/missing.safetensors",
        ]
    )
    capsys.readouterr()

    rc = models.main(["validate-profile", "missing-ltx"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "missing_model_file"


def test_models_reset_removes_config(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["init"])
    capsys.readouterr()

    rc = models.main(["reset", "--yes"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] is True
    assert not (tmp_path / ".comfy-agent-tools.json").exists()
