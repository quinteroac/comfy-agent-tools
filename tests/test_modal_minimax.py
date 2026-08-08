from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from comfy_agent_tools import modal_minimax


def _write_models(root: Path, paths: tuple[Path, ...]) -> None:
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")


def test_required_model_paths_are_mode_specific() -> None:
    assert "minimax_h3_fl2va" in str(modal_minimax.required_model_paths("t2v")[0])
    assert "minimax_h3_fl2va" in str(modal_minimax.required_model_paths("i2v")[0])
    assert "minimax_h3_ref2va" in str(modal_minimax.required_model_paths("r2v")[0])
    assert modal_minimax.DEFAULT_MINIMAX_TURBO_LORA in modal_minimax.required_model_paths("t2v")
    assert modal_minimax.DEFAULT_MINIMAX_TURBO_LORA in modal_minimax.required_model_paths("r2v")
    assert len(modal_minimax.required_model_paths("t2v")) == 5
    assert len(modal_minimax.required_model_paths("r2v")) == 5

    with pytest.raises(ValueError, match="unsupported"):
        modal_minimax.required_model_paths("bad")


def test_prepare_models_uploads_only_missing_volume_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    required = modal_minimax.required_model_paths("t2v")
    _write_models(tmp_path, required)
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-1] == "--json":
            return subprocess.CompletedProcess(command, 0, '{"paths":["' + required[0].as_posix() + '"]}', "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(modal_minimax, "_modal_command", lambda: "modal")
    result = modal_minimax.prepare_models("t2v", models_dir=tmp_path, volume="vol", runner=runner)

    assert result["skipped"] == [required[0].as_posix()]
    assert set(result["uploaded"]) == {path.as_posix() for path in required[1:]}
    upload_calls = [call for call in calls if len(call) > 3 and call[1:3] == ["volume", "put"]]
    assert len(upload_calls) == 4
    assert all(call[-1].startswith("/") for call in upload_calls)


def test_prepare_models_force_uploads_existing_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    required = modal_minimax.required_model_paths("r2v")
    _write_models(tmp_path, required)
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[-1] == "--json":
            return subprocess.CompletedProcess(command, 0, "{\"paths\": [\"all\"]}", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(modal_minimax, "_modal_command", lambda: "modal")
    result = modal_minimax.prepare_models("r2v", models_dir=tmp_path, volume="vol", force_upload=True, runner=runner)

    assert result["skipped"] == []
    assert len([call for call in calls if call[1:3] == ["volume", "put"]]) == 5


def test_prepare_models_downloads_missing_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    required = modal_minimax.required_model_paths("i2v")
    downloaded: list[str] = []

    def download(mode: str, *, runner: object) -> None:
        downloaded.append(mode)
        _write_models(tmp_path, required)

    monkeypatch.setattr(modal_minimax, "_download_capability", download)
    monkeypatch.setattr(modal_minimax, "_modal_command", lambda: "modal")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--json":
            return subprocess.CompletedProcess(command, 0, "{}", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    result = modal_minimax.prepare_models("i2v", models_dir=tmp_path, volume="vol", runner=runner)
    assert downloaded == ["i2v"]
    assert result["mode"] == "i2v"


def test_validate_modal_auth_accepts_environment_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_TOKEN_ID", "id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "secret")
    modal_minimax.validate_modal_auth()


def test_validate_modal_auth_rejects_missing_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(modal_minimax.ModalPreparationError, match="authentication"):
        modal_minimax.validate_modal_auth()
