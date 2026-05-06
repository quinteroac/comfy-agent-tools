from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from comfy_agent_tools import downloads
from comfy_agent_tools.cli import models
from comfy_agent_tools.profiles import BUILTIN_PROFILES, default_config, make_resolved_profile, resolve_profile


def test_download_registry_covers_builtin_profile_models() -> None:
    config = default_config()
    for profile_name, raw in BUILTIN_PROFILES.items():
        profile = make_resolved_profile(raw["supports"][0], profile_name, resolve_profile(profile_name, config), config)
        items = downloads.download_items_for_profile(profile)
        assert {item.model_key for item in items} == set(profile.models)


def test_models_download_dry_run_reports_missing_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "imagegen.upscale", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["downloaded"] == []
    assert payload["skipped"] == []
    assert payload["planned"] == [str(tmp_path / "models/upscale_models/4x-ClearRealityV1.pth")]
    assert not (tmp_path / "models").exists()


def test_models_download_writes_missing_file_and_skips_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    def fake_stream(_url: str, target: Path, *, headers: dict[str, str]) -> None:
        target.write_bytes(b"model")

    monkeypatch.setattr(downloads, "_stream_download", fake_stream)

    rc = models.main(["download", "imagegen.upscale", "--yes"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    artifact = tmp_path / "models/upscale_models/4x-ClearRealityV1.pth"
    assert artifact.read_bytes() == b"model"
    assert payload["downloaded"] == [str(artifact)]
    assert payload["skipped"] == []
    assert payload["total_downloaded_bytes"] == 5

    rc = models.main(["download", "imagegen.upscale", "--yes"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["downloaded"] == []
    assert payload["skipped"] == [str(artifact)]


def test_models_download_requires_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)

    rc = models.main(["download", "imagegen.generate"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "requires --yes" in payload["error"]


def test_download_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"bad")

    with pytest.raises(downloads.DownloadChecksumMismatchError):
        downloads._validate_download(
            path,
            downloads.DownloadSource(kind="http", url="https://example.test/model", sha256="0" * 64),
        )


def test_download_local_profile_reports_unsupported_source(tmp_path: Path) -> None:
    config = default_config()
    config["models_dir"] = str(tmp_path)
    config["profiles"] = {
        "local-anima": {
            "extends": "anima-preview3-turbo",
            "architecture": "anima",
            "models": {"unet": "diffusion_models/local.safetensors"},
        }
    }
    raw = resolve_profile("local-anima", config)
    profile = make_resolved_profile("imagegen.generate", "local-anima", raw, config)

    with pytest.raises(downloads.DownloadUnsupportedSourceError):
        downloads.download_items_for_profile(profile)
