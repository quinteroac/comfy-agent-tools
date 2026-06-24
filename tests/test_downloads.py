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
        if not profile.models:
            continue
        items = downloads.download_items_for_profile(profile)
        assert {item.model_key for item in items} == set(profile.models)


def test_download_remote_seedance_profile_reports_unsupported_source() -> None:
    config = default_config()
    raw = resolve_profile("seedance2-api", config)
    profile = make_resolved_profile("videogen.seedance2-t2v", "seedance2-api", raw, config)

    with pytest.raises(downloads.DownloadUnsupportedSourceError):
        downloads.download_items_for_profile(profile)


def test_download_remote_grok_profile_reports_unsupported_source() -> None:
    config = default_config()
    raw = resolve_profile("grok-imagine-api", config)
    profile = make_resolved_profile("imagegen.grok-generate", "grok-imagine-api", raw, config)

    with pytest.raises(downloads.DownloadUnsupportedSourceError):
        downloads.download_items_for_profile(profile)


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


def test_models_download_motion_track_dry_run_includes_ic_lora(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "videogen.motion-track", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "ltx23-motion-track"
    assert str(tmp_path / "models/loras/ltx23/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors") in payload["planned"]
    assert not (tmp_path / "models").exists()


def test_models_download_ideogram4_dry_run_includes_fp8_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "imagegen.ideogram4-generate", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "ideogram4-fp8"
    assert set(payload["sources"]) == {"huggingface"}
    assert str(tmp_path / "models/diffusion_models/ideogram4_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/text_encoders/qwen3vl_8b_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/vae/flux2-vae.safetensors") in payload["planned"]
    assert not (tmp_path / "models").exists()


def test_models_download_imagedescribe_reports_local_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "imagedescribe.describe", "--dry-run"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_type"] == "download_unsupported_source"
    assert "does not use local downloadable model files" in payload["error"]


def test_models_download_dasiwa_ltx23_dry_run_includes_civitai_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download-profile", "ltx23-dasiwa-golden-lace-v3", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "ltx23-dasiwa-golden-lace-v3"
    assert str(tmp_path / "models/checkpoints/DasiwaLTX23_goldenLaceV3.safetensors") in payload["planned"]
    assert {"civitai", "huggingface"} == set(payload["sources"])
    assert not (tmp_path / "models").exists()


def test_models_download_wan22_i2v_dry_run_includes_dual_unets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "videogen.wan22-i2v", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "wan22-i2v"
    assert str(tmp_path / "models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/vae/wan_2.1_vae.safetensors") in payload["planned"]
    assert not (tmp_path / "models").exists()


def test_models_download_wan22_t2v_dry_run_includes_dual_unets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "videogen.wan22-t2v", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "wan22-t2v"
    assert str(tmp_path / "models/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/vae/wan_2.1_vae.safetensors") in payload["planned"]
    assert set(payload["sources"]) == {"huggingface"}
    assert not (tmp_path / "models").exists()


def test_models_download_dasiwa_wan22_t2v_dry_run_marks_unets_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download-profile", "wan22-dasiwa-tastysin-t2v", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "wan22-dasiwa-tastysin-t2v"
    assert str(tmp_path / "models/diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinHighV81.safetensors") in payload["planned"]
    assert str(tmp_path / "models/diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinLowV81.safetensors") in payload["planned"]
    assert str(tmp_path / "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/vae/wan_2.1_vae.safetensors") in payload["planned"]
    assert set(payload["sources"]) == {"huggingface", "local"}
    assert not (tmp_path / "models").exists()


def test_models_download_wan22_s2v_dry_run_includes_audio_encoder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "videogen.wan22-s2v", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "wan22-s2v"
    assert str(tmp_path / "models/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/audio_encoders/wav2vec2_large_english_fp16.safetensors") in payload["planned"]
    assert str(tmp_path / "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/vae/wan_2.1_vae.safetensors") in payload["planned"]
    assert not (tmp_path / "models").exists()


def test_models_download_dasiwa_wan22_s2v_dry_run_includes_civitai_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download-profile", "wan22-dasiwa-littledemon-v2-s2v", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "wan22-dasiwa-littledemon-v2-s2v"
    assert str(tmp_path / "models/diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors") in payload["planned"]
    assert str(tmp_path / "models/audio_encoders/wav2vec2_large_english_fp16.safetensors") in payload["planned"]
    assert {"civitai", "huggingface"} == set(payload["sources"])
    assert not (tmp_path / "models").exists()


def test_models_download_wan22_video_audio_dry_run_includes_dasiwa_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download", "videogen.wan22-video-audio", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "wan22-dasiwa-littledemon-v2-video-audio"
    assert str(tmp_path / "models/diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors") in payload["planned"]
    assert str(tmp_path / "models/audio_encoders/wav2vec2_large_english_fp16.safetensors") in payload["planned"]
    assert str(tmp_path / "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors") in payload["planned"]
    assert str(tmp_path / "models/vae/wan_2.1_vae.safetensors") in payload["planned"]
    assert {"civitai", "huggingface"} == set(payload["sources"])
    assert not (tmp_path / "models").exists()


def test_models_download_flux_klein_snofs_dry_run_includes_lora(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    models.main(["set-models-dir", str(tmp_path / "models")])
    capsys.readouterr()

    rc = models.main(["download-profile", "flux-klein-9b-snofs", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["model_profile"] == "flux-klein-9b-snofs"
    assert str(tmp_path / "models/diffusion_models/flux-2-klein-9b-fp8.safetensors") in payload["planned"]
    assert str(tmp_path / "models/text_encoders/qwen_3_8b_fp8mixed.safetensors") in payload["planned"]
    assert str(tmp_path / "models/vae/flux2-vae.safetensors") in payload["planned"]
    assert str(tmp_path / "models/loras/flux-klein/klein_snofs_v1_1.safetensors") in payload["planned"]
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


def test_civitai_download_uses_query_token_instead_of_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIVITAI_API_TOKEN", "secret-token")
    url, headers = downloads._http_url_and_headers(
        downloads.DownloadSource(
            kind="http",
            url="https://civitai.com/api/download/models/123?type=Model&format=SafeTensor",
            token_env="CIVITAI_API_TOKEN",
        )
    )

    assert "token=secret-token" in url
    assert "Authorization" not in headers


def test_download_local_profile_reports_unsupported_source(tmp_path: Path) -> None:
    config = default_config()
    config["models_dir"] = str(tmp_path)
    config["profiles"] = {
        "local-anima": {
            "extends": "anima-base",
            "architecture": "anima",
            "models": {"unet": "diffusion_models/local.safetensors"},
        }
    }
    raw = resolve_profile("local-anima", config)
    profile = make_resolved_profile("imagegen.generate", "local-anima", raw, config)

    with pytest.raises(downloads.DownloadUnsupportedSourceError):
        downloads.download_items_for_profile(profile)
