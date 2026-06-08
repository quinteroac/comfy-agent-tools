from __future__ import annotations

from pathlib import Path

import pytest

from comfy_agent_tools.profiles import (
    ArchitectureMismatchError,
    BUILTIN_DEFAULTS,
    BUILTIN_PROFILES,
    default_config,
    resolve_capability,
    resolve_model_path,
    resolve_profile,
    validate_defaults,
)


def test_builtin_profiles_separate_architecture_and_profile() -> None:
    profile = BUILTIN_PROFILES["ltx23-10eros"]

    assert profile["architecture"] == "ltx23"
    assert "videogen.ia2av" in profile["supports"]
    assert "10Eros" in profile["label"]

    dasiwa_ltx = BUILTIN_PROFILES["ltx23-dasiwa-golden-lace-v3"]
    assert dasiwa_ltx["architecture"] == "ltx23"
    assert dasiwa_ltx["supports"] == ["videogen.t2v", "videogen.i2v", "videogen.flf2v", "videogen.ia2av"]
    assert dasiwa_ltx["models"]["checkpoint"] == "checkpoints/DasiwaLTX23_goldenLaceV3.safetensors"
    assert dasiwa_ltx["defaults"]["cfg"] == 1.0

    anima = BUILTIN_PROFILES["anima-base"]
    assert anima["architecture"] == "anima"
    assert anima["supports"] == ["imagegen.generate"]
    assert anima["defaults"]["steps"] == 8
    assert anima["defaults"]["cfg"] == 1.0
    assert anima["models"]["unet"] == "diffusion_models/anima-base-v1.0.safetensors"
    assert anima["models"]["lora"] == "loras/anima/anima-turbo-lora-v0.1.safetensors"

    flux = BUILTIN_PROFILES["flux-klein-9b-snofs"]
    assert flux["architecture"] == "flux-klein"
    assert flux["supports"] == ["imagegen.generate", "imagegen.edit"]
    assert flux["defaults"]["steps"] == 4
    assert flux["models"]["lora"] == "loras/flux-klein/klein_snofs_v1_1.safetensors"

    ideogram4 = BUILTIN_PROFILES["ideogram4-fp8"]
    assert ideogram4["architecture"] == "ideogram4"
    assert ideogram4["supports"] == ["imagegen.ideogram4-generate"]
    assert ideogram4["defaults"]["steps"] == 20
    assert ideogram4["defaults"]["cfg"] == 7.0
    assert ideogram4["models"]["unet"] == "diffusion_models/ideogram4_fp8_scaled.safetensors"
    assert ideogram4["models"]["uncond_unet"] == "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors"

    wan22 = BUILTIN_PROFILES["wan22-i2v"]
    assert wan22["architecture"] == "wan22"
    assert wan22["supports"] == ["videogen.wan22-i2v", "videogen.wan22-flf2v"]
    assert wan22["models"]["unet_high"] == "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    assert wan22["defaults"]["fps"] == 16

    wan22_s2v = BUILTIN_PROFILES["wan22-s2v"]
    assert wan22_s2v["architecture"] == "wan22"
    assert wan22_s2v["supports"] == ["videogen.wan22-s2v"]
    assert wan22_s2v["models"]["unet"] == "diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"
    assert wan22_s2v["models"]["audio_encoder"] == "audio_encoders/wav2vec2_large_english_fp16.safetensors"
    assert wan22_s2v["defaults"]["length"] == 77
    assert wan22_s2v["defaults"]["cfg"] == 6.0

    dasiwa_s2v = BUILTIN_PROFILES["wan22-dasiwa-littledemon-v2-s2v"]
    assert dasiwa_s2v["architecture"] == "wan22"
    assert dasiwa_s2v["supports"] == ["videogen.wan22-s2v"]
    assert dasiwa_s2v["models"]["unet"] == "diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors"
    assert dasiwa_s2v["defaults"]["steps"] == 4
    assert dasiwa_s2v["defaults"]["cfg"] == 1.0
    assert dasiwa_s2v["defaults"]["sampler"] == "euler"
    assert dasiwa_s2v["defaults"]["shift"] == 10.0

    dasiwa_video_audio = BUILTIN_PROFILES["wan22-dasiwa-littledemon-v2-video-audio"]
    assert dasiwa_video_audio["architecture"] == "wan22"
    assert dasiwa_video_audio["supports"] == ["videogen.wan22-video-audio"]
    assert dasiwa_video_audio["models"]["unet"] == "diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors"
    assert dasiwa_video_audio["defaults"]["chunk_length"] == 77
    assert dasiwa_video_audio["defaults"]["chunk_overlap"] == 4
    assert dasiwa_video_audio["defaults"]["steps"] == 4
    assert dasiwa_video_audio["defaults"]["denoise"] == 0.35
    assert dasiwa_video_audio["defaults"]["lipsync_second_steps"] == 2

    dasiwa = BUILTIN_PROFILES["wan22-dasiwa-tastysin-i2v"]
    assert dasiwa["architecture"] == "wan22"
    assert dasiwa["supports"] == ["videogen.wan22-i2v", "videogen.wan22-flf2v"]
    assert dasiwa["models"]["unet_high"] == "diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinHighV81.safetensors"
    assert dasiwa["models"]["unet_low"] == "diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinLowV81.safetensors"
    assert dasiwa["defaults"]["steps"] == 4
    assert dasiwa["defaults"]["i2v_cfg"] == 1.0
    assert dasiwa["defaults"]["flf2v_cfg"] == 1.0

    boundbite = BUILTIN_PROFILES["wan22-dasiwa-boundbite-i2v"]
    assert boundbite["architecture"] == "wan22"
    assert boundbite["supports"] == ["videogen.wan22-i2v", "videogen.wan22-flf2v"]
    assert boundbite["models"]["unet_high"] == "diffusion_models/DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors"
    assert boundbite["models"]["unet_low"] == "diffusion_models/DasiwaWAN22I2V14BLightspeed_boundbiteLowV10.safetensors"
    assert boundbite["defaults"]["steps"] == 4
    assert boundbite["defaults"]["i2v_cfg"] == 1.0
    assert boundbite["defaults"]["flf2v_cfg"] == 1.0

    seedance = BUILTIN_PROFILES["seedance2-api"]
    assert seedance["architecture"] == "seedance2-api"
    assert seedance["models"] == {}
    assert seedance["defaults"]["remote"] is True
    assert seedance["defaults"]["model"] == "Seedance 2.0"
    assert "videogen.seedance2-t2v" in seedance["supports"]

    grok = BUILTIN_PROFILES["grok-imagine-api"]
    assert grok["architecture"] == "grok-imagine-api"
    assert grok["models"] == {}
    assert grok["defaults"]["remote"] is True
    assert grok["defaults"]["model"] == "grok-imagine-image"
    assert grok["supports"] == ["imagegen.grok-generate", "imagegen.grok-edit"]


def test_builtin_defaults_point_to_supported_profiles() -> None:
    config = default_config()

    validate_defaults(config)
    assert BUILTIN_DEFAULTS["imagegen.generate"] == "anima-base"
    assert BUILTIN_DEFAULTS["imagegen.edit"] == "qwen-edit2511"
    assert BUILTIN_DEFAULTS["videogen.seedance2-t2v"] == "seedance2-api"
    assert BUILTIN_DEFAULTS["videogen.wan22-i2v"] == "wan22-i2v"
    assert BUILTIN_DEFAULTS["videogen.wan22-s2v"] == "wan22-s2v"
    assert BUILTIN_DEFAULTS["videogen.wan22-video-audio"] == "wan22-dasiwa-littledemon-v2-video-audio"
    assert BUILTIN_DEFAULTS["imagegen.grok-generate"] == "grok-imagine-api"
    assert BUILTIN_DEFAULTS["imagegen.grok-edit"] == "grok-imagine-api"
    assert BUILTIN_DEFAULTS["imagegen.ideogram4-generate"] == "ideogram4-fp8"
    for capability, profile_name in BUILTIN_DEFAULTS.items():
        profile = resolve_profile(profile_name, config)
        assert capability in profile["supports"]


def test_inherited_ltx_profile_overrides_checkpoint_only() -> None:
    config = default_config()
    config["profiles"] = {
        "my-ltx23-finetune": {
            "extends": "ltx23-10eros",
            "architecture": "ltx23",
            "models": {"checkpoint": "checkpoints/my_ltx23_finetune.safetensors"},
        }
    }
    config["defaults"]["videogen.t2v"] = "my-ltx23-finetune"

    profile = resolve_profile("my-ltx23-finetune", config)

    assert profile["architecture"] == "ltx23"
    assert profile["models"]["checkpoint"] == "checkpoints/my_ltx23_finetune.safetensors"
    assert profile["models"]["text_encoder"] == "text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"


def test_inherited_profile_rejects_architecture_mismatch() -> None:
    config = default_config()
    config["profiles"] = {
        "bad-ltx": {
            "extends": "ltx23-10eros",
            "architecture": "qwen-image-edit",
        }
    }

    with pytest.raises(ArchitectureMismatchError):
        resolve_profile("bad-ltx", config)


def test_resolve_capability_uses_local_models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / ".comfy-agent-tools.json"
    config_path.write_text(
        """
{
  "models_dir": "/tmp/custom-models",
  "defaults": {
    "imagegen.generate": "qwen-edit2511"
  },
  "profiles": {}
}
""".strip(),
        encoding="utf-8",
    )

    profile, source = resolve_capability("imagegen.generate", config_path)

    assert source == "local"
    assert profile.models_dir == Path("/tmp/custom-models")
    assert resolve_model_path(profile.models_dir, profile.models["unet"]) == Path(
        "/tmp/custom-models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"
    )
