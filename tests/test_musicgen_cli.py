from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock
import wave

import torch

from comfy_agent_tools.cli import musicgen
from comfy_agent_tools.musicgen.ace_step import run_ace_step
from comfy_agent_tools.musicgen.artifacts import save_wav
from comfy_agent_tools.musicgen.config import MusicgenConfig


def _audio(samples: int = 24000, sample_rate: int = 24000) -> dict[str, object]:
    tone = torch.sin(torch.linspace(0, math.tau * 440, samples))
    waveform = torch.stack([tone, tone * 0.5]).reshape(1, 2, samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _result() -> dict[str, object]:
    return {"audio": _audio()}


def test_parser_generate_defaults() -> None:
    args = musicgen.build_parser().parse_args(["generate", "--prompt", "hello"])

    assert args.command == "generate"
    assert args.models_dir is None
    assert args.unet is None
    assert args.clip_0_6b is None
    assert args.clip_1_7b is None
    assert args.vae is None
    assert args.duration is None
    assert args.bpm is None
    assert args.keyscale is None
    assert args.steps is None
    assert args.cfg is None
    assert args.extra_lora == []
    assert args.verbose is False


def test_parser_generate_overrides() -> None:
    args = musicgen.build_parser().parse_args(
        [
            "generate",
            "--prompt",
            "synth",
            "--lyrics",
            "hello",
            "--duration",
            "15",
            "--bpm",
            "90",
            "--time-signature",
            "3",
            "--language",
            "es",
            "--keyscale",
            "E minor",
            "--seed",
            "7",
            "--steps",
            "8",
            "--cfg",
            "1",
            "--sampler",
            "ddim",
            "--scheduler",
            "normal",
            "--verbose",
        ]
    )

    assert args.lyrics == "hello"
    assert args.duration == 15.0
    assert args.bpm == 90
    assert args.time_signature == "3"
    assert args.language == "es"
    assert args.keyscale == "E minor"
    assert args.seed == 7
    assert args.steps == 8
    assert args.cfg == 1.0
    assert args.sampler == "ddim"
    assert args.scheduler == "normal"
    assert args.verbose is True


def test_generate_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    lora_path = tmp_path / "loras" / "ace-step-1.5" / "vocal-polish.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")

    def fake_run_ace_step(*, prompt: str, lyrics: str, config: object) -> dict[str, object]:
        assert prompt == "make music"
        assert lyrics == "la la"
        assert config.extra_loras[0].path == Path("loras/ace-step-1.5/vocal-polish.safetensors")
        return _result()

    monkeypatch.setattr(musicgen, "run_ace_step", fake_run_ace_step)

    rc = musicgen.main(
        [
            "generate",
            "--prompt",
            "make music",
            "--lyrics",
            "la la",
            "--language",
            "es",
            "--time-signature",
            "4",
            "--duration",
            "120",
            "--models-dir",
            str(tmp_path),
            "--extra-lora",
            "loras/ace-step-1.5/vocal-polish.safetensors:0.5:0.1",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["kind"] == "music"
    assert payload["mode"] == "generate"
    assert payload["model"] == "acestep_v1.5_base.safetensors"
    assert payload["text_encoder"] == "qwen_1.7b_ace15.safetensors"
    assert payload["format"] == "wav"
    assert payload["sample_rate"] == 24000
    assert payload["channels"] == 2
    assert payload["duration_seconds"] == 1.0
    assert payload["requested_duration_seconds"] == 120.0
    assert payload["bpm"] == 120
    assert payload["keyscale"] == "C major"
    assert payload["language"] == "es"
    assert payload["time_signature"] == "4"
    assert payload["steps"] == 32
    assert payload["cfg"] == 7.0
    assert payload["capability"] == "musicgen.generate"
    assert payload["model_profile"] == "ace15-base"
    assert payload["architecture"] == "ace-step-1.5"
    assert payload["resolved_models"]["unet"].endswith("diffusion_models/acestep_v1.5_base.safetensors")
    assert payload["sampler"] == "euler"
    assert payload["scheduler"] == "simple"
    assert payload["extra_loras"] == [
        {"path": str(lora_path), "strength_model": 0.5, "strength_clip": 0.1}
    ]
    assert payload["lyrics_present"] is True
    assert Path(payload["artifacts"][0]).is_file()


def test_generate_suppresses_output_by_default(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    def noisy_run_ace_step(*, prompt: str, lyrics: str, config: object) -> dict[str, object]:
        print("progress should not leak")
        return _result()

    monkeypatch.setattr(musicgen, "run_ace_step", noisy_run_ace_step)

    rc = musicgen.main(["generate", "--prompt", "quiet", "--out", str(tmp_path)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "progress should not leak" not in captured.out
    assert json.loads(captured.out)["ok"] is True


def test_generate_verbose_allows_output(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    def noisy_run_ace_step(*, prompt: str, lyrics: str, config: object) -> dict[str, object]:
        print("progress is visible")
        return _result()

    monkeypatch.setattr(musicgen, "run_ace_step", noisy_run_ace_step)

    rc = musicgen.main(["generate", "--prompt", "verbose", "--out", str(tmp_path), "--verbose"])

    assert rc == 0
    assert "progress is visible" in capsys.readouterr().out


def test_missing_model_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    def fail(*, prompt: str, lyrics: str, config: object) -> dict[str, object]:
        raise FileNotFoundError("UNet model file not found: missing.safetensors")

    monkeypatch.setattr(musicgen, "run_ace_step", fail)

    rc = musicgen.main(["generate", "--prompt", "fail", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["kind"] == "music"
    assert payload["mode"] == "generate"
    assert payload["error_type"] == "not_found"


def test_runtime_exception_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    def fail(*, prompt: str, lyrics: str, config: object) -> dict[str, object]:
        raise RuntimeError("ComfyUI runtime not available: missing dependency")

    monkeypatch.setattr(musicgen, "run_ace_step", fail)

    rc = musicgen.main(["generate", "--prompt", "fail", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "runtime"


def test_validation_errors_return_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    cases = [
        (["--duration", "0"], "duration must be greater than 0"),
        (["--bpm", "0"], "bpm must be greater than 0"),
        (["--steps", "0"], "steps must be greater than 0"),
        (["--cfg", "0"], "cfg must be greater than 0"),
    ]

    for flags, message in cases:
        def fail(*, prompt: str, lyrics: str, config: object, message: str = message) -> dict[str, object]:
            raise ValueError(message)

        monkeypatch.setattr(musicgen, "run_ace_step", fail)
        rc = musicgen.main(["generate", "--prompt", "fail", "--out", str(tmp_path), *flags])

        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["error_type"] == "error"
        assert payload["error"] == message


def test_missing_audio_payload_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.setattr(musicgen, "run_ace_step", lambda *, prompt, lyrics, config: {})

    rc = musicgen.main(["generate", "--prompt", "fail", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "audio"
    assert "audio payload" in payload["error"]


def test_ace_step_validates_numeric_config_before_runtime() -> None:
    invalid_configs = [
        (MusicgenConfig(duration=0), "duration must be greater than 0"),
        (MusicgenConfig(bpm=0), "bpm must be greater than 0"),
        (MusicgenConfig(steps=0), "steps must be greater than 0"),
        (MusicgenConfig(cfg=0), "cfg must be greater than 0"),
    ]

    for config, message in invalid_configs:
        try:
            run_ace_step(prompt="tags", lyrics="", config=config)
        except ValueError as exc:
            assert str(exc) == message
        else:
            raise AssertionError(f"expected ValueError: {message}")


def test_save_wav_with_stereo_audio(tmp_path: Path) -> None:
    output = tmp_path / "music.wav"

    metadata = save_wav(_audio(samples=12000, sample_rate=24000), output)

    assert output.is_file()
    assert metadata["format"] == "wav"
    assert metadata["sample_rate"] == 24000
    assert metadata["channels"] == 2
    assert metadata["duration_seconds"] == 0.5
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 2
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 12000
