from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
import torch

from comfy_agent_tools.cli import videogen
from comfy_agent_tools.videogen.artifacts import save_mp4_with_audio


def _frames() -> list[Image.Image]:
    return [Image.new("RGB", (32, 18), "red"), Image.new("RGB", (32, 18), "blue")]


def _result() -> dict[str, object]:
    return {"frames": _frames(), "audio": {"waveform": object(), "sample_rate": 44100}}


def test_parser_t2v_defaults() -> None:
    args = videogen.build_parser().parse_args(["t2v", "--prompt", "hello"])

    assert args.command == "t2v"
    assert args.models_dir is None
    assert args.checkpoint is None
    assert args.width is None
    assert args.height is None
    assert args.length is None
    assert args.fps is None
    assert args.extra_lora == []
    assert args.verbose is False


def test_parser_accepts_verbose_for_all_modes(tmp_path: Path) -> None:
    parser = videogen.build_parser()

    t2v = parser.parse_args(["t2v", "--prompt", "hello", "--verbose"])
    i2v = parser.parse_args(
        ["i2v", "--input", str(tmp_path / "a.png"), "--prompt", "hello", "--verbose"]
    )
    ia2av = parser.parse_args(
        [
            "ia2av",
            "--input",
            str(tmp_path / "a.png"),
            "--audio",
            str(tmp_path / "a.wav"),
            "--prompt",
            "hello",
            "--verbose",
        ]
    )
    flf2v = parser.parse_args(
        [
            "flf2v",
            "--first",
            str(tmp_path / "a.png"),
            "--last",
            str(tmp_path / "b.png"),
            "--prompt",
            "hello",
            "--verbose",
        ]
    )

    assert t2v.verbose is True
    assert i2v.verbose is True
    assert ia2av.verbose is True
    assert flf2v.verbose is True


def test_parser_ia2av_defaults(tmp_path: Path) -> None:
    args = videogen.build_parser().parse_args(
        [
            "ia2av",
            "--input",
            str(tmp_path / "input.png"),
            "--audio",
            str(tmp_path / "song.wav"),
            "--prompt",
            "move with audio",
        ]
    )

    assert args.command == "ia2av"
    assert args.input == tmp_path / "input.png"
    assert args.audio == tmp_path / "song.wav"
    assert args.audio_start_time == 0.0
    assert args.audio_duration is None
    assert args.length is None
    assert args.fps is None


def test_parser_ia2av_audio_window(tmp_path: Path) -> None:
    args = videogen.build_parser().parse_args(
        [
            "ia2av",
            "--input",
            str(tmp_path / "input.png"),
            "--audio",
            str(tmp_path / "song.wav"),
            "--prompt",
            "move with audio",
            "--audio-start-time",
            "12.5",
            "--audio-duration",
            "4.0",
        ]
    )

    assert args.audio_start_time == 12.5
    assert args.audio_duration == 4.0


def test_parser_seedance2_defaults(tmp_path: Path) -> None:
    parser = videogen.build_parser()

    t2v = parser.parse_args(["seedance2-t2v", "--prompt", "hello"])
    r2v = parser.parse_args(["seedance2-r2v", "--input", str(tmp_path / "a.png"), "--prompt", "hello"])
    flf2v = parser.parse_args(
        [
            "seedance2-flf2v",
            "--first",
            str(tmp_path / "a.png"),
            "--last",
            str(tmp_path / "b.png"),
            "--prompt",
            "hello",
        ]
    )

    assert t2v.command == "seedance2-t2v"
    assert t2v.model == "Seedance 2.0"
    assert t2v.resolution == "480p"
    assert t2v.ratio == "16:9"
    assert t2v.duration == 7
    assert t2v.generate_audio is True
    assert t2v.watermark is False
    assert t2v.seed == 0
    assert r2v.input == tmp_path / "a.png"
    assert flf2v.first == tmp_path / "a.png"
    assert flf2v.last == tmp_path / "b.png"


def test_parser_seedance2_accepts_verbose_and_no_audio() -> None:
    args = videogen.build_parser().parse_args(
        ["seedance2-t2v", "--prompt", "hello", "--no-generate-audio", "--verbose"]
    )

    assert args.generate_audio is False
    assert args.verbose is True


def test_t2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.setattr(videogen, "run_t2v", lambda *, prompt, config: _result())
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(["t2v", "--prompt", "make a video", "--out", str(tmp_path)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["kind"] == "video"
    assert payload["mode"] == "t2v"
    assert payload["model"] == "10Eros_v1-fp8mixed_learned.safetensors"
    assert payload["width"] == 32
    assert payload["height"] == 18
    assert payload["frames"] == 2
    assert payload["fps"] == 24
    assert payload["audio_muxed"] is True
    assert payload["capability"] == "videogen.t2v"
    assert payload["model_profile"] == "ltx23-10eros"
    assert payload["architecture"] == "ltx23"
    assert payload["resolved_models"]["checkpoint"].endswith("checkpoints/10Eros_v1-fp8mixed_learned.safetensors")
    assert Path(payload["artifacts"][0]).is_file()


def test_i2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)
    monkeypatch.setattr(videogen, "run_i2v", lambda *, image, prompt, config: _result())
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(["i2v", "--input", str(input_path), "--prompt", "move", "--out", str(tmp_path)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "i2v"
    assert payload["input"] == str(input_path)
    assert payload["audio_muxed"] is True


def test_flf2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    lora_path = tmp_path / "loras" / "ltx23" / "detailer.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")
    Image.new("RGB", (8, 8), "green").save(first)
    Image.new("RGB", (8, 8), "blue").save(last)

    def fake_run_flf2v(*, first_image, last_image, prompt, config):
        assert config.extra_loras[0].path == Path("loras/ltx23/detailer.safetensors")
        return _result()

    monkeypatch.setattr(videogen, "run_flf2v", fake_run_flf2v)
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "flf2v",
            "--first",
            str(first),
            "--last",
            str(last),
            "--prompt",
            "transition",
            "--models-dir",
            str(tmp_path),
            "--extra-lora",
            "loras/ltx23/detailer.safetensors:0.6",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "flf2v"
    assert payload["first"] == str(first)
    assert payload["last"] == str(last)
    assert payload["extra_loras"] == [
        {"path": str(lora_path), "strength_model": 0.6, "strength_clip": 0.0}
    ]


def test_t2v_extra_lora_returns_clean_unsupported_error(tmp_path: Path, capsys: MagicMock) -> None:
    lora_path = tmp_path / "loras" / "ltx23" / "detailer.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")

    rc = videogen.main(
        [
            "t2v",
            "--prompt",
            "move",
            "--models-dir",
            str(tmp_path),
            "--extra-lora",
            "loras/ltx23/detailer.safetensors",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "t2v"
    assert "extra LoRAs are not supported for videogen.t2v yet" in payload["error"]


def test_ia2av_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    audio_path = tmp_path / "song.wav"
    Image.new("RGB", (8, 8), "green").save(input_path)
    audio_path.write_bytes(b"fake audio")
    seen: dict[str, object] = {}

    def fake_run_ia2av(*, image: Path, audio: Path, prompt: str, config: object) -> dict[str, object]:
        seen["image"] = image
        seen["audio"] = audio
        seen["prompt"] = prompt
        seen["audio_start_time"] = config.audio_start_time
        seen["audio_duration"] = config.audio_duration
        return _result()

    monkeypatch.setattr(videogen, "run_ia2av", fake_run_ia2av)
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "ia2av",
            "--input",
            str(input_path),
            "--audio",
            str(audio_path),
            "--prompt",
            "move with audio",
            "--audio-start-time",
            "1.5",
            "--audio-duration",
            "2.0",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["image"] == input_path
    assert seen["audio"] == audio_path
    assert seen["prompt"] == "move with audio"
    assert seen["audio_start_time"] == 1.5
    assert seen["audio_duration"] == 2.0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "ia2av"
    assert payload["input"] == str(input_path)
    assert payload["audio_input"] == str(audio_path)
    assert payload["audio_conditioned"] is True
    assert payload["capability"] == "videogen.ia2av"
    assert payload["model_profile"] == "ltx23-10eros"
    assert payload["architecture"] == "ltx23"
    assert payload["audio_start_time"] == 1.5
    assert payload["audio_duration_seconds"] == 2.0
    assert payload["audio_muxed"] is True
    assert Path(payload["artifacts"][0]).is_file()


def test_ia2av_success_json_uses_video_duration_when_audio_duration_omitted(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    input_path = tmp_path / "input.png"
    audio_path = tmp_path / "song.wav"
    Image.new("RGB", (8, 8), "green").save(input_path)
    audio_path.write_bytes(b"fake audio")
    monkeypatch.setattr(videogen, "run_ia2av", lambda *, image, audio, prompt, config: _result())
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(
        ["ia2av", "--input", str(input_path), "--audio", str(audio_path), "--prompt", "move", "--out", str(tmp_path)]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audio_duration_seconds"] == 2 / 24


def test_seedance2_t2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    artifact = tmp_path / "seedance.mp4"

    def fake_run(*, prompt, config, out_dir):
        artifact.write_bytes(b"mp4")
        return {"artifact": artifact}

    monkeypatch.setattr(videogen, "run_seedance2_t2v", fake_run)

    rc = videogen.main(["seedance2-t2v", "--prompt", "remote video", "--out", str(tmp_path)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "seedance2-t2v"
    assert payload["remote"] is True
    assert payload["provider"] == "comfy-api"
    assert payload["capability"] == "videogen.seedance2-t2v"
    assert payload["model_profile"] == "seedance2-api"
    assert payload["architecture"] == "seedance2-api"
    assert payload["model"] == "Seedance 2.0"
    assert payload["resolution"] == "480p"
    assert payload["ratio"] == "16:9"
    assert payload["duration_seconds"] == 7
    assert payload["generate_audio"] is True
    assert payload["watermark"] is False
    assert payload["resolved_models"] == {}
    assert payload["artifacts"] == [str(artifact)]


def test_seedance2_r2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    artifact = tmp_path / "seedance-r2v.mp4"
    Image.new("RGB", (8, 8), "green").save(input_path)

    def fake_run(*, image, prompt, config, out_dir):
        assert image == input_path
        assert prompt == "move"
        artifact.write_bytes(b"mp4")
        return {"artifact": artifact}

    monkeypatch.setattr(videogen, "run_seedance2_r2v", fake_run)

    rc = videogen.main(["seedance2-r2v", "--input", str(input_path), "--prompt", "move", "--out", str(tmp_path)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "seedance2-r2v"
    assert payload["input"] == str(input_path)
    assert payload["artifacts"] == [str(artifact)]


def test_seedance2_flf2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    artifact = tmp_path / "seedance-flf2v.mp4"
    Image.new("RGB", (8, 8), "green").save(first)
    Image.new("RGB", (8, 8), "blue").save(last)

    def fake_run(*, first_image, last_image, prompt, config, out_dir):
        assert first_image == first
        assert last_image == last
        artifact.write_bytes(b"mp4")
        return {"artifact": artifact}

    monkeypatch.setattr(videogen, "run_seedance2_flf2v", fake_run)

    rc = videogen.main(
        ["seedance2-flf2v", "--first", str(first), "--last", str(last), "--prompt", "transition", "--out", str(tmp_path)]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "seedance2-flf2v"
    assert payload["first"] == str(first)
    assert payload["last"] == str(last)
    assert payload["artifacts"] == [str(artifact)]


def test_i2v_missing_input_returns_json(tmp_path: Path, capsys: MagicMock) -> None:
    rc = videogen.main(["i2v", "--input", str(tmp_path / "missing.png"), "--prompt", "move"])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["kind"] == "video"
    assert payload["mode"] == "i2v"
    assert payload["error_type"] == "not_found"


def test_seedance2_missing_api_key_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.delenv("COMFY_ORG_API_KEY", raising=False)

    rc = videogen.main(["seedance2-t2v", "--prompt", "remote", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "seedance2-t2v"
    assert payload["error_type"] == "auth_required"


def test_seedance2_missing_dependency_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.setenv("COMFY_ORG_API_KEY", "test-key")

    def fail(*, prompt, config, out_dir):
        from comfy_agent_tools.videogen.seedance2 import Seedance2MissingDependencyError

        raise Seedance2MissingDependencyError("installed comfy-diffusion does not vendor Seedance 2.0")

    monkeypatch.setattr(videogen, "run_seedance2_t2v", fail)

    rc = videogen.main(["seedance2-t2v", "--prompt", "remote", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "missing_dependency"


def test_seedance2_remote_api_error_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.setenv("COMFY_ORG_API_KEY", "test-key")

    def fail(*, prompt, config, out_dir):
        from comfy_agent_tools.videogen.seedance2 import Seedance2Error

        raise Seedance2Error("Seedance 2.0 API request failed: bad gateway")

    monkeypatch.setattr(videogen, "run_seedance2_t2v", fail)

    rc = videogen.main(["seedance2-t2v", "--prompt", "remote", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "remote_api_error"


def test_ia2av_missing_audio_returns_json(tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)

    rc = videogen.main(
        [
            "ia2av",
            "--input",
            str(input_path),
            "--audio",
            str(tmp_path / "missing.wav"),
            "--prompt",
            "move",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["kind"] == "video"
    assert payload["mode"] == "ia2av"
    assert payload["error_type"] == "not_found"
    assert "input audio not found" in payload["error"]


def test_runtime_exception_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    def fail(*, prompt: str, config: object) -> dict[str, object]:
        raise RuntimeError("ComfyUI runtime not available: missing dependency")

    monkeypatch.setattr(videogen, "run_t2v", fail)

    rc = videogen.main(["t2v", "--prompt", "fail", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "runtime"


def test_audio_mux_failure_returns_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.setattr(videogen, "run_t2v", lambda *, prompt, config: _result())

    def fail_mux(frames: object, audio: object, path: object, fps: object) -> None:
        raise RuntimeError("audio mux failed")

    monkeypatch.setattr(videogen, "save_mp4_with_audio", fail_mux)

    rc = videogen.main(["t2v", "--prompt", "fail mux", "--out", str(tmp_path)])

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error_type"] == "audio_mux"


def test_t2v_suppresses_output_by_default(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    def noisy_run(*, prompt: str, config: object) -> dict[str, object]:
        print("progress should not leak")
        return _result()

    monkeypatch.setattr(videogen, "run_t2v", noisy_run)
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(["t2v", "--prompt", "quiet", "--out", str(tmp_path)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "progress should not leak" not in captured.out
    assert json.loads(captured.out)["ok"] is True


def test_save_mp4_with_stereo_audio(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (64, 36), "red"), Image.new("RGB", (64, 36), "blue")]
    sample_rate = 24000
    samples = sample_rate // 4
    tone = torch.sin(torch.linspace(0, math.tau * 440 * 0.25, samples))
    waveform = torch.stack([tone, tone * 0.5]).reshape(1, 2, samples)
    output = tmp_path / "stereo.mp4"

    save_mp4_with_audio(frames, {"waveform": waveform, "sample_rate": sample_rate}, output, 24)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_flf2v_wrapper_uses_latent_upscaler() -> None:
    source = Path("comfy_agent_tools/videogen/ltx23.py").read_text(encoding="utf-8")

    assert "ltxv_latent_upsample" in source
    assert "load_latent_upscale_model" in source
    assert "euler_cfg_pp" in source


def test_ia2av_wrapper_uses_upstream_ia2v_pipeline() -> None:
    source = Path("comfy_agent_tools/videogen/ltx23.py").read_text(encoding="utf-8")

    assert "from comfy_diffusion.pipelines.video.ltx.ltx23 import ia2v" in source
    assert "audio_path=audio" in source
    assert "audio_start_time=config.audio_start_time" in source
    assert "audio_duration=config.audio_duration" in source
