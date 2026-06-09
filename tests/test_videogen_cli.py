from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image
import pytest
import torch

from comfy_agent_tools.cli import videogen
from comfy_agent_tools.videogen.artifacts import save_mp4, save_mp4_with_audio
from comfy_agent_tools.loras import ExtraLora
from comfy_agent_tools.videogen import ltx23, wan22
from comfy_agent_tools.videogen.seedance2 import _disable_api_node_progress_display


def _frames() -> list[Image.Image]:
    return [Image.new("RGB", (32, 18), "red"), Image.new("RGB", (32, 18), "blue")]


def _result() -> dict[str, object]:
    return {"frames": _frames(), "audio": {"waveform": object(), "sample_rate": 44100}}


def test_seedance2_disables_api_node_progress_in_upload_helpers(monkeypatch: MagicMock) -> None:
    package = types.ModuleType("comfy_api_nodes")
    util = types.ModuleType("comfy_api_nodes.util")
    client = types.ModuleType("comfy_api_nodes.util.client")
    upload_helpers = types.ModuleType("comfy_api_nodes.util.upload_helpers")

    def noisy_progress(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("progress should be disabled")

    client._display_time_progress = noisy_progress
    upload_helpers._display_time_progress = noisy_progress
    util.client = client
    util.upload_helpers = upload_helpers
    package.util = util

    monkeypatch.setitem(sys.modules, "comfy_api_nodes", package)
    monkeypatch.setitem(sys.modules, "comfy_api_nodes.util", util)
    monkeypatch.setitem(sys.modules, "comfy_api_nodes.util.client", client)
    monkeypatch.setitem(sys.modules, "comfy_api_nodes.util.upload_helpers", upload_helpers)

    _disable_api_node_progress_display()

    client._display_time_progress()
    upload_helpers._display_time_progress()


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


def test_parser_motion_track_options(tmp_path: Path) -> None:
    args = videogen.build_parser().parse_args(
        [
            "motion-track",
            "--input",
            str(tmp_path / "input.png"),
            "--control-video",
            str(tmp_path / "tracks.mp4"),
            "--prompt",
            "follow the drawn tracks",
            "--attention-strength",
            "0.8",
            "--width",
            "512",
            "--height",
            "320",
            "--length",
            "49",
            "--fps",
            "24",
        ]
    )

    assert args.command == "motion-track"
    assert args.input == tmp_path / "input.png"
    assert args.control_video == tmp_path / "tracks.mp4"
    assert args.prompt == "follow the drawn tracks"
    assert args.attention_strength == 0.8
    assert args.width == 512
    assert args.height == 320
    assert args.length == 49
    assert args.fps == 24


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


def test_parser_wan22_defaults(tmp_path: Path) -> None:
    parser = videogen.build_parser()

    t2v = parser.parse_args(["wan22-t2v", "--prompt", "hello"])
    i2v = parser.parse_args(["wan22-i2v", "--input", str(tmp_path / "a.png"), "--prompt", "hello"])
    flf2v = parser.parse_args(
        [
            "wan22-flf2v",
            "--first",
            str(tmp_path / "a.png"),
            "--last",
            str(tmp_path / "b.png"),
            "--prompt",
            "hello",
        ]
    )
    s2v = parser.parse_args(
        [
            "wan22-s2v",
            "--input",
            str(tmp_path / "a.png"),
            "--audio",
            str(tmp_path / "a.wav"),
            "--prompt",
            "hello",
        ]
    )

    assert t2v.command == "wan22-t2v"
    assert t2v.prompt == "hello"
    assert t2v.width is None
    assert t2v.height is None
    assert t2v.length is None
    assert t2v.fps is None
    assert t2v.steps is None
    assert t2v.cfg is None
    assert t2v.extra_lora == []
    assert t2v.extra_lora_high == []
    assert t2v.extra_lora_low == []
    assert i2v.command == "wan22-i2v"
    assert i2v.width is None
    assert i2v.height is None
    assert i2v.length is None
    assert i2v.fps is None
    assert i2v.steps is None
    assert i2v.cfg is None
    assert i2v.extra_lora == []
    assert i2v.extra_lora_high == []
    assert i2v.extra_lora_low == []
    assert i2v.input == tmp_path / "a.png"
    assert flf2v.command == "wan22-flf2v"
    assert flf2v.extra_lora == []
    assert flf2v.extra_lora_high == []
    assert flf2v.extra_lora_low == []
    assert flf2v.first == tmp_path / "a.png"
    assert flf2v.last == tmp_path / "b.png"
    assert s2v.command == "wan22-s2v"
    assert s2v.input == tmp_path / "a.png"
    assert s2v.audio == tmp_path / "a.wav"
    assert s2v.length is None
    assert s2v.chunk_length is None
    assert s2v.steps is None
    assert s2v.cfg is None
    assert s2v.audio_start_time == 0.0
    assert s2v.audio_duration is None

    video_audio = parser.parse_args(
        [
            "wan22-video-audio",
            "--mode",
            "audio-driven",
            "--input-video",
            str(tmp_path / "input.mp4"),
            "--audio",
            str(tmp_path / "a.wav"),
        ]
    )
    lipsync = parser.parse_args(
        [
            "wan22-video-audio",
            "--mode",
            "lipsync",
            "--input-video",
            str(tmp_path / "input.mp4"),
            "--audio",
            str(tmp_path / "a.wav"),
            "--mask-video",
            str(tmp_path / "mask.mp4"),
        ]
    )

    assert video_audio.command == "wan22-video-audio"
    assert video_audio.mode == "audio-driven"
    assert video_audio.input_video == tmp_path / "input.mp4"
    assert video_audio.audio == tmp_path / "a.wav"
    assert video_audio.mask_video is None
    assert video_audio.prompt is None
    assert video_audio.chunk_length is None
    assert video_audio.chunk_overlap is None
    assert video_audio.steps is None
    assert video_audio.denoise is None
    assert lipsync.mode == "lipsync"
    assert lipsync.mask_video == tmp_path / "mask.mp4"


def test_parser_rejects_unknown_wan22_video_audio_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        videogen.build_parser().parse_args(
            [
                "wan22-video-audio",
                "--mode",
                "bad-mode",
                "--input-video",
                str(tmp_path / "input.mp4"),
                "--audio",
                str(tmp_path / "a.wav"),
            ]
        )


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
    assert Path(payload["manifests"][0]).is_file()


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


def test_motion_track_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    control_path = tmp_path / "tracks.mp4"
    Image.new("RGB", (8, 8), "green").save(input_path)
    control_path.write_bytes(b"fake video")
    seen: dict[str, object] = {}

    def fake_run_motion_track(*, image: Path, control_video: Path, prompt: str, config: object) -> dict[str, object]:
        seen["image"] = image
        seen["control_video"] = control_video
        seen["prompt"] = prompt
        seen["attention_strength"] = config.attention_strength
        return _result()

    monkeypatch.setattr(videogen, "run_motion_track", fake_run_motion_track)
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "motion-track",
            "--input",
            str(input_path),
            "--control-video",
            str(control_path),
            "--prompt",
            "follow tracks",
            "--attention-strength",
            "0.75",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["image"] == input_path
    assert seen["control_video"] == control_path
    assert seen["prompt"] == "follow tracks"
    assert seen["attention_strength"] == 0.75
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "motion-track"
    assert payload["input"] == str(input_path)
    assert payload["control_video"] == str(control_path)
    assert payload["attention_strength"] == 0.75
    assert payload["reference_downscale"] == 1.0
    assert payload["capability"] == "videogen.motion-track"
    assert payload["model_profile"] == "ltx23-motion-track"
    assert payload["architecture"] == "ltx23"
    assert payload["resolved_models"]["ic_lora"].endswith(
        "loras/ltx23/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
    )
    assert payload["ic_lora"] == payload["resolved_models"]["ic_lora"]
    assert Path(payload["artifacts"][0]).is_file()


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


def test_wan22_i2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)
    seen: dict[str, object] = {}

    def fake_run(*, image, prompt, config):
        seen["image"] = image
        seen["prompt"] = prompt
        seen["steps"] = config.steps
        seen["high_steps"] = config.high_steps
        seen["low_steps"] = config.low_steps
        seen["cfg"] = config.cfg
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_i2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(["wan22-i2v", "--input", str(input_path), "--prompt", "move", "--out", str(tmp_path)])

    assert rc == 0
    assert seen["image"] == input_path
    assert seen["prompt"] == "move"
    assert seen["steps"] == 20
    assert seen["high_steps"] == 10
    assert seen["low_steps"] == 10
    assert seen["cfg"] == 3.5
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "wan22-i2v"
    assert payload["input"] == str(input_path)
    assert payload["audio_muxed"] is False
    assert payload["capability"] == "videogen.wan22-i2v"
    assert payload["model_profile"] == "wan22-i2v"
    assert payload["architecture"] == "wan22"
    assert payload["fps"] == 16
    assert payload["steps"] == 20
    assert payload["high_steps"] == 10
    assert payload["low_steps"] == 10
    assert payload["resolved_models"]["unet_high"].endswith(
        "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    )
    assert Path(payload["artifacts"][0]).is_file()


def test_wan22_t2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    seen: dict[str, object] = {}

    def fake_run(*, prompt, config):
        seen["prompt"] = prompt
        seen["unet_high"] = config.unet_high
        seen["unet_low"] = config.unet_low
        seen["steps"] = config.steps
        seen["high_steps"] = config.high_steps
        seen["low_steps"] = config.low_steps
        seen["cfg"] = config.cfg
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_t2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(["wan22-t2v", "--prompt", "make a video", "--out", str(tmp_path)])

    assert rc == 0
    assert seen["prompt"] == "make a video"
    assert seen["unet_high"] == Path("diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors")
    assert seen["unet_low"] == Path("diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors")
    assert seen["steps"] == 20
    assert seen["high_steps"] == 10
    assert seen["low_steps"] == 10
    assert seen["cfg"] == 3.5
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "wan22-t2v"
    assert payload["audio_muxed"] is False
    assert payload["capability"] == "videogen.wan22-t2v"
    assert payload["model_profile"] == "wan22-t2v"
    assert payload["architecture"] == "wan22"
    assert payload["fps"] == 16
    assert payload["resolved_models"]["unet_high"].endswith(
        "diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    )
    assert Path(payload["artifacts"][0]).is_file()


def test_wan22_t2v_accepts_dasiwa_profile_defaults(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".comfy-agent-tools.json"
    config_path.write_text(
        json.dumps(
            {
                "models_dir": str(tmp_path / "models"),
                "defaults": {"videogen.wan22-t2v": "wan22-dasiwa-boundbite-t2v"},
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_run(*, prompt, config):
        seen["unet_high"] = config.unet_high
        seen["unet_low"] = config.unet_low
        seen["steps"] = config.steps
        seen["high_steps"] = config.high_steps
        seen["low_steps"] = config.low_steps
        seen["cfg"] = config.cfg
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_t2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(["wan22-t2v", "--prompt", "make a video", "--out", str(tmp_path)])

    assert rc == 0
    assert seen["unet_high"] == Path("diffusion_models/DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors")
    assert seen["unet_low"] == Path("diffusion_models/DasiwaWAN22I2V14BLightspeed_boundbiteLowV10.safetensors")
    assert seen["steps"] == 8
    assert seen["high_steps"] == 2
    assert seen["low_steps"] == 6
    assert seen["cfg"] == 1.0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_profile"] == "wan22-dasiwa-boundbite-t2v"
    assert payload["resolved_models"]["unet_high"].endswith(
        "diffusion_models/DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors"
    )


def test_wan22_i2v_accepts_targeted_extra_loras(
    monkeypatch: MagicMock,
    tmp_path: Path,
    capsys: MagicMock,
) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    both_lora = tmp_path / "loras" / "wan22" / "both.safetensors"
    high_lora = tmp_path / "loras" / "wan22" / "high.safetensors"
    low_lora = tmp_path / "loras" / "wan22" / "low.safetensors"
    both_lora.parent.mkdir(parents=True)
    for path in (both_lora, high_lora, low_lora):
        path.write_bytes(b"fake")
    Image.new("RGB", (8, 8), "green").save(input_path)
    seen: dict[str, object] = {}

    def fake_run(*, image, prompt, config):
        seen["extra_loras"] = config.extra_loras
        seen["extra_loras_high"] = config.extra_loras_high
        seen["extra_loras_low"] = config.extra_loras_low
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_i2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "wan22-i2v",
            "--input",
            str(input_path),
            "--prompt",
            "move",
            "--models-dir",
            str(tmp_path),
            "--extra-lora",
            "loras/wan22/both.safetensors:0.6:0.1",
            "--extra-lora-high",
            "loras/wan22/high.safetensors:0.7",
            "--extra-lora-low",
            "loras/wan22/low.safetensors:0.8",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["extra_loras"][0].path == Path("loras/wan22/both.safetensors")
    assert seen["extra_loras_high"][0].path == Path("loras/wan22/high.safetensors")
    assert seen["extra_loras_low"][0].path == Path("loras/wan22/low.safetensors")
    payload = json.loads(capsys.readouterr().out)
    assert payload["extra_loras"] == [
        {"path": str(both_lora), "strength_model": 0.6, "strength_clip": 0.1}
    ]
    assert payload["extra_loras_high"] == [
        {"path": str(high_lora), "strength_model": 0.7, "strength_clip": 0.0}
    ]
    assert payload["extra_loras_low"] == [
        {"path": str(low_lora), "strength_model": 0.8, "strength_clip": 0.0}
    ]


def test_wan22_i2v_accepts_custom_high_low_steps(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)
    seen: dict[str, object] = {}

    def fake_run(*, image, prompt, config):
        seen["steps"] = config.steps
        seen["high_steps"] = config.high_steps
        seen["low_steps"] = config.low_steps
        seen["split_step"] = config.split_step
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_i2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "wan22-i2v",
            "--input",
            str(input_path),
            "--prompt",
            "move",
            "--high-steps",
            "4",
            "--low-steps",
            "2",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["steps"] == 6
    assert seen["high_steps"] == 4
    assert seen["low_steps"] == 2
    assert seen["split_step"] == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == 6
    assert payload["high_steps"] == 4
    assert payload["low_steps"] == 2


def test_wan22_rejects_inconsistent_step_split(tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)

    rc = videogen.main(
        [
            "wan22-i2v",
            "--input",
            str(input_path),
            "--prompt",
            "move",
            "--steps",
            "10",
            "--high-steps",
            "4",
            "--low-steps",
            "4",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_type"] == "error"
    assert "--steps must equal --high-steps + --low-steps" in payload["error"]


def test_wan22_flf2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.new("RGB", (8, 8), "green").save(first)
    Image.new("RGB", (8, 8), "blue").save(last)
    seen: dict[str, object] = {}

    def fake_run(*, first_image, last_image, prompt, config):
        seen["first_image"] = first_image
        seen["last_image"] = last_image
        seen["cfg"] = config.cfg
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_flf2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(["wan22-flf2v", "--first", str(first), "--last", str(last), "--prompt", "transition", "--out", str(tmp_path)])

    assert rc == 0
    assert seen["first_image"] == first
    assert seen["last_image"] == last
    assert seen["cfg"] == 4.0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "wan22-flf2v"
    assert payload["first"] == str(first)
    assert payload["last"] == str(last)
    assert payload["audio_muxed"] is False
    assert payload["capability"] == "videogen.wan22-flf2v"
    assert payload["artifacts"]


def test_wan22_flf2v_accepts_low_only_extra_lora(
    monkeypatch: MagicMock,
    tmp_path: Path,
    capsys: MagicMock,
) -> None:
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    lora_path = tmp_path / "loras" / "wan22" / "low-detail.safetensors"
    lora_path.parent.mkdir(parents=True)
    lora_path.write_bytes(b"fake")
    Image.new("RGB", (8, 8), "green").save(first)
    Image.new("RGB", (8, 8), "blue").save(last)
    seen: dict[str, object] = {}

    def fake_run(*, first_image, last_image, prompt, config):
        seen["extra_loras"] = config.extra_loras
        seen["extra_loras_high"] = config.extra_loras_high
        seen["extra_loras_low"] = config.extra_loras_low
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_flf2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "wan22-flf2v",
            "--first",
            str(first),
            "--last",
            str(last),
            "--prompt",
            "transition",
            "--models-dir",
            str(tmp_path),
            "--extra-lora-low",
            "loras/wan22/low-detail.safetensors:0.65",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["extra_loras"] == []
    assert seen["extra_loras_high"] == []
    assert seen["extra_loras_low"][0].path == Path("loras/wan22/low-detail.safetensors")
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "wan22-flf2v"
    assert payload["extra_loras"] == []
    assert payload["extra_loras_high"] == []
    assert payload["extra_loras_low"] == [
        {"path": str(lora_path), "strength_model": 0.65, "strength_clip": 0.0}
    ]


def test_wan22_s2v_success_json(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    audio_path = tmp_path / "speech.wav"
    Image.new("RGB", (8, 8), "green").save(input_path)
    audio_path.write_bytes(b"fake audio")
    seen: dict[str, object] = {}

    def fake_run(*, image, audio, prompt, config):
        seen["image"] = image
        seen["audio"] = audio
        seen["prompt"] = prompt
        seen["length"] = config.length
        seen["chunk_length"] = config.chunk_length
        seen["steps"] = config.steps
        seen["cfg"] = config.cfg
        seen["sampler"] = config.sampler
        seen["scheduler"] = config.scheduler
        seen["shift"] = config.shift
        seen["audio_start_time"] = config.audio_start_time
        seen["audio_duration"] = config.audio_duration
        return _result()

    monkeypatch.setattr(videogen, "run_wan22_s2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "wan22-s2v",
            "--input",
            str(input_path),
            "--audio",
            str(audio_path),
            "--prompt",
            "speak",
            "--audio-start-time",
            "0.5",
            "--audio-duration",
            "3.0",
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["image"] == input_path
    assert seen["audio"] == audio_path
    assert seen["prompt"] == "speak"
    assert seen["length"] == 77
    assert seen["chunk_length"] == 77
    assert seen["steps"] == 20
    assert seen["cfg"] == 6.0
    assert seen["sampler"] == "uni_pc"
    assert seen["scheduler"] == "simple"
    assert seen["shift"] == 8.0
    assert seen["audio_start_time"] == 0.5
    assert seen["audio_duration"] == 3.0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "wan22-s2v"
    assert payload["input"] == str(input_path)
    assert payload["audio_input"] == str(audio_path)
    assert payload["audio_muxed"] is True
    assert payload["audio_conditioned"] is True
    assert payload["audio_duration_seconds"] == 3.0
    assert payload["capability"] == "videogen.wan22-s2v"
    assert payload["model_profile"] == "wan22-s2v"
    assert payload["architecture"] == "wan22"
    assert payload["resolved_models"]["unet"].endswith(
        "diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors"
    )
    assert payload["resolved_models"]["audio_encoder"].endswith(
        "audio_encoders/wav2vec2_large_english_fp16.safetensors"
    )
    assert Path(payload["artifacts"][0]).is_file()


def test_wan22_video_audio_audio_driven_success_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "input.mp4"
    audio_path = tmp_path / "speech.wav"
    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")
    seen: dict[str, object] = {}

    def fake_run(*, video, audio, mode, prompt, config, mask_video, mask_image):
        seen["video"] = video
        seen["audio"] = audio
        seen["mode"] = mode
        seen["prompt"] = prompt
        seen["unet"] = config.unet
        seen["chunk_length"] = config.chunk_length
        seen["chunk_overlap"] = config.chunk_overlap
        seen["steps"] = config.steps
        seen["denoise"] = config.denoise
        seen["mask_video"] = mask_video
        seen["mask_image"] = mask_image
        return {"frames": _frames(), "audio": {"waveform": object(), "sample_rate": 44100}, "chunks": [{"index": 0}]}

    monkeypatch.setattr(videogen, "run_wan22_video_audio", fake_run)
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "wan22-video-audio",
            "--mode",
            "audio-driven",
            "--input-video",
            str(video_path),
            "--audio",
            str(audio_path),
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["video"] == video_path
    assert seen["audio"] == audio_path
    assert seen["mode"] == "audio-driven"
    assert seen["prompt"] == "Audio-reactive motion, expressive movement, coherent video."
    assert seen["unet"] == Path("diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors")
    assert seen["chunk_length"] == 77
    assert seen["chunk_overlap"] == 4
    assert seen["steps"] == 4
    assert seen["denoise"] == 0.35
    assert seen["mask_video"] is None
    assert seen["mask_image"] is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "wan22-video-audio"
    assert payload["video_audio_mode"] == "audio-driven"
    assert payload["input_video"] == str(video_path)
    assert payload["audio_input"] == str(audio_path)
    assert payload["chunks"] == [{"index": 0}]
    assert payload["audio_muxed"] is True
    assert payload["capability"] == "videogen.wan22-video-audio"
    assert payload["model_profile"] == "wan22-dasiwa-littledemon-v2-video-audio"
    assert payload["resolved_models"]["unet"].endswith(
        "diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors"
    )
    assert Path(payload["artifacts"][0]).is_file()


def test_wan22_video_audio_lipsync_success_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "input.mp4"
    audio_path = tmp_path / "speech.wav"
    mask_path = tmp_path / "mask.png"
    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")
    Image.new("L", (8, 8), "white").save(mask_path)
    seen: dict[str, object] = {}

    def fake_run(*, video, audio, mode, prompt, config, mask_video, mask_image):
        seen["mode"] = mode
        seen["prompt"] = prompt
        seen["lipsync_steps"] = config.lipsync_steps
        seen["lipsync_denoise"] = config.lipsync_denoise
        seen["lipsync_second_steps"] = config.lipsync_second_steps
        seen["lipsync_second_denoise"] = config.lipsync_second_denoise
        seen["mask_video"] = mask_video
        seen["mask_image"] = mask_image
        return {"frames": _frames(), "audio": {"waveform": object(), "sample_rate": 44100}, "chunks": [{"index": 0}]}

    monkeypatch.setattr(videogen, "run_wan22_video_audio", fake_run)
    monkeypatch.setattr(videogen, "save_mp4_with_audio", lambda frames, audio, path, fps: Path(path).touch())

    rc = videogen.main(
        [
            "wan22-video-audio",
            "--mode",
            "lipsync",
            "--input-video",
            str(video_path),
            "--audio",
            str(audio_path),
            "--mask-image",
            str(mask_path),
            "--out",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert seen["mode"] == "lipsync"
    assert seen["prompt"] == "Speaking. Talking. Expressive lip movement."
    assert seen["lipsync_steps"] == 4
    assert seen["lipsync_denoise"] == 0.45
    assert seen["lipsync_second_steps"] == 2
    assert seen["lipsync_second_denoise"] == 0.25
    assert seen["mask_video"] is None
    assert seen["mask_image"] == mask_path
    payload = json.loads(capsys.readouterr().out)
    assert payload["video_audio_mode"] == "lipsync"
    assert payload["mask_input"] == str(mask_path)
    assert payload["mask_kind"] == "image"
    assert payload["lipsync_second_steps"] == 2
    assert payload["lipsync_second_denoise"] == 0.25


def test_wan22_video_audio_lipsync_requires_mask(tmp_path: Path, capsys: MagicMock) -> None:
    video_path = tmp_path / "input.mp4"
    audio_path = tmp_path / "speech.wav"
    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")

    rc = videogen.main(
        [
            "wan22-video-audio",
            "--mode",
            "lipsync",
            "--input-video",
            str(video_path),
            "--audio",
            str(audio_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "wan22-video-audio"
    assert payload["error_type"] == "error"
    assert "requires --mask-video or --mask-image" in payload["error"]


def test_wan22_video_audio_non_16_fps_returns_json(
    monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock
) -> None:
    video_path = tmp_path / "input.mp4"
    audio_path = tmp_path / "speech.wav"
    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")

    def fail(*_args, **_kwargs):
        raise ValueError("Wan 2.2 video+audio v1 supports only 16 fps input video")

    monkeypatch.setattr(videogen, "run_wan22_video_audio", fail)

    rc = videogen.main(
        [
            "wan22-video-audio",
            "--mode",
            "audio-driven",
            "--input-video",
            str(video_path),
            "--audio",
            str(audio_path),
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "wan22-video-audio"
    assert payload["error_type"] == "error"
    assert "16 fps" in payload["error"]


def test_wan22_dasiwa_profile_defaults(monkeypatch: MagicMock, tmp_path: Path, capsys: MagicMock) -> None:
    config_path = tmp_path / ".comfy-agent-tools.json"
    config_path.write_text(
        json.dumps(
            {
                "models_dir": str(tmp_path / "models"),
                "defaults": {
                    "videogen.wan22-i2v": "wan22-dasiwa-tastysin-i2v",
                },
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)
    seen: dict[str, object] = {}

    def fake_run(*, image, prompt, config):
        seen["steps"] = config.steps
        seen["high_steps"] = config.high_steps
        seen["low_steps"] = config.low_steps
        seen["cfg"] = config.cfg
        seen["unet_high"] = config.unet_high
        seen["unet_low"] = config.unet_low
        return {"frames": _frames()}

    monkeypatch.setattr(videogen, "run_wan22_i2v", fake_run)
    monkeypatch.setattr(videogen, "save_mp4", lambda frames, path, fps: Path(path).touch())

    rc = videogen.main(["wan22-i2v", "--input", str(input_path), "--prompt", "move", "--out", str(tmp_path)])

    assert rc == 0
    assert seen["steps"] == 4
    assert seen["high_steps"] == 2
    assert seen["low_steps"] == 2
    assert seen["cfg"] == 1.0
    assert seen["unet_high"] == Path("diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinHighV81.safetensors")
    assert seen["unet_low"] == Path("diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinLowV81.safetensors")
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_profile"] == "wan22-dasiwa-tastysin-i2v"


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


def test_wan22_s2v_missing_audio_returns_json(tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)

    rc = videogen.main(
        [
            "wan22-s2v",
            "--input",
            str(input_path),
            "--audio",
            str(tmp_path / "missing.wav"),
            "--prompt",
            "speak",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["kind"] == "video"
    assert payload["mode"] == "wan22-s2v"
    assert payload["error_type"] == "not_found"
    assert "input audio not found" in payload["error"]


def test_motion_track_missing_control_video_returns_json(tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8), "green").save(input_path)

    rc = videogen.main(
        [
            "motion-track",
            "--input",
            str(input_path),
            "--control-video",
            str(tmp_path / "missing.mp4"),
            "--prompt",
            "follow tracks",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "motion-track"
    assert payload["error_type"] == "not_found"
    assert "control video not found" in payload["error"]


def test_motion_track_missing_ic_helpers_returns_json(tmp_path: Path, capsys: MagicMock) -> None:
    input_path = tmp_path / "input.png"
    control_path = tmp_path / "tracks.mp4"
    Image.new("RGB", (8, 8), "green").save(input_path)
    control_path.write_bytes(b"fake video")
    original_require = ltx23._require_ic_lora_helpers
    ltx23._require_ic_lora_helpers = lambda: (_ for _ in ()).throw(
        ModuleNotFoundError(
            "installed comfy-diffusion does not expose IC-LoRA helpers; install comfy-diffusion v2.2.0 or newer "
            "for LTXICLoRALoaderModelOnly and LTXAddVideoICLoRAGuide wrappers"
        )
    )

    try:
        rc = videogen.main(
            [
                "motion-track",
                "--input",
                str(input_path),
                "--control-video",
                str(control_path),
                "--prompt",
                "follow tracks",
            ]
        )
    finally:
        ltx23._require_ic_lora_helpers = original_require

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["mode"] == "motion-track"
    assert payload["error_type"] == "missing_dependency"
    assert "LTXICLoRALoaderModelOnly" in payload["error"]


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


def test_save_mp4_without_audio(tmp_path: Path) -> None:
    frames = [Image.new("RGB", (64, 36), "red"), Image.new("RGB", (64, 36), "blue")]
    output = tmp_path / "silent.mp4"

    save_mp4(frames, output, 16)

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


def test_motion_track_wrapper_uses_comfy_diffusion_v220_ic_lora_helpers() -> None:
    source = Path("comfy_agent_tools/videogen/ltx23.py").read_text(encoding="utf-8")

    assert "apply_ic_lora_model_only" in source
    assert "ltx_add_video_ic_lora_guide" in source
    assert "latent_downscale_factor=latent_downscale_factor" in source


def test_wan22_i2v_wrapper_samples_high_noise_then_low_noise() -> None:
    source = Path("comfy_agent_tools/videogen/wan22.py").read_text(encoding="utf-8")

    high_call = source.index("latent = sample_advanced(\n        model_high")
    low_call = source.index("latent = sample_advanced(\n        model_low")
    assert high_call < low_call
    assert "start_at_step=0" in source
    assert "end_at_step=config.split_step" in source
    assert "start_at_step=config.split_step" in source
    assert "end_at_step=config.steps" in source


def test_wan22_t2v_wrapper_uses_empty_wan_latent_and_dual_sampling() -> None:
    source = Path("comfy_agent_tools/videogen/wan22.py").read_text(encoding="utf-8")

    t2v_start = source.index("def run_t2v")
    s2v_start = source.index("def run_s2v")
    t2v_source = source[t2v_start:s2v_start]
    high_call = t2v_source.index("latent = sample_advanced(\n        model_high")
    low_call = t2v_source.index("latent = sample_advanced(\n        model_low")
    assert "wan_image_to_video(" in t2v_source
    assert high_call < low_call
    assert "start_at_step=0" in t2v_source
    assert "end_at_step=config.split_step" in t2v_source
    assert "start_at_step=config.split_step" in t2v_source
    assert "end_at_step=config.steps" in t2v_source


def test_wan22_extra_loras_apply_to_selected_unets(monkeypatch: MagicMock, tmp_path: Path) -> None:
    for name in ("both.safetensors", "high.safetensors", "low.safetensors"):
        (tmp_path / name).write_bytes(b"fake")
    calls: dict[str, object] = {}

    def fake_both(models, clip, loras):
        calls["both"] = (models, clip, loras)
        return [f"{models[0]}+both", f"{models[1]}+both"], "clip+both"

    def fake_single(model, clip, loras):
        calls.setdefault("single", []).append((model, clip, loras))
        return f"{model}+single", f"{clip}+single"

    monkeypatch.setattr(wan22, "apply_extra_loras_to_models", fake_both)
    monkeypatch.setattr(wan22, "apply_extra_loras", fake_single)

    config = wan22.Wan22Config(
        models_dir=tmp_path,
        extra_loras=[ExtraLora(Path("both.safetensors"))],
        extra_loras_high=[ExtraLora(Path("high.safetensors"))],
        extra_loras_low=[ExtraLora(Path("low.safetensors"))],
    )

    model_high, model_low, clip = wan22._apply_extra_loras("high", "low", "clip", config)

    assert calls["both"][0] == ["high", "low"]
    assert [lora.path.name for lora in calls["both"][2]] == ["both.safetensors"]
    assert calls["single"][0][0] == "high+both"
    assert [lora.path.name for lora in calls["single"][0][2]] == ["high.safetensors"]
    assert calls["single"][1][0] == "low+both"
    assert [lora.path.name for lora in calls["single"][1][2]] == ["low.safetensors"]
    assert model_high == "high+both+single"
    assert model_low == "low+both+single"
    assert clip == "clip+both+single+single"


def test_wan22_s2v_wrapper_uses_audio_conditioning_and_extend() -> None:
    source = Path("comfy_agent_tools/videogen/wan22.py").read_text(encoding="utf-8")

    assert "wan_sound_image_to_video" in source
    assert "wan_sound_image_to_video_extend" in source
    assert "audio_encoder_encode" in source
    assert "latent_concat" in source
    assert "sample(" in source
