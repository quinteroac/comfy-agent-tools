from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from comfy_agent_tools.videogen import seedvr2_upscale
from comfy_agent_tools.videogen.seedvr2_upscale import SeedVR2UpscaleConfig, run_seedvr2_upscale_video


def test_run_seedvr2_upscale_builds_upstream_command(monkeypatch: MagicMock, tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "inference_cli.py").write_text("# test\n", encoding="utf-8")
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"mp4")
    output_video = tmp_path / "output.mp4"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        output_video.write_bytes(b"upscaled")
        return type("Completed", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(seedvr2_upscale.subprocess, "run", fake_run)

    result = run_seedvr2_upscale_video(
        video=input_video,
        output=output_video,
        config=SeedVR2UpscaleConfig(
            resolution="4k",
            max_edge=4096,
            model="seedvr2_ema_3b_fp16.safetensors",
            model_dir=tmp_path / "models" / "SEEDVR2",
            batch_size=9,
            chunk_size=330,
            temporal_overlap=3,
            cuda_device="0,1",
            blocks_to_swap=12,
            video_backend="ffmpeg",
            upstream_dir=upstream,
        ),
        verbose=False,
    )

    command = calls[0]
    assert str(upstream / "inference_cli.py") in command
    assert str(input_video.resolve()) in command
    assert command[command.index("--output") + 1] == str(output_video)
    assert command[command.index("--resolution") + 1] == "2160"
    assert command[command.index("--max_resolution") + 1] == "4096"
    assert command[command.index("--dit_model") + 1] == "seedvr2_ema_3b_fp16.safetensors"
    assert command[command.index("--batch_size") + 1] == "9"
    assert command[command.index("--chunk_size") + 1] == "330"
    assert command[command.index("--temporal_overlap") + 1] == "3"
    assert command[command.index("--model_dir") + 1] == str(tmp_path / "models" / "SEEDVR2")
    assert command[command.index("--cuda_device") + 1] == "0,1"
    assert command[command.index("--blocks_to_swap") + 1] == "12"
    assert command[command.index("--dit_offload_device") + 1] == "cpu"
    assert "--uniform_batch_size" in command
    assert result["target"] == "4k"
    assert result["short_edge"] == 2160
    assert result["max_edge"] == 4096
    assert result["upstream_commit"] == seedvr2_upscale.SEEDVR2_COMMIT
