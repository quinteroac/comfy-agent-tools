from __future__ import annotations

from pathlib import Path
import sys
import types
from typing import Any

from PIL import Image
import pytest
import torch

from comfy_agent_tools.videogen import wan22


def _install_fake_comfy_diffusion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    frame_count: int,
    fps: float = 16.0,
) -> dict[str, Any]:
    calls: dict[str, Any] = {
        "load_audio": [],
        "samples": [],
        "wan_sound_image_to_video": [],
        "loaded_models": [],
        "model_shift": None,
        "zeroed_negative": False,
    }

    package = types.ModuleType("comfy_diffusion")
    audio = types.ModuleType("comfy_diffusion.audio")
    conditioning = types.ModuleType("comfy_diffusion.conditioning")
    image = types.ModuleType("comfy_diffusion.image")
    models = types.ModuleType("comfy_diffusion.models")
    runtime = types.ModuleType("comfy_diffusion.runtime")
    sampling = types.ModuleType("comfy_diffusion.sampling")
    vae = types.ModuleType("comfy_diffusion.vae")
    video = types.ModuleType("comfy_diffusion.video")

    def load_audio(path: str | Path, start_time: float = 0.0, duration: float | None = None) -> dict[str, Any]:
        payload = {
            "path": Path(path),
            "start_time": start_time,
            "duration": duration,
            "waveform": torch.zeros(1, 1, 160),
            "sample_rate": 16_000,
        }
        calls["load_audio"].append(payload)
        return payload

    audio.load_audio = load_audio
    audio.audio_encoder_encode = lambda _encoder, payload: {"encoded_audio": payload}

    conditioning.encode_prompt = lambda _clip, prompt, negative: (["positive", prompt], ["negative", negative])

    def conditioning_zero_out(_conditioning: Any) -> list[str]:
        calls["zeroed_negative"] = True
        return ["zero"]

    conditioning.conditioning_zero_out = conditioning_zero_out

    def wan_sound_image_to_video(
        positive: Any,
        negative: Any,
        _vae: Any,
        *,
        width: int,
        height: int,
        length: int,
        batch_size: int,
        audio_encoder_output: Any,
        ref_image: Any,
    ) -> tuple[Any, Any, dict[str, int]]:
        calls["wan_sound_image_to_video"].append(
            {
                "positive": positive,
                "negative": negative,
                "width": width,
                "height": height,
                "length": length,
                "batch_size": batch_size,
                "audio_encoder_output": audio_encoder_output,
                "ref_image": ref_image,
            }
        )
        return positive, negative, {"length": length}

    conditioning.wan_sound_image_to_video = wan_sound_image_to_video

    def image_to_tensor(frame: Image.Image) -> torch.Tensor:
        return torch.zeros(1, frame.height, frame.width, 3)

    image.image_to_tensor = image_to_tensor

    class FakeModelManager:
        def __init__(self, models_dir: str | Path) -> None:
            calls["models_dir"] = Path(models_dir)

        def load_unet(self, path: str | Path) -> str:
            calls["loaded_models"].append(Path(path))
            return "model"

        def load_clip(self, path: str | Path, *, clip_type: str) -> str:
            calls["loaded_models"].append(Path(path))
            calls["clip_type"] = clip_type
            return "clip"

        def load_audio_encoder(self, path: str | Path) -> str:
            calls["loaded_models"].append(Path(path))
            return "audio_encoder"

        def load_vae(self, path: str | Path) -> str:
            calls["loaded_models"].append(Path(path))
            return "vae"

    models.ModelManager = FakeModelManager

    def model_sampling_sd3(model: str, *, shift: float) -> str:
        calls["model_shift"] = shift
        return model

    models.model_sampling_sd3 = model_sampling_sd3
    runtime.check_runtime = lambda: {}

    def sample(
        _model: Any,
        _positive: Any,
        _negative: Any,
        latent: dict[str, int],
        *,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        seed: int,
        denoise: float,
    ) -> dict[str, int | float | str]:
        payload = {
            "length": int(latent["length"]),
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "seed": seed,
            "denoise": denoise,
            "sample_index": len(calls["samples"]),
        }
        calls["samples"].append(payload)
        return payload

    sampling.sample = sample
    vae.vae_encode_tensor = lambda _vae, tensor: {"length": int(tensor.shape[0])}

    def vae_decode_batch(_vae: Any, latent: dict[str, Any]) -> list[Image.Image]:
        sample_index = int(latent.get("sample_index", 0))
        color = (40 + sample_index * 30, 80, 120)
        return [Image.new("RGB", (6, 4), color) for _ in range(int(latent["length"]))]

    vae.vae_decode_batch = vae_decode_batch

    source_frames = torch.zeros(frame_count, 4, 6, 3)
    for index in range(frame_count):
        source_frames[index] = index / max(frame_count, 1)
    mask_frames = torch.ones(frame_count, 4, 6, 3)

    video.get_video_metadata = lambda _path: {"frame_count": frame_count, "fps": fps, "width": 6, "height": 4}

    def load_video(path: str | Path) -> torch.Tensor:
        return mask_frames if "mask" in Path(path).name else source_frames

    video.load_video = load_video

    for module_name, module in {
        "comfy_diffusion": package,
        "comfy_diffusion.audio": audio,
        "comfy_diffusion.conditioning": conditioning,
        "comfy_diffusion.image": image,
        "comfy_diffusion.models": models,
        "comfy_diffusion.runtime": runtime,
        "comfy_diffusion.sampling": sampling,
        "comfy_diffusion.vae": vae,
        "comfy_diffusion.video": video,
    }.items():
        monkeypatch.setitem(sys.modules, module_name, module)

    return calls


def test_wan22_video_audio_chunks_and_crops_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fake_comfy_diffusion(monkeypatch, frame_count=150)
    input_video = tmp_path / "input.mp4"
    input_audio = tmp_path / "speech.wav"
    input_video.write_bytes(b"video")
    input_audio.write_bytes(b"audio")

    result = wan22.run_video_audio(
        video=input_video,
        audio=input_audio,
        mode="audio-driven",
        prompt="move with the audio",
        config=wan22.Wan22VideoAudioConfig(models_dir=tmp_path / "models"),
    )

    assert len(result["frames"]) == 150
    assert result["chunks"] == [
        {"index": 0, "start_frame": 0, "frames": 77, "steps": 4, "denoise": 0.35},
        {"index": 1, "start_frame": 73, "frames": 77, "steps": 4, "denoise": 0.35},
    ]
    audio_windows = calls["load_audio"]
    assert audio_windows[0]["start_time"] == 0.0
    assert audio_windows[0]["duration"] == pytest.approx(150 / 16)
    assert [item["start_time"] for item in audio_windows[1:]] == pytest.approx([0.0, 73 / 16])
    assert [item["duration"] for item in audio_windows[1:]] == pytest.approx([77 / 16, 77 / 16])
    assert [item["denoise"] for item in calls["samples"]] == [0.35, 0.35]
    assert [item["steps"] for item in calls["samples"]] == [4, 4]
    assert len(calls["wan_sound_image_to_video"]) == 2
    assert calls["model_shift"] == 10.0
    assert calls["zeroed_negative"] is True


def test_wan22_video_audio_lipsync_two_passes_and_composites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_fake_comfy_diffusion(monkeypatch, frame_count=80)
    input_video = tmp_path / "input.mp4"
    input_audio = tmp_path / "speech.wav"
    mask_image = tmp_path / "mask.png"
    input_video.write_bytes(b"video")
    input_audio.write_bytes(b"audio")
    Image.new("L", (6, 4), "white").save(mask_image)

    result = wan22.run_video_audio(
        video=input_video,
        audio=input_audio,
        mode="lipsync",
        prompt="talk",
        config=wan22.Wan22VideoAudioConfig(models_dir=tmp_path / "models"),
        mask_image=mask_image,
    )

    assert len(result["frames"]) == 80
    assert len(result["chunks"]) == 2
    assert [item["steps"] for item in calls["samples"]] == [4, 2, 4, 2]
    assert [item["denoise"] for item in calls["samples"]] == [0.45, 0.25, 0.45, 0.25]
    assert len(calls["wan_sound_image_to_video"]) == 4


def test_wan22_video_audio_rejects_non_16_fps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_comfy_diffusion(monkeypatch, frame_count=10, fps=24.0)
    input_video = tmp_path / "input.mp4"
    input_audio = tmp_path / "speech.wav"
    input_video.write_bytes(b"video")
    input_audio.write_bytes(b"audio")

    with pytest.raises(ValueError, match="16 fps"):
        wan22.run_video_audio(
            video=input_video,
            audio=input_audio,
            mode="audio-driven",
            prompt="move",
            config=wan22.Wan22VideoAudioConfig(models_dir=tmp_path / "models"),
        )
