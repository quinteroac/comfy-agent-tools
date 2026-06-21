"""Video artifact helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any
from uuid import uuid4


def make_video_path(out_dir: str | Path, *, prefix: str) -> Path:
    """Return a collision-resistant MP4 path under out_dir."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return output_dir / f"{prefix}-{stamp}-{suffix}.mp4"


def frame_metadata(frames: list[Any], fps: int) -> dict[str, Any]:
    """Return simple frame metadata for decoded PIL frames."""
    if not frames:
        raise ValueError("no frames were produced")
    first = frames[0]
    width, height = getattr(first, "size", (None, None))
    return {
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "frames": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps if fps else None,
    }


def save_mp4(frames: list[Any], path: str | Path, fps: int) -> None:
    """Save frames into an MP4 file using PyAV."""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if not frames:
        raise ValueError("no frames were produced")

    import av
    import numpy as np

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_width, first_height = frames[0].size

    with av.open(str(output_path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = int(first_width)
        video_stream.height = int(first_height)
        video_stream.pix_fmt = "yuv420p"

        for image in frames:
            if image.size != (first_width, first_height):
                raise ValueError("all frames must have identical width and height")
            array = np.asarray(image.convert("RGB"))
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in video_stream.encode(video_frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)


def save_mp4_with_audio(frames: list[Any], audio: dict[str, Any], path: str | Path, fps: int) -> None:
    """Save frames and generated audio into one MP4 file using PyAV."""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if not frames:
        raise ValueError("no frames were produced")
    if not audio or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio with waveform and sample_rate is required")

    import av
    import numpy as np

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_width, first_height = frames[0].size
    waveform = _audio_to_numpy(audio["waveform"])
    sample_rate = int(audio["sample_rate"])

    with av.open(str(output_path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = int(first_width)
        video_stream.height = int(first_height)
        video_stream.pix_fmt = "yuv420p"

        audio_stream = container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = "stereo" if waveform.shape[0] == 2 else "mono"

        for image in frames:
            if image.size != (first_width, first_height):
                raise ValueError("all frames must have identical width and height")
            array = np.asarray(image.convert("RGB"))
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in video_stream.encode(video_frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)

        for audio_frame in _audio_frames(waveform, sample_rate, audio_stream.layout.name):
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

        for packet in audio_stream.encode():
            container.mux(packet)


def save_mp4_with_source_audio(frames: list[Any], source_video: str | Path, path: str | Path, fps: int) -> bool:
    """Save frames into an MP4 and mux audio from a source video when present."""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    if not frames:
        raise ValueError("no frames were produced")

    import av
    import numpy as np

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_width, first_height = frames[0].size
    audio_muxed = False

    with av.open(str(source_video)) as source, av.open(str(output_path), mode="w") as container:
        source_audio = next((stream for stream in source.streams.audio), None)
        audio_stream = None
        resampler = None
        sample_rate = 44100
        layout = "stereo"
        if source_audio is not None:
            sample_rate = int(source_audio.rate or sample_rate)
            layout = source_audio.layout.name if source_audio.layout is not None else layout
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.layout = layout
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)

        video_stream = container.add_stream("libx264", rate=fps)
        video_stream.width = int(first_width)
        video_stream.height = int(first_height)
        video_stream.pix_fmt = "yuv420p"

        for image in frames:
            if image.size != (first_width, first_height):
                raise ValueError("all frames must have identical width and height")
            array = np.asarray(image.convert("RGB"))
            video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in video_stream.encode(video_frame):
                container.mux(packet)

        for packet in video_stream.encode():
            container.mux(packet)

        if source_audio is None or audio_stream is None or resampler is None:
            return False

        audio_pts = 0
        for audio_frame in source.decode(source_audio):
            for resampled_frame in _resampled_audio_frames(resampler, audio_frame):
                resampled_frame.pts = audio_pts
                resampled_frame.time_base = Fraction(1, sample_rate)
                audio_pts += resampled_frame.samples
                for packet in audio_stream.encode(resampled_frame):
                    container.mux(packet)
                audio_muxed = True

        for resampled_frame in _resampled_audio_frames(resampler, None):
            resampled_frame.pts = audio_pts
            resampled_frame.time_base = Fraction(1, sample_rate)
            audio_pts += resampled_frame.samples
            for packet in audio_stream.encode(resampled_frame):
                container.mux(packet)
            audio_muxed = True

        for packet in audio_stream.encode():
            container.mux(packet)

    return audio_muxed


def _audio_to_numpy(waveform: Any) -> Any:
    import numpy as np

    data = waveform
    if hasattr(data, "detach"):
        data = data.detach()
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        array = data.numpy()
    else:
        array = np.asarray(data)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"expected audio waveform shape [B,C,N], [C,N], or [N], got {array.shape}")
    if array.shape[0] > 2:
        array = array[:2]
    return np.clip(array, -1.0, 1.0)


def _audio_frames(waveform: Any, sample_rate: int, layout: str) -> list[Any]:
    import av

    chunk_size = 1024
    frames = []
    for start in range(0, waveform.shape[1], chunk_size):
        chunk = waveform[:, start : start + chunk_size]
        frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
        frame.sample_rate = sample_rate
        frames.append(frame)
    return frames


def _resampled_audio_frames(resampler: Any, frame: Any) -> list[Any]:
    resampled = resampler.resample(frame)
    if resampled is None:
        return []
    if isinstance(resampled, list):
        return resampled
    return [resampled]
