"""Audio artifact helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave


def make_audio_path(out_dir: str | Path, *, prefix: str) -> Path:
    """Return a collision-resistant WAV path under out_dir."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return output_dir / f"{prefix}-{stamp}-{suffix}.wav"


def audio_metadata(audio: dict[str, Any]) -> dict[str, Any]:
    """Return stable metadata for a generated audio payload."""
    waveform, sample_rate = _validated_audio(audio)
    samples = waveform.shape[1]
    return {
        "format": "wav",
        "sample_rate": sample_rate,
        "channels": int(waveform.shape[0]),
        "duration_seconds": samples / sample_rate if sample_rate else None,
    }


def save_wav(audio: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Save a generated audio payload as 16-bit PCM WAV and return metadata."""
    waveform, sample_rate = _validated_audio(audio)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcm = _float_to_int16_interleaved(waveform)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(int(waveform.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)

    return audio_metadata(audio)


def _validated_audio(audio: dict[str, Any]) -> tuple[Any, int]:
    if not audio or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("audio with waveform and sample_rate is required")

    waveform = _audio_to_numpy(audio["waveform"])
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be greater than 0")
    if waveform.shape[1] <= 0:
        raise ValueError("audio waveform must contain at least one sample")
    return waveform, sample_rate


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


def _float_to_int16_interleaved(waveform: Any) -> bytes:
    import numpy as np

    pcm = (waveform.T * 32767.0).round().astype(np.int16)
    return pcm.tobytes()

