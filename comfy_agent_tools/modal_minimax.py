"""Model preparation helpers for the MiniMax H3 Modal app.

This module deliberately has no dependency on the Modal Python SDK.  The local
entrypoint uses the ``modal`` CLI so the base comfy-agent-tools installation
does not need to install Modal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Sequence

from comfy_agent_tools.profiles import load_config
from comfy_agent_tools.videogen.minimax import DEFAULT_MINIMAX_TURBO_LORA


DEFAULT_MODAL_VOLUME = "comfy-minimax-h3-models"
DEFAULT_MODAL_GPU = "RTX-PRO-6000"
MODEL_MOUNT = Path("/mnt/models/comfyui")
MODAL_TURBO_NODE = Path("/root/.cache/comfy-diffusion/custom_nodes/ComfyUI-MiniMax-H3-Turbo")

MODEL_PATHS: dict[str, Path] = {
    "ref2va_unet": Path("diffusion_models/minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
    "fl2va_unet": Path("diffusion_models/minimax/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    "text_encoder": Path("text_encoders/minimax/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
    "audio_vae": Path("vae/minimax/minimax_h3_audio_vae_fp32.safetensors"),
    "video_vae": Path("vae/minimax/minimax_h3_video_vae_fp16.safetensors"),
    "turbo_lora": DEFAULT_MINIMAX_TURBO_LORA,
}

CAPABILITIES = {
    "t2v": "videogen.minimax-h3-t2v",
    "i2v": "videogen.minimax-h3-i2v",
    "r2v": "videogen.minimax-h3-r2v",
}


class ModalPreparationError(RuntimeError):
    """Raised when the local Modal/model preparation workflow cannot proceed."""

    error_type = "modal_preparation_error"


def required_model_paths(mode: str) -> tuple[Path, ...]:
    """Return the H3 model paths needed by a generation mode."""
    try:
        capability = CAPABILITIES[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported MiniMax Modal mode: {mode}") from exc
    keys = ("ref2va_unet",) if capability.endswith("r2v") else ("fl2va_unet",)
    return tuple(MODEL_PATHS[key] for key in (*keys, "text_encoder", "audio_vae", "video_vae", "turbo_lora"))


def configured_models_dir() -> Path:
    """Read the configured models directory without creating local config."""
    config, _source = load_config()
    value = os.environ.get("COMFY_MODELS_DIR") or config.get("models_dir")
    if not value:
        raise ModalPreparationError("models_dir is not configured; run comfy-models set-models-dir <models_dir>")
    return Path(value).expanduser()


def _modal_command() -> str:
    command = shutil.which("modal")
    if not command:
        raise ModalPreparationError("Modal CLI is required; install it with 'uv tool install modal'")
    return command


def _run(command: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> str:
    try:
        result = runner(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ModalPreparationError(f"command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise ModalPreparationError(f"command failed ({' '.join(command)}): {detail}") from exc
    return result.stdout or ""


def validate_modal_auth() -> None:
    """Validate that Modal has credentials without printing secrets."""
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return
    modal_file = Path.home() / ".modal.toml"
    if not modal_file.exists():
        raise ModalPreparationError(
            "Modal authentication is missing; run 'modal token new' or set "
            "MODAL_TOKEN_ID and MODAL_TOKEN_SECRET"
        )


def _download_capability(mode: str, *, runner: Callable[..., subprocess.CompletedProcess[str]]) -> None:
    capability = CAPABILITIES[mode]
    if shutil.which("comfy-models"):
        command = ["comfy-models", "download", capability, "--yes"]
    elif shutil.which("uv"):
        command = ["uv", "run", "comfy-models", "download", capability, "--yes"]
    else:
        raise ModalPreparationError("comfy-models is required to download missing MiniMax files")
    _run(command, runner=runner)


def _volume_listing(
    volume: str,
    paths: Sequence[Path],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
    modal = _modal_command()
    listings: list[str] = []
    locations = {Path("."), *(path.parent for path in paths)}
    for location in locations:
        command = [modal, "volume", "ls", volume]
        if location != Path("."):
            command.append(location.as_posix())
        try:
            listings.append(_run([*command, "--json"], runner=runner))
        except ModalPreparationError:
            try:
                listings.append(_run(command, runner=runner))
            except ModalPreparationError as exc:
                # A newly added model family (for example loras/minimax) may
                # not have a directory in the Volume yet; treat it as empty.
                if "no such file or directory" not in str(exc).lower():
                    raise
    return "\n".join(listings)


def _ensure_volume(volume: str, *, runner: Callable[..., subprocess.CompletedProcess[str]]) -> None:
    """Create a named Volume, tolerating the already-exists response."""
    modal = _modal_command()
    try:
        _run([modal, "volume", "create", volume], runner=runner)
    except ModalPreparationError as exc:
        if "already exists" not in str(exc).lower():
            raise


def _listing_contains(listing: str, relative_path: Path) -> bool:
    needle = relative_path.as_posix()
    try:
        payload = json.loads(listing)
    except json.JSONDecodeError:
        return needle in listing
    return needle in json.dumps(payload, sort_keys=True)


def prepare_models(
    mode: str,
    *,
    models_dir: Path | None = None,
    volume: str = DEFAULT_MODAL_VOLUME,
    force_upload: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, object]:
    """Download missing local H3 files and incrementally upload them to Modal."""
    required = required_model_paths(mode)
    local_root = models_dir or configured_models_dir()
    missing_local = [path for path in required if not (local_root / path).is_file()]
    if missing_local:
        _download_capability(mode, runner=runner)
    missing_local = [path for path in required if not (local_root / path).is_file()]
    if missing_local:
        raise ModalPreparationError(
            "MiniMax model files are still missing locally: " + ", ".join(str(path) for path in missing_local)
        )

    modal = _modal_command()
    _ensure_volume(volume, runner=runner)
    listing = _volume_listing(volume, required, runner=runner)
    uploaded: list[str] = []
    skipped: list[str] = []
    for relative_path in required:
        if not force_upload and _listing_contains(listing, relative_path):
            skipped.append(relative_path.as_posix())
            continue
        _run(
            [modal, "volume", "put", volume, str(local_root / relative_path), f"/{relative_path.as_posix()}"],
            runner=runner,
        )
        uploaded.append(relative_path.as_posix())
    return {
        "volume": volume,
        "models_dir": str(local_root),
        "mode": mode,
        "required": [path.as_posix() for path in required],
        "uploaded": uploaded,
        "skipped": skipped,
    }
