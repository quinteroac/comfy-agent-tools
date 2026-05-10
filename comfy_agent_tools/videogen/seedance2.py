"""Seedance 2.0 API node wrappers for comfy-videogen."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
from typing import Any

from comfy_agent_tools.videogen.artifacts import make_video_path

SEEDANCE2_MODEL = "Seedance 2.0"
SEEDANCE2_MODEL_ID = "dreamina-seedance-2-0-260128"
SEEDANCE2_PROVIDER = "comfy-api"

DEFAULT_SEEDANCE2_RESOLUTION = "480p"
DEFAULT_SEEDANCE2_RATIO = "16:9"
DEFAULT_SEEDANCE2_DURATION = 7
DEFAULT_SEEDANCE2_GENERATE_AUDIO = True
DEFAULT_SEEDANCE2_WATERMARK = False
DEFAULT_SEEDANCE2_SEED = 0

SEEDANCE2_RESOLUTIONS = ("480p", "720p", "1080p")
SEEDANCE2_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive")


class Seedance2Error(RuntimeError):
    """Base Seedance 2.0 error with stable error_type."""

    error_type = "remote_api_error"


class Seedance2AuthRequiredError(Seedance2Error):
    error_type = "auth_required"


class Seedance2MissingDependencyError(Seedance2Error):
    error_type = "missing_dependency"


class Seedance2UnsupportedModelError(Seedance2Error):
    error_type = "unsupported_remote_model"


@dataclass(frozen=True)
class Seedance2Config:
    """Runtime configuration for remote Seedance 2.0 API commands."""

    model: str = SEEDANCE2_MODEL
    resolution: str = DEFAULT_SEEDANCE2_RESOLUTION
    ratio: str = DEFAULT_SEEDANCE2_RATIO
    duration: int = DEFAULT_SEEDANCE2_DURATION
    generate_audio: bool = DEFAULT_SEEDANCE2_GENERATE_AUDIO
    watermark: bool = DEFAULT_SEEDANCE2_WATERMARK
    seed: int = DEFAULT_SEEDANCE2_SEED
    api_key: str | None = None

    def validate(self) -> None:
        """Validate user-facing Seedance options."""
        if self.model != SEEDANCE2_MODEL:
            raise Seedance2UnsupportedModelError(f"unsupported Seedance model: {self.model}")
        if self.resolution not in SEEDANCE2_RESOLUTIONS:
            raise ValueError(f"unsupported Seedance 2.0 resolution: {self.resolution}")
        if self.ratio not in SEEDANCE2_RATIOS:
            raise ValueError(f"unsupported Seedance 2.0 ratio: {self.ratio}")
        if self.duration < 4 or self.duration > 15:
            raise ValueError("Seedance 2.0 duration must be between 4 and 15 seconds")
        if not (0 <= self.seed <= 2147483647):
            raise ValueError("Seedance 2.0 seed must be between 0 and 2147483647")
        if not (self.api_key or os.environ.get("COMFY_ORG_API_KEY", "")).strip():
            raise Seedance2AuthRequiredError("COMFY_ORG_API_KEY is required for Seedance 2.0 API generation")

    def model_payload(self, prompt: str) -> dict[str, Any]:
        """Return the dynamic combo payload expected by the Seedance 2.0 nodes."""
        return {
            "model": self.model,
            "prompt": prompt,
            "resolution": self.resolution,
            "ratio": self.ratio,
            "duration": self.duration,
            "generate_audio": self.generate_audio,
        }

    @property
    def effective_api_key(self) -> str:
        """Return API key from config or environment."""
        return (self.api_key or os.environ.get("COMFY_ORG_API_KEY", "")).strip()


def run_t2v(*, prompt: str, config: Seedance2Config, out_dir: Path) -> dict[str, Any]:
    """Run ByteDance Seedance 2.0 text-to-video through Comfy API nodes."""
    _validate_prompt(prompt)
    config.validate()
    return _run_async(_run_t2v(prompt=prompt, config=config, out_dir=out_dir))


def run_r2v(*, image: Path, prompt: str, config: Seedance2Config, out_dir: Path) -> dict[str, Any]:
    """Run ByteDance Seedance 2.0 reference image-to-video through Comfy API nodes."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    _validate_prompt(prompt)
    config.validate()
    return _run_async(_run_r2v(image=image, prompt=prompt, config=config, out_dir=out_dir))


def run_flf2v(*, first_image: Path, last_image: Path, prompt: str, config: Seedance2Config, out_dir: Path) -> dict[str, Any]:
    """Run ByteDance Seedance 2.0 first/last-frame-to-video through Comfy API nodes."""
    if not first_image.is_file():
        raise FileNotFoundError(f"first image not found: {first_image}")
    if not last_image.is_file():
        raise FileNotFoundError(f"last image not found: {last_image}")
    _validate_prompt(prompt)
    config.validate()
    return _run_async(_run_flf2v(first_image=first_image, last_image=last_image, prompt=prompt, config=config, out_dir=out_dir))


def _validate_prompt(prompt: str) -> None:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")


def _run_async(coro: Any) -> dict[str, Any]:
    try:
        return asyncio.run(coro)
    except Seedance2Error:
        raise
    except Exception as exc:
        message = str(exc).strip() or type(exc).__name__
        raise Seedance2Error(f"Seedance 2.0 API request failed: {message}") from exc


async def _run_t2v(*, prompt: str, config: Seedance2Config, out_dir: Path) -> dict[str, Any]:
    nodes = _seedance2_nodes(config)
    output = await nodes["t2v"].execute(config.model_payload(prompt), config.seed, config.watermark)
    artifact = _save_node_video(output, out_dir, prefix="comfy-videogen-seedance2-t2v")
    return {"artifact": artifact}


async def _run_r2v(*, image: Path, prompt: str, config: Seedance2Config, out_dir: Path) -> dict[str, Any]:
    nodes = _seedance2_nodes(config)
    image_tensor = _load_image_tensor(image)
    model = config.model_payload(prompt)
    model["reference_images"] = {"image_1": image_tensor}
    output = await nodes["r2v"].execute(model, config.seed, config.watermark)
    artifact = _save_node_video(output, out_dir, prefix="comfy-videogen-seedance2-r2v")
    return {"artifact": artifact}


async def _run_flf2v(*, first_image: Path, last_image: Path, prompt: str, config: Seedance2Config, out_dir: Path) -> dict[str, Any]:
    nodes = _seedance2_nodes(config)
    output = await nodes["flf2v"].execute(
        config.model_payload(prompt),
        config.seed,
        config.watermark,
        first_frame=_load_image_tensor(first_image),
        last_frame=_load_image_tensor(last_image),
    )
    artifact = _save_node_video(output, out_dir, prefix="comfy-videogen-seedance2-flf2v")
    return {"artifact": artifact}


def _seedance2_nodes(config: Seedance2Config) -> dict[str, Any]:
    try:
        from comfy_diffusion._runtime import ensure_comfyui_on_path

        comfyui_root = ensure_comfyui_on_path()
        _prioritize_comfyui_root(comfyui_root)
        _disable_api_node_progress_display()
        from comfy_api_nodes import nodes_bytedance
    except Exception as exc:
        raise Seedance2MissingDependencyError(
            "Seedance 2.0 API nodes are not available. Update comfy-diffusion to a release that vendors ComfyUI Seedance 2.0 nodes."
        ) from exc

    required = {
        "t2v": "ByteDance2TextToVideoNode",
        "r2v": "ByteDance2ReferenceNode",
        "flf2v": "ByteDance2FirstLastFrameNode",
    }
    missing = [name for name in required.values() if not hasattr(nodes_bytedance, name)]
    models = getattr(nodes_bytedance, "SEEDANCE_MODELS", {})
    if missing or models.get(SEEDANCE2_MODEL) != SEEDANCE2_MODEL_ID:
        raise Seedance2MissingDependencyError(
            "installed comfy-diffusion does not vendor the ByteDance Seedance 2.0 API nodes"
        )

    nodes = {key: getattr(nodes_bytedance, name) for key, name in required.items()}
    for node in nodes.values():
        _configure_node_auth(node, config.effective_api_key)
    return nodes


def _prioritize_comfyui_root(comfyui_root: Path) -> None:
    """Put ComfyUI first so its top-level packages win import resolution."""
    root = str(comfyui_root)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    _install_comfyui_utils_package(comfyui_root)


def _install_comfyui_utils_package(comfyui_root: Path) -> None:
    """Ensure ComfyUI's ``utils`` package is not shadowed by ``comfy/utils.py``."""
    package_dir = comfyui_root / "utils"
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        return
    spec = importlib.util.spec_from_file_location(
        "utils",
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = module
    spec.loader.exec_module(module)


def _disable_api_node_progress_display() -> None:
    """Disable ComfyUI server UI progress calls for headless CLI execution."""
    def no_op_progress(*args: Any, **kwargs: Any) -> None:
        return None

    for module_name in ("client", "upload_helpers"):
        try:
            module = __import__(
                f"comfy_api_nodes.util.{module_name}",
                fromlist=["_display_time_progress"],
            )
            module._display_time_progress = no_op_progress
        except Exception:
            pass


def _configure_node_auth(node: Any, api_key: str) -> None:
    hidden = getattr(node, "hidden", None)
    if hidden is None:
        hidden = SimpleNamespace()
        setattr(node, "hidden", hidden)
    setattr(hidden, "auth_token_comfy_org", "")
    setattr(hidden, "api_key_comfy_org", api_key)
    setattr(hidden, "unique_id", "comfy-agent-tools-seedance2")


def _load_image_tensor(path: Path) -> Any:
    try:
        from comfy_diffusion.image import load_image

        image, _mask = load_image(path)
        return image
    except Exception as exc:
        raise RuntimeError(f"failed to load image for Seedance 2.0: {path}: {exc}") from exc


def _save_node_video(output: Any, out_dir: Path, *, prefix: str) -> Path:
    video = _extract_video(output)
    path = make_video_path(out_dir, prefix=prefix)
    if hasattr(video, "save_to"):
        video.save_to(str(path))
        return path
    source = video.get_stream_source() if hasattr(video, "get_stream_source") else video
    if hasattr(source, "getvalue"):
        path.write_bytes(source.getvalue())
        return path
    if isinstance(source, (str, Path)):
        shutil.copyfile(source, path)
        return path
    raise RuntimeError("Seedance 2.0 node did not return a saveable video output")


def _extract_video(output: Any) -> Any:
    result = getattr(output, "result", output)
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("Seedance 2.0 node returned an empty result")
        return result[0]
    return result
