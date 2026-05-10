"""Grok Imagine API node wrappers for comfy-imagegen."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

from PIL import Image

from comfy_agent_tools.imagegen.artifacts import tensor_to_pil

GROK_PROVIDER = "comfy-api"
DEFAULT_GROK_MODEL = "grok-imagine-image"
DEFAULT_GROK_RESOLUTION = "1K"
DEFAULT_GROK_ASPECT_RATIO = "1:1"
DEFAULT_GROK_EDIT_ASPECT_RATIO = "auto"
DEFAULT_GROK_NUMBER_OF_IMAGES = 1
DEFAULT_GROK_SEED = 0

GROK_MODELS = ("grok-imagine-image-pro", "grok-imagine-image", "grok-imagine-image-beta")
GROK_RESOLUTIONS = ("1K", "2K")
GROK_ASPECT_RATIOS = (
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "9:16",
    "16:9",
    "9:19.5",
    "19.5:9",
    "9:20",
    "20:9",
    "1:2",
    "2:1",
)
GROK_EDIT_ASPECT_RATIOS = ("auto", *GROK_ASPECT_RATIOS)


class GrokImagineError(RuntimeError):
    """Base Grok Imagine error with stable error_type."""

    error_type = "remote_api_error"


class GrokImagineAuthRequiredError(GrokImagineError):
    error_type = "auth_required"


class GrokImagineMissingDependencyError(GrokImagineError):
    error_type = "missing_dependency"


class GrokImagineUnsupportedModelError(GrokImagineError):
    error_type = "unsupported_remote_model"


@dataclass(frozen=True)
class GrokImagineConfig:
    """Runtime configuration for remote Grok Imagine API commands."""

    model: str = DEFAULT_GROK_MODEL
    resolution: str = DEFAULT_GROK_RESOLUTION
    aspect_ratio: str = DEFAULT_GROK_ASPECT_RATIO
    number_of_images: int = DEFAULT_GROK_NUMBER_OF_IMAGES
    seed: int = DEFAULT_GROK_SEED
    api_key: str | None = None

    def validate(self, *, edit: bool = False) -> None:
        """Validate user-facing Grok Imagine options."""
        if self.model not in GROK_MODELS:
            raise GrokImagineUnsupportedModelError(f"unsupported Grok Imagine model: {self.model}")
        if self.resolution not in GROK_RESOLUTIONS:
            raise ValueError(f"unsupported Grok Imagine resolution: {self.resolution}")
        valid_ratios = GROK_EDIT_ASPECT_RATIOS if edit else GROK_ASPECT_RATIOS
        if self.aspect_ratio not in valid_ratios:
            raise ValueError(f"unsupported Grok Imagine aspect ratio: {self.aspect_ratio}")
        if self.number_of_images < 1 or self.number_of_images > 10:
            raise ValueError("Grok Imagine number_of_images must be between 1 and 10")
        if not (0 <= self.seed <= 2147483647):
            raise ValueError("Grok Imagine seed must be between 0 and 2147483647")
        if not (self.api_key or os.environ.get("COMFY_ORG_API_KEY", "")).strip():
            raise GrokImagineAuthRequiredError("COMFY_ORG_API_KEY is required for Grok Imagine API generation")

    @property
    def effective_api_key(self) -> str:
        """Return API key from config or environment."""
        return (self.api_key or os.environ.get("COMFY_ORG_API_KEY", "")).strip()


def run_generate(*, prompt: str, config: GrokImagineConfig) -> list[Image.Image]:
    """Run Grok Imagine text-to-image through Comfy API nodes."""
    _validate_prompt(prompt)
    config.validate(edit=False)
    return _run_async(_run_generate(prompt=prompt, config=config))


def run_edit(*, image: Path, prompt: str, config: GrokImagineConfig) -> list[Image.Image]:
    """Run Grok Imagine image editing through Comfy API nodes."""
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    _validate_prompt(prompt)
    config.validate(edit=True)
    return _run_async(_run_edit(image=image, prompt=prompt, config=config))


def _validate_prompt(prompt: str) -> None:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")


def _run_async(coro: Any) -> list[Image.Image]:
    try:
        return asyncio.run(coro)
    except GrokImagineError:
        raise
    except Exception as exc:
        message = str(exc).strip() or type(exc).__name__
        raise GrokImagineError(f"Grok Imagine API request failed: {message}") from exc


async def _run_generate(*, prompt: str, config: GrokImagineConfig) -> list[Image.Image]:
    nodes = _grok_nodes(config)
    output = await nodes["generate"].execute(
        config.model,
        prompt,
        config.aspect_ratio,
        config.number_of_images,
        config.seed,
        config.resolution,
    )
    return _node_output_to_images(output)


async def _run_edit(*, image: Path, prompt: str, config: GrokImagineConfig) -> list[Image.Image]:
    nodes = _grok_nodes(config)
    image_tensor = _load_image_tensor(image)
    output = await nodes["edit"].execute(
        config.model,
        image_tensor,
        prompt,
        config.resolution,
        config.number_of_images,
        config.seed,
        config.aspect_ratio,
    )
    return _node_output_to_images(output)


def _grok_nodes(config: GrokImagineConfig) -> dict[str, Any]:
    try:
        from comfy_diffusion._runtime import ensure_comfyui_on_path

        comfyui_root = ensure_comfyui_on_path()
        _prioritize_comfyui_root(comfyui_root)
        _disable_api_node_progress_display()
        from comfy_api_nodes import nodes_grok
    except Exception as exc:
        raise GrokImagineMissingDependencyError(
            "Grok Imagine API nodes are not available. Update comfy-diffusion to a release that vendors ComfyUI Grok nodes."
        ) from exc

    required = {
        "generate": "GrokImageNode",
        "edit": "GrokImageEditNode",
    }
    missing = [name for name in required.values() if not hasattr(nodes_grok, name)]
    if missing:
        raise GrokImagineMissingDependencyError("installed comfy-diffusion does not vendor the Grok Imagine API nodes")

    nodes = {key: getattr(nodes_grok, name) for key, name in required.items()}
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
    setattr(hidden, "unique_id", "comfy-agent-tools-grok-imagine")


def _load_image_tensor(path: Path) -> Any:
    try:
        from comfy_diffusion.image import load_image

        image, _mask = load_image(path)
        return image
    except Exception as exc:
        raise RuntimeError(f"failed to load image for Grok Imagine: {path}: {exc}") from exc


def _node_output_to_images(output: Any) -> list[Image.Image]:
    result = getattr(output, "result", output)
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("Grok Imagine node returned an empty result")
        result = result[0]

    tensor = result
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        array = tensor.numpy()
    else:
        import numpy as np

        array = np.asarray(tensor)

    if array.ndim == 4:
        return [tensor_to_pil(array[index]) for index in range(array.shape[0])]
    return [tensor_to_pil(array)]
