"""Runtime validation helpers."""

from __future__ import annotations

from typing import Any


def require_comfy_runtime() -> dict[str, Any]:
    """Validate comfy-diffusion runtime and raise a readable error on failure."""
    from comfy_diffusion.runtime import check_runtime

    result = check_runtime()
    if result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {result['error']}")
    return result
