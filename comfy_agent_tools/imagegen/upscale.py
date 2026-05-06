"""Image upscaling helpers."""

from __future__ import annotations

from PIL import Image

from .artifacts import tensor_to_pil
from .config import ImagegenConfig
from .runtime import require_comfy_runtime


def run_upscale(*, image: Image.Image, config: ImagegenConfig) -> list[Image.Image]:
    """Upscale an image with the configured ClearReality model."""
    require_comfy_runtime()

    from comfy_diffusion.image import image_to_tensor, image_upscale_with_model
    from comfy_diffusion.models import ModelManager

    mm = ModelManager(config.models_dir)
    upscale_model = mm.load_upscale_model(config.resolve_model_path(config.upscaler))
    tensor = image_to_tensor(image)
    upscaled = image_upscale_with_model(upscale_model, tensor)
    return [tensor_to_pil(upscaled)]
