"""CLI for local image generation, editing, and upscaling."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from comfy_agent_tools.loras import extra_loras_json, parse_extra_lora
from comfy_agent_tools.imagegen.artifacts import create_seed_image, load_rgb_image, save_images
from comfy_agent_tools.imagegen.config import (
    DEFAULT_CFG,
    DEFAULT_CLIP,
    DEFAULT_HEIGHT,
    DEFAULT_LORA,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUT,
    DEFAULT_SAMPLER,
    DEFAULT_SEED,
    DEFAULT_SCHEDULER,
    DEFAULT_STEPS,
    DEFAULT_UNET,
    DEFAULT_UPSCALER,
    DEFAULT_VAE,
    DEFAULT_WIDTH,
    ImagegenConfig,
)
from comfy_agent_tools.imagegen.anima import run_anima_t2i
from comfy_agent_tools.imagegen.qwen import run_qwen_edit
from comfy_agent_tools.imagegen.upscale import run_upscale
from comfy_agent_tools.profiles import ProfileError, ResolvedProfile, resolve_capability


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the comfy-imagegen argument parser."""
    parser = argparse.ArgumentParser(
        prog="comfy-imagegen",
        description="Generate, edit, and upscale images with local comfy-diffusion models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--models-dir", type=_path, default=None)
        subparser.add_argument("--out", type=_path, default=DEFAULT_OUT)
        subparser.add_argument(
            "--verbose",
            action="store_true",
            help="Show ComfyUI warnings and progress output while running.",
        )

    def add_qwen(subparser: argparse.ArgumentParser) -> None:
        add_common(subparser)
        subparser.add_argument("--prompt", required=True)
        subparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        subparser.add_argument("--steps", type=int, default=None)
        subparser.add_argument("--cfg", type=float, default=None)
        subparser.add_argument("--unet", type=_path, default=None)
        subparser.add_argument("--clip", type=_path, default=None)
        subparser.add_argument("--vae", type=_path, default=None)
        subparser.add_argument("--lora", type=_path, default=None)
        subparser.add_argument(
            "--extra-lora",
            action="append",
            type=parse_extra_lora,
            default=[],
            help="Apply an extra LoRA as PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]. Repeatable.",
        )

    generate = subparsers.add_parser("generate", help="Generate an image from a prompt.")
    add_qwen(generate)
    generate.add_argument("--width", type=int, default=None)
    generate.add_argument("--height", type=int, default=None)

    edit = subparsers.add_parser("edit", help="Edit an input image with a prompt.")
    add_qwen(edit)
    edit.add_argument("--input", type=_path, required=True)

    upscale = subparsers.add_parser("upscale", help="Upscale an input image.")
    add_common(upscale)
    upscale.add_argument("--input", type=_path, required=True)
    upscale.add_argument("--upscaler", type=_path, default=None)

    return parser


def _capability(command: str) -> str:
    return f"imagegen.{command}"


def _config(args: argparse.Namespace, profile: ResolvedProfile) -> ImagegenConfig:
    return ImagegenConfig(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        unet=getattr(args, "unet", None) if getattr(args, "unet", None) is not None else profile.models.get("unet", DEFAULT_UNET),
        clip=getattr(args, "clip", None) if getattr(args, "clip", None) is not None else profile.models.get("clip", DEFAULT_CLIP),
        vae=getattr(args, "vae", None) if getattr(args, "vae", None) is not None else profile.models.get("vae", DEFAULT_VAE),
        lora=getattr(args, "lora", None) if getattr(args, "lora", None) is not None else profile.models.get("lora", DEFAULT_LORA),
        upscaler=getattr(args, "upscaler", None) if getattr(args, "upscaler", None) is not None else profile.models.get("upscaler", DEFAULT_UPSCALER),
        steps=getattr(args, "steps", None) if getattr(args, "steps", None) is not None else int(profile.defaults.get("steps", DEFAULT_STEPS)),
        cfg=getattr(args, "cfg", None) if getattr(args, "cfg", None) is not None else float(profile.defaults.get("cfg", DEFAULT_CFG)),
        sampler=str(profile.defaults.get("sampler", DEFAULT_SAMPLER)),
        scheduler=str(profile.defaults.get("scheduler", DEFAULT_SCHEDULER)),
        seed=getattr(args, "seed", DEFAULT_SEED),
        extra_loras=list(getattr(args, "extra_lora", []) or []),
    )


def _success(
    *,
    mode: str,
    artifacts: list[Path],
    config: ImagegenConfig,
    upscaled: bool,
    images: list[object],
    input_path: Path | None = None,
    requested_width: int | None = None,
    requested_height: int | None = None,
    profile: ResolvedProfile,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "image",
        "mode": mode,
        "artifacts": [str(path) for path in artifacts],
        "seed": config.seed,
        "models_dir": str(config.models_dir),
        "model": str(config.unet.name),
        "upscaled": upscaled,
        "steps": config.steps,
        "cfg": config.cfg,
        "outputs": [_image_metadata(image) for image in images],
        **profile.metadata(),
        "models_dir": str(config.models_dir),
        "resolved_models": _resolved_image_models(config, upscaled=upscaled),
        "extra_loras": extra_loras_json(config.models_dir, config.extra_loras or []),
    }
    if input_path is not None:
        payload["input"] = str(input_path)
    if requested_width is not None and requested_height is not None:
        payload["requested_width"] = requested_width
        payload["requested_height"] = requested_height
    if upscaled:
        payload["upscaler"] = str(config.upscaler.name)
    return payload


def _error(*, mode: str, error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(error),
        "error_type": _classify_error(error),
        "kind": "image",
        "mode": mode,
    }


def _image_metadata(image: object) -> dict[str, Any]:
    size = getattr(image, "size", None)
    width: int | None = None
    height: int | None = None
    if isinstance(size, tuple) and len(size) >= 2:
        width = int(size[0])
        height = int(size[1])
    return {
        "width": width,
        "height": height,
        "mode": getattr(image, "mode", None),
    }


def _resolved_image_models(config: ImagegenConfig, *, upscaled: bool) -> dict[str, str]:
    models = {
        "unet": config.resolve_model_path(config.unet),
        "clip": config.resolve_model_path(config.clip),
        "vae": config.resolve_model_path(config.vae),
        "lora": config.resolve_model_path(config.lora),
    }
    if upscaled:
        models["upscaler"] = config.resolve_model_path(config.upscaler)
    return {key: str(value) for key, value in models.items()}


def _classify_error(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, FileNotFoundError):
        return "not_found"
    if isinstance(error, ModuleNotFoundError) or "no module named" in message:
        return "missing_dependency"
    if isinstance(error, MemoryError) or "out of memory" in message or "cuda out of memory" in message:
        return "out_of_memory"
    if "runtime not available" in message or "runtime bootstrap failed" in message:
        return "runtime"
    if isinstance(error, ProfileError):
        return error.error_type
    return "error"


@contextlib.contextmanager
def _maybe_silence(enabled: bool) -> Any:
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed comfy-imagegen command and return its JSON payload."""
    profile, _source = resolve_capability(_capability(args.command))
    config = _config(args, profile)
    _ = config.resolved_extra_loras

    if args.command == "generate":
        width = args.width if args.width is not None else int(profile.defaults.get("width", DEFAULT_WIDTH))
        height = args.height if args.height is not None else int(profile.defaults.get("height", DEFAULT_HEIGHT))
        with _maybe_silence(not args.verbose):
            if profile.architecture == "qwen-image-edit":
                image = create_seed_image(width, height)
                images = run_qwen_edit(prompt=args.prompt, image=image, config=config)
            elif profile.architecture == "anima":
                images = run_anima_t2i(
                    prompt=args.prompt,
                    width=width,
                    height=height,
                    config=config,
                )
            else:
                raise ValueError(f"unsupported image generation architecture: {profile.architecture}")
        artifacts = save_images(images, args.out, prefix="comfy-imagegen-generate")
        return _success(
            mode="generate",
            artifacts=artifacts,
            config=config,
            upscaled=False,
            images=images,
            requested_width=width,
            requested_height=height,
            profile=profile,
        )

    if args.command == "edit":
        image = load_rgb_image(args.input)
        with _maybe_silence(not args.verbose):
            images = run_qwen_edit(prompt=args.prompt, image=image, config=config)
        artifacts = save_images(images, args.out, prefix="comfy-imagegen-edit")
        return _success(
            mode="edit",
            artifacts=artifacts,
            config=config,
            upscaled=False,
            images=images,
            input_path=args.input,
            profile=profile,
        )

    if args.command == "upscale":
        image = load_rgb_image(args.input)
        with _maybe_silence(not args.verbose):
            images = run_upscale(image=image, config=config)
        artifacts = save_images(images, args.out, prefix="comfy-imagegen-upscale")
        return _success(
            mode="upscale",
            artifacts=artifacts,
            config=config,
            upscaled=True,
            images=images,
            input_path=args.input,
            profile=profile,
        )

    raise ValueError(f"unknown command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = args.command or "unknown"

    try:
        payload = run_command(args)
    except Exception as exc:
        payload = _error(mode=mode, error=exc)
        print(json.dumps(payload, indent=2))
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
