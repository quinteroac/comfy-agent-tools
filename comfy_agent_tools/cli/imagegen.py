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
from comfy_agent_tools.imagegen.flux_klein import run_flux_klein_edit, run_flux_klein_t2i
from comfy_agent_tools.imagegen.grok import (
    DEFAULT_GROK_ASPECT_RATIO,
    DEFAULT_GROK_EDIT_ASPECT_RATIO,
    DEFAULT_GROK_MODEL,
    DEFAULT_GROK_NUMBER_OF_IMAGES,
    DEFAULT_GROK_RESOLUTION,
    DEFAULT_GROK_SEED,
    GROK_ASPECT_RATIOS,
    GROK_EDIT_ASPECT_RATIOS,
    GROK_MODELS,
    GROK_PROVIDER,
    GROK_RESOLUTIONS,
    GrokImagineConfig,
    GrokImagineError,
    run_edit as run_grok_edit,
    run_generate as run_grok_generate,
)
from comfy_agent_tools.imagegen.ideogram4 import (
    DEFAULT_IDEOGRAM4_CFG,
    DEFAULT_IDEOGRAM4_CFG_OVERRIDE_END,
    DEFAULT_IDEOGRAM4_CFG_OVERRIDE_START,
    DEFAULT_IDEOGRAM4_CFG_OVERRIDE_VALUE,
    DEFAULT_IDEOGRAM4_HEIGHT,
    DEFAULT_IDEOGRAM4_MU,
    DEFAULT_IDEOGRAM4_SAMPLER,
    DEFAULT_IDEOGRAM4_SEED,
    DEFAULT_IDEOGRAM4_STD,
    DEFAULT_IDEOGRAM4_STEPS,
    DEFAULT_IDEOGRAM4_WIDTH,
    Ideogram4Config,
    load_prompt_from_args,
    run_ideogram4_t2i,
)
from comfy_agent_tools.imagegen.qwen import run_qwen_edit
from comfy_agent_tools.imagegen.upscale import run_upscale
from comfy_agent_tools.media import write_run_manifest
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
            "--no-manifest",
            action="store_true",
            help="Do not write a comfy-media run manifest for this generation.",
        )
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
    edit.add_argument("--width", type=int, default=None, help="FLUX.2 Klein edit output width. Defaults to input width.")
    edit.add_argument("--height", type=int, default=None, help="FLUX.2 Klein edit output height. Defaults to input height.")

    upscale = subparsers.add_parser("upscale", help="Upscale an input image.")
    add_common(upscale)
    upscale.add_argument("--input", type=_path, required=True)
    upscale.add_argument("--upscaler", type=_path, default=None)

    def add_grok_common(subparser: argparse.ArgumentParser, *, edit_mode: bool = False) -> None:
        subparser.add_argument("--out", type=_path, default=DEFAULT_OUT)
        subparser.add_argument(
            "--no-manifest",
            action="store_true",
            help="Do not write a comfy-media run manifest for this generation.",
        )
        subparser.add_argument("--model", default=None, choices=GROK_MODELS)
        subparser.add_argument("--resolution", default=None, choices=GROK_RESOLUTIONS)
        ratios = GROK_EDIT_ASPECT_RATIOS if edit_mode else GROK_ASPECT_RATIOS
        default_ratio = DEFAULT_GROK_EDIT_ASPECT_RATIO if edit_mode else DEFAULT_GROK_ASPECT_RATIO
        subparser.add_argument("--aspect-ratio", default=None, choices=ratios, help=f"Defaults to {default_ratio}.")
        subparser.add_argument("--number-of-images", type=int, default=None)
        subparser.add_argument("--seed", type=int, default=None)
        subparser.add_argument(
            "--verbose",
            action="store_true",
            help="Show ComfyUI API node logs and progress output while running.",
        )

    grok_generate = subparsers.add_parser("grok-generate", help="Generate remote Grok Imagine images from text.")
    add_grok_common(grok_generate)
    grok_generate.add_argument("--prompt", required=True)

    grok_edit = subparsers.add_parser("grok-edit", help="Edit an image with remote Grok Imagine.")
    add_grok_common(grok_edit, edit_mode=True)
    grok_edit.add_argument("--input", type=_path, required=True)
    grok_edit.add_argument("--prompt", required=True)

    ideogram4 = subparsers.add_parser("ideogram4-generate", help="Generate local Ideogram 4 images.")
    add_common(ideogram4)
    ideogram4.add_argument("--prompt", required=True, help="High-level Ideogram 4 description.")
    ideogram4.add_argument("--width", type=int, default=None)
    ideogram4.add_argument("--height", type=int, default=None)
    ideogram4.add_argument("--steps", type=int, default=None)
    ideogram4.add_argument("--cfg", type=float, default=None)
    ideogram4.add_argument("--cfg-override-value", type=float, default=None)
    ideogram4.add_argument("--disable-cfg-override", action="store_true")
    ideogram4.add_argument("--cfg-override-start", type=float, default=None)
    ideogram4.add_argument("--cfg-override-end", type=float, default=None)
    ideogram4.add_argument("--seed", type=int, default=None)
    ideogram4.add_argument("--mu", type=float, default=None)
    ideogram4.add_argument("--std", type=float, default=None)
    ideogram4.add_argument("--sampler", default=None)
    ideogram4.add_argument("--unet", type=_path, default=None)
    ideogram4.add_argument("--uncond-unet", type=_path, default=None)
    ideogram4.add_argument("--clip", type=_path, default=None)
    ideogram4.add_argument("--vae", type=_path, default=None)
    ideogram4.add_argument(
        "--extra-lora",
        action="append",
        type=parse_extra_lora,
        default=[],
        help="Apply an extra LoRA as PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]. Repeatable.",
    )
    ideogram4.add_argument("--style-aesthetics", required=True)
    ideogram4.add_argument("--style-lighting", required=True)
    ideogram4.add_argument("--style-medium", required=True)
    ideogram4.add_argument("--style-photo")
    ideogram4.add_argument("--style-art-style")
    ideogram4.add_argument("--style-color", action="append", default=[])
    ideogram4.add_argument("--background", required=True)
    ideogram4.add_argument("--object", action="append", default=[])
    ideogram4.add_argument("--text", action="append", default=[])
    ideogram4.add_argument(
        "--output-json",
        type=_path,
        default=None,
        help="Write the generated structured Ideogram prompt JSON to this path.",
    )

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


def _grok_config(args: argparse.Namespace, profile: ResolvedProfile) -> GrokImagineConfig:
    default_ratio = (
        profile.defaults.get("edit_aspect_ratio", DEFAULT_GROK_EDIT_ASPECT_RATIO)
        if args.command == "grok-edit"
        else profile.defaults.get("aspect_ratio", DEFAULT_GROK_ASPECT_RATIO)
    )
    return GrokImagineConfig(
        model=args.model if args.model is not None else str(profile.defaults.get("model", DEFAULT_GROK_MODEL)),
        resolution=args.resolution if args.resolution is not None else str(profile.defaults.get("resolution", DEFAULT_GROK_RESOLUTION)),
        aspect_ratio=args.aspect_ratio if args.aspect_ratio is not None else str(default_ratio),
        number_of_images=(
            args.number_of_images
            if args.number_of_images is not None
            else int(profile.defaults.get("number_of_images", DEFAULT_GROK_NUMBER_OF_IMAGES))
        ),
        seed=args.seed if args.seed is not None else int(profile.defaults.get("seed", DEFAULT_GROK_SEED)),
    )


def _ideogram4_config(args: argparse.Namespace, profile: ResolvedProfile) -> Ideogram4Config:
    cfg_override_value: float | None
    if args.disable_cfg_override:
        cfg_override_value = None
    elif args.cfg_override_value is not None:
        cfg_override_value = args.cfg_override_value
    else:
        default_override = profile.defaults.get("cfg_override_value", DEFAULT_IDEOGRAM4_CFG_OVERRIDE_VALUE)
        cfg_override_value = None if default_override is None else float(default_override)
    return Ideogram4Config(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        unet=args.unet if args.unet is not None else profile.models.get("unet"),
        uncond_unet=args.uncond_unet if args.uncond_unet is not None else profile.models.get("uncond_unet"),
        clip=args.clip if args.clip is not None else profile.models.get("clip"),
        vae=args.vae if args.vae is not None else profile.models.get("vae"),
        width=args.width if args.width is not None else int(profile.defaults.get("width", DEFAULT_IDEOGRAM4_WIDTH)),
        height=args.height if args.height is not None else int(profile.defaults.get("height", DEFAULT_IDEOGRAM4_HEIGHT)),
        steps=args.steps if args.steps is not None else int(profile.defaults.get("steps", DEFAULT_IDEOGRAM4_STEPS)),
        cfg=args.cfg if args.cfg is not None else float(profile.defaults.get("cfg", DEFAULT_IDEOGRAM4_CFG)),
        cfg_override_value=cfg_override_value,
        cfg_override_start=(
            args.cfg_override_start
            if args.cfg_override_start is not None
            else float(profile.defaults.get("cfg_override_start", DEFAULT_IDEOGRAM4_CFG_OVERRIDE_START))
        ),
        cfg_override_end=(
            args.cfg_override_end
            if args.cfg_override_end is not None
            else float(profile.defaults.get("cfg_override_end", DEFAULT_IDEOGRAM4_CFG_OVERRIDE_END))
        ),
        seed=args.seed if args.seed is not None else int(profile.defaults.get("seed", DEFAULT_IDEOGRAM4_SEED)),
        mu=args.mu if args.mu is not None else float(profile.defaults.get("mu", DEFAULT_IDEOGRAM4_MU)),
        std=args.std if args.std is not None else float(profile.defaults.get("std", DEFAULT_IDEOGRAM4_STD)),
        sampler=args.sampler if args.sampler is not None else str(profile.defaults.get("sampler", DEFAULT_IDEOGRAM4_SAMPLER)),
        extra_loras=list(args.extra_lora or []),
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


def _grok_success(
    *,
    mode: str,
    artifacts: list[Path],
    images: list[object],
    config: GrokImagineConfig,
    profile: ResolvedProfile,
    input_path: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "image",
        "mode": mode,
        "remote": True,
        "provider": GROK_PROVIDER,
        "artifacts": [str(path) for path in artifacts],
        "seed": config.seed,
        "model": config.model,
        "resolution": config.resolution,
        "aspect_ratio": config.aspect_ratio,
        "number_of_images": config.number_of_images,
        "outputs": [_image_metadata(image) for image in images],
        "capability": profile.capability,
        "model_profile": profile.name,
        "architecture": profile.architecture,
        "resolved_models": {},
    }
    if input_path is not None:
        payload["input"] = str(input_path)
    return payload


def _ideogram4_success(
    *,
    artifacts: list[Path],
    images: list[object],
    config: Ideogram4Config,
    profile: ResolvedProfile,
    prompt_json: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "image",
        "mode": "ideogram4-generate",
        "artifacts": [str(path) for path in artifacts],
        "seed": config.seed,
        "model": config.unet.name,
        "steps": config.steps,
        "cfg": config.cfg,
        "cfg_override_value": config.cfg_override_value,
        "cfg_override_start": config.cfg_override_start,
        "cfg_override_end": config.cfg_override_end,
        "mu": config.mu,
        "std": config.std,
        "sampler": config.sampler,
        "requested_width": config.width,
        "requested_height": config.height,
        "outputs": [_image_metadata(image) for image in images],
        "capability": profile.capability,
        "model_profile": profile.name,
        "architecture": profile.architecture,
        "models_dir": str(config.models_dir),
        "resolved_models": _resolved_ideogram4_models(config),
        "extra_loras": extra_loras_json(config.models_dir, config.extra_loras or []),
    }
    if prompt_json is not None:
        payload["prompt_json"] = str(prompt_json)
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
    }
    if config.lora is not None:
        models["lora"] = config.resolve_model_path(config.lora)
    if upscaled:
        models["upscaler"] = config.resolve_model_path(config.upscaler)
    return {key: str(value) for key, value in models.items()}


def _resolved_ideogram4_models(config: Ideogram4Config) -> dict[str, str]:
    return {
        "unet": str(config.resolve_model_path(config.unet)),
        "uncond_unet": str(config.resolve_model_path(config.uncond_unet)),
        "clip": str(config.resolve_model_path(config.clip)),
        "vae": str(config.resolve_model_path(config.vae)),
    }


def _write_prompt_json(prompt: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt + "\n", encoding="utf-8")
    return path


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
    if isinstance(error, GrokImagineError):
        return error.error_type
    if "api key" in message or "comfy_org_api_key" in message:
        return "auth_required"
    if "grok imagine api" in message or "api request" in message:
        return "remote_api_error"
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

    if args.command == "grok-generate":
        config = _grok_config(args, profile)
        with _maybe_silence(not args.verbose):
            images = run_grok_generate(prompt=args.prompt, config=config)
        artifacts = save_images(images, args.out, prefix="comfy-imagegen-grok-generate")
        return _grok_success(
            mode="grok-generate",
            artifacts=artifacts,
            images=images,
            config=config,
            profile=profile,
        )

    if args.command == "grok-edit":
        if not args.input.is_file():
            raise FileNotFoundError(f"input image not found: {args.input}")
        config = _grok_config(args, profile)
        with _maybe_silence(not args.verbose):
            images = run_grok_edit(image=args.input, prompt=args.prompt, config=config)
        artifacts = save_images(images, args.out, prefix="comfy-imagegen-grok-edit")
        return _grok_success(
            mode="grok-edit",
            artifacts=artifacts,
            images=images,
            config=config,
            profile=profile,
            input_path=args.input,
        )

    if args.command == "ideogram4-generate":
        config = _ideogram4_config(args, profile)
        prompt = load_prompt_from_args(args)
        with _maybe_silence(not args.verbose):
            images = run_ideogram4_t2i(prompt=prompt, config=config)
        artifacts = save_images(images, args.out, prefix="comfy-imagegen-ideogram4-generate")
        prompt_json = _write_prompt_json(prompt, args.output_json) if args.output_json is not None else None
        return _ideogram4_success(
            artifacts=artifacts,
            images=images,
            config=config,
            profile=profile,
            prompt_json=prompt_json,
        )

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
            elif profile.architecture == "flux-klein":
                images = run_flux_klein_t2i(
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
        requested_width: int | None = None
        requested_height: int | None = None
        with _maybe_silence(not args.verbose):
            if profile.architecture == "qwen-image-edit":
                if args.width is not None or args.height is not None:
                    raise ValueError("edit --width/--height are only supported by FLUX.2 Klein profiles")
                images = run_qwen_edit(prompt=args.prompt, image=image, config=config)
            elif profile.architecture == "flux-klein":
                requested_width = args.width if args.width is not None else int(image.size[0])
                requested_height = args.height if args.height is not None else int(image.size[1])
                images = run_flux_klein_edit(
                    prompt=args.prompt,
                    image=image,
                    width=requested_width,
                    height=requested_height,
                    config=config,
                )
            else:
                raise ValueError(f"unsupported image editing architecture: {profile.architecture}")
        artifacts = save_images(images, args.out, prefix="comfy-imagegen-edit")
        return _success(
            mode="edit",
            artifacts=artifacts,
            config=config,
            upscaled=False,
            images=images,
            input_path=args.input,
            requested_width=requested_width,
            requested_height=requested_height,
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
        if payload.get("ok") is True and not getattr(args, "no_manifest", False):
            payload["manifests"] = [str(write_run_manifest(out_dir=args.out, tool="comfy-imagegen", payload=payload, args=args))]
    except Exception as exc:
        payload = _error(mode=mode, error=exc)
        print(json.dumps(payload, indent=2))
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
