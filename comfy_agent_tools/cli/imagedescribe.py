"""CLI for local image description with Qwen3-VL 2B Instruct."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from comfy_agent_tools.imagegen.artifacts import load_rgb_image
from comfy_agent_tools.imagedescribe.config import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_LLM,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MIN_P,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUT,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ImagedescribeConfig,
)
from comfy_agent_tools.imagedescribe.qwen3vl import run_qwen3vl_describe
from comfy_agent_tools.media import write_run_manifest
from comfy_agent_tools.profiles import ProfileError, ResolvedProfile, resolve_capability


def _path(value: str) -> Path:
    return Path(value)


DEFAULT_PROMPT = "Describe this image in detail."

CAPABILITY = "imagedescribe.describe"


def build_parser() -> argparse.ArgumentParser:
    """Build the comfy-imagedescribe argument parser."""
    parser = argparse.ArgumentParser(
        prog="comfy-imagedescribe",
        description="Describe images with the local Qwen3-VL 2B Instruct vision-language model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Describe an input image with Qwen3-VL 2B Instruct.")
    describe.add_argument("--models-dir", type=_path, default=None)
    describe.add_argument("--out", type=_path, default=DEFAULT_OUT)
    describe.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write a comfy-media run manifest for this description.",
    )
    describe.add_argument("--llm", type=_path, default=None, help="Qwen3-VL Instruct model directory path.")
    describe.add_argument("--input", type=_path, required=True, help="Input image to describe.")
    describe.add_argument("--prompt", default=DEFAULT_PROMPT, help="Instruction sent to the model.")
    describe.add_argument("--seed", type=int, default=DEFAULT_SEED)
    describe.add_argument("--max-length", type=int, default=None)
    describe.add_argument("--temperature", type=float, default=None)
    describe.add_argument("--top-k", type=int, default=None)
    describe.add_argument("--top-p", type=float, default=None)
    describe.add_argument("--min-p", type=float, default=None)
    describe.add_argument("--repetition-penalty", type=float, default=None)
    describe.add_argument(
        "--greedy",
        action="store_true",
        help="Disable sampling and use greedy decoding (temperature/top-k/top-p are ignored).",
    )
    describe.add_argument(
        "--verbose",
        action="store_true",
        help="Show ComfyUI warnings and progress output while running.",
    )

    return parser


def _config(args: argparse.Namespace, profile: ResolvedProfile) -> ImagedescribeConfig:
    do_sample = not args.greedy
    return ImagedescribeConfig(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        llm=args.llm if args.llm is not None else Path(
            str(profile.defaults.get("llm"))
            if profile.defaults.get("llm")
            else str(profile.models.get("llm", DEFAULT_LLM))
        ),
        seed=args.seed,
        max_length=(
            args.max_length
            if args.max_length is not None
            else int(profile.defaults.get("max_length", DEFAULT_MAX_LENGTH))
        ),
        do_sample=bool(profile.defaults.get("do_sample", do_sample)) if not args.greedy else False,
        temperature=(
            args.temperature
            if args.temperature is not None
            else float(profile.defaults.get("temperature", DEFAULT_TEMPERATURE))
        ),
        top_k=(
            args.top_k
            if args.top_k is not None
            else int(profile.defaults.get("top_k", DEFAULT_TOP_K))
        ),
        top_p=(
            args.top_p
            if args.top_p is not None
            else float(profile.defaults.get("top_p", DEFAULT_TOP_P))
        ),
        min_p=(
            args.min_p
            if args.min_p is not None
            else float(profile.defaults.get("min_p", DEFAULT_MIN_P))
        ),
        repetition_penalty=(
            args.repetition_penalty
            if args.repetition_penalty is not None
            else float(profile.defaults.get("repetition_penalty", DEFAULT_REPETITION_PENALTY))
        ),
    )


@contextlib.contextmanager
def _maybe_silence(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _success(
    *,
    description: str,
    config: ImagedescribeConfig,
    input_path: Path,
    prompt: str,
    profile: ResolvedProfile,
) -> dict[str, Any]:
    return {
        "ok": True,
        "kind": "text",
        "mode": "describe",
        "input": str(input_path),
        "prompt": prompt,
        "description": description,
        "seed": config.seed,
        "max_length": config.max_length,
        "do_sample": config.do_sample,
        "temperature": config.temperature,
        "top_k": config.top_k,
        "top_p": config.top_p,
        "min_p": config.min_p,
        "repetition_penalty": config.repetition_penalty,
        "model": config.llm.name,
        **profile.metadata(),
        "models_dir": str(config.models_dir),
        "resolved_models": {"llm": str(config.resolve_model_path(config.llm))},
    }


def _error(*, mode: str, error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(error),
        "error_type": _classify_error(error),
        "kind": "text",
        "mode": mode,
    }


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


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed comfy-imagedescribe command and return its JSON payload."""
    profile, _source = resolve_capability(CAPABILITY)
    config = _config(args, profile)

    if args.command == "describe":
        image = load_rgb_image(args.input)
        with _maybe_silence(not args.verbose):
            description = run_qwen3vl_describe(
                image=image,
                prompt=args.prompt,
                config=config,
            )
        return _success(
            description=description,
            config=config,
            input_path=args.input,
            prompt=args.prompt,
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
            payload["manifests"] = [
                str(write_run_manifest(out_dir=args.out, tool="comfy-imagedescribe", payload=payload, args=args))
            ]
    except Exception as exc:
        payload = _error(mode=mode, error=exc)
        print(json.dumps(payload, indent=2))
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
