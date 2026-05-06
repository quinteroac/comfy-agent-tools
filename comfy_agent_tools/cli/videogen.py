"""CLI for local LTX 2.3 video generation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from comfy_agent_tools.loras import extra_loras_json, parse_extra_lora
from comfy_agent_tools.videogen.artifacts import frame_metadata, make_video_path, save_mp4_with_audio
from comfy_agent_tools.videogen.config import (
    DEFAULT_CFG,
    DEFAULT_AUDIO_START_TIME,
    DEFAULT_CHECKPOINT,
    DEFAULT_DISTILLED_LORA,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_MODELS_DIR,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OUT,
    DEFAULT_SEED,
    DEFAULT_TE_LORA,
    DEFAULT_TEXT_ENCODER,
    DEFAULT_UPSCALER,
    DEFAULT_WIDTH,
    VideogenConfig,
)
from comfy_agent_tools.videogen.ltx23 import run_flf2v, run_i2v, run_ia2av, run_t2v
from comfy_agent_tools.profiles import ProfileError, ResolvedProfile, resolve_capability


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the comfy-videogen argument parser."""
    parser = argparse.ArgumentParser(
        prog="comfy-videogen",
        description="Generate MP4 videos with local LTX 2.3 models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--models-dir", type=_path, default=None)
        subparser.add_argument("--out", type=_path, default=DEFAULT_OUT)
        subparser.add_argument("--checkpoint", type=_path, default=None)
        subparser.add_argument("--text-encoder", type=_path, default=None)
        subparser.add_argument("--distilled-lora", type=_path, default=None)
        subparser.add_argument("--te-lora", type=_path, default=None)
        subparser.add_argument("--upscaler", type=_path, default=None)
        subparser.add_argument("--width", type=int, default=None)
        subparser.add_argument("--height", type=int, default=None)
        subparser.add_argument("--length", type=int, default=None)
        subparser.add_argument("--fps", type=int, default=None)
        subparser.add_argument("--cfg", type=float, default=None)
        subparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        subparser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
        subparser.add_argument(
            "--extra-lora",
            action="append",
            type=parse_extra_lora,
            default=[],
            help="Apply an extra LoRA as PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]. Repeatable.",
        )
        subparser.add_argument(
            "--verbose",
            action="store_true",
            help="Show ComfyUI warnings and progress output while running.",
        )

    t2v = subparsers.add_parser("t2v", help="Generate a video from a text prompt.")
    add_common(t2v)
    t2v.add_argument("--prompt", required=True)

    i2v = subparsers.add_parser("i2v", help="Animate an input image with a prompt.")
    add_common(i2v)
    i2v.add_argument("--input", type=_path, required=True)
    i2v.add_argument("--prompt", required=True)

    ia2av = subparsers.add_parser(
        "ia2av",
        help="Animate an input image into a video conditioned on an input audio file.",
    )
    add_common(ia2av)
    ia2av.add_argument("--input", type=_path, required=True)
    ia2av.add_argument("--audio", type=_path, required=True)
    ia2av.add_argument("--prompt", required=True)
    ia2av.add_argument("--audio-start-time", type=float, default=DEFAULT_AUDIO_START_TIME)
    ia2av.add_argument("--audio-duration", type=float, default=None)

    flf2v = subparsers.add_parser(
        "flf2v",
        help="Generate a transition video between first and last frame images.",
    )
    add_common(flf2v)
    flf2v.add_argument("--first", type=_path, required=True)
    flf2v.add_argument("--last", type=_path, required=True)
    flf2v.add_argument("--prompt", required=True)

    return parser


def _capability(command: str) -> str:
    return f"videogen.{command}"


def _config(args: argparse.Namespace, profile: ResolvedProfile) -> VideogenConfig:
    return VideogenConfig(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        checkpoint=args.checkpoint if args.checkpoint is not None else profile.models.get("checkpoint", DEFAULT_CHECKPOINT),
        text_encoder=args.text_encoder if args.text_encoder is not None else profile.models.get("text_encoder", DEFAULT_TEXT_ENCODER),
        distilled_lora=args.distilled_lora if args.distilled_lora is not None else profile.models.get("distilled_lora", DEFAULT_DISTILLED_LORA),
        te_lora=args.te_lora if args.te_lora is not None else profile.models.get("te_lora", DEFAULT_TE_LORA),
        upscaler=args.upscaler if args.upscaler is not None else profile.models.get("upscaler", DEFAULT_UPSCALER),
        width=args.width if args.width is not None else int(profile.defaults.get("width", DEFAULT_WIDTH)),
        height=args.height if args.height is not None else int(profile.defaults.get("height", DEFAULT_HEIGHT)),
        length=args.length if args.length is not None else int(profile.defaults.get("length", DEFAULT_LENGTH)),
        fps=args.fps if args.fps is not None else int(profile.defaults.get("fps", DEFAULT_FPS)),
        cfg=args.cfg if args.cfg is not None else float(profile.defaults.get("cfg", DEFAULT_CFG)),
        seed=args.seed,
        audio_start_time=getattr(args, "audio_start_time", DEFAULT_AUDIO_START_TIME),
        audio_duration=getattr(args, "audio_duration", None),
        negative_prompt=args.negative_prompt,
        extra_loras=list(getattr(args, "extra_lora", []) or []),
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
    mode: str,
    artifact: Path,
    config: VideogenConfig,
    frames: list[object],
    input_path: Path | None = None,
    first_path: Path | None = None,
    last_path: Path | None = None,
    audio_path: Path | None = None,
    audio_start_time: float | None = None,
    audio_duration: float | None = None,
    profile: ResolvedProfile,
) -> dict[str, Any]:
    meta = frame_metadata(frames, config.fps)
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "video",
        "mode": mode,
        "artifacts": [str(artifact)],
        "seed": config.seed,
        "models_dir": str(config.models_dir),
        "model": config.checkpoint.name,
        "width": meta["width"],
        "height": meta["height"],
        "frames": meta["frames"],
        "fps": meta["fps"],
        "duration_seconds": meta["duration_seconds"],
        "audio_muxed": True,
        **profile.metadata(),
        "models_dir": str(config.models_dir),
        "resolved_models": _resolved_video_models(config),
        "extra_loras": extra_loras_json(config.models_dir, config.extra_loras or []),
    }
    if input_path is not None:
        payload["input"] = str(input_path)
    if audio_path is not None:
        payload["audio_input"] = str(audio_path)
        payload["audio_conditioned"] = True
        payload["audio_start_time"] = audio_start_time if audio_start_time is not None else config.audio_start_time
        payload["audio_duration_seconds"] = (
            audio_duration
            if audio_duration is not None
            else config.audio_duration
            if config.audio_duration is not None
            else meta["duration_seconds"]
        )
    if first_path is not None:
        payload["first"] = str(first_path)
    if last_path is not None:
        payload["last"] = str(last_path)
    return payload


def _resolved_video_models(config: VideogenConfig) -> dict[str, str]:
    return {
        "checkpoint": str(config.resolve_model_path(config.checkpoint)),
        "text_encoder": str(config.resolve_model_path(config.text_encoder)),
        "distilled_lora": str(config.resolve_model_path(config.distilled_lora)),
        "te_lora": str(config.resolve_model_path(config.te_lora)),
        "upscaler": str(config.resolve_model_path(config.upscaler)),
    }


def _error(*, mode: str, error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(error),
        "error_type": _classify_error(error),
        "kind": "video",
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
    if "audio" in message or "mux" in message or "aac" in message:
        return "audio_mux"
    if isinstance(error, ProfileError):
        return error.error_type
    return "error"


def _write_result_video(result: dict[str, Any], out_dir: Path, *, prefix: str, fps: int) -> Path:
    frames = result.get("frames")
    audio = result.get("audio")
    if not isinstance(frames, list):
        raise ValueError("pipeline result did not include a frames list")
    path = make_video_path(out_dir, prefix=prefix)
    save_mp4_with_audio(frames, audio, path, fps)
    return path


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed comfy-videogen command and return its JSON payload."""
    profile, _source = resolve_capability(_capability(args.command))
    config = _config(args, profile)
    _ = config.resolved_extra_loras

    if args.command == "t2v":
        if config.extra_loras:
            raise ValueError("extra LoRAs are not supported for videogen.t2v yet; use flf2v or a profile-level structural LoRA")
        with _maybe_silence(not args.verbose):
            result = run_t2v(prompt=args.prompt, config=config)
            artifact = _write_result_video(result, args.out, prefix="comfy-videogen-t2v", fps=config.fps)
        return _success(
            mode="t2v",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            profile=profile,
        )

    if args.command == "i2v":
        if not args.input.is_file():
            raise FileNotFoundError(f"input image not found: {args.input}")
        if config.extra_loras:
            raise ValueError("extra LoRAs are not supported for videogen.i2v yet; use flf2v or a profile-level structural LoRA")
        with _maybe_silence(not args.verbose):
            result = run_i2v(image=args.input, prompt=args.prompt, config=config)
            artifact = _write_result_video(result, args.out, prefix="comfy-videogen-i2v", fps=config.fps)
        return _success(
            mode="i2v",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            input_path=args.input,
            profile=profile,
        )

    if args.command == "ia2av":
        if not args.input.is_file():
            raise FileNotFoundError(f"input image not found: {args.input}")
        if not args.audio.is_file():
            raise FileNotFoundError(f"input audio not found: {args.audio}")
        if config.extra_loras:
            raise ValueError("extra LoRAs are not supported for videogen.ia2av yet; use flf2v or a profile-level structural LoRA")
        with _maybe_silence(not args.verbose):
            result = run_ia2av(image=args.input, audio=args.audio, prompt=args.prompt, config=config)
            artifact = _write_result_video(result, args.out, prefix="comfy-videogen-ia2av", fps=config.fps)
        return _success(
            mode="ia2av",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            input_path=args.input,
            audio_path=args.audio,
            audio_start_time=args.audio_start_time,
            audio_duration=args.audio_duration,
            profile=profile,
        )

    if args.command == "flf2v":
        if not args.first.is_file():
            raise FileNotFoundError(f"first image not found: {args.first}")
        if not args.last.is_file():
            raise FileNotFoundError(f"last image not found: {args.last}")
        with _maybe_silence(not args.verbose):
            result = run_flf2v(
                first_image=args.first,
                last_image=args.last,
                prompt=args.prompt,
                config=config,
            )
            artifact = _write_result_video(result, args.out, prefix="comfy-videogen-flf2v", fps=config.fps)
        return _success(
            mode="flf2v",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            first_path=args.first,
            last_path=args.last,
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
