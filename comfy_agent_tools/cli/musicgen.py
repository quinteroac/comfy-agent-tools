"""CLI for local ACE-Step music generation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from comfy_agent_tools.loras import extra_loras_json, parse_extra_lora
from comfy_agent_tools.musicgen.ace_step import run_ace_step
from comfy_agent_tools.musicgen.artifacts import audio_metadata, make_audio_path, save_wav
from comfy_agent_tools.musicgen.config import (
    DEFAULT_BPM,
    DEFAULT_CFG,
    DEFAULT_CLIP_0_6B,
    DEFAULT_CLIP_1_7B,
    DEFAULT_DURATION,
    DEFAULT_KEYSCALE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODELS_DIR,
    DEFAULT_OUT,
    DEFAULT_SAMPLER,
    DEFAULT_SCHEDULER,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_TIME_SIGNATURE,
    DEFAULT_UNET,
    DEFAULT_VAE,
    MusicgenConfig,
)
from comfy_agent_tools.profiles import ProfileError, ResolvedProfile, resolve_capability


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the comfy-musicgen argument parser."""
    parser = argparse.ArgumentParser(
        prog="comfy-musicgen",
        description="Generate WAV music with local ACE-Step 1.5 models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate music from prompt tags and optional lyrics.")
    generate.add_argument("--models-dir", type=_path, default=None)
    generate.add_argument("--out", type=_path, default=DEFAULT_OUT)
    generate.add_argument("--unet", type=_path, default=None)
    generate.add_argument("--clip-0-6b", type=_path, default=None)
    generate.add_argument("--clip-1-7b", type=_path, default=None)
    generate.add_argument("--vae", type=_path, default=None)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--lyrics", default="")
    generate.add_argument("--duration", type=float, default=None)
    generate.add_argument("--bpm", type=int, default=None)
    generate.add_argument("--time-signature", default=None)
    generate.add_argument("--language", default=None)
    generate.add_argument("--keyscale", default=None)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--steps", type=int, default=None)
    generate.add_argument("--cfg", type=float, default=None)
    generate.add_argument("--sampler", default=None)
    generate.add_argument("--scheduler", default=None)
    generate.add_argument(
        "--extra-lora",
        action="append",
        type=parse_extra_lora,
        default=[],
        help="Apply an extra LoRA as PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]. Repeatable.",
    )
    generate.add_argument(
        "--verbose",
        action="store_true",
        help="Show ComfyUI warnings and progress output while running.",
    )

    return parser


def _config(args: argparse.Namespace, profile: ResolvedProfile) -> MusicgenConfig:
    return MusicgenConfig(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        unet=args.unet if args.unet is not None else profile.models.get("unet", DEFAULT_UNET),
        clip_0_6b=args.clip_0_6b if args.clip_0_6b is not None else profile.models.get("clip_0_6b", DEFAULT_CLIP_0_6B),
        clip_1_7b=args.clip_1_7b if args.clip_1_7b is not None else profile.models.get("clip_1_7b", DEFAULT_CLIP_1_7B),
        vae=args.vae if args.vae is not None else profile.models.get("vae", DEFAULT_VAE),
        duration=args.duration if args.duration is not None else float(profile.defaults.get("duration", DEFAULT_DURATION)),
        bpm=args.bpm if args.bpm is not None else int(profile.defaults.get("bpm", DEFAULT_BPM)),
        time_signature=args.time_signature if args.time_signature is not None else str(profile.defaults.get("time_signature", DEFAULT_TIME_SIGNATURE)),
        language=args.language if args.language is not None else str(profile.defaults.get("language", DEFAULT_LANGUAGE)),
        keyscale=args.keyscale if args.keyscale is not None else str(profile.defaults.get("keyscale", DEFAULT_KEYSCALE)),
        seed=args.seed,
        steps=args.steps if args.steps is not None else int(profile.defaults.get("steps", DEFAULT_STEPS)),
        cfg=args.cfg if args.cfg is not None else float(profile.defaults.get("cfg", DEFAULT_CFG)),
        sampler=args.sampler if args.sampler is not None else str(profile.defaults.get("sampler", DEFAULT_SAMPLER)),
        scheduler=args.scheduler if args.scheduler is not None else str(profile.defaults.get("scheduler", DEFAULT_SCHEDULER)),
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
    config: MusicgenConfig,
    audio: dict[str, Any],
    lyrics_present: bool,
    profile: ResolvedProfile,
) -> dict[str, Any]:
    meta = audio_metadata(audio)
    return {
        "ok": True,
        "kind": "music",
        "mode": mode,
        "artifacts": [str(artifact)],
        "seed": config.seed,
        "models_dir": str(config.models_dir),
        "model": config.unet.name,
        "text_encoder": config.clip_1_7b.name,
        "format": meta["format"],
        "sample_rate": meta["sample_rate"],
        "channels": meta["channels"],
        "duration_seconds": meta["duration_seconds"],
        "requested_duration_seconds": config.duration,
        "bpm": config.bpm,
        "keyscale": config.keyscale,
        "language": config.language,
        "time_signature": config.time_signature,
        "steps": config.steps,
        "cfg": config.cfg,
        "sampler": config.sampler,
        "scheduler": config.scheduler,
        "lyrics_present": lyrics_present,
        **profile.metadata(),
        "models_dir": str(config.models_dir),
        "resolved_models": _resolved_music_models(config),
        "extra_loras": extra_loras_json(config.models_dir, config.extra_loras or []),
    }


def _resolved_music_models(config: MusicgenConfig) -> dict[str, str]:
    return {
        "unet": str(config.resolve_model_path(config.unet)),
        "clip_0_6b": str(config.resolve_model_path(config.clip_0_6b)),
        "clip_1_7b": str(config.resolve_model_path(config.clip_1_7b)),
        "vae": str(config.resolve_model_path(config.vae)),
    }


def _error(*, mode: str, error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(error),
        "error_type": _classify_error(error),
        "kind": "music",
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
    if "audio" in message or "waveform" in message or "sample_rate" in message or "wav" in message:
        return "audio"
    if isinstance(error, ProfileError):
        return error.error_type
    return "error"


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed comfy-musicgen command and return its JSON payload."""
    profile, _source = resolve_capability("musicgen.generate")
    config = _config(args, profile)
    _ = config.resolved_extra_loras

    if args.command == "generate":
        with _maybe_silence(not args.verbose):
            result = run_ace_step(prompt=args.prompt, lyrics=args.lyrics, config=config)
            audio = result.get("audio")
            if not isinstance(audio, dict):
                raise ValueError("pipeline result did not include an audio payload")
            artifact = make_audio_path(args.out, prefix="comfy-musicgen-generate")
            save_wav(audio, artifact)
        return _success(
            mode="generate",
            artifact=artifact,
            config=config,
            audio=audio,
            lyrics_present=bool(args.lyrics.strip()),
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
