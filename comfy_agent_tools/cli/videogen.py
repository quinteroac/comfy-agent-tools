"""CLI for local LTX 2.3, WAN 2.2, and remote Seedance 2.0 video generation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from comfy_agent_tools.loras import extra_loras_json, parse_extra_lora
from comfy_agent_tools.media import write_run_manifest
from comfy_agent_tools.videogen.artifacts import frame_metadata, make_video_path, save_mp4, save_mp4_with_audio
from comfy_agent_tools.videogen.config import (
    DEFAULT_CFG,
    DEFAULT_AUDIO_START_TIME,
    DEFAULT_ATTENTION_STRENGTH,
    DEFAULT_CHECKPOINT,
    DEFAULT_DISTILLED_LORA,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_IC_LORA,
    DEFAULT_LENGTH,
    DEFAULT_MODELS_DIR,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OUT,
    DEFAULT_REFERENCE_DOWNSCALE,
    DEFAULT_SEED,
    DEFAULT_TE_LORA,
    DEFAULT_TEXT_ENCODER,
    DEFAULT_UPSCALER,
    DEFAULT_WIDTH,
    VideogenConfig,
)
from comfy_agent_tools.videogen.ltx23 import run_flf2v, run_i2v, run_ia2av, run_motion_track, run_t2v
from comfy_agent_tools.videogen.wan22 import (
    DEFAULT_WAN22_AUDIO_ENCODER,
    DEFAULT_WAN22_FLF2V_CFG,
    DEFAULT_WAN22_FPS,
    DEFAULT_WAN22_HEIGHT,
    DEFAULT_WAN22_I2V_CFG,
    DEFAULT_WAN22_LENGTH,
    DEFAULT_WAN22_NEGATIVE_PROMPT,
    DEFAULT_WAN22_S2V_CFG,
    DEFAULT_WAN22_S2V_CHUNK_LENGTH,
    DEFAULT_WAN22_S2V_LENGTH,
    DEFAULT_WAN22_S2V_LORA_STRENGTH,
    DEFAULT_WAN22_S2V_SAMPLER,
    DEFAULT_WAN22_S2V_SCHEDULER,
    DEFAULT_WAN22_S2V_SHIFT,
    DEFAULT_WAN22_S2V_UNET,
    DEFAULT_WAN22_STEPS,
    DEFAULT_WAN22_TEXT_ENCODER,
    DEFAULT_WAN22_UNET_HIGH,
    DEFAULT_WAN22_UNET_LOW,
    DEFAULT_WAN22_VAE,
    DEFAULT_WAN22_VIDEO_AUDIO_CHUNK_OVERLAP,
    DEFAULT_WAN22_VIDEO_AUDIO_DENOISE,
    DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_DENOISE,
    DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_PROMPT,
    DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_DENOISE,
    DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_STEPS,
    DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_STEPS,
    DEFAULT_WAN22_VIDEO_AUDIO_PROMPT,
    DEFAULT_WAN22_VIDEO_AUDIO_STEPS,
    DEFAULT_WAN22_WIDTH,
    Wan22Config,
    Wan22S2VConfig,
    Wan22VideoAudioConfig,
    run_flf2v as run_wan22_flf2v,
    run_i2v as run_wan22_i2v,
    run_s2v as run_wan22_s2v,
    run_video_audio as run_wan22_video_audio,
)
from comfy_agent_tools.videogen.seedance2 import (
    DEFAULT_SEEDANCE2_DURATION,
    DEFAULT_SEEDANCE2_GENERATE_AUDIO,
    DEFAULT_SEEDANCE2_RATIO,
    DEFAULT_SEEDANCE2_RESOLUTION,
    DEFAULT_SEEDANCE2_SEED,
    DEFAULT_SEEDANCE2_WATERMARK,
    SEEDANCE2_MODEL,
    SEEDANCE2_PROVIDER,
    Seedance2Config,
    Seedance2Error,
    run_flf2v as run_seedance2_flf2v,
    run_r2v as run_seedance2_r2v,
    run_t2v as run_seedance2_t2v,
)
from comfy_agent_tools.profiles import ProfileError, ResolvedProfile, resolve_capability


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the comfy-videogen argument parser."""
    parser = argparse.ArgumentParser(
        prog="comfy-videogen",
        description="Generate MP4 videos with local LTX 2.3/WAN 2.2 models or remote Seedance 2.0 API nodes.",
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
        subparser.add_argument("--checkpoint", type=_path, default=None)
        subparser.add_argument("--text-encoder", type=_path, default=None)
        subparser.add_argument("--distilled-lora", type=_path, default=None)
        subparser.add_argument("--te-lora", type=_path, default=None)
        subparser.add_argument("--upscaler", type=_path, default=None)
        subparser.add_argument("--ic-lora", type=_path, default=None)
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

    motion_track = subparsers.add_parser(
        "motion-track",
        help="Generate a video from an image and a motion-track IC-LoRA control video.",
    )
    add_common(motion_track)
    motion_track.add_argument("--input", type=_path, required=True)
    motion_track.add_argument("--control-video", type=_path, required=True)
    motion_track.add_argument("--prompt", required=True)
    motion_track.add_argument("--attention-strength", type=float, default=None)
    motion_track.add_argument("--reference-downscale", type=float, default=None)

    def add_wan22_common(subparser: argparse.ArgumentParser, *, default_cfg: float) -> None:
        subparser.add_argument("--models-dir", type=_path, default=None)
        subparser.add_argument("--out", type=_path, default=DEFAULT_OUT)
        subparser.add_argument(
            "--no-manifest",
            action="store_true",
            help="Do not write a comfy-media run manifest for this generation.",
        )
        subparser.add_argument("--unet-high", type=_path, default=None)
        subparser.add_argument("--unet-low", type=_path, default=None)
        subparser.add_argument("--text-encoder", type=_path, default=None)
        subparser.add_argument("--vae", type=_path, default=None)
        subparser.add_argument("--width", type=int, default=None)
        subparser.add_argument("--height", type=int, default=None)
        subparser.add_argument("--length", type=int, default=None)
        subparser.add_argument("--fps", type=int, default=None)
        subparser.add_argument("--steps", type=int, default=None)
        subparser.add_argument(
            "--high-steps",
            type=int,
            default=None,
            help="WAN 2.2 high-noise model steps. More high steps usually means more motion.",
        )
        subparser.add_argument(
            "--low-steps",
            type=int,
            default=None,
            help="WAN 2.2 low-noise model steps. More low steps usually means more detail/refinement.",
        )
        subparser.add_argument("--cfg", type=float, default=None)
        subparser.add_argument("--seed", type=int, default=DEFAULT_SEED)
        subparser.add_argument("--negative-prompt", default=DEFAULT_WAN22_NEGATIVE_PROMPT)
        subparser.set_defaults(default_cfg=default_cfg)
        subparser.add_argument(
            "--verbose",
            action="store_true",
            help="Show ComfyUI warnings and progress output while running.",
        )

    wan22_i2v = subparsers.add_parser("wan22-i2v", help="Animate an input image with WAN 2.2.")
    add_wan22_common(wan22_i2v, default_cfg=DEFAULT_WAN22_I2V_CFG)
    wan22_i2v.add_argument("--input", type=_path, required=True)
    wan22_i2v.add_argument("--prompt", required=True)

    wan22_flf2v = subparsers.add_parser("wan22-flf2v", help="Generate a WAN 2.2 first/last-frame video.")
    add_wan22_common(wan22_flf2v, default_cfg=DEFAULT_WAN22_FLF2V_CFG)
    wan22_flf2v.add_argument("--first", type=_path, required=True)
    wan22_flf2v.add_argument("--last", type=_path, required=True)
    wan22_flf2v.add_argument("--prompt", required=True)

    wan22_s2v = subparsers.add_parser("wan22-s2v", help="Generate a WAN 2.2 sound-to-video clip from image and audio.")
    wan22_s2v.add_argument("--models-dir", type=_path, default=None)
    wan22_s2v.add_argument("--out", type=_path, default=DEFAULT_OUT)
    wan22_s2v.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write a comfy-media run manifest for this generation.",
    )
    wan22_s2v.add_argument("--unet", type=_path, default=None)
    wan22_s2v.add_argument("--text-encoder", type=_path, default=None)
    wan22_s2v.add_argument("--audio-encoder", type=_path, default=None)
    wan22_s2v.add_argument("--vae", type=_path, default=None)
    wan22_s2v.add_argument("--lora", type=_path, default=None, help="Optional model-only S2V LoRA path.")
    wan22_s2v.add_argument("--lora-strength", type=float, default=None)
    wan22_s2v.add_argument("--width", type=int, default=None)
    wan22_s2v.add_argument("--height", type=int, default=None)
    wan22_s2v.add_argument("--length", type=int, default=None)
    wan22_s2v.add_argument("--chunk-length", type=int, default=None)
    wan22_s2v.add_argument("--fps", type=int, default=None)
    wan22_s2v.add_argument("--steps", type=int, default=None)
    wan22_s2v.add_argument("--cfg", type=float, default=None)
    wan22_s2v.add_argument("--sampler", default=None)
    wan22_s2v.add_argument("--scheduler", default=None)
    wan22_s2v.add_argument("--shift", type=float, default=None)
    wan22_s2v.add_argument("--seed", type=int, default=DEFAULT_SEED)
    wan22_s2v.add_argument("--negative-prompt", default=DEFAULT_WAN22_NEGATIVE_PROMPT)
    wan22_s2v.add_argument("--audio-start-time", type=float, default=DEFAULT_AUDIO_START_TIME)
    wan22_s2v.add_argument("--audio-duration", type=float, default=None)
    wan22_s2v.add_argument(
        "--verbose",
        action="store_true",
        help="Show ComfyUI warnings and progress output while running.",
    )
    wan22_s2v.add_argument("--input", type=_path, required=True)
    wan22_s2v.add_argument("--audio", type=_path, required=True)
    wan22_s2v.add_argument("--prompt", required=True)

    wan22_video_audio = subparsers.add_parser(
        "wan22-video-audio",
        help="Process an input video with audio using WAN 2.2 S2V video-to-video presets.",
    )
    wan22_video_audio.add_argument("--models-dir", type=_path, default=None)
    wan22_video_audio.add_argument("--out", type=_path, default=DEFAULT_OUT)
    wan22_video_audio.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write a comfy-media run manifest for this generation.",
    )
    wan22_video_audio.add_argument("--mode", required=True, choices=["audio-driven", "lipsync"])
    wan22_video_audio.add_argument("--input-video", type=_path, required=True)
    wan22_video_audio.add_argument("--audio", type=_path, required=True)
    wan22_video_audio.add_argument("--mask-video", type=_path, default=None)
    wan22_video_audio.add_argument("--mask-image", type=_path, default=None)
    wan22_video_audio.add_argument("--prompt", default=None)
    wan22_video_audio.add_argument("--unet", type=_path, default=None)
    wan22_video_audio.add_argument("--text-encoder", type=_path, default=None)
    wan22_video_audio.add_argument("--audio-encoder", type=_path, default=None)
    wan22_video_audio.add_argument("--vae", type=_path, default=None)
    wan22_video_audio.add_argument("--chunk-length", type=int, default=None)
    wan22_video_audio.add_argument("--chunk-overlap", type=int, default=None)
    wan22_video_audio.add_argument("--steps", type=int, default=None)
    wan22_video_audio.add_argument("--denoise", type=float, default=None)
    wan22_video_audio.add_argument("--cfg", type=float, default=None)
    wan22_video_audio.add_argument("--sampler", default=None)
    wan22_video_audio.add_argument("--scheduler", default=None)
    wan22_video_audio.add_argument("--shift", type=float, default=None)
    wan22_video_audio.add_argument("--seed", type=int, default=DEFAULT_SEED)
    wan22_video_audio.add_argument("--negative-prompt", default=None)
    wan22_video_audio.add_argument("--audio-start-time", type=float, default=DEFAULT_AUDIO_START_TIME)
    wan22_video_audio.add_argument(
        "--verbose",
        action="store_true",
        help="Show ComfyUI warnings and progress output while running.",
    )

    def add_seedance2_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--out", type=_path, default=DEFAULT_OUT)
        subparser.add_argument(
            "--no-manifest",
            action="store_true",
            help="Do not write a comfy-media run manifest for this generation.",
        )
        subparser.add_argument("--model", default=SEEDANCE2_MODEL, choices=[SEEDANCE2_MODEL])
        subparser.add_argument("--resolution", default=DEFAULT_SEEDANCE2_RESOLUTION, choices=["480p", "720p", "1080p"])
        subparser.add_argument("--ratio", default=DEFAULT_SEEDANCE2_RATIO, choices=["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"])
        subparser.add_argument("--duration", type=int, default=DEFAULT_SEEDANCE2_DURATION)
        subparser.add_argument(
            "--generate-audio",
            action=argparse.BooleanOptionalAction,
            default=DEFAULT_SEEDANCE2_GENERATE_AUDIO,
            help="Generate audio in the remote Seedance output. Use --no-generate-audio to disable.",
        )
        subparser.add_argument("--watermark", action="store_true", default=DEFAULT_SEEDANCE2_WATERMARK)
        subparser.add_argument("--seed", type=int, default=DEFAULT_SEEDANCE2_SEED)
        subparser.add_argument(
            "--verbose",
            action="store_true",
            help="Show ComfyUI API node logs and progress output while running.",
        )

    seedance2_t2v = subparsers.add_parser("seedance2-t2v", help="Generate a remote Seedance 2.0 video from text.")
    add_seedance2_common(seedance2_t2v)
    seedance2_t2v.add_argument("--prompt", required=True)

    seedance2_r2v = subparsers.add_parser("seedance2-r2v", help="Generate a remote Seedance 2.0 video from a reference image.")
    add_seedance2_common(seedance2_r2v)
    seedance2_r2v.add_argument("--input", type=_path, required=True)
    seedance2_r2v.add_argument("--prompt", required=True)

    seedance2_flf2v = subparsers.add_parser("seedance2-flf2v", help="Generate a remote Seedance 2.0 first/last-frame video.")
    add_seedance2_common(seedance2_flf2v)
    seedance2_flf2v.add_argument("--first", type=_path, required=True)
    seedance2_flf2v.add_argument("--last", type=_path, required=True)
    seedance2_flf2v.add_argument("--prompt", required=True)

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
        ic_lora=args.ic_lora if args.ic_lora is not None else profile.models.get("ic_lora", DEFAULT_IC_LORA),
        width=args.width if args.width is not None else int(profile.defaults.get("width", DEFAULT_WIDTH)),
        height=args.height if args.height is not None else int(profile.defaults.get("height", DEFAULT_HEIGHT)),
        length=args.length if args.length is not None else int(profile.defaults.get("length", DEFAULT_LENGTH)),
        fps=args.fps if args.fps is not None else int(profile.defaults.get("fps", DEFAULT_FPS)),
        cfg=args.cfg if args.cfg is not None else float(profile.defaults.get("cfg", DEFAULT_CFG)),
        seed=args.seed,
        audio_start_time=getattr(args, "audio_start_time", DEFAULT_AUDIO_START_TIME),
        audio_duration=getattr(args, "audio_duration", None),
        negative_prompt=args.negative_prompt,
        attention_strength=(
            args.attention_strength
            if getattr(args, "attention_strength", None) is not None
            else float(profile.defaults.get("attention_strength", DEFAULT_ATTENTION_STRENGTH))
        ),
        reference_downscale=(
            args.reference_downscale
            if getattr(args, "reference_downscale", None) is not None
            else float(profile.defaults.get("reference_downscale", DEFAULT_REFERENCE_DOWNSCALE))
        ),
        extra_loras=list(getattr(args, "extra_lora", []) or []),
    )


def _seedance2_config(args: argparse.Namespace, profile: ResolvedProfile) -> Seedance2Config:
    return Seedance2Config(
        model=args.model if args.model is not None else str(profile.defaults.get("model", SEEDANCE2_MODEL)),
        resolution=args.resolution if args.resolution is not None else str(profile.defaults.get("resolution", DEFAULT_SEEDANCE2_RESOLUTION)),
        ratio=args.ratio if args.ratio is not None else str(profile.defaults.get("ratio", DEFAULT_SEEDANCE2_RATIO)),
        duration=args.duration if args.duration is not None else int(profile.defaults.get("duration", DEFAULT_SEEDANCE2_DURATION)),
        generate_audio=(
            args.generate_audio
            if args.generate_audio is not None
            else bool(profile.defaults.get("generate_audio", DEFAULT_SEEDANCE2_GENERATE_AUDIO))
        ),
        watermark=args.watermark if args.watermark is not None else bool(profile.defaults.get("watermark", DEFAULT_SEEDANCE2_WATERMARK)),
        seed=args.seed if args.seed is not None else int(profile.defaults.get("seed", DEFAULT_SEEDANCE2_SEED)),
    )


def _wan22_config(args: argparse.Namespace, profile: ResolvedProfile) -> Wan22Config:
    command_default_cfg = DEFAULT_WAN22_FLF2V_CFG if args.command == "wan22-flf2v" else DEFAULT_WAN22_I2V_CFG
    profile_cfg_key = "flf2v_cfg" if args.command == "wan22-flf2v" else "i2v_cfg"
    steps, high_steps, low_steps = _wan22_step_counts(args, profile)
    return Wan22Config(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        unet_high=args.unet_high if args.unet_high is not None else profile.models.get("unet_high", DEFAULT_WAN22_UNET_HIGH),
        unet_low=args.unet_low if args.unet_low is not None else profile.models.get("unet_low", DEFAULT_WAN22_UNET_LOW),
        text_encoder=args.text_encoder if args.text_encoder is not None else profile.models.get("text_encoder", DEFAULT_WAN22_TEXT_ENCODER),
        vae=args.vae if args.vae is not None else profile.models.get("vae", DEFAULT_WAN22_VAE),
        width=args.width if args.width is not None else int(profile.defaults.get("width", DEFAULT_WAN22_WIDTH)),
        height=args.height if args.height is not None else int(profile.defaults.get("height", DEFAULT_WAN22_HEIGHT)),
        length=args.length if args.length is not None else int(profile.defaults.get("length", DEFAULT_WAN22_LENGTH)),
        fps=args.fps if args.fps is not None else int(profile.defaults.get("fps", DEFAULT_WAN22_FPS)),
        steps=steps,
        high_steps=high_steps,
        low_steps=low_steps,
        cfg=args.cfg if args.cfg is not None else float(profile.defaults.get(profile_cfg_key, command_default_cfg)),
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )


def _wan22_s2v_config(args: argparse.Namespace, profile: ResolvedProfile) -> Wan22S2VConfig:
    return Wan22S2VConfig(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        unet=args.unet if args.unet is not None else profile.models.get("unet", DEFAULT_WAN22_S2V_UNET),
        text_encoder=args.text_encoder if args.text_encoder is not None else profile.models.get("text_encoder", DEFAULT_WAN22_TEXT_ENCODER),
        audio_encoder=args.audio_encoder if args.audio_encoder is not None else profile.models.get("audio_encoder", DEFAULT_WAN22_AUDIO_ENCODER),
        vae=args.vae if args.vae is not None else profile.models.get("vae", DEFAULT_WAN22_VAE),
        lora=args.lora if args.lora is not None else profile.models.get("lora"),
        width=args.width if args.width is not None else int(profile.defaults.get("width", DEFAULT_WAN22_WIDTH)),
        height=args.height if args.height is not None else int(profile.defaults.get("height", DEFAULT_WAN22_HEIGHT)),
        length=args.length if args.length is not None else int(profile.defaults.get("length", DEFAULT_WAN22_S2V_LENGTH)),
        chunk_length=(
            args.chunk_length
            if args.chunk_length is not None
            else int(profile.defaults.get("chunk_length", DEFAULT_WAN22_S2V_CHUNK_LENGTH))
        ),
        fps=args.fps if args.fps is not None else int(profile.defaults.get("fps", DEFAULT_WAN22_FPS)),
        steps=args.steps if args.steps is not None else int(profile.defaults.get("steps", DEFAULT_WAN22_STEPS)),
        cfg=args.cfg if args.cfg is not None else float(profile.defaults.get("cfg", DEFAULT_WAN22_S2V_CFG)),
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        sampler=args.sampler if args.sampler is not None else str(profile.defaults.get("sampler", DEFAULT_WAN22_S2V_SAMPLER)),
        scheduler=args.scheduler if args.scheduler is not None else str(profile.defaults.get("scheduler", DEFAULT_WAN22_S2V_SCHEDULER)),
        shift=args.shift if args.shift is not None else float(profile.defaults.get("shift", DEFAULT_WAN22_S2V_SHIFT)),
        lora_strength=(
            args.lora_strength
            if args.lora_strength is not None
            else float(profile.defaults.get("lora_strength", DEFAULT_WAN22_S2V_LORA_STRENGTH))
        ),
        audio_start_time=args.audio_start_time,
        audio_duration=args.audio_duration,
    )


def _wan22_video_audio_config(args: argparse.Namespace, profile: ResolvedProfile) -> Wan22VideoAudioConfig:
    steps_key = "lipsync_steps" if args.mode == "lipsync" else "steps"
    denoise_key = "lipsync_denoise" if args.mode == "lipsync" else "denoise"
    primary_steps_default = (
        DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_STEPS
        if args.mode == "lipsync"
        else DEFAULT_WAN22_VIDEO_AUDIO_STEPS
    )
    primary_denoise_default = (
        DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_DENOISE
        if args.mode == "lipsync"
        else DEFAULT_WAN22_VIDEO_AUDIO_DENOISE
    )
    steps = args.steps if args.steps is not None else int(profile.defaults.get(steps_key, primary_steps_default))
    denoise = (
        args.denoise
        if args.denoise is not None
        else float(profile.defaults.get(denoise_key, primary_denoise_default))
    )
    return Wan22VideoAudioConfig(
        models_dir=args.models_dir if args.models_dir is not None else profile.models_dir,
        unet=args.unet if args.unet is not None else profile.models.get("unet", DEFAULT_WAN22_S2V_UNET),
        text_encoder=args.text_encoder if args.text_encoder is not None else profile.models.get("text_encoder", DEFAULT_WAN22_TEXT_ENCODER),
        audio_encoder=args.audio_encoder if args.audio_encoder is not None else profile.models.get("audio_encoder", DEFAULT_WAN22_AUDIO_ENCODER),
        vae=args.vae if args.vae is not None else profile.models.get("vae", DEFAULT_WAN22_VAE),
        chunk_length=(
            args.chunk_length
            if args.chunk_length is not None
            else int(profile.defaults.get("chunk_length", DEFAULT_WAN22_S2V_CHUNK_LENGTH))
        ),
        chunk_overlap=(
            args.chunk_overlap
            if args.chunk_overlap is not None
            else int(profile.defaults.get("chunk_overlap", DEFAULT_WAN22_VIDEO_AUDIO_CHUNK_OVERLAP))
        ),
        fps=int(profile.defaults.get("fps", DEFAULT_WAN22_FPS)),
        steps=steps,
        denoise=denoise,
        lipsync_steps=steps if args.mode == "lipsync" else int(profile.defaults.get("lipsync_steps", DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_STEPS)),
        lipsync_denoise=(
            denoise
            if args.mode == "lipsync"
            else float(profile.defaults.get("lipsync_denoise", DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_DENOISE))
        ),
        lipsync_second_steps=int(
            profile.defaults.get("lipsync_second_steps", DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_STEPS)
        ),
        lipsync_second_denoise=float(
            profile.defaults.get("lipsync_second_denoise", DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_SECOND_DENOISE)
        ),
        cfg=args.cfg if args.cfg is not None else float(profile.defaults.get("cfg", 1.0)),
        seed=args.seed,
        sampler=args.sampler if args.sampler is not None else str(profile.defaults.get("sampler", "euler")),
        scheduler=args.scheduler if args.scheduler is not None else str(profile.defaults.get("scheduler", "simple")),
        shift=args.shift if args.shift is not None else float(profile.defaults.get("shift", 10.0)),
        negative_prompt=(
            args.negative_prompt
            if args.negative_prompt is not None
            else str(profile.defaults.get("negative_prompt", ""))
        ),
        audio_start_time=args.audio_start_time,
    )


def _wan22_video_audio_prompt(args: argparse.Namespace, profile: ResolvedProfile) -> str:
    if args.prompt is not None:
        return args.prompt
    default_key = "lipsync_prompt" if args.mode == "lipsync" else "audio_driven_prompt"
    fallback = DEFAULT_WAN22_VIDEO_AUDIO_LIPSYNC_PROMPT if args.mode == "lipsync" else DEFAULT_WAN22_VIDEO_AUDIO_PROMPT
    return str(profile.defaults.get(default_key, fallback))


def _wan22_step_counts(args: argparse.Namespace, profile: ResolvedProfile) -> tuple[int, int, int]:
    profile_steps = int(profile.defaults.get("steps", DEFAULT_WAN22_STEPS))
    total_steps = args.steps if args.steps is not None else profile_steps
    default_high_steps = int(profile.defaults.get("high_steps", total_steps // 2))
    default_low_steps = int(profile.defaults.get("low_steps", total_steps - default_high_steps))

    high_steps = args.high_steps
    low_steps = args.low_steps
    if high_steps is None and low_steps is None:
        if args.steps is None:
            high_steps = default_high_steps
            low_steps = default_low_steps
            total_steps = high_steps + low_steps
        else:
            high_steps = total_steps // 2
            low_steps = total_steps - high_steps
    elif high_steps is None:
        low_steps = int(low_steps)
        high_steps = total_steps - low_steps
    elif low_steps is None:
        high_steps = int(high_steps)
        low_steps = total_steps - high_steps
    else:
        high_steps = int(high_steps)
        low_steps = int(low_steps)
        if args.steps is None:
            total_steps = high_steps + low_steps
        elif high_steps + low_steps != total_steps:
            raise ValueError("--steps must equal --high-steps + --low-steps when all three are provided")

    if total_steps <= 0:
        raise ValueError("steps must be greater than 0")
    if high_steps <= 0:
        raise ValueError("high_steps must be greater than 0")
    if low_steps <= 0:
        raise ValueError("low_steps must be greater than 0")
    if high_steps + low_steps != total_steps:
        raise ValueError("high_steps + low_steps must equal steps")
    return total_steps, high_steps, low_steps


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
    control_video_path: Path | None = None,
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
    if control_video_path is not None:
        payload["control_video"] = str(control_video_path)
        payload["ic_lora"] = str(config.resolve_model_path(config.ic_lora))
        payload["attention_strength"] = config.attention_strength
        payload["reference_downscale"] = config.reference_downscale
    if first_path is not None:
        payload["first"] = str(first_path)
    if last_path is not None:
        payload["last"] = str(last_path)
    return payload


def _seedance2_success(
    *,
    mode: str,
    artifact: Path,
    config: Seedance2Config,
    profile: ResolvedProfile,
    input_path: Path | None = None,
    first_path: Path | None = None,
    last_path: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "video",
        "mode": mode,
        "remote": True,
        "provider": SEEDANCE2_PROVIDER,
        "artifacts": [str(artifact)],
        "seed": config.seed,
        "model": config.model,
        "resolution": config.resolution,
        "ratio": config.ratio,
        "duration_seconds": config.duration,
        "generate_audio": config.generate_audio,
        "watermark": config.watermark,
        "capability": profile.capability,
        "model_profile": profile.name,
        "architecture": profile.architecture,
        "resolved_models": {},
    }
    if input_path is not None:
        payload["input"] = str(input_path)
    if first_path is not None:
        payload["first"] = str(first_path)
    if last_path is not None:
        payload["last"] = str(last_path)
    return payload


def _wan22_success(
    *,
    mode: str,
    artifact: Path,
    config: Wan22Config,
    frames: list[object],
    profile: ResolvedProfile,
    input_path: Path | None = None,
    first_path: Path | None = None,
    last_path: Path | None = None,
) -> dict[str, Any]:
    meta = frame_metadata(frames, config.fps)
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "video",
        "mode": mode,
        "artifacts": [str(artifact)],
        "seed": config.seed,
        "models_dir": str(config.models_dir),
        "model": config.unet_high.name,
        "width": meta["width"],
        "height": meta["height"],
        "frames": meta["frames"],
        "fps": meta["fps"],
        "duration_seconds": meta["duration_seconds"],
        "steps": config.steps,
        "high_steps": config.high_steps,
        "low_steps": config.low_steps,
        "cfg": config.cfg,
        "audio_muxed": False,
        "capability": profile.capability,
        "model_profile": profile.name,
        "architecture": profile.architecture,
        "resolved_models": _resolved_wan22_models(config),
    }
    if input_path is not None:
        payload["input"] = str(input_path)
    if first_path is not None:
        payload["first"] = str(first_path)
    if last_path is not None:
        payload["last"] = str(last_path)
    return payload


def _wan22_s2v_success(
    *,
    mode: str,
    artifact: Path,
    config: Wan22S2VConfig,
    frames: list[object],
    profile: ResolvedProfile,
    input_path: Path,
    audio_path: Path,
) -> dict[str, Any]:
    meta = frame_metadata(frames, config.fps)
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "video",
        "mode": mode,
        "artifacts": [str(artifact)],
        "seed": config.seed,
        "models_dir": str(config.models_dir),
        "model": config.unet.name,
        "width": meta["width"],
        "height": meta["height"],
        "frames": meta["frames"],
        "fps": meta["fps"],
        "duration_seconds": meta["duration_seconds"],
        "steps": config.steps,
        "cfg": config.cfg,
        "chunk_length": config.chunk_length,
        "sampler": config.sampler,
        "scheduler": config.scheduler,
        "shift": config.shift,
        "audio_muxed": True,
        "audio_conditioned": True,
        "input": str(input_path),
        "audio_input": str(audio_path),
        "audio_start_time": config.audio_start_time,
        "audio_duration_seconds": (
            config.audio_duration if config.audio_duration is not None else meta["duration_seconds"]
        ),
        "capability": profile.capability,
        "model_profile": profile.name,
        "architecture": profile.architecture,
        "resolved_models": _resolved_wan22_s2v_models(config),
    }
    if config.lora is not None:
        payload["lora"] = str(config.resolve_model_path(config.lora))
        payload["lora_strength"] = config.lora_strength
    return payload


def _wan22_video_audio_success(
    *,
    artifact: Path,
    config: Wan22VideoAudioConfig,
    frames: list[object],
    profile: ResolvedProfile,
    video_audio_mode: str,
    input_video: Path,
    audio_path: Path,
    chunks: list[dict[str, Any]],
    mask_video: Path | None = None,
    mask_image: Path | None = None,
) -> dict[str, Any]:
    meta = frame_metadata(frames, config.fps)
    payload: dict[str, Any] = {
        "ok": True,
        "kind": "video",
        "mode": "wan22-video-audio",
        "video_audio_mode": video_audio_mode,
        "artifacts": [str(artifact)],
        "seed": config.seed,
        "models_dir": str(config.models_dir),
        "model": config.unet.name,
        "width": meta["width"],
        "height": meta["height"],
        "frames": meta["frames"],
        "fps": meta["fps"],
        "duration_seconds": meta["duration_seconds"],
        "chunk_length": config.chunk_length,
        "chunk_overlap": config.chunk_overlap,
        "chunks": chunks,
        "cfg": config.cfg,
        "sampler": config.sampler,
        "scheduler": config.scheduler,
        "shift": config.shift,
        "steps": config.lipsync_steps if video_audio_mode == "lipsync" else config.steps,
        "denoise": config.lipsync_denoise if video_audio_mode == "lipsync" else config.denoise,
        "audio_muxed": True,
        "audio_conditioned": True,
        "input_video": str(input_video),
        "audio_input": str(audio_path),
        "audio_start_time": config.audio_start_time,
        "audio_duration_seconds": meta["duration_seconds"],
        "capability": profile.capability,
        "model_profile": profile.name,
        "architecture": profile.architecture,
        "resolved_models": _resolved_wan22_video_audio_models(config),
    }
    if video_audio_mode == "lipsync":
        payload["lipsync_second_steps"] = config.lipsync_second_steps
        payload["lipsync_second_denoise"] = config.lipsync_second_denoise
    if mask_video is not None:
        payload["mask_input"] = str(mask_video)
        payload["mask_kind"] = "video"
    if mask_image is not None:
        payload["mask_input"] = str(mask_image)
        payload["mask_kind"] = "image"
    return payload


def _resolved_video_models(config: VideogenConfig) -> dict[str, str]:
    return {
        "checkpoint": str(config.resolve_model_path(config.checkpoint)),
        "text_encoder": str(config.resolve_model_path(config.text_encoder)),
        "distilled_lora": str(config.resolve_model_path(config.distilled_lora)),
        "te_lora": str(config.resolve_model_path(config.te_lora)),
        "upscaler": str(config.resolve_model_path(config.upscaler)),
        "ic_lora": str(config.resolve_model_path(config.ic_lora)),
    }


def _resolved_wan22_models(config: Wan22Config) -> dict[str, str]:
    return {
        "unet_high": str(config.resolve_model_path(config.unet_high)),
        "unet_low": str(config.resolve_model_path(config.unet_low)),
        "text_encoder": str(config.resolve_model_path(config.text_encoder)),
        "vae": str(config.resolve_model_path(config.vae)),
    }


def _resolved_wan22_s2v_models(config: Wan22S2VConfig) -> dict[str, str]:
    models = {
        "unet": str(config.resolve_model_path(config.unet)),
        "text_encoder": str(config.resolve_model_path(config.text_encoder)),
        "audio_encoder": str(config.resolve_model_path(config.audio_encoder)),
        "vae": str(config.resolve_model_path(config.vae)),
    }
    if config.lora is not None:
        models["lora"] = str(config.resolve_model_path(config.lora))
    return models


def _resolved_wan22_video_audio_models(config: Wan22VideoAudioConfig) -> dict[str, str]:
    return {
        "unet": str(config.resolve_model_path(config.unet)),
        "text_encoder": str(config.resolve_model_path(config.text_encoder)),
        "audio_encoder": str(config.resolve_model_path(config.audio_encoder)),
        "vae": str(config.resolve_model_path(config.vae)),
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
    if "ic-lora" in message and "helper" in message:
        return "missing_dependency"
    if "mux" in message or "aac" in message or "audio with waveform" in message:
        return "audio_mux"
    if isinstance(error, ProfileError):
        return error.error_type
    if isinstance(error, Seedance2Error):
        return error.error_type
    if "api key" in message or "comfy_org_api_key" in message:
        return "auth_required"
    if "seedance 2.0 api" in message or "api request" in message:
        return "remote_api_error"
    return "error"


def _write_result_video(result: dict[str, Any], out_dir: Path, *, prefix: str, fps: int) -> Path:
    frames = result.get("frames")
    audio = result.get("audio")
    if not isinstance(frames, list):
        raise ValueError("pipeline result did not include a frames list")
    path = make_video_path(out_dir, prefix=prefix)
    save_mp4_with_audio(frames, audio, path, fps)
    return path


def _write_result_video_no_audio(result: dict[str, Any], out_dir: Path, *, prefix: str, fps: int) -> Path:
    frames = result.get("frames")
    if not isinstance(frames, list):
        raise ValueError("pipeline result did not include a frames list")
    path = make_video_path(out_dir, prefix=prefix)
    save_mp4(frames, path, fps)
    return path


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed comfy-videogen command and return its JSON payload."""
    profile, _source = resolve_capability(_capability(args.command))

    if args.command == "seedance2-t2v":
        config = _seedance2_config(args, profile)
        with _maybe_silence(not args.verbose):
            result = run_seedance2_t2v(prompt=args.prompt, config=config, out_dir=args.out)
        return _seedance2_success(
            mode="seedance2-t2v",
            artifact=result["artifact"],
            config=config,
            profile=profile,
        )

    if args.command == "seedance2-r2v":
        if not args.input.is_file():
            raise FileNotFoundError(f"input image not found: {args.input}")
        config = _seedance2_config(args, profile)
        with _maybe_silence(not args.verbose):
            result = run_seedance2_r2v(image=args.input, prompt=args.prompt, config=config, out_dir=args.out)
        return _seedance2_success(
            mode="seedance2-r2v",
            artifact=result["artifact"],
            config=config,
            profile=profile,
            input_path=args.input,
        )

    if args.command == "seedance2-flf2v":
        if not args.first.is_file():
            raise FileNotFoundError(f"first image not found: {args.first}")
        if not args.last.is_file():
            raise FileNotFoundError(f"last image not found: {args.last}")
        config = _seedance2_config(args, profile)
        with _maybe_silence(not args.verbose):
            result = run_seedance2_flf2v(first_image=args.first, last_image=args.last, prompt=args.prompt, config=config, out_dir=args.out)
        return _seedance2_success(
            mode="seedance2-flf2v",
            artifact=result["artifact"],
            config=config,
            profile=profile,
            first_path=args.first,
            last_path=args.last,
        )

    if args.command == "wan22-i2v":
        if not args.input.is_file():
            raise FileNotFoundError(f"input image not found: {args.input}")
        config = _wan22_config(args, profile)
        with _maybe_silence(not args.verbose):
            result = run_wan22_i2v(image=args.input, prompt=args.prompt, config=config)
            artifact = _write_result_video_no_audio(result, args.out, prefix="comfy-videogen-wan22-i2v", fps=config.fps)
        return _wan22_success(
            mode="wan22-i2v",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            profile=profile,
            input_path=args.input,
        )

    if args.command == "wan22-flf2v":
        if not args.first.is_file():
            raise FileNotFoundError(f"first image not found: {args.first}")
        if not args.last.is_file():
            raise FileNotFoundError(f"last image not found: {args.last}")
        config = _wan22_config(args, profile)
        with _maybe_silence(not args.verbose):
            result = run_wan22_flf2v(first_image=args.first, last_image=args.last, prompt=args.prompt, config=config)
            artifact = _write_result_video_no_audio(result, args.out, prefix="comfy-videogen-wan22-flf2v", fps=config.fps)
        return _wan22_success(
            mode="wan22-flf2v",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            profile=profile,
            first_path=args.first,
            last_path=args.last,
        )

    if args.command == "wan22-s2v":
        if not args.input.is_file():
            raise FileNotFoundError(f"input image not found: {args.input}")
        if not args.audio.is_file():
            raise FileNotFoundError(f"input audio not found: {args.audio}")
        config = _wan22_s2v_config(args, profile)
        with _maybe_silence(not args.verbose):
            result = run_wan22_s2v(image=args.input, audio=args.audio, prompt=args.prompt, config=config)
            artifact = _write_result_video(result, args.out, prefix="comfy-videogen-wan22-s2v", fps=config.fps)
        return _wan22_s2v_success(
            mode="wan22-s2v",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            profile=profile,
            input_path=args.input,
            audio_path=args.audio,
        )

    if args.command == "wan22-video-audio":
        if not args.input_video.is_file():
            raise FileNotFoundError(f"input video not found: {args.input_video}")
        if not args.audio.is_file():
            raise FileNotFoundError(f"input audio not found: {args.audio}")
        if args.mode == "lipsync":
            if args.mask_video is None and args.mask_image is None:
                raise ValueError("lipsync mode requires --mask-video or --mask-image")
            if args.mask_video is not None and not args.mask_video.is_file():
                raise FileNotFoundError(f"mask video not found: {args.mask_video}")
            if args.mask_image is not None and not args.mask_image.is_file():
                raise FileNotFoundError(f"mask image not found: {args.mask_image}")
        config = _wan22_video_audio_config(args, profile)
        prompt = _wan22_video_audio_prompt(args, profile)
        with _maybe_silence(not args.verbose):
            result = run_wan22_video_audio(
                video=args.input_video,
                audio=args.audio,
                mode=args.mode,
                prompt=prompt,
                config=config,
                mask_video=args.mask_video,
                mask_image=args.mask_image,
            )
            artifact = _write_result_video(result, args.out, prefix="comfy-videogen-wan22-video-audio", fps=config.fps)
        return _wan22_video_audio_success(
            artifact=artifact,
            config=config,
            frames=result["frames"],
            profile=profile,
            video_audio_mode=args.mode,
            input_video=args.input_video,
            audio_path=args.audio,
            chunks=list(result.get("chunks", [])),
            mask_video=args.mask_video,
            mask_image=args.mask_image,
        )

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

    if args.command == "motion-track":
        if not args.input.is_file():
            raise FileNotFoundError(f"input image not found: {args.input}")
        if not args.control_video.is_file():
            raise FileNotFoundError(f"control video not found: {args.control_video}")
        with _maybe_silence(not args.verbose):
            result = run_motion_track(
                image=args.input,
                control_video=args.control_video,
                prompt=args.prompt,
                config=config,
            )
            artifact = _write_result_video(result, args.out, prefix="comfy-videogen-motion-track", fps=config.fps)
        return _success(
            mode="motion-track",
            artifact=artifact,
            config=config,
            frames=result["frames"],
            input_path=args.input,
            control_video_path=args.control_video,
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
            payload["manifests"] = [str(write_run_manifest(out_dir=args.out, tool="comfy-videogen", payload=payload, args=args))]
    except Exception as exc:
        payload = _error(mode=mode, error=exc)
        print(json.dumps(payload, indent=2))
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
