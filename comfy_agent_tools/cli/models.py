"""CLI for local model profile configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from comfy_agent_tools.profiles import (
    BUILTIN_PROFILES,
    CONFIG_FILENAME,
    ProfileError,
    all_profile_names,
    config_path,
    default_config,
    load_config,
    make_resolved_profile,
    profiles_by_architecture,
    resolve_capability,
    resolve_profile,
    validate_defaults,
    validate_profile_files,
    write_config,
)
from comfy_agent_tools.downloads import download_profile_models


def build_parser() -> argparse.ArgumentParser:
    """Build the comfy-models parser."""
    parser = argparse.ArgumentParser(
        prog="comfy-models",
        description="Configure local model profiles and defaults.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available model profiles grouped by architecture.")
    subparsers.add_parser("show", help="Show effective local profile configuration.")
    subparsers.add_parser("init", help=f"Create {CONFIG_FILENAME} with built-in defaults.")
    subparsers.add_parser("validate", help="Validate effective defaults and model files.")
    download = subparsers.add_parser("download", help="Download missing model files for a capability.")
    download.add_argument("capability")
    download.add_argument("--dry-run", action="store_true", help="Show missing files without downloading.")
    download.add_argument("--yes", action="store_true", help="Confirm downloading missing files.")

    download_profile = subparsers.add_parser("download-profile", help="Download missing model files for a profile.")
    download_profile.add_argument("profile")
    download_profile.add_argument("--dry-run", action="store_true", help="Show missing files without downloading.")
    download_profile.add_argument("--yes", action="store_true", help="Confirm downloading missing files.")

    reset = subparsers.add_parser("reset", help=f"Remove {CONFIG_FILENAME}.")
    reset.add_argument("--yes", action="store_true", help="Confirm reset without prompting.")

    set_models_dir = subparsers.add_parser("set-models-dir", help="Set global models_dir.")
    set_models_dir.add_argument("models_dir")

    set_default = subparsers.add_parser("set-default", help="Set default profile for a capability.")
    set_default.add_argument("capability")
    set_default.add_argument("profile")

    add_profile = subparsers.add_parser("add-profile", help="Add a local profile extending an existing profile.")
    add_profile.add_argument("name")
    add_profile.add_argument("--extends", required=True, dest="extends_profile")
    add_profile.add_argument("--label")
    add_profile.add_argument("--architecture")
    add_profile.add_argument("--models-dir")
    add_profile.add_argument("--checkpoint")
    add_profile.add_argument("--unet")
    add_profile.add_argument("--uncond-unet")
    add_profile.add_argument("--clip")
    add_profile.add_argument("--clip-0-6b")
    add_profile.add_argument("--clip-1-7b")
    add_profile.add_argument("--vae")
    add_profile.add_argument("--lora")
    add_profile.add_argument("--text-encoder")
    add_profile.add_argument("--audio-encoder")
    add_profile.add_argument("--distilled-lora")
    add_profile.add_argument("--te-lora")
    add_profile.add_argument("--upscaler")
    add_profile.add_argument("--ic-lora")
    add_profile.add_argument("--unet-high")
    add_profile.add_argument("--unet-low")

    validate_profile = subparsers.add_parser("validate-profile", help="Validate one resolved profile and its files.")
    validate_profile.add_argument("profile")

    return parser


def _success(payload: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, **payload}


def _error(error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(error),
        "error_type": getattr(error, "error_type", "error"),
    }


def _load_or_default() -> tuple[dict[str, Any], str]:
    return load_config()


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed comfy-models command and return a JSON payload."""
    if args.command == "list":
        config, source = _load_or_default()
        profiles = {}
        for name in all_profile_names(config):
            profile = resolve_profile(name, config)
            profiles[name] = {
                "label": profile.get("label", name),
                "architecture": profile["architecture"],
                "supports": profile.get("supports", []),
                "builtin": name in BUILTIN_PROFILES,
            }
        return _success(
            {
                "config_source": source,
                "architectures": profiles_by_architecture(config),
                "profiles": profiles,
            }
        )

    if args.command == "show":
        config, source = _load_or_default()
        resolved = {}
        for capability in dict(config.get("defaults", {})):
            profile, profile_source = resolve_capability(capability)
            resolved[capability] = {
                "profile": profile.name,
                "architecture": profile.architecture,
                "models_dir": str(profile.models_dir),
                "source": profile_source,
            }
        return _success(
            {
                "config_path": str(config_path()),
                "config_source": source,
                "models_dir": str(config.get("models_dir")),
                "defaults": config.get("defaults", {}),
                "resolved_defaults": resolved,
            }
        )

    if args.command == "init":
        path = write_config(default_config())
        return _success({"config_path": str(path), "created": True})

    if args.command == "set-models-dir":
        config, _source = _load_or_default()
        config["models_dir"] = args.models_dir
        path = write_config(config)
        return _success({"config_path": str(path), "models_dir": args.models_dir})

    if args.command == "set-default":
        config, _source = _load_or_default()
        profile = resolve_profile(args.profile, config)
        if args.capability not in list(profile.get("supports", [])):
            from comfy_agent_tools.profiles import UnsupportedCapabilityError

            raise UnsupportedCapabilityError(f"profile '{args.profile}' does not support {args.capability}")
        config.setdefault("defaults", {})[args.capability] = args.profile
        path = write_config(config)
        return _success({"config_path": str(path), "capability": args.capability, "profile": args.profile})

    if args.command == "add-profile":
        config, _source = _load_or_default()
        base = resolve_profile(args.extends_profile, config)
        architecture = args.architecture or base["architecture"]
        models = _models_from_args(args)
        profile: dict[str, Any] = {
            "extends": args.extends_profile,
            "architecture": architecture,
        }
        if args.label:
            profile["label"] = args.label
        if args.models_dir:
            profile["models_dir"] = args.models_dir
        if models:
            profile["models"] = models
        config.setdefault("profiles", {})[args.name] = profile
        resolve_profile(args.name, config)
        path = write_config(config)
        return _success({"config_path": str(path), "profile": args.name, "extends": args.extends_profile})

    if args.command == "validate":
        config, source = _load_or_default()
        validate_defaults(config)
        resolved = {}
        for capability in dict(config.get("defaults", {})):
            profile, _profile_source = resolve_capability(capability)
            validate_profile_files(profile)
            resolved[capability] = profile.name
        return _success({"config_source": source, "validated_defaults": resolved})

    if args.command == "download":
        if not args.dry_run and not args.yes:
            raise ProfileError("download requires --yes unless --dry-run is used")
        profile, _source = resolve_capability(args.capability)
        result = download_profile_models(profile, dry_run=args.dry_run)
        return _success(
            {
                "mode": "download",
                "capability": args.capability,
                "model_profile": profile.name,
                "architecture": profile.architecture,
                "models_dir": str(profile.models_dir),
                **result,
            }
        )

    if args.command == "download-profile":
        if not args.dry_run and not args.yes:
            raise ProfileError("download-profile requires --yes unless --dry-run is used")
        config, _source = _load_or_default()
        raw = resolve_profile(args.profile, config)
        supports = list(raw.get("supports", []))
        if not supports:
            raise ProfileError(f"profile '{args.profile}' does not declare supported capabilities")
        profile = make_resolved_profile(supports[0], args.profile, raw, config)
        result = download_profile_models(profile, dry_run=args.dry_run)
        return _success(
            {
                "mode": "download",
                "capability": profile.capability,
                "model_profile": profile.name,
                "architecture": profile.architecture,
                "models_dir": str(profile.models_dir),
                **result,
            }
        )

    if args.command == "validate-profile":
        config, source = _load_or_default()
        raw = resolve_profile(args.profile, config)
        supports = list(raw.get("supports", []))
        if not supports:
            raise ProfileError(f"profile '{args.profile}' does not declare supported capabilities")
        capability = supports[0]
        resolved = make_resolved_profile(capability, args.profile, raw, config)
        validate_profile_files(resolved)
        return _success({"config_source": source, "profile": args.profile, "capability": capability})

    if args.command == "reset":
        path = config_path()
        if path.exists():
            path.unlink()
        return _success({"config_path": str(path), "removed": True})

    raise ValueError(f"unknown command: {args.command}")


def _models_from_args(args: argparse.Namespace) -> dict[str, str]:
    mapping = {
        "checkpoint": args.checkpoint,
        "unet": args.unet,
        "uncond_unet": args.uncond_unet,
        "clip": args.clip,
        "clip_0_6b": args.clip_0_6b,
        "clip_1_7b": args.clip_1_7b,
        "vae": args.vae,
        "lora": args.lora,
        "text_encoder": args.text_encoder,
        "audio_encoder": args.audio_encoder,
        "distilled_lora": args.distilled_lora,
        "te_lora": args.te_lora,
        "upscaler": args.upscaler,
        "ic_lora": args.ic_lora,
        "unet_high": args.unet_high,
        "unet_low": args.unet_low,
    }
    return {key: value for key, value in mapping.items() if value}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = run_command(args)
    except Exception as exc:
        payload = _error(exc)
        print(json.dumps(payload, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
