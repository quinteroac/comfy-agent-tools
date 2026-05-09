"""Model profile registry and local defaults for comfy-agent-tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CONFIG_FILENAME = ".comfy-agent-tools.json"
BUILTIN_MODELS_DIR = Path("/mnt/models/comfyui")

BUILTIN_DEFAULTS: dict[str, str] = {
    "imagegen.generate": "anima-preview3-turbo",
    "imagegen.edit": "qwen-edit2511",
    "imagegen.upscale": "clear-reality",
    "videogen.t2v": "ltx23-10eros",
    "videogen.i2v": "ltx23-10eros",
    "videogen.flf2v": "ltx23-10eros",
    "videogen.ia2av": "ltx23-10eros",
    "videogen.motion-track": "ltx23-motion-track",
    "videogen.seedance2-t2v": "seedance2-api",
    "videogen.seedance2-r2v": "seedance2-api",
    "videogen.seedance2-flf2v": "seedance2-api",
    "imagegen.grok-generate": "grok-imagine-api",
    "imagegen.grok-edit": "grok-imagine-api",
    "musicgen.generate": "ace15-base",
}

BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "qwen-edit2511": {
        "label": "Qwen Image Edit 2511",
        "architecture": "qwen-image-edit",
        "supports": ["imagegen.generate", "imagegen.edit"],
        "models": {
            "unet": "diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
            "clip": "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "vae": "vae/qwen_image_vae.safetensors",
            "lora": "loras/qwen-image-edit/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        },
        "defaults": {
            "steps": 4,
            "cfg": 3.0,
            "width": 1024,
            "height": 1024,
        },
    },
    "clear-reality": {
        "label": "ClearReality 4x upscaler",
        "architecture": "upscale-model",
        "supports": ["imagegen.upscale"],
        "models": {
            "upscaler": "upscale_models/4x-ClearRealityV1.pth",
        },
        "defaults": {},
    },
    "anima-preview3-turbo": {
        "label": "Anima Preview3 Base + Turbo LoRA",
        "architecture": "anima",
        "supports": ["imagegen.generate"],
        "models": {
            "unet": "diffusion_models/animaOfficial_preview3Base.safetensors",
            "clip": "text_encoders/qwen_3_06b_base.safetensors",
            "vae": "vae/qwen_image_vae.safetensors",
            "lora": "loras/anima/anima-turbo-lora-v0.1.safetensors",
        },
        "defaults": {
            "width": 1024,
            "height": 1024,
            "steps": 8,
            "cfg": 1.0,
            "sampler": "er_sde",
            "scheduler": "normal",
        },
    },
    "flux-klein-9b-snofs": {
        "label": "FLUX.2 Klein 9B FP8 + SNOFS LoRA",
        "architecture": "flux-klein",
        "supports": ["imagegen.generate", "imagegen.edit"],
        "models": {
            "unet": "diffusion_models/flux-2-klein-9b-fp8.safetensors",
            "clip": "text_encoders/qwen_3_8b_fp8mixed.safetensors",
            "vae": "vae/flux2-vae.safetensors",
            "lora": "loras/flux-klein/klein_snofs_v1_1.safetensors",
        },
        "defaults": {
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "flux2",
        },
    },
    "ltx23-10eros": {
        "label": "10Eros LTX 2.3",
        "architecture": "ltx23",
        "supports": ["videogen.t2v", "videogen.i2v", "videogen.flf2v", "videogen.ia2av"],
        "models": {
            "checkpoint": "checkpoints/10Eros_v1-fp8mixed_learned.safetensors",
            "text_encoder": "text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            "distilled_lora": "loras/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors",
            "te_lora": "loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            "upscaler": "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        },
        "defaults": {
            "width": 512,
            "height": 320,
            "length": 49,
            "fps": 24,
            "cfg": 1.0,
        },
    },
    "ltx23-motion-track": {
        "label": "10Eros LTX 2.3 Motion Track IC-LoRA",
        "architecture": "ltx23",
        "supports": ["videogen.motion-track"],
        "models": {
            "checkpoint": "checkpoints/10Eros_v1-fp8mixed_learned.safetensors",
            "text_encoder": "text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            "distilled_lora": "loras/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors",
            "te_lora": "loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            "upscaler": "latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "ic_lora": "loras/ltx23/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
        },
        "defaults": {
            "width": 512,
            "height": 320,
            "length": 49,
            "fps": 24,
            "cfg": 1.0,
            "attention_strength": 1.0,
            "reference_downscale": 0.5,
        },
    },
    "ace15-base": {
        "label": "ACE-Step 1.5 Base",
        "architecture": "ace-step-1.5",
        "supports": ["musicgen.generate"],
        "models": {
            "unet": "diffusion_models/acestep_v1.5_base.safetensors",
            "clip_0_6b": "text_encoders/qwen_0.6b_ace15.safetensors",
            "clip_1_7b": "text_encoders/qwen_1.7b_ace15.safetensors",
            "vae": "vae/ace_1.5_vae.safetensors",
        },
        "defaults": {
            "duration": 120.0,
            "bpm": 120,
            "time_signature": "4",
            "language": "en",
            "keyscale": "C major",
            "steps": 32,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "simple",
        },
    },
    "seedance2-api": {
        "label": "ByteDance Seedance 2.0 API",
        "architecture": "seedance2-api",
        "supports": ["videogen.seedance2-t2v", "videogen.seedance2-r2v", "videogen.seedance2-flf2v"],
        "models": {},
        "defaults": {
            "provider": "comfy-api",
            "model": "Seedance 2.0",
            "resolution": "480p",
            "ratio": "16:9",
            "duration": 7,
            "generate_audio": True,
            "watermark": False,
            "seed": 0,
            "remote": True,
        },
    },
    "grok-imagine-api": {
        "label": "Grok Imagine API",
        "architecture": "grok-imagine-api",
        "supports": ["imagegen.grok-generate", "imagegen.grok-edit"],
        "models": {},
        "defaults": {
            "provider": "comfy-api",
            "model": "grok-imagine-image",
            "resolution": "1K",
            "aspect_ratio": "1:1",
            "edit_aspect_ratio": "auto",
            "number_of_images": 1,
            "seed": 0,
            "remote": True,
        },
    },
}

SUPPORTED_ARCHITECTURES = {
    "qwen-image-edit",
    "anima",
    "flux-klein",
    "upscale-model",
    "ltx23",
    "ace-step-1.5",
    "seedance2-api",
    "grok-imagine-api",
}


class ProfileError(Exception):
    """Base profile/config error with a stable error_type."""

    error_type = "config_error"


class ConfigError(ProfileError):
    error_type = "config_error"


class UnknownProfileError(ProfileError):
    error_type = "unknown_profile"


class UnsupportedCapabilityError(ProfileError):
    error_type = "unsupported_capability"


class ArchitectureMismatchError(ProfileError):
    error_type = "architecture_mismatch"


class UnsupportedArchitectureError(ProfileError):
    error_type = "unsupported_architecture"


class MissingModelFileError(ProfileError):
    error_type = "missing_model_file"


@dataclass(frozen=True)
class ResolvedProfile:
    """Fully resolved profile for a specific capability."""

    capability: str
    name: str
    architecture: str
    label: str
    models_dir: Path
    models: dict[str, Path]
    defaults: dict[str, Any]
    supports: list[str]

    def metadata(self) -> dict[str, Any]:
        """Return stable JSON metadata for command results."""
        return {
            "capability": self.capability,
            "model_profile": self.name,
            "architecture": self.architecture,
            "models_dir": str(self.models_dir),
            "resolved_models": {
                key: str(resolve_model_path(self.models_dir, value))
                for key, value in self.models.items()
            },
        }


def default_config() -> dict[str, Any]:
    """Return a new local config with built-in defaults."""
    return {
        "models_dir": str(BUILTIN_MODELS_DIR),
        "defaults": dict(BUILTIN_DEFAULTS),
        "profiles": {},
    }


def config_path(cwd: str | Path | None = None) -> Path:
    """Return the repo-local config path for cwd."""
    return Path(cwd or Path.cwd()) / CONFIG_FILENAME


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """Load local config or return built-in fallback plus source label."""
    cfg_path = Path(path) if path is not None else config_path()
    if not cfg_path.exists():
        return default_config(), "builtin"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {cfg_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config must be a JSON object: {cfg_path}")
    merged = default_config()
    merged.update(data)
    merged["defaults"] = {**BUILTIN_DEFAULTS, **dict(data.get("defaults", {}))}
    merged["profiles"] = dict(data.get("profiles", {}))
    return merged, "local"


def write_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write local config as pretty JSON."""
    cfg_path = Path(path) if path is not None else config_path()
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cfg_path


def all_profile_names(config: dict[str, Any] | None = None) -> list[str]:
    """Return all profile names, built-ins first."""
    names = list(BUILTIN_PROFILES)
    if config:
        for name in config.get("profiles", {}):
            if name not in names:
                names.append(name)
    return names


def resolve_profile(name: str, config: dict[str, Any] | None = None, _stack: tuple[str, ...] = ()) -> dict[str, Any]:
    """Resolve a profile by name, applying local inheritance."""
    config = config or default_config()
    local_profiles = dict(config.get("profiles", {}))

    if name in local_profiles:
        raw = dict(local_profiles[name])
        base_name = raw.get("extends")
        if base_name:
            if name in _stack:
                raise ConfigError(f"profile inheritance cycle: {' -> '.join((*_stack, name))}")
            base = resolve_profile(str(base_name), config, (*_stack, name))
        else:
            base = {}
        profile = _merge_profile(base, raw)
    elif name in BUILTIN_PROFILES:
        profile = dict(BUILTIN_PROFILES[name])
    else:
        raise UnknownProfileError(f"unknown profile: {name}")

    architecture = profile.get("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise UnsupportedArchitectureError(f"unsupported architecture for profile '{name}': {architecture}")
    if "extends" in profile:
        profile.pop("extends", None)
    profile["name"] = name
    _validate_architecture_inheritance(name, profile, config)
    return profile


def resolve_capability(capability: str, config_path_override: str | Path | None = None) -> tuple[ResolvedProfile, str]:
    """Resolve the default profile for a capability."""
    config, source = load_config(config_path_override)
    profile_name = str(dict(config.get("defaults", {})).get(capability, ""))
    if not profile_name:
        raise UnsupportedCapabilityError(f"no default profile configured for capability: {capability}")
    profile = resolve_profile(profile_name, config)
    supports = list(profile.get("supports", []))
    if capability not in supports:
        raise UnsupportedCapabilityError(f"profile '{profile_name}' does not support {capability}")

    models_dir = Path(profile.get("models_dir") or config.get("models_dir") or BUILTIN_MODELS_DIR)
    models = {key: Path(value) for key, value in dict(profile.get("models", {})).items()}
    return (
        make_resolved_profile(capability, profile_name, profile, config),
        source,
    )


def make_resolved_profile(
    capability: str,
    profile_name: str,
    profile: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> ResolvedProfile:
    """Build a ResolvedProfile from an already resolved profile dict."""
    config = config or default_config()
    supports = list(profile.get("supports", []))
    if capability not in supports:
        raise UnsupportedCapabilityError(f"profile '{profile_name}' does not support {capability}")
    models_dir = Path(profile.get("models_dir") or config.get("models_dir") or BUILTIN_MODELS_DIR)
    models = {key: Path(value) for key, value in dict(profile.get("models", {})).items()}
    return ResolvedProfile(
        capability=capability,
        name=profile_name,
        architecture=str(profile["architecture"]),
        label=str(profile.get("label", profile_name)),
        models_dir=models_dir,
        models=models,
        defaults=dict(profile.get("defaults", {})),
        supports=supports,
    )


def resolve_model_path(models_dir: Path, path: Path) -> Path:
    """Resolve relative model path against models_dir."""
    return path if path.is_absolute() else models_dir / path


def validate_profile_files(profile: ResolvedProfile) -> None:
    """Raise if a resolved profile references missing files."""
    for key, path in profile.models.items():
        resolved = resolve_model_path(profile.models_dir, path)
        if not resolved.is_file():
            raise MissingModelFileError(f"{key} model file not found: {resolved}")


def validate_defaults(config: dict[str, Any] | None = None) -> None:
    """Validate that every default points to a supporting profile."""
    config = config or default_config()
    for capability, profile_name in dict(config.get("defaults", {})).items():
        profile = resolve_profile(str(profile_name), config)
        if capability not in list(profile.get("supports", [])):
            raise UnsupportedCapabilityError(f"profile '{profile_name}' does not support {capability}")


def profiles_by_architecture(config: dict[str, Any] | None = None) -> dict[str, list[str]]:
    """Return profile names grouped by architecture."""
    config = config or default_config()
    grouped: dict[str, list[str]] = {}
    for name in all_profile_names(config):
        profile = resolve_profile(name, config)
        grouped.setdefault(str(profile["architecture"]), []).append(name)
    return grouped


def _merge_profile(base: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    profile = dict(base)
    profile.update({key: value for key, value in raw.items() if key not in {"models", "defaults", "supports"}})
    profile["models"] = {**dict(base.get("models", {})), **dict(raw.get("models", {}))}
    profile["defaults"] = {**dict(base.get("defaults", {})), **dict(raw.get("defaults", {}))}
    if "supports" in raw:
        profile["supports"] = list(raw["supports"])
    elif "supports" in base:
        profile["supports"] = list(base["supports"])
    return profile


def _validate_architecture_inheritance(name: str, profile: dict[str, Any], config: dict[str, Any]) -> None:
    raw = dict(config.get("profiles", {})).get(name)
    if not raw or not raw.get("extends") or not raw.get("architecture"):
        return
    base = resolve_profile(str(raw["extends"]), config)
    if raw["architecture"] != base.get("architecture"):
        raise ArchitectureMismatchError(
            f"profile '{name}' architecture '{raw['architecture']}' does not match base '{raw['extends']}' architecture '{base.get('architecture')}'"
        )
