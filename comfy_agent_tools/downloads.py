"""On-demand model download registry and helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from comfy_agent_tools.profiles import (
    BUILTIN_PROFILES,
    ProfileError,
    ResolvedProfile,
    resolve_model_path,
)


class DownloadError(ProfileError):
    error_type = "download_source_error"


class DownloadChecksumMismatchError(DownloadError):
    error_type = "download_checksum_mismatch"


class DownloadAuthRequiredError(DownloadError):
    error_type = "download_auth_required"


class DownloadUnsupportedSourceError(DownloadError):
    error_type = "download_unsupported_source"


@dataclass(frozen=True)
class DownloadSource:
    """Source for one model file."""

    kind: str
    repo_id: str | None = None
    filename: str | None = None
    revision: str = "main"
    url: str | None = None
    token_env: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    def source_name(self) -> str:
        if self.kind == "hf":
            return "huggingface"
        if self.kind == "http":
            if self.url and "civitai" in self.url:
                return "civitai"
            if self.url and "openmodeldb" in self.url:
                return "openmodeldb"
            return "http"
        return self.kind


@dataclass(frozen=True)
class DownloadItem:
    """One downloadable file tied to a profile model key."""

    profile: str
    model_key: str
    target_path: str
    source: DownloadSource


DOWNLOAD_REGISTRY: dict[str, dict[str, DownloadSource]] = {
    "anima-preview3-turbo": {
        "unet": DownloadSource(
            kind="hf",
            repo_id="circlestone-labs/Anima",
            filename="split_files/diffusion_models/anima-preview3-base.safetensors",
        ),
        "clip": DownloadSource(
            kind="hf",
            repo_id="circlestone-labs/Anima",
            filename="split_files/text_encoders/qwen_3_06b_base.safetensors",
        ),
        "vae": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            filename="split_files/vae/qwen_image_vae.safetensors",
        ),
        "lora": DownloadSource(
            kind="http",
            url="https://civitai.com/api/download/models/2877687?type=Model&format=SafeTensor",
            token_env="CIVITAI_API_TOKEN",
            sha256="68ed0aec6ff4ebc3add1180e191797adb5aa6b69dd8b0fc8aa9e680145f65aac",
        ),
    },
    "qwen-edit2511": {
        "unet": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI",
            filename="split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        ),
        "clip": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        ),
        "vae": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/Qwen-Image_ComfyUI",
            filename="split_files/vae/qwen_image_vae.safetensors",
        ),
        "lora": DownloadSource(
            kind="hf",
            repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            filename="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        ),
    },
    "clear-reality": {
        "upscaler": DownloadSource(
            kind="hf",
            repo_id="skbhadra/ClearRealityV1",
            filename="4x-ClearRealityV1.pth",
        ),
    },
    "ltx23-10eros": {
        "checkpoint": DownloadSource(
            kind="hf",
            repo_id="TenStrip/LTX2.3-10Eros",
            filename="10Eros_v1-fp8mixed_learned.safetensors",
            size_bytes=29_161_842_950,
        ),
        "text_encoder": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/ltx-2",
            filename="split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
            sha256="aaca463d11e6d8d2a4bdb0d6299214c15ef78a3f73e0ef8113d5a9d0219b3f6d",
            size_bytes=9_447_702_218,
        ),
        "distilled_lora": DownloadSource(
            kind="hf",
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-22b-distilled-lora-384.safetensors",
        ),
        "te_lora": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/ltx-2",
            filename="split_files/loras/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors",
            sha256="87bcabeac9bec9f374232b5122d6511c2b2112d479e50176149e944b3712eb4a",
            size_bytes=628_203_616,
        ),
        "upscaler": DownloadSource(
            kind="hf",
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            sha256="5f416311fa8172b65af67530758964708d29a317b830d689a51143b7f91913ed",
            size_bytes=996_000_000,
        ),
    },
    "ace15-base": {
        "unet": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/ace_step_1.5_ComfyUI_files",
            filename="split_files/diffusion_models/acestep_v1.5_base.safetensors",
        ),
        "clip_0_6b": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/ace_step_1.5_ComfyUI_files",
            filename="split_files/text_encoders/qwen_0.6b_ace15.safetensors",
        ),
        "clip_1_7b": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/ace_step_1.5_ComfyUI_files",
            filename="split_files/text_encoders/qwen_1.7b_ace15.safetensors",
        ),
        "vae": DownloadSource(
            kind="hf",
            repo_id="Comfy-Org/ace_step_1.5_ComfyUI_files",
            filename="split_files/vae/ace_1.5_vae.safetensors",
        ),
    },
}


def download_items_for_profile(profile: ResolvedProfile) -> list[DownloadItem]:
    """Return downloadable items for a resolved profile."""
    registry = DOWNLOAD_REGISTRY.get(profile.name)
    if registry is None and profile.name in BUILTIN_PROFILES:
        raise DownloadUnsupportedSourceError(f"no download registry for profile: {profile.name}")
    if registry is None:
        raise DownloadUnsupportedSourceError(
            f"profile '{profile.name}' is local/custom; add model files manually or extend a built-in profile"
        )

    items: list[DownloadItem] = []
    for model_key, target_path in profile.models.items():
        source = registry.get(model_key)
        if source is None:
            raise DownloadUnsupportedSourceError(f"no download source for {profile.name}.{model_key}")
        items.append(
            DownloadItem(
                profile=profile.name,
                model_key=model_key,
                target_path=str(target_path),
                source=source,
            )
        )
    return items


def download_profile_models(profile: ResolvedProfile, *, dry_run: bool = False) -> dict[str, Any]:
    """Download missing models for a profile and return JSON-friendly metadata."""
    downloaded: list[str] = []
    skipped: list[str] = []
    planned: list[str] = []
    files: list[dict[str, Any]] = []
    total_downloaded_bytes = 0
    planned_download_bytes = 0
    sources: set[str] = set()

    for item in download_items_for_profile(profile):
        target = resolve_model_path(profile.models_dir, Path(item.target_path))
        source_name = item.source.source_name()
        sources.add(source_name)
        if target.is_file() and target.stat().st_size > 0:
            skipped.append(str(target))
            files.append(
                {
                    "path": str(target),
                    "model_key": item.model_key,
                    "source": source_name,
                    "status": "skipped",
                    "size_bytes": target.stat().st_size,
                }
            )
            continue
        planned.append(str(target))
        planned_download_bytes += item.source.size_bytes or 0
        file_info = {
            "path": str(target),
            "model_key": item.model_key,
            "source": source_name,
            "status": "planned" if dry_run else "pending",
            "size_bytes": item.source.size_bytes,
        }
        if dry_run:
            files.append(file_info)
            continue
        size = _download_item(item, target)
        downloaded.append(str(target))
        total_downloaded_bytes += size
        file_info["status"] = "downloaded"
        file_info["size_bytes"] = size
        files.append(file_info)

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "planned": planned,
        "files": files,
        "sources": sorted(sources),
        "total_downloaded_bytes": total_downloaded_bytes,
        "planned_download_bytes": planned_download_bytes,
        "dry_run": dry_run,
    }


def _download_item(item: DownloadItem, target: Path) -> int:
    source = item.source
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.",
        suffix=".part",
        dir=str(target.parent),
        delete=False,
    ) as handle:
        part_path = Path(handle.name)
    try:
        if source.kind == "hf":
            url = _hf_url(source)
            headers = _auth_headers(source.token_env or "HF_TOKEN")
            _stream_download(url, part_path, headers=headers)
        elif source.kind == "http":
            url, headers = _http_url_and_headers(source)
            _stream_download(url, part_path, headers=headers)
        else:
            raise DownloadUnsupportedSourceError(f"unsupported download source kind: {source.kind}")

        _validate_download(part_path, source)
        shutil.move(str(part_path), target)
        return target.stat().st_size
    except Exception:
        part_path.unlink(missing_ok=True)
        raise


def _hf_url(source: DownloadSource) -> str:
    if not source.repo_id or not source.filename:
        raise DownloadUnsupportedSourceError("huggingface source requires repo_id and filename")
    from huggingface_hub import hf_hub_url

    return hf_hub_url(repo_id=source.repo_id, filename=source.filename, revision=source.revision)


def _auth_headers(token_env: str | None) -> dict[str, str]:
    if not token_env:
        return {}
    token = os.environ.get(token_env)
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _http_url_and_headers(source: DownloadSource) -> tuple[str, dict[str, str]]:
    """Return URL and headers for direct HTTP downloads without leaking tokens."""
    url = str(source.url)
    if source.token_env and "civitai.com" in url:
        token = os.environ.get(source.token_env)
        if token:
            return _url_with_query_param(url, "token", token), {}
    return url, _auth_headers(source.token_env)


def _url_with_query_param(url: str, key: str, value: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[key] = value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _stream_download(url: str, target: Path, *, headers: dict[str, str]) -> None:
    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise DownloadAuthRequiredError(f"download requires authentication: {url}") from exc
        raise DownloadError(f"download failed with HTTP {exc.code}: {url}") from exc
    except URLError as exc:
        raise DownloadError(f"download failed: {url}: {exc.reason}") from exc


def _validate_download(path: Path, source: DownloadSource) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise DownloadError(f"download produced an empty file: {path}")
    if source.sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != source.sha256.lower():
            raise DownloadChecksumMismatchError(f"sha256 mismatch for {path}: expected {source.sha256}, got {actual}")
