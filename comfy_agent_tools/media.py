"""Media manifests, indexing, gallery serving, and HyperFrames export."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4


SCHEMA_VERSION = "1.0"
MEDIA_DIRNAME = ".comfy-media"
RUNS_DIRNAME = "runs"
INDEX_FILENAME = "index.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3"}
COMPOSITION_EXTENSIONS = {".html", ".htm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS | COMPOSITION_EXTENSIONS


def write_run_manifest(*, out_dir: str | Path, tool: str, payload: dict[str, Any], args: Any) -> Path:
    """Write one run manifest and return its path."""
    output_dir = Path(out_dir)
    runs_dir = output_dir / MEDIA_DIRNAME / RUNS_DIRNAME
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid4().hex
    manifest = create_run_manifest(run_id=run_id, tool=tool, payload=payload, args=args)
    manifest_path = runs_dir / f"{run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def create_run_manifest(*, run_id: str, tool: str, payload: dict[str, Any], args: Any) -> dict[str, Any]:
    """Create the stable run manifest from a successful CLI payload."""
    kind = str(payload.get("kind", "media"))
    mode = str(payload.get("mode", "unknown"))
    artifacts = [_artifact_entry(path, kind=kind) for path in payload.get("artifacts", [])]
    duration = payload.get("duration_seconds")

    generation = _generation_metadata(payload)
    prompt = getattr(args, "prompt", None)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": tool,
        "kind": kind,
        "mode": mode,
        "artifacts": artifacts,
        "inputs": _input_metadata(payload),
        "prompt": prompt,
        "generation": generation,
        "model_profile": payload.get("model_profile"),
        "architecture": payload.get("architecture"),
        "capability": payload.get("capability"),
        "resolved_models": payload.get("resolved_models", {}),
        "extra_loras": payload.get("extra_loras", []),
        "hyperframes": {
            "asset_kind": _hyperframes_asset_kind(kind, artifacts),
            "duration_seconds": duration,
            "width": payload.get("width") or _first_output_value(payload, "width"),
            "height": payload.get("height") or _first_output_value(payload, "height"),
            "composition_compatible": bool(artifacts),
        },
    }
    return manifest


def build_index(out_dir: str | Path) -> dict[str, Any]:
    """Build a gallery index from run manifests and orphan media files."""
    output_dir = Path(out_dir)
    manifests = load_manifests(output_dir)
    items = [_index_item_from_manifest(manifest, output_dir) for manifest in manifests]
    known_paths = {
        str(Path(artifact["path"]).resolve())
        for manifest in manifests
        for artifact in manifest.get("artifacts", [])
        if artifact.get("path")
    }

    for path in sorted(output_dir.rglob("*")) if output_dir.exists() else []:
        if not path.is_file() or MEDIA_DIRNAME in path.parts:
            continue
        if path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if str(path.resolve()) in known_paths:
            continue
        items.append(_orphan_item(path, output_dir))

    items.sort(key=lambda item: (item.get("created_at") or "", item.get("title") or ""), reverse=True)
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "out_dir": str(output_dir),
        "items": items,
    }


def write_index(out_dir: str | Path) -> Path:
    """Write the current media index and return the path."""
    output_dir = Path(out_dir)
    index = build_index(output_dir)
    media_dir = output_dir / MEDIA_DIRNAME
    media_dir.mkdir(parents=True, exist_ok=True)
    path = media_dir / INDEX_FILENAME
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifests(out_dir: str | Path) -> list[dict[str, Any]]:
    """Load all valid run manifests under an output directory."""
    runs_dir = Path(out_dir) / MEDIA_DIRNAME / RUNS_DIRNAME
    manifests: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("schema_version") == SCHEMA_VERSION:
            data["_manifest_path"] = str(path)
            manifests.append(data)
    return manifests


def load_selection(path: str | Path, out_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load a manifest, index, or list of selected items."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "items" in data:
        return list(data["items"])
    if isinstance(data, dict) and "artifacts" in data:
        base = Path(out_dir) if out_dir is not None else Path(path).parent
        return [_index_item_from_manifest(data, base)]
    if isinstance(data, list):
        return data
    raise ValueError("selection must be a manifest, index, or JSON item list")


def export_hyperframes_project(
    *,
    selection_path: str | Path,
    out_dir: str | Path,
    project_dir: str | Path,
    copy_assets: bool = False,
) -> dict[str, Any]:
    """Create a minimal HyperFrames review-reel project from selected media."""
    output_dir = Path(out_dir)
    project = Path(project_dir)
    assets_dir = project / "assets"
    project.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    items = load_selection(selection_path, output_dir)
    if not items:
        raise ValueError("selection did not include any media items")

    normalized_items: list[dict[str, Any]] = []
    for item in items:
        normalized = dict(item)
        artifact_path = _primary_artifact_path(normalized)
        if artifact_path is None:
            continue
        source = _resolve_artifact_path(artifact_path, output_dir)
        if copy_assets:
            target = _unique_asset_path(assets_dir, source.name)
            shutil.copy2(source, target)
            src = os.path.relpath(target, project)
        else:
            src = os.path.relpath(source, project)
        normalized["export_src"] = src.replace(os.sep, "/")
        normalized_items.append(normalized)

    if not normalized_items:
        raise ValueError("selection did not include readable media artifacts")

    selection_json = project / "comfy-media-selection.json"
    selection_json.write_text(json.dumps(normalized_items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index_html = project / "index.html"
    index_html.write_text(_review_reel_html(normalized_items), encoding="utf-8")

    return {
        "ok": True,
        "project_dir": str(project),
        "index": str(index_html),
        "selection": str(selection_json),
        "items": len(normalized_items),
        "copied_assets": copy_assets,
    }


def gallery_payload(*, out_dir: str | Path, host: str, port: int) -> dict[str, Any]:
    """Return startup metadata for the gallery server."""
    url = f"http://{host}:{port}"
    return {
        "ok": True,
        "url": url,
        "out_dir": str(Path(out_dir)),
        "api": f"{url}/api/index",
    }


def serve_gallery(*, out_dir: str | Path, host: str, port: int) -> None:
    """Serve the local gallery until interrupted."""
    output_dir = Path(out_dir).resolve()
    handler = partial(_GalleryHandler, out_dir=output_dir)
    server = ThreadingHTTPServer((host, port), handler)
    server.serve_forever()


class _GalleryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, out_dir: Path, **kwargs: Any) -> None:
        self.out_dir = out_dir
        super().__init__(*args, directory=str(out_dir), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"/", "/index.html"}:
            self._send_text(_gallery_html(), "text/html; charset=utf-8")
            return
        if path == "/api/index":
            self._send_json(build_index(self.out_dir))
            return
        if path == "/static/hyperframes-player.js":
            player_path = Path(__file__).with_name("media_static") / "hyperframes-player.js"
            self._send_text(player_path.read_text(encoding="utf-8"), "text/javascript; charset=utf-8")
            return
        if path == "/static/gallery.css":
            css_path = Path(__file__).with_name("media_static") / "gallery.css"
            self._send_text(css_path.read_text(encoding="utf-8"), "text/css; charset=utf-8")
            return
        if path == "/static/gallery.js":
            js_path = Path(__file__).with_name("media_static") / "gallery.js"
            self._send_text(js_path.read_text(encoding="utf-8"), "text/javascript; charset=utf-8")
            return
        return super().do_GET()

    def _send_json(self, data: dict[str, Any]) -> None:
        self._send_text(json.dumps(data, indent=2, sort_keys=True), "application/json; charset=utf-8")

    def _send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _generation_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "ok",
        "artifacts",
        "manifests",
        "resolved_models",
        "extra_loras",
        "models_dir",
        "error",
        "error_type",
    }
    return {key: value for key, value in payload.items() if key not in excluded}


def _input_metadata(payload: dict[str, Any]) -> dict[str, str]:
    keys = ("input", "first", "last", "audio_input", "control_video")
    return {key: str(payload[key]) for key in keys if payload.get(key)}


def _artifact_entry(path: str | Path, *, kind: str) -> dict[str, Any]:
    artifact = Path(path)
    mime, _encoding = mimetypes.guess_type(str(artifact))
    return {
        "path": str(artifact),
        "filename": artifact.name,
        "kind": _kind_from_path(artifact, fallback=kind),
        "mime_type": mime,
        "size_bytes": artifact.stat().st_size if artifact.exists() else None,
    }


def _kind_from_path(path: Path, *, fallback: str = "media") -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in AUDIO_EXTENSIONS:
        return "music"
    if ext in COMPOSITION_EXTENSIONS:
        return "hyperframes"
    return fallback


def _hyperframes_asset_kind(kind: str, artifacts: list[dict[str, Any]]) -> str:
    if artifacts:
        return str(artifacts[0].get("kind") or kind)
    return kind


def _first_output_value(payload: dict[str, Any], key: str) -> Any:
    outputs = payload.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        return outputs[0].get(key)
    return None


def _index_item_from_manifest(manifest: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", [])
    primary = artifacts[0] if artifacts else {}
    path = primary.get("path")
    return {
        "id": manifest.get("run_id"),
        "manifest": manifest.get("_manifest_path"),
        "created_at": manifest.get("created_at"),
        "title": Path(path).name if path else f"{manifest.get('tool', 'media')} {manifest.get('mode', '')}".strip(),
        "kind": manifest.get("kind") or primary.get("kind"),
        "mode": manifest.get("mode"),
        "tool": manifest.get("tool"),
        "prompt": manifest.get("prompt"),
        "profile": manifest.get("model_profile"),
        "seed": manifest.get("generation", {}).get("seed"),
        "artifacts": artifacts,
        "media_url": _media_url(path, out_dir) if path else None,
        "hyperframes": manifest.get("hyperframes", {}),
        "inputs": manifest.get("inputs", {}),
    }


def _orphan_item(path: Path, out_dir: Path) -> dict[str, Any]:
    stat = path.stat()
    created_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": f"orphan-{path.relative_to(out_dir).as_posix()}",
        "manifest": None,
        "created_at": created_at,
        "title": path.name,
        "kind": _kind_from_path(path),
        "mode": "orphan",
        "tool": "comfy-media",
        "prompt": None,
        "profile": None,
        "seed": None,
        "artifacts": [_artifact_entry(path, kind=_kind_from_path(path))],
        "media_url": _media_url(str(path), out_dir),
        "hyperframes": {"asset_kind": _kind_from_path(path), "composition_compatible": True},
        "inputs": {},
    }


def _media_url(path: str | Path, out_dir: Path) -> str:
    artifact = _resolve_artifact_path(path, out_dir)
    rel = artifact.relative_to(out_dir.resolve())
    return "/" + rel.as_posix()


def _resolve_artifact_path(path: str | Path, out_dir: str | Path) -> Path:
    artifact = Path(path)
    if not artifact.is_absolute():
        candidate = Path(out_dir) / artifact
        artifact = candidate if candidate.exists() else artifact
    return artifact.resolve()


def _primary_artifact_path(item: dict[str, Any]) -> str | None:
    artifacts = item.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        first = artifacts[0]
        if isinstance(first, dict):
            return first.get("path")
        if isinstance(first, str):
            return first
    return None


def _unique_asset_path(assets_dir: Path, filename: str) -> Path:
    target = assets_dir / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    for index in range(2, 10_000):
        candidate = assets_dir / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise ValueError(f"could not choose a unique asset filename for {filename}")


def _review_reel_html(items: list[dict[str, Any]]) -> str:
    scenes: list[str] = []
    audio_clips: list[str] = []
    start = 0.0
    track = 0
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or item.get("hyperframes", {}).get("asset_kind") or "media")
        src = str(item["export_src"])
        duration = _scene_duration(item, kind)
        label = _scene_label(item)
        meta = " · ".join(part for part in [kind, str(item.get("mode") or ""), str(item.get("profile") or "")] if part)
        if kind == "music":
            audio_clips.append(
                f'<audio id="audio-{index}" data-start="0" data-duration="{duration:.3f}" '
                f'data-track-index="20" src="{escape(src)}" data-volume="1"></audio>'
            )
            scenes.append(_audio_scene(index, start, duration, label, meta))
        elif kind == "video":
            scenes.append(_video_scene(index, start, duration, src, label, meta, track))
        elif kind == "hyperframes":
            scenes.append(_composition_scene(index, start, duration, src, track))
        else:
            scenes.append(_image_scene(index, start, duration, src, label, meta, track))
        start += duration
        track += 1

    total_duration = max(start, 1.0)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comfy Media Review Reel</title>
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; background: #101114; color: #f4f0e8; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    [data-composition-id="comfy-media-review"] {{ position: relative; width: 100%; height: 100%; overflow: hidden; background: #101114; }}
    [data-start] {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }}
    .scene {{ display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 24px; box-sizing: border-box; padding: 72px; background: #101114; }}
    .scene img, .scene video {{ max-width: 100%; max-height: 82%; object-fit: contain; border-radius: 6px; }}
    .caption {{ width: 100%; display: flex; justify-content: space-between; gap: 24px; font-size: 24px; line-height: 1.3; color: #f4f0e8; }}
    .meta {{ color: #b8c7c2; text-align: right; }}
    .audio-card {{ min-width: 52%; padding: 56px; border: 1px solid #3e4648; border-radius: 8px; background: #181b1e; }}
    .audio-card h1 {{ margin: 0 0 16px; font-size: 56px; font-weight: 760; }}
    .audio-card p {{ margin: 0; color: #b8c7c2; font-size: 28px; }}
  </style>
</head>
<body>
  <div data-composition-id="comfy-media-review" data-start="0" data-duration="{total_duration:.3f}" data-width="1920" data-height="1080">
    {''.join(scenes)}
    {''.join(audio_clips)}
  </div>
  <script>
    window.__timelines = window.__timelines || {{}};
    window.__timelines["comfy-media-review"] = {{ duration: () => {total_duration:.3f}, paused: true }};
  </script>
</body>
</html>
"""


def _scene_duration(item: dict[str, Any], kind: str) -> float:
    value = item.get("hyperframes", {}).get("duration_seconds") or item.get("duration_seconds")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        duration = 4.0 if kind == "image" else 6.0
    return max(duration, 1.0)


def _scene_label(item: dict[str, Any]) -> str:
    seed = item.get("seed")
    suffix = f" seed {seed}" if seed is not None else ""
    return f"{item.get('title') or 'media'}{suffix}"


def _image_scene(index: int, start: float, duration: float, src: str, label: str, meta: str, track: int) -> str:
    return (
        f'<section id="scene-{index}" class="scene" data-start="{start:.3f}" data-duration="{duration:.3f}" data-track-index="{track}">'
        f'<img src="{escape(src)}" alt="">'
        f'<div class="caption"><span>{escape(label)}</span><span class="meta">{escape(meta)}</span></div>'
        f"</section>"
    )


def _video_scene(index: int, start: float, duration: float, src: str, label: str, meta: str, track: int) -> str:
    return (
        f'<section id="scene-{index}" class="scene" data-start="{start:.3f}" data-duration="{duration:.3f}" data-track-index="{track}">'
        f'<video src="{escape(src)}" muted playsinline></video>'
        f'<div class="caption"><span>{escape(label)}</span><span class="meta">{escape(meta)}</span></div>'
        f"</section>"
    )


def _audio_scene(index: int, start: float, duration: float, label: str, meta: str) -> str:
    return (
        f'<section id="scene-{index}" class="scene" data-start="{start:.3f}" data-duration="{duration:.3f}" data-track-index="{index}">'
        f'<div class="audio-card"><h1>{escape(label)}</h1><p>{escape(meta)}</p></div>'
        f"</section>"
    )


def _composition_scene(index: int, start: float, duration: float, src: str, track: int) -> str:
    return (
        f'<div id="scene-{index}" data-composition-id="exported-{index}" data-composition-src="{escape(src)}" '
        f'data-start="{start:.3f}" data-duration="{duration:.3f}" data-track-index="{track}"></div>'
    )


def _gallery_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comfy Media</title>
  <link rel="stylesheet" href="/static/gallery.css">
  <script type="module" src="/static/hyperframes-player.js"></script>
</head>
<body>
  <main>
    <header>
      <h1>Comfy Media</h1>
      <input id="search" type="search" placeholder="Search prompt, filename, profile">
      <select id="kind">
        <option value="">All media</option>
        <option value="image">Images</option>
        <option value="video">Videos</option>
        <option value="music">Audio</option>
        <option value="hyperframes">HyperFrames</option>
      </select>
    </header>
    <section id="stage" aria-live="polite"></section>
    <aside id="info" aria-live="polite"></aside>
    <section id="items" class="filmstrip" aria-label="Media thumbnails"></section>
  </main>
  <script src="/static/gallery.js"></script>
</body>
</html>
"""
