from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from comfy_agent_tools.cli import media
from comfy_agent_tools.media import create_run_manifest


def _image_payload(path: Path) -> dict[str, object]:
    return {
        "ok": True,
        "kind": "image",
        "mode": "generate",
        "artifacts": [str(path)],
        "seed": 42,
        "capability": "imagegen.generate",
        "model_profile": "anima-base",
        "architecture": "anima",
        "resolved_models": {"unet": "/models/anima.safetensors"},
        "extra_loras": [],
        "outputs": [{"width": 64, "height": 64, "mode": "RGB"}],
    }


def test_create_run_manifest_from_success_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(artifact)

    class Args:
        prompt = "make an icon"

    manifest = create_run_manifest(
        run_id="run-1",
        tool="comfy-imagegen",
        payload=_image_payload(artifact),
        args=Args(),
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["run_id"] == "run-1"
    assert manifest["tool"] == "comfy-imagegen"
    assert manifest["kind"] == "image"
    assert manifest["prompt"] == "make an icon"
    assert manifest["artifacts"][0]["path"] == str(artifact)
    assert manifest["hyperframes"]["width"] == 64
    assert manifest["resolved_models"]["unet"] == "/models/anima.safetensors"


def test_index_reads_manifests_and_orphan_media(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "image.png"
    orphan = tmp_path / "loose.wav"
    Image.new("RGB", (4, 4), "red").save(artifact)
    orphan.write_bytes(b"wav")

    class Args:
        prompt = "indexed image"

    manifest = create_run_manifest(
        run_id="run-1",
        tool="comfy-imagegen",
        payload=_image_payload(artifact),
        args=Args(),
    )
    runs = tmp_path / ".comfy-media" / "runs"
    runs.mkdir(parents=True)
    (runs / "run-1.json").write_text(json.dumps(manifest), encoding="utf-8")

    rc = media.main(["index", "--out", str(tmp_path)])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert Path(payload["index_path"]).is_file()
    titles = {item["title"] for item in payload["items"]}
    assert {"image.png", "loose.wav"} <= titles


def test_gallery_dry_run_outputs_startup_metadata(tmp_path: Path, capsys) -> None:
    rc = media.main(["gallery", "--out", str(tmp_path), "--port", "9876", "--dry-run"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["url"] == "http://127.0.0.1:9876"
    assert payload["api"] == "http://127.0.0.1:9876/api/index"


def test_export_hyperframes_creates_review_project(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(artifact)
    index = {
        "ok": True,
        "items": [
            {
                "id": "run-1",
                "title": "image.png",
                "kind": "image",
                "mode": "generate",
                "profile": "anima-base",
                "seed": 42,
                "artifacts": [{"path": str(artifact), "kind": "image"}],
                "hyperframes": {"duration_seconds": 4, "width": 64, "height": 64},
            }
        ],
    }
    selection = tmp_path / "index.json"
    selection.write_text(json.dumps(index), encoding="utf-8")
    project = tmp_path / "hf-review"

    rc = media.main(
        [
            "export-hyperframes",
            "--out",
            str(tmp_path),
            "--selection",
            str(selection),
            "--project-dir",
            str(project),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["items"] == 1
    assert (project / "index.html").is_file()
    assert (project / "comfy-media-selection.json").is_file()
    html = (project / "index.html").read_text(encoding="utf-8")
    assert 'data-composition-id="comfy-media-review"' in html
    assert "image.png" in html
