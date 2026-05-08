"""CLI for local media review, indexing, and HyperFrames export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from comfy_agent_tools.media import (
    build_index,
    export_hyperframes_project,
    gallery_payload,
    serve_gallery,
    write_index,
)


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the comfy-media argument parser."""
    parser = argparse.ArgumentParser(
        prog="comfy-media",
        description="Review generated media, build indexes, and export HyperFrames projects.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build a JSON index for generated media.")
    index.add_argument("--out", type=_path, default=Path("outputs"))
    index.add_argument("--write", action=argparse.BooleanOptionalAction, default=True)

    gallery = subparsers.add_parser("gallery", help="Serve the local media gallery.")
    gallery.add_argument("--out", type=_path, default=Path("outputs"))
    gallery.add_argument("--host", default="127.0.0.1")
    gallery.add_argument("--port", type=int, default=8765)
    gallery.add_argument(
        "--dry-run",
        action="store_true",
        help="Print startup metadata without starting the blocking HTTP server.",
    )

    export = subparsers.add_parser("export-hyperframes", help="Export a HyperFrames review-reel project.")
    export.add_argument("--out", type=_path, default=Path("outputs"))
    export.add_argument("--selection", type=_path, required=True)
    export.add_argument("--project-dir", type=_path, required=True)
    export.add_argument("--copy-assets", action="store_true")

    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    """Run a parsed comfy-media command and return its JSON payload."""
    if args.command == "index":
        index = build_index(args.out)
        if args.write:
            index["index_path"] = str(write_index(args.out))
        return index

    if args.command == "gallery":
        if args.dry_run:
            return gallery_payload(out_dir=args.out, host=args.host, port=args.port)
        payload = gallery_payload(out_dir=args.out, host=args.host, port=args.port)
        print(json.dumps(payload, indent=2), flush=True)
        serve_gallery(out_dir=args.out, host=args.host, port=args.port)
        return payload

    if args.command == "export-hyperframes":
        return export_hyperframes_project(
            selection_path=args.selection,
            out_dir=args.out,
            project_dir=args.project_dir,
            copy_assets=args.copy_assets,
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
        payload = {"ok": False, "mode": mode, "error": str(exc), "error_type": "error"}
        print(json.dumps(payload, indent=2))
        return 1

    if not (mode == "gallery" and not getattr(args, "dry_run", False)):
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
