---
name: comfy-media
description: Review generated comfy-agent-tools media, build local indexes, serve the gallery, and export selected artifacts into a HyperFrames review-reel project. Use after comfy-imagegen, comfy-videogen, or comfy-musicgen creates outputs, or when the user wants to browse, compare, select, or compose generated media.
---

# comfy-media

Use this skill to inspect and organize generated media from `comfy-agent-tools`.
If `comfy-media` is not available, use `comfy-tools-setup` first. In this
repository, prefer `uv run comfy-media`.

Every comfy-agent-tools media workflow should begin by starting or reusing the
local gallery for the active output directory, then index the output directory
after each successful generation command.

Generated media commands write run manifests by default under:

```text
outputs/.comfy-media/runs/
```

## Index Media

Build a JSON index from run manifests and existing files under `outputs`:

```bash
uv run comfy-media index --out outputs
```

Read `items` from stdout. The command also writes:

```text
outputs/.comfy-media/index.json
```

## Open Gallery

Serve a local gallery bound to localhost:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Use the printed `url`. The gallery previews images, videos, WAV/MP3 audio, and
HyperFrames composition HTML. It serves only from the selected output directory.

For a non-blocking startup check:

```bash
uv run comfy-media gallery --out outputs --dry-run
```

## Export To HyperFrames

Export a selected manifest or index into a minimal HyperFrames review reel:

```bash
uv run comfy-media export-hyperframes \
  --out outputs \
  --selection outputs/.comfy-media/index.json \
  --project-dir outputs/hyperframes-review
```

The export writes:

- `index.html`: HyperFrames-compatible review-reel composition.
- `comfy-media-selection.json`: preserved selection metadata.
- `assets/`: created for optional copied assets.

By default, media files are referenced in place. Use `--copy-assets` when the
HyperFrames project should be portable.

Preview or render the exported project with HyperFrames:

```bash
npx hyperframes preview outputs/hyperframes-review
```

## Notes For Agents

- Do not expose the gallery on public interfaces unless the user explicitly asks.
- Do not commit generated media, manifests, indexes, or exported review projects.
- Use `--no-manifest` only when the user explicitly wants no generated metadata.
- The manifest excludes API keys, binary media contents, and model file contents.
