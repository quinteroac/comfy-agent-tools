---
name: comfy-bernini-videoedit
description: Run Bernini WAN 2.2 video edit workflows with comfy-videogen. Use when the user asks for Bernini video editing, video-to-video V2V, reference-video-to-video RV2V, video-reference-to-video VV2V, or reference-to-video R2V using the local wan22-bernini profile. Do not use for generic WAN generation, Seedance remote API video, image-only generation, model onboarding, or unsupported custom ComfyUI server workflows.
---

# comfy-bernini-videoedit

Use this skill for Bernini video edit workflows through the existing
`comfy-videogen wan22-bernini` CLI command. This is a workflow skill, not a new
model profile system: it always uses the local `videogen.wan22-bernini`
capability and the `wan22-bernini` model profile.

If `comfy-videogen`, `comfy-models`, or `comfy-media` is missing, use
`comfy-tools-setup` first. In this repository, prefer `uv run comfy-videogen`;
outside the repo, let `comfy-tools-setup` install the CLIs with `uv tool`.

At the start of every Bernini workflow, start or reuse the local Comfy Media
gallery for the active output directory:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Keep outputs under `outputs/`. The CLI is quiet by default and prints final JSON;
use `--verbose` only for debugging ComfyUI runtime output, warnings, or progress.

If model validation or generation fails with `missing_model_file`, use
`comfy-model-downloader` for the exact local capability before running inference:

```bash
uv run comfy-models download videogen.wan22-bernini --dry-run
uv run comfy-models download videogen.wan22-bernini --yes
```

Do not download models automatically. Preview first, explain what will be
downloaded, and wait for user approval before using `--yes`.

## Profiles

- `V2V`: source video edit. Use a video and a prompt; preserve camera, lighting,
  and scene continuity unless the user asks otherwise.
- `RV2V`: reference-guided video edit. Use a source video plus one or more
  reference images for subject, style, or object replacement.
- `R2V`: reference-to-video generation. Use one or more reference images without
  a source video.
- `VV2V`: video-reference-to-video is not supported by the current CLI. Do not
  pretend it works and do not silently map it to another mode. Ask for extracted
  frames or still images from the reference video, then run `RV2V`.

## Commands

V2V source video edit:

```bash
uv run comfy-videogen wan22-bernini \
  --input-video path/to/source.mp4 \
  --prompt "Edit the video while preserving camera motion, lighting, and background continuity." \
  --out outputs
```

RV2V reference-guided video edit:

```bash
uv run comfy-videogen wan22-bernini \
  --input-video path/to/source.mp4 \
  --reference-image path/to/reference.png \
  --prompt "Replace the character with the subject in image 0. Keep camera motion, lighting, and background unchanged." \
  --out outputs
```

Use repeated `--reference-image` flags for multi-reference conditioning:

```bash
uv run comfy-videogen wan22-bernini \
  --input-video path/to/source.mp4 \
  --reference-image path/to/subject.png \
  --reference-image path/to/outfit.png \
  --prompt "Use image 0 for identity and image 1 for outfit. Preserve the source video's camera motion and scene." \
  --out outputs
```

R2V reference-to-video:

```bash
uv run comfy-videogen wan22-bernini \
  --reference-image path/to/reference.png \
  --prompt "Create a cinematic short video of the subject in image 0 with natural motion and coherent lighting." \
  --out outputs
```

VV2V requested with a reference video:

```text
Bernini VV2V is not supported by the current comfy-videogen CLI. Extract one or
more still frames from the reference video, then use RV2V with --reference-image.
```

After each successful generation, run:

```bash
uv run comfy-media index --out outputs
```

Use the final JSON to report the saved MP4 path, profile, dimensions, frame
count, seed, and any media manifest path.
