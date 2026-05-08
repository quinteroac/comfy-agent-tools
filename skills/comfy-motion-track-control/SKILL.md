---
name: comfy-motion-track-control
description: Generate local LTX 2.3 videos guided by the Motion Track Control IC-LoRA. Use when the user wants an image animated along drawn sparse motion trajectories, spline overlays, point tracks, or a prepared motion-control reference video. Do not use for style-only LoRAs, Seedance API video, image-only generation, music-only generation, automatic point tracking from real video, or ComfyUI server-only workflows.
---

# comfy-motion-track-control

Use this skill for local LTX 2.3 motion-track control through the
`comfy-videogen motion-track` CLI. The mode uses the
`Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control` IC-LoRA, not a normal
style LoRA. It requires an input image, a prepared control video containing
colored spline/point trajectories, and a prompt.

If `comfy-videogen` or `comfy-models` is not available, use
`comfy-tools-setup` first. In this repository, prefer `uv run comfy-videogen`;
outside the repo, let `comfy-tools-setup` install the CLIs with `uv tool`.

At the start of every motion-track workflow, start or reuse the local Comfy Media
gallery for the active output directory:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Use `comfy-media --help` only if the CLI is missing or behaves unexpectedly; do
not skip the gallery just because generation can run headless.

If `.comfy-agent-tools.json` is missing or the user needs a different
`models_dir`, use `comfy-model-onboarding` first. Do not assume a local models
directory on a new machine.

If model validation fails with `missing_model_file`, use
`comfy-model-downloader` for only this capability:

```bash
uv run comfy-models download videogen.motion-track --dry-run
uv run comfy-models download videogen.motion-track --yes
```

## Command

```bash
uv run comfy-videogen motion-track \
  --input path/to/start.png \
  --control-video path/to/motion-reference.mp4 \
  --prompt "cinematic portrait, hair and camera follow the drawn motion paths, natural motion, soft light" \
  --attention-strength 1.0 \
  --out outputs
```

The CLI prints JSON only. On success, read `artifacts` for the MP4 path. On
failure, read `error_type`; if it is `missing_dependency`, the installed
`comfy-diffusion` build is older than v2.2.0 or does not expose the IC-LoRA wrappers for
`LTXICLoRALoaderModelOnly` and `LTXAddVideoICLoRAGuide` yet.

After every successful motion-track command, immediately index the same output
directory so the new artifact appears in Comfy Media:

```bash
uv run comfy-media index --out outputs
```

## Control Video Requirements

The control video is the motion instruction. It should be temporally aligned to
the generation settings:

- Match `--fps` and the intended duration (`--length / --fps`).
- Keep tracks smooth and physically plausible. Start with 3-4 keypoints per
  track before adding complex curves.
- Use visible colored trails, dots, or splines over the reference image.
- Keep the reference visually simple enough that the trajectories are clear.
- The official checkpoint is `ref0.5`, so the reference control is expected at
  0.5x output resolution internally. The CLI profile records
  `reference_downscale=0.5`.

## HyperFrames Reference Workflow

Use HyperFrames when the user starts from a still image and wants an agent-made
motion reference:

1. Create a short HyperFrames composition with the still image as the background.
2. Define tracks as arrays of `{x, y}` points, for example
   `[[{"x":420,"y":210},{"x":390,"y":250},{"x":340,"y":290}]]`.
3. Interpolate each track across the target frame count and render colored
   circles/splines along the path.
4. Render the HyperFrames composition to MP4 with the same FPS and duration that
   will be passed to `comfy-videogen motion-track`.
5. Run the CLI using the rendered MP4 as `--control-video`.

Keep HyperFrames as a preparation tool only; it is not a Python package
dependency of `comfy-agent-tools`.

## ComfyUI Workflow Reference

When the user wants an interactive visual editor for tracks, use the official
ComfyUI-LTXVideo workflow:

`LTX-2.3_ICLoRA_Motion_Track_Distilled.json`

The relevant nodes are:

- `LTXVDrawTracks`
- `LTXVSparseTrackEditor`
- `LTXICLoRALoaderModelOnly`
- `LTXAddVideoICLoRAGuide`

Use ComfyUI only to prepare or inspect the control signal unless the user
explicitly asks for a ComfyUI server workflow.

## Defaults

- Capability: `videogen.motion-track`
- Profile: `ltx23-motion-track`
- IC-LoRA:
  `loras/ltx23/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors`
- Base profile: 10Eros LTX 2.3 with distilled and text-encoder LoRAs.
- Video params: `width=512`, `height=320`, `length=49`, `fps=24`, `cfg=1.0`
- Control params: `attention_strength=1.0`, `reference_downscale=0.5`

This mode does not extract tracks automatically from real videos in v1. If the
user provides a real video, ask for or create a rendered trajectory/control
video first.
