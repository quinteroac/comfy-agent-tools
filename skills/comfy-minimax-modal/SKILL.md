---
name: comfy-minimax-modal
description: Prepare and run MiniMax H3 T2V, I2V, or R2V on a Modal GPU, including Modal authentication checks, Volume creation, selective model downloads, incremental model uploads, and local MP4 retrieval. Use when the user wants MiniMax H3 executed remotely on Modal. Do not use for local MiniMax, hosted MiniMax APIs, Seedance, or ComfyUI server workflows.
---

# comfy-minimax-modal

Use this skill when MiniMax H3 should run on Modal rather than on the local
GPU. The workflow uses the repository's `modal/minimax_h3.py` app and keeps the
result as an MP4 in the local workspace.

## Bootstrap

If the `modal` command is missing, install it with:

```bash
uv tool install modal
```

Validate authentication without printing credentials:

```bash
modal profile current
```

If no Modal profile exists, guide the user through `modal token new`, or use
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Never put these values in the repo.

The agent must also ensure `comfy-models` is available and that a local
`models_dir` is configured. If the MiniMax files are absent, download only the
capability needed by the requested mode; do not download every supported model.

## Prepare and run

The normal command prepares the Volume automatically, downloads missing local
files, uploads only missing files, runs the remote GPU function, and saves the
MP4 locally:

```bash
modal run modal/minimax_h3.py \
  --mode t2v \
  --prompt 'A cinematic aerial shot of a misty mountain valley at sunrise.' \
  --prepare-models \
  --out outputs
```

Preparation is enabled by default; `--prepare-models` makes the intent
explicit. Use `--no-prepare-models` only when the Volume is already known to be
complete.

Image-to-video:

```bash
modal run modal/minimax_h3.py \
  --mode i2v \
  --input path/to/first-frame.png \
  --prompt 'Slow camera push-in with subtle natural motion and synchronized ambience.' \
  --prepare-models \
  --out outputs
```

Reference-to-video accepts repeated `--input` arguments:

```bash
modal run modal/minimax_h3.py \
  --mode r2v \
  --input character.png \
  --input outfit.png \
  --prompt 'Use <Picture 1> for identity and <Picture 2> for clothing.' \
  --prepare-models \
  --out outputs
```

## Long videos with chained shots

Use `--multishot` when the requested duration is longer than one H3 shot. The
agent should write one prompt per shot separated by `---`; the runner renders
the first shot as T2V, carries each final frame into the next shot as I2V,
removes duplicated seam frames, stitches audio, and trims the final MP4 to the
requested duration:

```bash
modal run modal/minimax_h3.py \
  --mode t2v \
  --multishot \
  --prompt '{"prompts":["Establishing shot of a rainy neon street.","A courier runs past the camera and turns into an alley.","The courier reaches a warm doorway and looks back."]}' \
  --duration 30 \
  --shot-duration 10 \
  --sage-attention --easycache \
  --prepare-models \
  --out outputs
```

`--duration 30 --shot-duration 5` means six shots; `--duration 30
--shot-duration 10` means three. Use `--shot-count` to force the number of
shots. The prompts can also be passed as JSON with a `prompts` array. Both
accelerators remain opt-in, and EasyCache should be omitted when quality is
more important than speed. The first shot can be seeded with `--input
first-frame.png --first-mode i2v`, or with repeated `--input` references and
`--first-mode r2v`. With `--first-mode auto` (the default), no input means T2V,
one input means I2V, and multiple inputs mean R2V.

The default persistent Volume is `comfy-minimax-h3-models`. Override it before
running both preparation and inference:

```bash
MINIMAX_MODAL_VOLUME=my-minimax-volume modal run modal/minimax_h3.py ...
```

The default GPU is `RTX-PRO-6000`; override it when deploying/running with
`MINIMAX_MODAL_GPU`. Existing files in the Volume are skipped. Use
`--force-upload` only when intentionally replacing the remote copies.

The command prints one JSON object containing `ok`, `provider: "modal"`, the
selected mode, preparation results, and the local MP4 path. Use `comfy-media`
afterwards to index or review the artifact.

## Model layout

The Volume mirrors the configured local `models_dir` and contains only the H3
files required by the selected mode:

- T2V/I2V: FL2VA UNet, text encoder, audio VAE, and video VAE.
- R2V: REF2VA UNet, text encoder, audio VAE, and video VAE.

Do not put API keys, `.comfy-agent-tools.json`, or unrelated models in the
Volume preparation commands.
