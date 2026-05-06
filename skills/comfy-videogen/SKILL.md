---
name: comfy-videogen
description: Generate MP4 videos locally with comfy-diffusion and the LTX 2.3 10Eros checkpoint. Use when the user wants local GPU-backed text-to-video, image-to-video, image+audio-to-video, or first/last-frame video generation with audio saved into the workspace. Do not use for hosted video APIs, image-only generation, music-only generation, voice generation, model downloads, ComfyUI server workflows, UI work, or custom node installation.
---

# comfy-videogen

Use this skill for local video generation through the `comfy-videogen` CLI. The
CLI uses LTX 2.3 model files under `/mnt/models/comfyui`. If the built-in LTX
2.3 profile is missing files, use `comfy-model-downloader` for the requested
`videogen.<mode>` capability before running inference.

The CLI is quiet by default and prints only final JSON. Use `--verbose` only when
debugging ComfyUI runtime output, warnings, or progress bars.

If `comfy-videogen` or `comfy-models` is not available, use
`comfy-tools-setup` first. In this repository, prefer `uv run comfy-videogen`;
outside the repo, let `comfy-tools-setup` install the CLIs with `uv tool`.

If `.comfy-agent-tools.json` is missing or the user wants to configure a new
checkpoint/fine-tune/default such as an LTX 2.3 variant, use
`comfy-model-onboarding` first.

If model validation fails with `missing_model_file`, use `comfy-model-downloader`
for the exact mode: `videogen.t2v`, `videogen.i2v`, `videogen.flf2v`, or
`videogen.ia2av`.

If the user asks to use or organize a LoRA by name or purpose, use
`comfy-lora-onboarding` to search `loras/ltx23/` first and pass the chosen file
with `--extra-lora` only to modes that support ad hoc LoRA insertion.

## Modes

- `t2v`: text prompt to MP4 with audio.
- `i2v`: input image plus prompt to MP4 with audio.
- `ia2av`: input image plus input audio plus prompt to MP4 with audio. Use this
  to animate a still image in relation to an existing WAV/MP3/FLAC, including
  WAV files created by `comfy-musicgen`.
- `flf2v`: first image plus last image plus prompt to MP4 with audio. This mode
  uses the experimental 10Eros first/last-frame adaptation with latent
  upscaling/refinement.

## Commands

Text to video:

```bash
uv run comfy-videogen t2v \
  --prompt "a slow cinematic camera push through a warm coffee shop, soft ambient room tone" \
  --out outputs
```

Image to video:

```bash
uv run comfy-videogen i2v \
  --input path/to/image.png \
  --prompt "steam rises gently while the camera slowly pushes in, warm cinematic ambience" \
  --out outputs
```

Image plus audio to audiovisual video:

```bash
uv run comfy-videogen ia2av \
  --input path/to/image.png \
  --audio path/to/song.wav \
  --prompt "a slow expressive portrait animation, subtle head movement and lighting pulses synchronized with the song, cinematic shallow depth of field" \
  --length 97 \
  --fps 24 \
  --out outputs
```

First/last frame:

```bash
uv run comfy-videogen flf2v \
  --first path/to/start.png \
  --last path/to/end.png \
  --prompt "a smooth transition between the two frames with subtle camera motion and ambient sound" \
  --extra-lora /mnt/models/comfyui/loras/ltx23/detailer.safetensors:0.7:0.0 \
  --out outputs
```

## Prompt Guidance

Describe visual motion, camera movement, subject action, scene mood, and audio
texture. Keep prompts concrete and short enough for a single shot. For `i2v`,
`ia2av`, and `flf2v`, name what should remain anchored to the input image or
guide frames.

For `ia2av`, the prompt should describe how the image should move in relation to
the audio: tempo-synced lighting pulses, breathing portrait motion, subtle camera
drift, dance movement, performance gestures, or environmental reaction. The
video duration is controlled by `--length / --fps`; long songs are trimmed to
that window unless `--audio-start-time` or `--audio-duration` is passed.

All modes use latent upscaling/refinement. The saved MP4 can be larger than the
requested `--width`/`--height`; read `width` and `height` from the final JSON.

## Defaults

- Models directory: `/mnt/models/comfyui`
- Checkpoint: `checkpoints/10Eros_v1-fp8mixed_learned.safetensors`
- Text encoder: `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors`
- Distilled LoRA: `loras/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors`
- Text-encoder LoRA: `loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors`
- Upscaler: `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- Video params: `width=512`, `height=320`, `length=49`, `fps=24`, `cfg=1.0`, `seed=0`
- IA2AV audio params: `audio_start_time=0.0`, `audio_duration=length/fps` by default
- Dependency: `comfy-diffusion[comfyui,video,audio]`

Extra LoRAs are optional and ad hoc. Use repeatable
`--extra-lora PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]` after resolving the file
through `loras/ltx23/` or the loose `loras/` fallback. In this cut, extra LoRAs
are supported for `flf2v`; `t2v`, `i2v`, and `ia2av` return clean JSON errors if
an extra LoRA is supplied because those modes still use upstream wrappers without
a safe insertion point.

## Output Handling

The CLI prints JSON to stdout. On success, read `artifacts` for the saved MP4 path.
On failure, read `error` and `error_type`; do not parse logs for control flow.

Audio is required in v1. If MP4 audio muxing fails, the command fails instead of
silently saving a no-audio video.
