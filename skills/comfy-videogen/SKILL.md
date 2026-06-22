---
name: comfy-videogen
description: Generate or post-process MP4 videos with comfy-diffusion using local LTX 2.3, local WAN 2.2, NVIDIA RTX Video Super Resolution, local SeedVR2 video upscaling, or remote ByteDance Seedance 2.0 API nodes. Use when the user wants local GPU-backed text-to-video, image-to-video, image+audio-to-video, video+audio processing, first/last-frame video generation, WAN 2.2 image/first-last-frame/sound-to-video/video+audio/Bernini video editing, LTX motion-track IC-LoRA control, RTX or SeedVR2 upscaling to target resolutions such as 720p/1080p/4k, or Seedance 2.0 API text/reference/first-last-frame video saved into the workspace. Do not use for image-only generation, music-only generation, voice generation, model downloads, ComfyUI server workflows, UI work, custom node installation, or non-Seedance hosted video APIs.
---

# comfy-videogen

Use this skill for video generation and video upscaling through the
`comfy-videogen` CLI. Local LTX 2.3 and WAN 2.2 modes use model files under
`/mnt/models/comfyui`. RTX upscaling uses `nvidia-vfx`, not model files.
SeedVR2 upscaling uses its pinned upstream CLI and auto-downloads its own model
files. Remote Seedance 2.0 modes use ComfyUI API Nodes vendored by
`comfy-diffusion` and require `COMFY_ORG_API_KEY`.

If a built-in local video profile is missing files, use `comfy-model-downloader`
for the requested local `videogen.<mode>` capability before running inference.
Do not use the downloader for Seedance 2.0, RTX upscaling, or SeedVR2
upscaling; `seedance2-api`, `rtx-vsr`, and `seedvr2` are not managed by
`comfy-models download`.

The CLI is quiet by default and prints only final JSON. Use `--verbose` only when
debugging ComfyUI runtime output, warnings, or progress bars.

If `comfy-videogen` or `comfy-models` is not available, use
`comfy-tools-setup` first. In this repository, prefer `uv run comfy-videogen`;
outside the repo, let `comfy-tools-setup` install the CLIs with `uv tool`.

At the start of every video workflow, start or reuse the local Comfy Media
gallery for the active output directory:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Use `comfy-media --help` only if the CLI is missing or behaves unexpectedly; do
not skip the gallery just because generation can run headless.

If `.comfy-agent-tools.json` is missing or the user wants to configure a new
checkpoint/fine-tune/default such as an LTX 2.3 variant, use
`comfy-model-onboarding` first.

If model validation fails with `missing_model_file`, use `comfy-model-downloader`
for the exact mode: `videogen.t2v`, `videogen.i2v`, `videogen.flf2v`,
`videogen.ia2av`, `videogen.wan22-t2v`, `videogen.wan22-i2v`, `videogen.wan22-flf2v`,
`videogen.wan22-s2v`, `videogen.wan22-video-audio`, `videogen.wan22-bernini`, or
`videogen.motion-track`.

Do not use `comfy-model-downloader` for `videogen.rtx-upscale`; install/validate
`nvidia-vfx` and RTX GPU support instead. Do not use it for
`videogen.seedvr2-upscale`; SeedVR2 downloads its own DiT/VAE weights.

If the user asks to use or organize a LoRA by name or purpose, use
`comfy-lora-onboarding` to search the architecture folder first, such as
`loras/ltx23/` for LTX or `loras/wan22/` for WAN I2V/FLF2V. Pass the chosen
file only to modes that support ad hoc LoRA insertion.

## Modes

- `t2v`: text prompt to MP4 with audio.
- `i2v`: input image plus prompt to MP4 with audio.
- `ia2av`: input image plus input audio plus prompt to MP4 with audio. Use this
  to animate a still image in relation to an existing WAV/MP3/FLAC, including
  WAV files created by `comfy-musicgen`.
- `flf2v`: first image plus last image plus prompt to MP4 with audio. This mode
  uses the experimental 10Eros first/last-frame adaptation with latent
  upscaling/refinement.
- `wan22-t2v`: WAN 2.2 text prompt to silent MP4.
- `wan22-i2v`: WAN 2.2 input image plus prompt to silent MP4.
- `wan22-flf2v`: WAN 2.2 first image plus last image plus prompt to silent MP4.
- `wan22-s2v`: WAN 2.2 reference image plus input audio plus prompt to MP4
  with the input audio muxed into the output.
- `wan22-video-audio`: WAN 2.2 input video plus input audio to MP4. Use
  `--mode audio-driven` for full-frame audio-reactive V2V, or `--mode lipsync`
  with `--mask-video`/`--mask-image` for external-mask mouth recomposition.
- `wan22-bernini`: WAN 2.2 Bernini source-video editing or reference-guided
  generation to silent MP4. Use `--input-video` for video edits and one or more
  `--reference-image` values for subject/style references.
- `motion-track`: input image plus motion-track control video plus prompt to MP4
  with audio. Use `comfy-motion-track-control` for IC-LoRA setup and control
  video preparation.
- `rtx-upscale`: existing MP4 to upscaled MP4 with NVIDIA RTX Video Super
  Resolution. Preserves input audio when present. Use for target presets such
  as `720p`, `1080p`, and `4k`.
- `seedvr2-upscale`: existing MP4 to upscaled MP4 with SeedVR2. Preserves input
  audio when present. Use for diffusion-based detail restoration/upscaling to
  `720p`, `1080p`, `1440p`, or `4k`.
- `seedance2-t2v`: remote Seedance 2.0 text prompt to MP4.
- `seedance2-r2v`: remote Seedance 2.0 reference image plus prompt to MP4.
- `seedance2-flf2v`: remote Seedance 2.0 first image plus last image plus
  prompt to MP4.

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
  --width 540 \
  --height 360 \
  --extra-lora /mnt/models/comfyui/loras/ltx23/detailer.safetensors:0.7:0.0 \
  --out outputs
```

WAN 2.2 text to video:

```bash
uv run comfy-videogen wan22-t2v \
  --prompt "a cinematic slow dolly through a sunlit atelier, elegant fabric motion, realistic camera drift" \
  --extra-lora-high /mnt/models/comfyui/loras/wan22/cinematic-motion.safetensors:0.7 \
  --out outputs
```

WAN 2.2 image to video:

```bash
uv run comfy-videogen wan22-i2v \
  --input path/to/image.png \
  --prompt "the subject begins moving naturally, cinematic camera drift, detailed motion" \
  --extra-lora-high /mnt/models/comfyui/loras/wan22/cinematic-motion.safetensors:0.7 \
  --out outputs
```

WAN 2.2 first/last frame:

```bash
uv run comfy-videogen wan22-flf2v \
  --first path/to/start.png \
  --last path/to/end.png \
  --prompt "a smooth cinematic transition between the two frames, coherent motion" \
  --extra-lora-low /mnt/models/comfyui/loras/wan22/low-noise-detail.safetensors:0.6 \
  --out outputs
```

WAN 2.2 sound to video:

```bash
uv run comfy-videogen wan22-s2v \
  --input path/to/portrait.png \
  --audio path/to/speech-or-song.wav \
  --prompt "the subject speaks naturally with expressive face motion, subtle body movement, cinematic camera drift" \
  --out outputs
```

WAN 2.2 video plus audio:

```bash
uv run comfy-videogen wan22-video-audio \
  --mode audio-driven \
  --input-video path/to/input.mp4 \
  --audio path/to/speech-or-music.wav \
  --out outputs
```

WAN 2.2 lipsync with an external mask:

```bash
uv run comfy-videogen wan22-video-audio \
  --mode lipsync \
  --input-video path/to/input.mp4 \
  --audio path/to/speech.wav \
  --mask-video path/to/mouth-mask.mp4 \
  --out outputs
```

WAN 2.2 Bernini video edit:

```bash
uv run comfy-videogen wan22-bernini \
  --input-video path/to/input.mp4 \
  --reference-image path/to/reference.png \
  --prompt "Replace the character with the subject in image 0. Keep camera motion, lighting, and background unchanged." \
  --out outputs
```

HDR IC-LoRA:

```bash
uv run comfy-videogen motion-track \
  --input path/to/start.png \
  --control-video path/to/motion-reference.mp4 \
  --prompt "cinematic portrait, hair and camera follow the drawn motion paths, natural motion" \
  --attention-strength 1.0 \
  --out outputs
```

Seedance 2.0 text to video:

```bash
COMFY_ORG_API_KEY=... uv run comfy-videogen seedance2-t2v \
  --prompt "cinematic shot of a futuristic city at sunset, slow camera drift" \
  --out outputs
```

Seedance 2.0 reference image to video:

```bash
COMFY_ORG_API_KEY=... uv run comfy-videogen seedance2-r2v \
  --input path/to/image.png \
  --prompt "slow expressive portrait animation, subtle head movement and soft lighting changes" \
  --out outputs
```

Seedance 2.0 first/last frame:

```bash
COMFY_ORG_API_KEY=... uv run comfy-videogen seedance2-flf2v \
  --first path/to/start.png \
  --last path/to/end.png \
  --prompt "smooth cinematic transition between both frames" \
  --out outputs
```

## Prompt Guidance

Describe visual motion, camera movement, subject action, scene mood, and audio
texture. Keep prompts concrete and short enough for a single shot. For `i2v`,
`ia2av`, and `flf2v`, name what should remain anchored to the input image or
guide frames.

For `ia2av`, `wan22-s2v`, and `wan22-video-audio`, the prompt should describe
how the visual subject should move in relation to the audio: speech or singing
mouth motion, tempo-synced lighting pulses, breathing portrait motion, subtle
camera drift, dance movement, performance gestures, or environmental reaction.
For `wan22-video-audio`, omit `--prompt` unless the default preset prompt is
wrong: audio-driven defaults to "Audio-reactive motion, expressive movement,
coherent video.", and lipsync defaults to "Speaking. Talking. Expressive lip
movement." The `wan22-video-audio` input video must be 16 fps.

This sizing rule applies only to local LTX 2.3 modes. It does not apply to
WAN 2.2, Seedance 2.0 remote API modes, or other non-LTX pipelines. Local LTX 2.3 runs a
two-step pipeline with latent x2 spatial upscaling/refinement, so treat
`--width` and `--height` as the base generation size, not the desired final MP4
size. When the user asks for a final resolution, pass half the requested
dimensions to avoid OOM and to hit the intended output: for final `1080x720`,
use `--width 540 --height 360`; for final `768x512`, use `--width 384 --height
256`. Read `width` and `height` from the final JSON to confirm the actual saved
MP4 size.

For WAN 2.2, the high-noise UNet controls most of the broad motion and the
low-noise UNet controls detail/refinement. The CLI accepts `--high-steps` and
`--low-steps`; when neither is supplied, the default is a 50/50 split of
`--steps`. For restrained motion, use a low high-step count and more low steps,
for example `--high-steps 1 --low-steps 3`. For more dynamic motion, give the
high-noise model more of the schedule, for example `--high-steps 4 --low-steps
2`. If both are provided without `--steps`, total `steps` is their sum.

## Defaults

### LTX 2.3 Local

- Models directory: `/mnt/models/comfyui`
- Checkpoint: `checkpoints/10Eros_v1-fp8mixed_learned.safetensors`
- Optional Dasiwa profile: `ltx23-dasiwa-golden-lace-v3`
- Dasiwa checkpoint: `checkpoints/DasiwaLTX23_goldenLaceV3.safetensors`
- Text encoder: `text_encoders/gemma_3_12B_it_fp4_mixed.safetensors`
- Distilled LoRA: `loras/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors`
- Text-encoder LoRA: `loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors`
- Upscaler: `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- HDR IC-LoRA: `loras/ltx23/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`
- Video params: `width=512`, `height=320`, `length=49`, `fps=24`, `cfg=1.0`, `seed=0`
- Motion-track params: `attention_strength=1.0`, `reference_downscale=1.0`
- IA2AV audio params: `audio_start_time=0.0`, `audio_duration=length/fps` by default
- Dependency: `comfy-diffusion[video,audio]` v2.2.0 or newer plus the vendored
  ComfyUI requirements for HDR IC-LoRA

To use Dasiwa Golden Lace v3 for standard LTX modes, set
`ltx23-dasiwa-golden-lace-v3` as the default for one or more of
`videogen.t2v`, `videogen.i2v`, `videogen.flf2v`, or `videogen.ia2av`.

Extra LoRAs are optional and ad hoc. Use repeatable
`--extra-lora PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]` after resolving the file
through the matching architecture folder or the loose `loras/` fallback. For
LTX, extra LoRAs are supported for `flf2v`; `t2v`, `i2v`, and `ia2av` return
clean JSON errors if an extra LoRA is supplied because those modes still use
upstream wrappers without a safe insertion point.

For WAN 2.2 I2V/FLF2V, `--extra-lora` applies the LoRA to both high-noise and
low-noise UNets. Use `--extra-lora-high` to target only the high-noise pass, or
`--extra-lora-low` to target only the low-noise pass. WAN uses a shared text
encoder, so any nonzero `CLIP_STRENGTH` patches that shared text encoder once
for the given LoRA flag. WAN S2V uses its separate model-only `--lora` flag.

### Seedance 2.0 Remote API

- Auth: `COMFY_ORG_API_KEY`
- Profile: `seedance2-api`
- Provider: `comfy-api`
- Model: `Seedance 2.0`
- Params: `resolution=480p`, `ratio=16:9`, `duration=7`, `generate_audio=true`, `watermark=false`, `seed=0`
- No local weights, no `models_dir`, no downloader, no LoRAs, no ComfyUI server.
- Requires a `comfy-diffusion` version that vendors ComfyUI API Nodes with
  `ByteDance Seedance 2.0`. If missing, the CLI returns `missing_dependency`.
- Do not use Seedance 1.x/1.5, OAuth, or token auth in this skill.

### NVIDIA RTX Video Super Resolution

- Profile: `rtx-vsr`
- Capability: `videogen.rtx-upscale`
- Command: `uv run comfy-videogen rtx-upscale --input-video input.mp4 --resolution 1080p --quality ULTRA --out outputs`
- Presets: `480p`, `720p`, `1080p`, `1440p`, `4k`, `8k`
- Presets preserve aspect ratio by fitting inside the target bounds. For
  example, square input plus `--resolution 1080p` outputs `1080x1080`, 4:3
  input outputs `1440x1080`, and portrait input outputs up to `1080x1920`.
- Custom exact sizing: use `--width` and `--height` together. This can change
  aspect ratio. Use `--scale 1.0..4.0` to preserve aspect ratio by multiplier.
- Requires a CUDA-capable NVIDIA RTX GPU and the `nvidia-vfx` Python package.
  If missing, the CLI returns `missing_dependency`; if no compatible GPU is
  available, it returns `hardware_required`.
- No local model files, no `models_dir`, no downloader, and no LoRAs.
- Preserves and re-encodes the input MP4 audio track when present
  (`audio_muxed=true`). Inputs without audio remain silent (`audio_muxed=false`).

### SeedVR2 Video Upscaler

- Profile: `seedvr2`
- Capability: `videogen.seedvr2-upscale`
- Command: `uv run comfy-videogen seedvr2-upscale --input-video input.mp4 --resolution 1080p --out outputs`
- Presets: `720p`, `1080p`, `1440p`, `4k`
- Presets preserve aspect ratio by using SeedVR2 short-edge targets with
  long-edge caps: `720/1280`, `1080/1920`, `1440/2560`, `2160/3840`.
- Use `--max-edge` to override the long-edge cap for a preset.
- Optional controls: `--model`, `--models-dir`/`--model-dir`, `--batch-size`,
  `--chunk-size`, `--temporal-overlap`, `--cuda-device`, `--blocks-to-swap`,
  `--video-backend`.
- Use `--models-dir PATH` to choose where SeedVR2 downloads/loads DiT/VAE
  weights. Omit it to keep SeedVR2's upstream default.
- No `COMFY_ORG_API_KEY`, no LoRAs, and no `comfy-model-downloader`; SeedVR2
  auto-downloads its own model files when missing.
- Preserves and re-encodes the input MP4 audio track when present
  (`audio_muxed=true`). Inputs without audio remain silent (`audio_muxed=false`).
- Prefer RTX for fast hardware VSR on compatible RTX GPUs; prefer SeedVR2 when
  diffusion restoration/detail is desired and runtime cost is acceptable.

### WAN 2.2 Local

- Profile: `wan22-i2v`
- T2V profile: `wan22-t2v`
- S2V profile: `wan22-s2v`
- Video+audio profile: `wan22-dasiwa-littledemon-v2-video-audio`
- Optional tuned profiles: `wan22-dasiwa-tastysin-t2v`, `wan22-dasiwa-boundbite-t2v`, `wan22-dasiwa-tastysin-i2v`, `wan22-dasiwa-boundbite-i2v`
- Optional S2V tuned profile: `wan22-dasiwa-littledemon-v2-s2v`
- Capabilities: `videogen.wan22-t2v`, `videogen.wan22-i2v`, `videogen.wan22-flf2v`, `videogen.wan22-s2v`, `videogen.wan22-video-audio`
- Models: high/low T2V or I2V UNets, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `vae/wan_2.1_vae.safetensors`
- S2V models: `diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors`, `audio_encoders/wav2vec2_large_english_fp16.safetensors`, the same UMT5 text encoder, and the same VAE
- Dasiwa S2V model: `diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors`
- Params: `width=640`, `height=640`, `length=81`, `fps=16`, `steps=20`, `high_steps=10`, `low_steps=10`, `i2v_cfg=3.5`, `flf2v_cfg=4.0`, `seed=0`
- I2V/FLF2V LoRAs: use `--extra-lora` for both UNets, `--extra-lora-high` for high-noise only, or `--extra-lora-low` for low-noise only.
- S2V params: `width=640`, `height=640`, `length=77`, `chunk_length=77`, `fps=16`, `steps=20`, `cfg=6.0`, `sampler=uni_pc`, `scheduler=simple`, `shift=8.0`, `seed=0`
- Dasiwa TastySin/BoundBite params: high UNet first, low UNet second, `steps=8`, `high_steps=2`, `low_steps=6`, `cfg=1.0` for T2V and `steps=4`, `high_steps=2`, `low_steps=2`, `i2v_cfg=1.0`, `flf2v_cfg=1.0` for I2V/FLF2V. T2V Dasiwa profiles reuse I2V checkpoints; run a GPU smoke test before making them production defaults.
- Dasiwa LittleDemon S2V params: baked fast distillation, `steps=4`, `cfg=1.0`, `sampler=euler`, `scheduler=simple`, `shift=10.0`; do not add extra Lightning/speed-up LoRAs.
- Dasiwa video+audio params: `fps=16`, `chunk_length=77`, `chunk_overlap=4`, `steps=4`, `denoise=0.35`, `cfg=1.0`, `sampler=euler`, `scheduler=simple`, `shift=10.0`.
- Dasiwa lipsync params: pass 1 `steps=4`, `denoise=0.45`; pass 2 `steps=2`, `denoise=0.25`; requires an external mask and does not run SAM3 automatically.
- To make Dasiwa LittleDemon the active S2V profile, run `uv run comfy-models set-default videogen.wan22-s2v wan22-dasiwa-littledemon-v2-s2v`.
- I2V/FLF2V output is silent MP4 (`audio_muxed=false`). S2V and video+audio output mux the input audio (`audio_muxed=true`).
- Use `comfy-model-downloader` for `videogen.wan22-t2v`, `videogen.wan22-i2v`, `videogen.wan22-flf2v`, `videogen.wan22-s2v`, or `videogen.wan22-video-audio` when files are missing.

## Output Handling

The CLI prints JSON to stdout. On success, read `artifacts` for the saved MP4 path.
On failure, read `error` and `error_type`; do not parse logs for control flow.

After every successful video command, immediately index the same output directory
so the new artifact appears in Comfy Media:

```bash
uv run comfy-media index --out outputs
```

Audio is required for LTX 2.3 audiovisual modes, WAN 2.2 S2V, and WAN 2.2
video+audio. RTX upscaling preserves source audio when the input has an audio
track. If MP4 audio muxing fails, the command fails instead of silently saving a
no-audio video. WAN 2.2 I2V and FLF2V intentionally save silent MP4 and report
`audio_muxed=false`.
