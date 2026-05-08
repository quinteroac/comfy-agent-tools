---
name: comfy-musicgen
description: Generate music locally with comfy-diffusion and ACE-Step 1.5 Base. Use when the user wants local GPU-backed music or song generation saved as WAV in the workspace. Do not use for hosted audio APIs, image generation, video generation, voice cloning, speech-only TTS, model downloads, ComfyUI server workflows, UI work, or custom node installation.
---

# comfy-musicgen

Use this skill for local music generation through the `comfy-musicgen` CLI. The
CLI uses ACE-Step 1.5 model files under `/mnt/models/comfyui`. If built-in
ACE-Step files are missing, use `comfy-model-downloader` for
`musicgen.generate` before running inference.

The CLI is quiet by default and prints only final JSON. Use `--verbose` only when
debugging ComfyUI runtime output, warnings, or progress bars.

If `comfy-musicgen` or `comfy-models` is not available, use
`comfy-tools-setup` first. In this repository, prefer `uv run comfy-musicgen`;
outside the repo, let `comfy-tools-setup` install the CLIs with `uv tool`.

At the start of every music workflow, start or reuse the local Comfy Media
gallery for the active output directory:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Use `comfy-media --help` only if the CLI is missing or behaves unexpectedly; do
not skip the gallery just because generation can run headless.

If `.comfy-agent-tools.json` is missing or the user wants to configure a new
ACE-Step checkpoint/fine-tune/default, use `comfy-model-onboarding` first.

If model validation fails with `missing_model_file`, use `comfy-model-downloader`
for `musicgen.generate`.

If the user asks to use or organize a LoRA by name or purpose, use
`comfy-lora-onboarding` to search `loras/ace-step-1.5/` first and pass the
chosen file with `--extra-lora`. Treat ACE-Step LoRAs as experimental until a
local smoke test confirms compatibility.

## Mode

- `generate`: music prompt tags plus optional lyrics to a WAV file.

## Command

Generate instrumental music:

```bash
uv run comfy-musicgen generate \
  --prompt "lofi hip hop beat, warm Rhodes piano, mellow drums, tape texture" \
  --lyrics "[instrumental]" \
  --duration 120 \
  --seed 0 \
  --extra-lora /mnt/models/comfyui/loras/ace-step-1.5/vocal-polish.safetensors:0.5:0.0 \
  --out outputs
```

Generate with lyrics:

```bash
uv run comfy-musicgen generate \
  --prompt "Latin pop ballad, polished radio production, nylon guitar, warm electric bass, soft drums, intimate male vocal, emotional chorus, 92 BPM" \
  --lyrics "[verse]\nLa luna cae sobre el mar\nTu nombre vuelve a respirar\n\n[chorus]\nQuedate cerca de mi\nCuando la noche quiera caer\n\n[instrumental]\n\n[bridge]\nNo hay distancia para olvidar\nLo que aprendimos a cuidar\n\n[chorus]\nQuedate cerca de mi\nCuando la noche quiera caer" \
  --language es \
  --bpm 92 \
  --keyscale "A minor" \
  --out outputs
```

Fast smoke test:

```bash
uv run comfy-musicgen generate \
  --prompt "lofi hip hop beat, warm Rhodes piano, mellow drums, tape texture" \
  --lyrics "[instrumental]" \
  --steps 8 \
  --cfg 1 \
  --duration 30 \
  --out outputs
```

## Prompt Guidance

ACE-Step works best when `--prompt` and `--lyrics` are treated as separate inputs.
Use `--prompt` for comma-separated music tags: primary genre, era or scene,
instruments, mood, vocal type, production texture, and tempo. Keep tags focused
and non-contradictory; prefer one main style with secondary influences instead of
a long list of clashing genres.

Good tag shape:

```text
Japanese pop, bright synth hooks, upbeat drums, electric bass, catchy female vocal, 132 BPM
```

Use `--lyrics` only when sung or spoken content is desired. Structure lyrics with
section markers such as `[verse]`, `[chorus]`, `[bridge]`, `[instrumental]`, and
`[outro]`. Keep lines singable: short phrases, natural vowels, clear emotional
turns, and enough breathing room. Add `[instrumental]` between sections for solos
or breaks.

For instrumental tracks, put `[instrumental]` or `[inst]` in `--lyrics` and also
include an instrumental cue in `--prompt`. If lyrics are left empty, the model may
still infer vocal texture from tags like `female vocal` or `male voice`.

Match `--language` to the lyrics language, for example `en`, `es`, `ja`, `zh`,
`fr`, or `ko`. For multilingual lyrics, prefer one dominant language per run; if
the language is not pronounced well, try romanized lyrics plus the matching
language code.

Use `--duration` for target song length in seconds. The final WAV metadata is in
the JSON response; read `duration_seconds`, `sample_rate`, and `channels` from
the response instead of assuming the model returned the exact target length.

The CLI defaults to the quality pass validated in real local tests:
`steps=32` and `cfg=7.0`. For quick smoke tests, override with `--steps 8
--cfg 1 --duration 30`.

## Prompt Templates

Instrumental:

```bash
uv run comfy-musicgen generate \
  --prompt "cinematic orchestral, slow tempo, strings, brass, emotional, film score" \
  --lyrics "[instrumental]" \
  --duration 120 \
  --out outputs
```

Song with structure:

```bash
uv run comfy-musicgen generate \
  --prompt "English pop, glossy radio production, bright synth hooks, punchy drums, catchy female vocal, 124 BPM" \
  --lyrics "[verse]\nI see the morning in your eyes\nA little fire in the city lights\n\n[chorus]\nHold on, we are running higher\nHands up into the open sky\n\n[instrumental]\n\n[chorus]\nHold on, we are running higher\nHands up into the open sky" \
  --language en \
  --bpm 124 \
  --keyscale "C major" \
  --out outputs
```

Troubleshooting:

- Rushed vocals: reduce lyric word count, shorten lines, or add `[instrumental]` breaks.
- Weak backing track: add concrete instrument tags and production terms.
- Muddled style: remove contradictory tags and choose a clearer primary genre.
- Wrong language or pronunciation: set `--language` explicitly and simplify lyrics.

## Defaults

- Models directory: `/mnt/models/comfyui`
- Diffusion model: `diffusion_models/acestep_v1.5_base.safetensors`
- Text encoder 0.6B: `text_encoders/qwen_0.6b_ace15.safetensors`
- Text encoder 1.7B: `text_encoders/qwen_1.7b_ace15.safetensors`
- VAE: `vae/ace_1.5_vae.safetensors`
- Audio params: `duration=120`, `bpm=120`, `time_signature=4`, `keyscale=C major`, `language=en`
- Sampling params: `steps=32`, `cfg=7.0`, `sampler=euler`, `scheduler=simple`, `seed=0`
- Output format: WAV PCM 16-bit
- Dependency: `comfy-diffusion[comfyui,video,audio]`

Extra LoRAs are optional and ad hoc. Use repeatable
`--extra-lora PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]` after resolving the file
through `loras/ace-step-1.5/` or the loose `loras/` fallback. If ComfyUI rejects
the LoRA, report that it may not be compatible with ACE-Step.

## Output Handling

The CLI prints JSON to stdout. On success, read `artifacts` for the saved WAV
path and audio metadata fields for playback details. On failure, read `error`
and `error_type`; do not parse logs for control flow.

Project-bound audio should be saved under the current workspace. Do not overwrite
existing assets unless the user explicitly asks for replacement.

After every successful music command, immediately index the same output
directory so the new artifact appears in Comfy Media:

```bash
uv run comfy-media index --out outputs
```
