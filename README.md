# comfy-agent-tools

Skills-first local media generation tools backed by
[`comfy-diffusion`](https://github.com/quinteroac/comfy-diffusion).

`comfy-agent-tools` gives agents a small set of skills and Python CLIs for local
image, video, and music generation. Users install the skills; the skills teach
the agent how to install or update the Python tools with `uv`, validate the local
runtime, initialize model profiles, and run generation commands.

Local model files are not distributed with this repo. Local Anima, Qwen, LTX,
ACE-Step, and upscaler profiles use model files under the user's configured
`models_dir`; remote API profiles such as Seedance 2.0 and Grok Imagine use
provider credentials instead of local weights.

Contributions use fork-based pull requests. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## How It Runs

Generation runs through
[`comfy-diffusion`](https://github.com/quinteroac/comfy-diffusion), a Python
library that vendors ComfyUI runtime pieces and exposes them as importable
pipelines. `comfy-agent-tools` calls those pipelines directly from the CLIs; it
does not require launching a ComfyUI server, opening the ComfyUI web UI, or
managing a separate ComfyUI process.

## Install The Skills

The primary installation flow is skills-first:

```bash
npx skills add quinteroac/comfy-agent-tools
```

Then ask your agent:

```text
setup comfy-agent-tools and generate an image with Anima
```

The agent should use the installed `comfy-tools-setup` skill to bootstrap the
Python CLIs on demand, initialize local config if needed, and validate models.

## What You Get

- `comfy-imagegen`: image generation, image editing, upscaling, and remote Grok Imagine.
- `comfy-videogen`: local LTX 2.3/WAN 2.2 video plus remote Seedance 2.0 API video.
- `comfy-motion-track-control`: LTX 2.3 HDR IC-LoRA guidance.
- `comfy-musicgen`: ACE-Step 1.5 music generation to WAV.
- `comfy-media`: local media gallery, indexing, and HyperFrames review-reel export.
- `comfy-models`: local model profiles, defaults, validation, and onboarding.
- `comfy-model-onboarding`: skill guidance for new checkpoints and fine-tunes.
- `comfy-model-downloader`: skill guidance for downloading supported base models on demand.
- `comfy-lora-onboarding`: skill guidance for organizing and applying ad hoc LoRAs.
- `comfy-tools-setup`: skill guidance for installing, updating, and validating the CLIs.

## Agent Bootstrap

When a CLI is missing, `comfy-tools-setup` installs the Python tools with:

```bash
uv tool install git+https://github.com/quinteroac/comfy-agent-tools
```

For updates:

```bash
uv tool upgrade comfy-agent-tools
```

If `uv` is missing, install `uv` first. Skills should not install system package
managers or model files without explicit user approval.

After the tools are available, initialize project-local config only when needed:

```bash
comfy-models init
comfy-models set-models-dir <models_dir>
comfy-models validate
comfy-models show
```

On a new machine, the agent should ask where the user wants to store or use
ComfyUI model files before running `set-models-dir`. `/mnt/models/comfyui` is a
common example, not something setup should assume automatically.

If validation reports missing built-in model files, the agent should use
`comfy-model-downloader` to download only the capability requested by the user:

```bash
comfy-models download imagegen.generate --dry-run
comfy-models download imagegen.generate --yes
```

In a local checkout of this repo, prefer dev mode:

```bash
uv sync --extra dev
uv run comfy-models validate
```

The Python runtime dependency is `comfy-diffusion[comfyui,video,audio]` pinned
to `v2.2.0` or newer for LTX 2.3 HDR IC-LoRA helpers. The
media extras are required because ComfyUI imports media nodes during runtime
startup, video generation writes MP4 files with audio, and music generation uses
audio helpers.

## Local Models

The default model directory is:

```text
/mnt/models/comfyui
```

Local models are downloaded only on demand. A request to generate an image
downloads the active `imagegen.generate` profile if missing; a music request
downloads `musicgen.generate`; a local LTX video request downloads the specific
`videogen.<mode>` profile. The tools do not download every supported model unless
explicitly asked. Remote API profiles such as `seedance2-api` and
`grok-imagine-api` are not downloaded by `comfy-models`.

Use `HF_TOKEN` for gated Hugging Face repositories and `CIVITAI_API_TOKEN` if a
Civitai direct download requires authentication.

Built-in profiles assume these local files exist:

```text
diffusion_models/anima-base-v1.0.safetensors
text_encoders/qwen_3_06b_base.safetensors
vae/qwen_image_vae.safetensors
loras/anima/anima-turbo-lora-v0.1.safetensors

diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors
text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
loras/qwen-image-edit/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
upscale_models/4x-ClearRealityV1.pth

checkpoints/10Eros_v1-fp8mixed_learned.safetensors
text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
loras/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors
loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
loras/ltx23/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors
latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors
diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors
text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
audio_encoders/wav2vec2_large_english_fp16.safetensors
vae/wan_2.1_vae.safetensors

diffusion_models/acestep_v1.5_base.safetensors
text_encoders/qwen_0.6b_ace15.safetensors
text_encoders/qwen_1.7b_ace15.safetensors
vae/ace_1.5_vae.safetensors
```

## Model Profiles

Model selection is configured by capability. If `.comfy-agent-tools.json` is
absent, the CLIs use built-in defaults:

| Capability | Default profile | Architecture |
| --- | --- | --- |
| `imagegen.generate` | `anima-base` | `anima` |
| `imagegen.edit` | `qwen-edit2511` | `qwen-image-edit` |
| `imagegen.upscale` | `clear-reality` | `upscale-model` |
| `imagegen.grok-generate` | `grok-imagine-api` | `grok-imagine-api` |
| `imagegen.grok-edit` | `grok-imagine-api` | `grok-imagine-api` |
| `videogen.t2v` | `ltx23-10eros` | `ltx23` |
| `videogen.i2v` | `ltx23-10eros` | `ltx23` |
| `videogen.flf2v` | `ltx23-10eros` | `ltx23` |
| `videogen.ia2av` | `ltx23-10eros` | `ltx23` |
| `videogen.motion-track` | `ltx23-motion-track` | `ltx23` |
| `videogen.wan22-i2v` | `wan22-i2v` | `wan22` |
| `videogen.wan22-flf2v` | `wan22-i2v` | `wan22` |
| `videogen.wan22-s2v` | `wan22-s2v` | `wan22` |
| `videogen.seedance2-t2v` | `seedance2-api` | `seedance2-api` |
| `videogen.seedance2-r2v` | `seedance2-api` | `seedance2-api` |
| `videogen.seedance2-flf2v` | `seedance2-api` | `seedance2-api` |
| `musicgen.generate` | `ace15-base` | `ace-step-1.5` |

`ltx23` is the architecture/adapter; `ltx23-10eros` is the built-in profile
validated for that architecture.

`wan22` is the local WAN 2.2 adapter. The default `wan22-i2v` profile uses the
Comfy-Org FP8 I2V high/low UNets. The optional
`wan22-dasiwa-tastysin-i2v` and `wan22-dasiwa-boundbite-i2v` point to local
Dasiwa high/low UNets and use `steps=4`, `cfg=1.0`; the WAN 2.2 i2v wrapper
samples the high-noise model in the first tranche and the low-noise model in
the second. The high-noise tranche controls broad motion, while the low-noise
tranche controls detail/refinement. By default WAN splits steps 50/50; pass
`--high-steps` and `--low-steps` to bias motion vs detail.

The `wan22-s2v` profile uses the Wan 2.2 S2V FP8 model plus wav2vec2 audio
encoder for reference-image and input-audio driven video. Defaults mirror the
native ComfyUI workflow: `length=77`, `fps=16`, `steps=20`, `cfg=6.0`,
`sampler=uni_pc`, `scheduler=simple`, and `shift=8.0`.

The optional `wan22-dasiwa-littledemon-v2-s2v` profile points to Dasiwa
LittleDemon v2 S2V. Its fast distillation is baked into the checkpoint, so the
profile uses `steps=4`, `cfg=1.0`, `sampler=euler`, `scheduler=simple`, and
`shift=10.0`; do not add extra Lightning/speed-up LoRAs on top of it.

An optional built-in image profile, `flux-klein-9b-snofs`, supports both
`imagegen.generate` and `imagegen.edit` with architecture `flux-klein`. It uses
`diffusion_models/flux-2-klein-9b-fp8.safetensors`,
`text_encoders/qwen_3_8b_fp8mixed.safetensors`, `vae/flux2-vae.safetensors`,
and `loras/flux-klein/klein_snofs_v1_1.safetensors`. The FLUX.2 Klein weights
are gated and non-commercial; SNOFS permits local image generation and selling
outputs, but does not permit public/commercial generation services, derivative
models, or weight redistribution without a separate license. Flux Klein edits
follow the official distilled image-edit workflow from `comfy-diffusion`, with
reference latents attached to both positive and negative conditioning.

Add a compatible LTX 2.3 fine-tune profile:

```bash
uv run comfy-models add-profile my-ltx23-finetune \
  --extends ltx23-10eros \
  --checkpoint checkpoints/my_ltx23_finetune.safetensors

uv run comfy-models validate-profile my-ltx23-finetune
uv run comfy-models set-default videogen.t2v my-ltx23-finetune
```

Use `comfy-model-onboarding` when adding a checkpoint, changing defaults, or
resolving config errors. V1 supports only architectures already implemented by
the tools.

`seedance2-api` is remote: it has no local model files, does not use
`models_dir`, is not handled by `comfy-models download`, and requires
`COMFY_ORG_API_KEY`.

`grok-imagine-api` is also remote: it has no local model files, does not use
`models_dir`, is not handled by `comfy-models download`, and requires
`COMFY_ORG_API_KEY`.

Download missing files for the effective profile of a capability:

```bash
uv run comfy-models download imagegen.generate --dry-run
uv run comfy-models download imagegen.generate --yes
uv run comfy-models download-profile anima-base --yes
uv run comfy-models download-profile flux-klein-9b-snofs --dry-run
```

`download` skips files that already exist, writes temporary `*.part` files while
streaming, and returns JSON with `downloaded`, `skipped`, `planned`, `sources`,
and `total_downloaded_bytes`.

## Ad Hoc LoRAs

Extra LoRAs are selected per command. There is no persistent LoRA catalog.

Use this folder convention under `/mnt/models/comfyui/loras` so agents can find
LoRAs by architecture and purpose:

```text
loras/
  anima/
    realism-portrait.safetensors
    blue-dawn-style.safetensors
    animation-sheet.safetensors
  qwen-image-edit/
    product-retouch.safetensors
  flux-klein/
    klein_snofs_v1_1.safetensors
  ltx23/
    camera-static.safetensors
    detailer.safetensors
  ace-step-1.5/
    vocal-polish.safetensors
```

Use descriptive filenames. Good names explain intent, such as
`realism-portrait.safetensors`; avoid names like `final.safetensors` or
`test2.safetensors`.

CLIs accept repeatable LoRA flags:

```text
--extra-lora PATH[:MODEL_STRENGTH[:CLIP_STRENGTH]]
```

Defaults are `MODEL_STRENGTH=1.0` and `CLIP_STRENGTH=0.0`. Paths may be absolute
or relative to `models_dir`.

Example:

```bash
uv run comfy-imagegen generate \
  --prompt "masterpiece, best quality, score_7, safe, realistic portrait, soft window light" \
  --extra-lora /mnt/models/comfyui/loras/anima/realism-portrait.safetensors:0.8:0.0 \
  --out outputs
```

Use `comfy-lora-onboarding` when a user wants to organize LoRAs, find a LoRA by
intent, or resolve ambiguity. Do not move or rename model files without explicit
approval.

## Image Generation

`comfy-imagegen generate` defaults to Anima Base v1.0 with the Anima turbo LoRA.
It is designed for anime, illustration, game art, and stylized assets. It does
not edit images.

```bash
uv run comfy-imagegen generate \
  --prompt "masterpiece, best quality, score_7, safe, 1girl, anime style, cinematic lighting, detailed background" \
  --width 1024 \
  --height 1024 \
  --seed 42 \
  --out outputs
```

Anima works best around 1MP, such as `1024x1024`, `896x1152`, or `1152x896`.
The turbo profile uses `steps=8` and `cfg=1.0`; increasing CFG can degrade the
turbo LoRA behavior.

Prompt guidance:

- Positive prefix: `masterpiece, best quality, score_7, safe, ...`
- Negative guidance when adapting prompts: `worst quality, low quality, score_1, score_2, score_3, artist name`
- Use lowercase tags with spaces, natural language, or a mix.

## Image Editing And Upscale

`comfy-imagegen edit` uses Qwen Image Edit 2511, because Anima generation does
not support image editing.

```bash
uv run comfy-imagegen edit \
  --input outputs/comfy-imagegen-generate-example.png \
  --prompt "Transform this photo into a polished animated film still style while preserving composition" \
  --seed 43 \
  --out outputs
```

For `flux-klein-9b-snofs` image editing, add `--width` and/or `--height` to set
the output canvas. Omitted dimensions default to the input image dimensions.

Upscale uses ClearReality:

```bash
uv run comfy-imagegen upscale \
  --input outputs/comfy-imagegen-edit-example.png \
  --out outputs
```

Qwen Image Edit may rescale internally, so final dimensions can differ from the
input or requested canvas. Read the final JSON metadata for actual dimensions.

### Grok Imagine API Images

Grok Imagine runs remotely through ComfyUI API Nodes vendored by
`comfy-diffusion`; it does not require a ComfyUI server, does not use local model
files, and is not downloaded with `comfy-models download`. Set
`COMFY_ORG_API_KEY` before running.

```bash
COMFY_ORG_API_KEY=... uv run comfy-imagegen grok-generate \
  --prompt "A cinematic product photo of a translucent orange cassette player on wet asphalt" \
  --model grok-imagine-image \
  --resolution 1K \
  --aspect-ratio 1:1 \
  --out outputs
```

```bash
COMFY_ORG_API_KEY=... uv run comfy-imagegen grok-edit \
  --input outputs/comfy-imagegen-generate-example.png \
  --prompt "Keep the subject and composition, change the background to a clean moonlit studio" \
  --resolution 1K \
  --out outputs
```

Grok defaults are `model="grok-imagine-image"`, `resolution=1K`,
`aspect_ratio=1:1`, `number_of_images=1`, and `seed=0`. Supported image models
are `grok-imagine-image-pro`, `grok-imagine-image`, and
`grok-imagine-image-beta`.

## Video Generation

`comfy-videogen` supports local LTX 2.3 generation, local WAN 2.2 image-driven
and sound-driven generation, and remote Seedance 2.0 API generation. It is quiet by default and prints final JSON only; pass
`--verbose` to show ComfyUI logs.

Local LTX 2.3 uses the `ltx23-10eros` profile and writes MP4 files with audio.
Local WAN 2.2 uses the `wan22-i2v` profile and writes silent MP4 files.
For Dasiwa TastySin or BoundBite, set the matching Dasiwa profile as the default
or pass its high/low UNet paths explicitly; use `--steps 4 --cfg 1.0`.
For Dasiwa LittleDemon S2V, set `wan22-dasiwa-littledemon-v2-s2v` as the
`videogen.wan22-s2v` default or pass its checkpoint with `--unet`; use the
profile defaults for 4-step generation.
WAN high steps influence motion; WAN low steps influence detail. Omit
`--high-steps`/`--low-steps` for the default 50/50 split, use
`--high-steps 1 --low-steps 3` for restrained motion, or use
`--high-steps 4 --low-steps 2` for more dynamic motion.

Text to video:

```bash
uv run comfy-videogen t2v \
  --prompt "a slow cinematic camera push through a warm coffee shop, soft ambient room tone" \
  --out outputs
```

Image to video:

```bash
uv run comfy-videogen i2v \
  --input outputs/comfy-imagegen-generate-example.png \
  --prompt "steam rises gently while the camera slowly pushes in, warm cinematic ambience" \
  --out outputs
```

Image plus audio to audiovisual video:

```bash
uv run comfy-videogen ia2av \
  --input outputs/comfy-imagegen-generate-example.png \
  --audio outputs/comfy-musicgen-generate-example.wav \
  --prompt "a slow expressive portrait animation, subtle head movement and lighting pulses synchronized with the song, cinematic shallow depth of field" \
  --length 97 \
  --fps 24 \
  --out outputs
```

First/last frame:

```bash
uv run comfy-videogen flf2v \
  --first start.png \
  --last end.png \
  --prompt "a smooth transition between the two frames with subtle camera motion and ambient sound" \
  --width 540 \
  --height 360 \
  --out outputs
```

WAN 2.2 image to video:

```bash
uv run comfy-videogen wan22-i2v \
  --input outputs/comfy-imagegen-generate-example.png \
  --prompt "the subject begins moving naturally, cinematic camera drift, detailed motion" \
  --high-steps 4 \
  --low-steps 2 \
  --out outputs
```

WAN 2.2 first/last frame:

```bash
uv run comfy-videogen wan22-flf2v \
  --first start.png \
  --last end.png \
  --prompt "a smooth cinematic transition between the two frames, coherent motion" \
  --out outputs
```

WAN 2.2 sound to video:

```bash
uv run comfy-videogen wan22-s2v \
  --input portrait.png \
  --audio speech-or-song.wav \
  --prompt "the subject speaks naturally with expressive face motion, subtle body movement, cinematic camera drift" \
  --out outputs
```

HDR IC-LoRA from an input image plus prepared trajectory control video:

```bash
uv run comfy-videogen motion-track \
  --input start.png \
  --control-video motion-reference.mp4 \
  --prompt "cinematic portrait, hair and camera follow the drawn motion paths, natural motion" \
  --attention-strength 1.0 \
  --out outputs
```

Use the `comfy-motion-track-control` skill to prepare motion references. A
common flow is to render colored point/spline trajectories over a still image
with HyperFrames, then pass that MP4 as `--control-video`. The control profile
uses the `HDR` IC-LoRA and records `reference_downscale=1.0`.

This sizing rule applies only to local LTX 2.3 modes. It does not apply to
Seedance 2.0 remote API modes or other non-LTX pipelines. Local LTX 2.3 runs a
two-step pipeline with latent x2 spatial upscaling/refinement, so treat
`--width` and `--height` as the base generation size, not the desired final MP4
size. If the desired final output is `1080x720`, run with `--width 540 --height
360`; requesting `--width 1080 --height 720` would try to produce about
`2160x1440`, which can cause OOM and is not the requested output. Read the JSON
metadata for actual `width` and `height`.

For `ia2av` and `wan22-s2v`, `--length` and `--fps` control video duration.
Long audio files, including 120s WAVs from `comfy-musicgen`, are trimmed to
`length / fps` by default; use `--audio-start-time` and `--audio-duration` to
pick a specific window.

Audio muxing is required for LTX audiovisual modes and WAN 2.2 S2V. If audio
cannot be written into the MP4, the command returns `ok:false` instead of saving
a silent video. WAN 2.2 I2V/FLF2V intentionally save silent MP4 files.

### Seedance 2.0 API Video

Seedance 2.0 runs remotely through ComfyUI API Nodes vendored by
`comfy-diffusion`; it does not require a ComfyUI server, does not use local model
files, and is not downloaded with `comfy-models download`. Set
`COMFY_ORG_API_KEY` before running. Only **Seedance 2.0** is exposed; Seedance
1.x/1.5 and OAuth/token auth are intentionally out of scope.

Text to video:

```bash
COMFY_ORG_API_KEY=... uv run comfy-videogen seedance2-t2v \
  --prompt "cinematic shot of a futuristic city at sunset, slow camera drift" \
  --out outputs
```

Reference image to video:

```bash
COMFY_ORG_API_KEY=... uv run comfy-videogen seedance2-r2v \
  --input portrait.png \
  --prompt "slow expressive portrait animation, subtle head movement and soft lighting changes" \
  --out outputs
```

First/last frame to video:

```bash
COMFY_ORG_API_KEY=... uv run comfy-videogen seedance2-flf2v \
  --first start.png \
  --last end.png \
  --prompt "smooth cinematic transition between both frames" \
  --out outputs
```

Seedance defaults are `model="Seedance 2.0"`, `resolution=480p`, `ratio=16:9`,
`duration=7`, `generate_audio=true`, `watermark=false`, and `seed=0`. The
installed `comfy-diffusion` version must vendor ComfyUI API Nodes with
`ByteDance Seedance 2.0`; otherwise the CLI returns `missing_dependency`.

## Music Generation

`comfy-musicgen` uses ACE-Step 1.5 Base and writes WAV PCM 16-bit files. It is
quiet by default and prints final JSON only; pass `--verbose` to show ComfyUI
logs.

Instrumental:

```bash
uv run comfy-musicgen generate \
  --prompt "lofi hip hop beat, warm Rhodes piano, mellow drums, tape texture" \
  --lyrics "[instrumental]" \
  --duration 120 \
  --seed 0 \
  --out outputs
```

Song with lyrics:

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

Current music defaults favor quality over speed:

- `duration=120`
- `bpm=120`
- `keyscale=C major`
- `steps=32`
- `cfg=7.0`

`--prompt` maps to ACE-Step music tags. Use structured lyrics with `[verse]`,
`[chorus]`, `[bridge]`, and `[instrumental]`. Set `--language` explicitly when
lyrics are not English.

## JSON Output

All CLIs are quiet by default and print one final JSON object. Success responses
include stable metadata such as:

- `ok`
- `kind`
- `mode`
- `artifacts`
- `manifests`
- `seed`
- `capability`
- `model_profile`
- `architecture`
- `models_dir`
- `resolved_models`
- `extra_loras`

Failures return a nonzero exit code and JSON with `ok:false`, `error`, and
`error_type`.

Use `--verbose` to show ComfyUI warnings, progress bars, and debug output.

Successful image, video, and music commands write a comfy-media run manifest
under `outputs/.comfy-media/runs/` by default. Use `--no-manifest` only when you
do not want gallery/index metadata.

## Media Gallery And HyperFrames Export

Index generated images, videos, and audio:

```bash
uv run comfy-media index --out outputs
```

Open the local gallery:

```bash
uv run comfy-media gallery --out outputs --host 127.0.0.1 --port 8765
```

Export an index or manifest into a minimal HyperFrames review-reel project:

```bash
uv run comfy-media export-hyperframes \
  --out outputs \
  --selection outputs/.comfy-media/index.json \
  --project-dir outputs/hyperframes-review
```

The export writes `index.html` and `comfy-media-selection.json` without requiring
HyperFrames as a Python dependency. Preview or render it with your HyperFrames
fork, for example `npx hyperframes preview outputs/hyperframes-review`.

## Development

Use `uv`:

```bash
uv sync --extra dev
uv run pytest
```

For profile/path changes:

```bash
uv run comfy-models validate
```

Install the repo git hooks:

```bash
git config core.hooksPath .githooks
```

The pre-commit hook checks that meaningful project changes include a staged
`CHANGELOG.md` update. Add an entry with:

```bash
uv run python scripts/update_changelog.py --entry "Describe the user-visible change"
```

See [AGENTS.md](AGENTS.md) for agent-facing repository guidance.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow details and
[CHANGELOG.md](CHANGELOG.md) for release notes.

Generated files belong in `outputs/`, which is ignored by git. Local config is
stored in `.comfy-agent-tools.json`, also ignored by git.

## License

MIT. See [LICENSE](LICENSE).
