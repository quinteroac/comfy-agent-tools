# comfy-agent-tools

Skills-first local media generation tools backed by
[`comfy-diffusion`](https://github.com/quinteroac/comfy-diffusion).

`comfy-agent-tools` gives agents a small set of skills and Python CLIs for local
image, video, and music generation. Users install the skills; the skills teach
the agent how to install or update the Python tools with `uv`, validate the local
runtime, initialize model profiles, and run generation commands.

Local model files are not distributed with this repo. Local Anima, Qwen,
Ideogram 4, LTX, ACE-Step, and upscaler profiles use model files under the
user's configured `models_dir`; remote API profiles such as Seedance 2.0 and
Grok Imagine use provider credentials instead of local weights.

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

- `comfy-imagegen`: image generation, Ideogram 4 structured prompting, Krea2 Turbo, image editing, upscaling, and remote Grok Imagine.
- `comfy-imagedescribe`: local Qwen3-VL 2B Instruct image description, captioning, tagging, and visual QA.
- `comfy-videogen`: local LTX 2.3/WAN 2.2 video plus remote Seedance 2.0 API video.
- `comfy-bernini-videoedit`: Bernini WAN 2.2 video edit workflows for V2V, RV2V, VV2V planning, and R2V.
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

The Python runtime dependency is `comfy-diffusion[video,audio]` pinned to
`v2.4.5` or newer for LTX 2.3 HDR IC-LoRA helpers, Ideogram 4, Bernini video
editing, the VAE decode fix required by LTX 2.3 video, and the Krea2 Turbo
pipeline. ComfyUI runtime packages are pinned directly from the ComfyUI
`0.24.1` requirements vendored by `comfy-diffusion`, because the media extras
are required during runtime startup,
video generation writes MP4 files with audio, and music generation uses audio
helpers.

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
checkpoints/DasiwaLTX23_goldenLaceV3.safetensors
text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
loras/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors
loras/ltx23/gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors
loras/ltx23/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors
latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors
diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors
diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors
diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinHighV81.safetensors
diffusion_models/DasiwaWAN22I2V14BV8V1_tastysinLowV81.safetensors
diffusion_models/DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors
diffusion_models/DasiwaWAN22I2V14BLightspeed_boundbiteLowV10.safetensors
diffusion_models/DasiwaWan2214BS2V_littledemonV2.safetensors
text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
audio_encoders/wav2vec2_large_english_fp16.safetensors
vae/wan_2.1_vae.safetensors

diffusion_models/acestep_v1.5_base.safetensors
text_encoders/qwen_0.6b_ace15.safetensors
text_encoders/qwen_1.7b_ace15.safetensors
vae/ace_1.5_vae.safetensors

diffusion_models/ideogram4_fp8_scaled.safetensors
diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors
text_encoders/qwen3vl_8b_fp8_scaled.safetensors
vae/flux2-vae.safetensors
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
| `imagegen.ideogram4-generate` | `ideogram4-fp8` | `ideogram4` |
| `imagegen.krea2-generate` | `krea2-turbo` | `krea2` |
| `videogen.t2v` | `ltx23-10eros` | `ltx23` |
| `videogen.i2v` | `ltx23-10eros` | `ltx23` |
| `videogen.flf2v` | `ltx23-10eros` | `ltx23` |
| `videogen.ia2av` | `ltx23-10eros` | `ltx23` |
| `videogen.motion-track` | `ltx23-motion-track` | `ltx23` |
| `videogen.wan22-t2v` | `wan22-t2v` | `wan22` |
| `videogen.wan22-i2v` | `wan22-i2v` | `wan22` |
| `videogen.wan22-flf2v` | `wan22-i2v` | `wan22` |
| `videogen.wan22-s2v` | `wan22-s2v` | `wan22` |
| `videogen.wan22-video-audio` | `wan22-dasiwa-littledemon-v2-video-audio` | `wan22` |
| `videogen.wan22-bernini` | `wan22-bernini` | `wan22` |
| `videogen.rtx-upscale` | `rtx-vsr` | `rtx-vsr` |
| `videogen.seedvr2-upscale` | `seedvr2` | `seedvr2` |
| `videogen.seedance2-t2v` | `seedance2-api` | `seedance2-api` |
| `videogen.seedance2-r2v` | `seedance2-api` | `seedance2-api` |
| `videogen.seedance2-flf2v` | `seedance2-api` | `seedance2-api` |
| `musicgen.generate` | `ace15-base` | `ace-step-1.5` |
| `imagedescribe.describe` | `qwen3vl-2b-instruct` | `qwen3-vl` |

`ltx23` is the architecture/adapter; `ltx23-10eros` is the default built-in
profile validated for that architecture. The optional
`ltx23-dasiwa-golden-lace-v3` profile points to Dasiwa LTX 2.3 Golden Lace v3
and reuses the same LTX 2.3 text encoder, distilled LoRA, text-encoder LoRA, and
latent upscaler files as the 10Eros profile.

`wan22` is the local WAN 2.2 adapter. The default `wan22-t2v` profile uses the
Comfy-Org FP8 T2V high/low UNets, while `wan22-i2v` uses the Comfy-Org FP8 I2V
high/low UNets. The optional `wan22-dasiwa-tastysin-t2v`,
`wan22-dasiwa-boundbite-t2v`, `wan22-dasiwa-tastysin-i2v`, and
`wan22-dasiwa-boundbite-i2v` profiles point to local Dasiwa I2V high/low UNets.
Dasiwa T2V defaults to `steps=8`, `high_steps=2`, `low_steps=6`, `cfg=1.0`;
Dasiwa I2V/FLF2V uses `steps=4`, `cfg=1.0`. The WAN 2.2 wrappers sample the high-noise model
in the first tranche and the low-noise model in the second. The high-noise
tranche controls broad motion, while the low-noise tranche controls
detail/refinement. By default WAN splits steps 50/50; pass `--high-steps` and
`--low-steps` to bias motion vs detail. Dasiwa T2V profiles intentionally reuse
I2V checkpoints, so validate quality with a GPU smoke test before treating them
as production defaults.

The `wan22-s2v` profile uses the Wan 2.2 S2V FP8 model plus wav2vec2 audio
encoder for reference-image and input-audio driven video. Defaults mirror the
native ComfyUI workflow: `length=77`, `fps=16`, `steps=20`, `cfg=6.0`,
`sampler=uni_pc`, `scheduler=simple`, and `shift=8.0`.

The optional `wan22-dasiwa-littledemon-v2-s2v` profile points to Dasiwa
LittleDemon v2 S2V. Its fast distillation is baked into the checkpoint, so the
profile uses `steps=4`, `cfg=1.0`, `sampler=euler`, `scheduler=simple`, and
`shift=10.0`; do not add extra Lightning/speed-up LoRAs on top of it.

The `wan22-bernini` profile uses the Bernini WAN 2.2 video-edit UNets plus the
LightX2V LoRA and the NSFW WAN UMT5 text encoder. It supports source-video
editing, reference-guided generation, and multi-reference conditioning through
`comfy-diffusion` v2.4.5 or newer.
The `wan22-dasiwa-littledemon-v2-video-audio` profile is the default for
`videogen.wan22-video-audio`; it reuses the Dasiwa S2V model for 16 fps
video+audio processing with 77-frame chunks and 4-frame crossfade overlap.

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

The built-in `ideogram4-fp8` profile is local text-to-image generation with
architecture `ideogram4`. It uses the Comfy-Org Ideogram 4 FP8 files and does
not use `COMFY_ORG_API_KEY`. Ideogram 4 accepts plain text, but works best with
structured JSON prompts that include `compositional_deconstruction`, typed
`obj`/`text` elements, uppercase `#RRGGBB` palettes, and optional `bbox`
coordinates in `[y_min, x_min, y_max, x_max]` order normalized from `0` to
`1000`.

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
uv run comfy-models download-profile ltx23-dasiwa-golden-lace-v3 --dry-run
uv run comfy-models download-profile flux-klein-9b-snofs --dry-run
uv run comfy-models download imagegen.ideogram4-generate --dry-run
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
  ideogram4/
    poster-detail.safetensors
  ltx23/
    camera-static.safetensors
    detailer.safetensors
  wan22/
    cinematic-motion.safetensors
    low-noise-detail.safetensors
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

For local WAN 2.2 I2V/FLF2V, `--extra-lora` applies the LoRA to both high-noise
and low-noise UNets. Use `--extra-lora-high` or `--extra-lora-low` when the LoRA
should affect only one pass. WAN uses a shared text encoder, so any nonzero
`CLIP_STRENGTH` patches that shared text encoder once for the given LoRA flag.
For local Ideogram 4, `--extra-lora` applies the LoRA to both the main UNet and
the unconditional UNet. Any nonzero `CLIP_STRENGTH` patches the shared Ideogram
text encoder once.

Example:

```bash
uv run comfy-imagegen generate \
  --prompt "masterpiece, best quality, score_7, safe, realistic portrait, soft window light" \
  --extra-lora /mnt/models/comfyui/loras/anima/realism-portrait.safetensors:0.8:0.0 \
  --out outputs
```

```bash
uv run comfy-imagegen ideogram4-generate \
  --prompt "A poster for a jazz night" \
  --style-aesthetics "minimal, geometric, high contrast" \
  --style-lighting "flat graphic design lighting" \
  --style-medium "graphic_design" \
  --style-art-style "clean vector poster, sans-serif typography" \
  --background "Matte black paper background with subtle grain." \
  --text "90,120,260,880|JAZZ NIGHT|Large cream uppercase headline." \
  --extra-lora /mnt/models/comfyui/loras/ideogram4/poster-detail.safetensors:0.8:0.0 \
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

### Ideogram 4 Images

Ideogram 4 runs locally through `comfy-diffusion` and writes PNG files. Download
only its profile when needed:

```bash
uv run comfy-models download imagegen.ideogram4-generate --dry-run
uv run comfy-models download imagegen.ideogram4-generate --yes
```

Ideogram 4 uses structured JSON internally. The CLI builds that JSON from
required layout/style flags; do not pass raw JSON on the command line.

```bash
uv run comfy-imagegen ideogram4-generate \
  --prompt "A modern concert poster for a jazz trio." \
  --style-aesthetics "minimal, elegant, high contrast" \
  --style-lighting "flat graphic design lighting" \
  --style-medium "graphic_design" \
  --style-art-style "clean vector poster, sans-serif typography" \
  --style-color "#101010" \
  --style-color "#F4D35E" \
  --background "Solid black background with subtle paper texture." \
  --object "420,120,900,880|A golden saxophone centered in the lower half." \
  --text "80,120,220,880|JAZZ NIGHT|Large condensed yellow headline." \
  --output-json outputs/ideogram4-prompt.json \
  --out outputs
```

Bboxes are `y_min,x_min,y_max,x_max` on a `0..1000` normalized canvas, not pixel
coordinates. `--prompt`, `--style-aesthetics`, `--style-lighting`,
`--style-medium`, and `--background` are required. Pass exactly one of
`--style-photo` or `--style-art-style`, plus at least one `--object` or `--text`
element. Add `--output-json PATH` to write the structured prompt JSON that the
CLI sent to Ideogram 4.

### Krea2 Turbo Images

Krea2 Turbo runs locally through `comfy-diffusion` (v2.4.5+) and writes PNG
files. Download only its profile when needed:

```bash
uv run comfy-models download imagegen.krea2-generate --dry-run
uv run comfy-models download imagegen.krea2-generate --yes
```

Generate an image:

```bash
uv run comfy-imagegen krea2-generate \
  --prompt "a cinematic portrait of an astronaut floating in a nebula, dramatic rim light" \
  --width 1024 \
  --height 1024 \
  --steps 8 \
  --cfg 1 \
  --rebalance-multiplier 4.0 \
  --out outputs
```

Krea2 Turbo defaults to `steps=8`, `cfg=1.0`, `sampler=euler`, `scheduler=simple`,
`rebalance_multiplier=4.0`, and `seed=0`. The `--rebalance-multiplier` controls
the Krea2 conditioning rebalance applied before sampling; lower it for more
literal prompt adherence or raise it for more creative interpretation.

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

`comfy-videogen` supports local LTX 2.3 generation, local WAN 2.2 text/image-driven
and sound-driven generation, and remote Seedance 2.0 API generation. It is quiet by default and prints final JSON only; pass
`--verbose` to show ComfyUI logs.

Local LTX 2.3 uses the `ltx23-10eros` profile by default and writes MP4 files
with audio. To use Dasiwa LTX 2.3 Golden Lace v3, set
`ltx23-dasiwa-golden-lace-v3` as the default for the desired LTX capabilities
(`videogen.t2v`, `videogen.i2v`, `videogen.flf2v`, or `videogen.ia2av`).
Local WAN 2.2 T2V uses the `wan22-t2v` profile by default. I2V/FLF2V uses the
`wan22-i2v` profile. These modes write silent MP4 files and accept ad hoc LoRAs
with `--extra-lora` for both UNets,
`--extra-lora-high` for the high-noise pass, and `--extra-lora-low` for the
low-noise pass. WAN 2.2 S2V and video+audio modes mux the input audio into the
output.
For Dasiwa TastySin or BoundBite, set the matching Dasiwa T2V/I2V profile as the
default or pass its high/low UNet paths explicitly; use `--steps 4 --cfg 1.0`.
For Dasiwa LittleDemon S2V, set `wan22-dasiwa-littledemon-v2-s2v` as the
`videogen.wan22-s2v` default or pass its checkpoint with `--unet`; use the
profile defaults for 4-step generation.
The `wan22-video-audio` command uses Dasiwa LittleDemon by default and accepts
two presets: `audio-driven` for full-frame audio-reactive V2V, and `lipsync` for
external-mask mouth recomposition. V1 requires a 16 fps input video.
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

WAN 2.2 text to video:

```bash
uv run comfy-videogen wan22-t2v \
  --prompt "a cinematic slow dolly through a sunlit atelier, elegant fabric motion, realistic camera drift" \
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
  --extra-lora-high /mnt/models/comfyui/loras/wan22/cinematic-motion.safetensors:0.7 \
  --out outputs
```

WAN 2.2 first/last frame:

```bash
uv run comfy-videogen wan22-flf2v \
  --first start.png \
  --last end.png \
  --prompt "a smooth cinematic transition between the two frames, coherent motion" \
  --extra-lora-low /mnt/models/comfyui/loras/wan22/low-noise-detail.safetensors:0.6 \
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

WAN 2.2 video plus audio, full-frame audio-driven V2V:

```bash
uv run comfy-videogen wan22-video-audio \
  --mode audio-driven \
  --input-video input.mp4 \
  --audio speech-or-music.wav \
  --out outputs
```

WAN 2.2 video plus audio lipsync with an external mask:

```bash
uv run comfy-videogen wan22-video-audio \
  --mode lipsync \
  --input-video input.mp4 \
  --audio speech.wav \
  --mask-video mouth-mask.mp4 \
  --out outputs
```

WAN 2.2 Bernini video edit with a source video and reference image:

```bash
uv run comfy-videogen wan22-bernini \
  --input-video input.mp4 \
  --reference-image replacement-subject.png \
  --prompt "Replace the character with the subject in image 0. Keep camera motion, lighting, and background unchanged." \
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
pick a specific window. `wan22-video-audio` processes the whole input video in
chunks; the input video must already be 16 fps, and `lipsync` requires
`--mask-video` or `--mask-image`.

Audio muxing is required for LTX audiovisual modes, WAN 2.2 S2V, and WAN 2.2
video+audio. If audio cannot be written into the MP4, the command returns
`ok:false` instead of saving a silent video. WAN 2.2 I2V/FLF2V intentionally
save silent MP4 files.

### RTX Video Super Resolution

`rtx-upscale` upscales an existing MP4 with NVIDIA RTX Video Super Resolution.
It wraps the same `nvidia-vfx` backend used by
`Comfy-Org/Nvidia_RTX_Nodes_ComfyUI` and requires a CUDA-capable NVIDIA RTX GPU.
It does not use local model files, `models_dir`, LoRAs, or the model downloader.
If `nvidia-vfx` is not installed, the CLI returns `missing_dependency`.

```bash
uv run comfy-videogen rtx-upscale \
  --input-video outputs/source.mp4 \
  --resolution 1080p \
  --quality ULTRA \
  --out outputs
```

Supported target presets are `480p`, `720p`, `1080p`, `1440p`, `4k`, and `8k`.
Presets preserve the input aspect ratio by fitting inside the target bounds; for
example, square input plus `--resolution 1080p` outputs `1080x1080`, and 4:3
input outputs `1440x1080`. Use `--width` plus `--height` for exact custom
dimensions, or `--scale 2.0` for a multiplier from the input dimensions. Output
dimensions are rounded to multiples of 8. If the input MP4 has an audio track,
the output re-encodes and muxes it into the upscaled MP4 (`audio_muxed=true`);
inputs without audio remain silent.

### SeedVR2 Video Upscaler

`seedvr2-upscale` upscales an existing MP4 with
`numz/ComfyUI-SeedVR2_VideoUpscaler`. The wrapper fetches the pinned upstream
repo on first use, then SeedVR2 downloads its DiT/VAE model files automatically
when they are missing. It does not use `comfy-models download`,
`COMFY_ORG_API_KEY`, or LoRAs.

```bash
uv run comfy-videogen seedvr2-upscale \
  --input-video outputs/source.mp4 \
  --resolution 1080p \
  --models-dir /mnt/models/seedvr2 \
  --out outputs
```

Supported target presets are `720p`, `1080p`, `1440p`, and `4k`. Presets map to
SeedVR2 short-edge targets and long-edge caps: `720p` is `720/1280`, `1080p` is
`1080/1920`, `1440p` is `1440/2560`, and `4k` is `2160/3840`, preserving input
aspect ratio. Use `--max-edge` to override the preset long-edge cap. Use
`--models-dir` (alias: `--model-dir`) to choose where SeedVR2 downloads/loads
its DiT/VAE weights; when omitted, SeedVR2 keeps its upstream default. Advanced
controls include `--model`, `--batch-size`, `--chunk-size`, `--temporal-overlap`,
`--cuda-device`, `--blocks-to-swap`, and `--video-backend`. If the input MP4 has
an audio track, the output muxes it into the upscaled MP4 (`audio_muxed=true`);
inputs without audio remain silent.

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

## Image Description

`comfy-imagedescribe` describes images with the local Qwen3-VL 2B Instruct
vision-language model (a HuggingFace model directory under
`LLM/Qwen-VL/Qwen3-VL-2B-Instruct`, loaded with `transformers`). It is quiet by
default and prints final JSON only; pass `--verbose` to show runtime logs.

Describe an image in detail:

```bash
uv run comfy-imagedescribe describe \
  --input outputs/comfy-imagegen-generate-example.png \
  --prompt "Describe this image in detail." \
  --out outputs
```

Short caption:

```bash
uv run comfy-imagedescribe describe \
  --input outputs/comfy-imagegen-generate-example.png \
  --prompt "Write a concise one-sentence caption for this image." \
  --max-length 128 \
  --out outputs
```

Visual QA:

```bash
uv run comfy-imagedescribe describe \
  --input outputs/comfy-imagegen-generate-example.png \
  --prompt "What is the dominant color palette, and how many people are visible?" \
  --out outputs
```

Current description defaults:

- `max_length=512`
- `temperature=0.7`, `top_k=64`, `top_p=0.95`, `min_p=0.05`
- `repetition_penalty=1.05`
- `do_sample=true`, `seed=0`

Read the generated text from the `description` field of the JSON response. Pass
`--greedy` for reproducible descriptions; raise `--max-length` for long-form
output and lower it for short captions or tags.

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
