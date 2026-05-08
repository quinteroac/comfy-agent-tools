# comfy-agent-tools

Skills-first local media generation tools backed by
[`comfy-diffusion`](https://github.com/quinteroac/comfy-diffusion).

`comfy-agent-tools` gives agents a small set of skills and Python CLIs for local
image, video, and music generation. Users install the skills; the skills teach
the agent how to install or update the Python tools with `uv`, validate the local
runtime, initialize model profiles, and run generation commands.

Local model files are not distributed with this repo. Local Anima, Qwen, LTX,
ACE-Step, and upscaler profiles use model files under the user's configured
`models_dir`; remote API profiles such as Seedance 2.0 use provider credentials
instead of local weights.

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

- `comfy-imagegen`: image generation, image editing, and upscaling.
- `comfy-videogen`: local LTX 2.3 video plus remote Seedance 2.0 API video.
- `comfy-motion-track-control`: LTX 2.3 Motion Track IC-LoRA guidance.
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
to `v2.2.0` or newer for LTX 2.3 Motion Track IC-LoRA helpers. The
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
explicitly asked. Remote API profiles such as `seedance2-api` are not downloaded
by `comfy-models`.

Use `HF_TOKEN` for gated Hugging Face repositories and `CIVITAI_API_TOKEN` if a
Civitai direct download requires authentication.

Built-in profiles assume these local files exist:

```text
diffusion_models/animaOfficial_preview3Base.safetensors
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
loras/ltx23/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors
latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

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
| `imagegen.generate` | `anima-preview3-turbo` | `anima` |
| `imagegen.edit` | `qwen-edit2511` | `qwen-image-edit` |
| `imagegen.upscale` | `clear-reality` | `upscale-model` |
| `videogen.t2v` | `ltx23-10eros` | `ltx23` |
| `videogen.i2v` | `ltx23-10eros` | `ltx23` |
| `videogen.flf2v` | `ltx23-10eros` | `ltx23` |
| `videogen.ia2av` | `ltx23-10eros` | `ltx23` |
| `videogen.motion-track` | `ltx23-motion-track` | `ltx23` |
| `videogen.seedance2-t2v` | `seedance2-api` | `seedance2-api` |
| `videogen.seedance2-r2v` | `seedance2-api` | `seedance2-api` |
| `videogen.seedance2-flf2v` | `seedance2-api` | `seedance2-api` |
| `musicgen.generate` | `ace15-base` | `ace-step-1.5` |

`ltx23` is the architecture/adapter; `ltx23-10eros` is the built-in profile
validated for that architecture.

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

Download missing files for the effective profile of a capability:

```bash
uv run comfy-models download imagegen.generate --dry-run
uv run comfy-models download imagegen.generate --yes
uv run comfy-models download-profile anima-preview3-turbo --yes
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

`comfy-imagegen generate` defaults to Anima Preview3 with the Anima turbo LoRA.
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

Upscale uses ClearReality:

```bash
uv run comfy-imagegen upscale \
  --input outputs/comfy-imagegen-edit-example.png \
  --out outputs
```

Qwen Image Edit may rescale internally, so final dimensions can differ from the
input or requested canvas. Read the final JSON metadata for actual dimensions.

## Video Generation

`comfy-videogen` supports both local LTX 2.3 generation and remote Seedance 2.0
API generation. It is quiet by default and prints final JSON only; pass
`--verbose` to show ComfyUI logs.

Local LTX 2.3 uses the `ltx23-10eros` profile and writes MP4 files with audio.

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

Motion-track IC-LoRA from an input image plus prepared trajectory control video:

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
uses the `ref0.5` IC-LoRA and records `reference_downscale=0.5`.

This sizing rule applies only to local LTX 2.3 modes. It does not apply to
Seedance 2.0 remote API modes or other non-LTX pipelines. Local LTX 2.3 runs a
two-step pipeline with latent x2 spatial upscaling/refinement, so treat
`--width` and `--height` as the base generation size, not the desired final MP4
size. If the desired final output is `1080x720`, run with `--width 540 --height
360`; requesting `--width 1080 --height 720` would try to produce about
`2160x1440`, which can cause OOM and is not the requested output. Read the JSON
metadata for actual `width` and `height`.

For `ia2av`, `--length` and `--fps` control video duration. Long audio files,
including 120s WAVs from `comfy-musicgen`, are trimmed to `length / fps` by
default; use `--audio-start-time` and `--audio-duration` to pick a specific
window.

Audio muxing is required. If audio cannot be written into the MP4, the command
returns `ok:false` instead of saving a silent video.

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
