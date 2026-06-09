# Changelog

All notable changes to this project will be documented in this file.

This project follows the spirit of [Keep a Changelog](https://keepachangelog.com/)
and uses semantic versioning once releases are tagged.

## [Unreleased]

### Fixed

- Fix headless Comfy API node reference-image uploads by disabling progress UI
  callbacks in both client and upload helper modules.
- Fix Civitai authenticated downloads by passing CIVITAI_API_TOKEN as a query token across redirects.

### Added

- Add local WAN 2.2 T2V command, profile, downloader entry, and Dasiwa T2V
  profiles that reuse the existing Dasiwa I2V high/low UNets.
- Add optional Dasiwa LTX 2.3 Golden Lace v3 built-in video profile and
  downloader entry.
- Add remote Grok Imagine API image generation and editing commands to
  `comfy-imagegen`.
- Add remote Seedance 2.0 API video commands to comfy-videogen.
- Add capability-scoped built-in model downloader and comfy-model-downloader skill.
- Add skills-first local media generation tools for image, video, music, model
  profiles, LoRA onboarding, and setup bootstrap.
- Add `comfy-imagegen`, `comfy-videogen`, `comfy-musicgen`, and `comfy-models`
  CLIs with JSON-clean default output.
- Add Anima Preview3 turbo image generation, Qwen Image Edit 2511 editing,
  ClearReality upscaling, LTX 2.3 10Eros video generation, IA2AV video mode,
  ACE-Step 1.5 music generation, model profiles, and ad hoc LoRA support.
- Add project docs for agents, contributors, license, and changelog maintenance.

### Changed

- Switch the default Anima image generation profile to Anima Base v1.0 while
  keeping the Anima turbo LoRA.
- Switch the built-in LTX 2.3 motion-track IC-LoRA default to the Lightricks
  HDR IC-LoRA profile.
- Document fork-based pull requests as the contribution flow.
- Document that generation runs through comfy-diffusion without requiring a
  ComfyUI server.
- Update setup guidance so agents ask for `models_dir` on new machines instead
  of assuming `/mnt/models/comfyui`.
