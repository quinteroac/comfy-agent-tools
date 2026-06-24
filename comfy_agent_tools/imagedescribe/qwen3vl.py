"""Qwen3-VL Instruct image description execution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .config import ImagedescribeConfig
from comfy_agent_tools.imagegen.runtime import require_comfy_runtime


def run_qwen3vl_describe(
    *,
    image: Image.Image,
    prompt: str,
    config: ImagedescribeConfig,
) -> str:
    """Describe an image with a Qwen3-VL Instruct VLM and return the generated text."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    require_comfy_runtime()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = config.resolve_model_path(config.llm)
    if not _looks_like_instruct_dir(model_path):
        raise FileNotFoundError(
            "Qwen3-VL Instruct model directory not found. The --llm path must point to a "
            "HuggingFace Qwen3-VL Instruct folder (with config.json and model.safetensors), "
            f"not a single text-encoder safetensors file. Looked at: {model_path}"
        )

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=dtype,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(str(model_path))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(device)

    do_sample = config.do_sample
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_length,
        "do_sample": do_sample,
        "repetition_penalty": config.repetition_penalty,
        "pad_token_id": processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update(
            {
                "temperature": config.temperature,
                "top_k": config.top_k,
                "top_p": config.top_p,
                "min_p": config.min_p,
            }
        )
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
    else:
        # Greedy decoding ignores temperature/top-k/top-p.
        gen_kwargs["num_beams"] = 1

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


def _looks_like_instruct_dir(path: Path) -> bool:
    """Return True when path is a HF model directory with config + weights."""
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    has_weights = any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    return has_weights
