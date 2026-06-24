"""Validate that Krea2 conditioning rebalance is actually applied.

Instruments comfy_diffusion.conditioning.rebalance_krea2_conditioning during a
real GPU run and checks:
  1. The function is called with the expected multiplier and per-layer weights.
  2. The output tensor differs from the input (not a no-op).
  3. The scaling matches the documented behavior (multiplier * per-layer gains).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comfy_agent_tools.imagegen.krea2_config import Krea2Config
from comfy_agent_tools.imagegen.krea2 import run_krea2_t2i


def main() -> int:
    import comfy_diffusion.conditioning as cond

    calls: list[dict] = []
    original = cond.rebalance_krea2_conditioning

    def spy(conditioning, multiplier=4.0, per_layer_weights=cond.KREA2_REBALANCE_DEFAULT_WEIGHTS):
        import torch

        # Capture a reference to the input tensor for comparison.
        ref_in = None
        if isinstance(conditioning, list) and conditioning:
            entry = conditioning[0]
            if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], torch.Tensor):
                ref_in = entry[0].detach().clone()

        result = original(conditioning, multiplier=multiplier, per_layer_weights=per_layer_weights)

        ref_out = None
        if isinstance(result, list) and result:
            entry = result[0]
            if isinstance(entry, (list, tuple)) and len(entry) == 2 and isinstance(entry[0], torch.Tensor):
                ref_out = entry[0].detach().clone()

        calls.append({
            "multiplier": multiplier,
            "per_layer_weights": tuple(per_layer_weights) if per_layer_weights is not None else None,
            "input_shape": tuple(ref_in.shape) if ref_in is not None else None,
            "output_shape": tuple(ref_out.shape) if ref_out is not None else None,
            "input_mean": float(ref_in.float().mean()) if ref_in is not None else None,
            "output_mean": float(ref_out.float().mean()) if ref_out is not None else None,
            "input_abs_mean": float(ref_in.float().abs().mean()) if ref_in is not None else None,
            "output_abs_mean": float(ref_out.float().abs().mean()) if ref_out is not None else None,
            "changed": (ref_in is not None and ref_out is not None and not torch.equal(ref_in, ref_out)),
        })
        return result

    cond.rebalance_krea2_conditioning = spy
    # Also patch the symbol imported into the pipeline module namespace.
    import comfy_diffusion.pipelines.image.krea2.turbo as turbo
    turbo.rebalance_krea2_conditioning = spy

    try:
        config = Krea2Config()
        images = run_krea2_t2i(
            prompt="a small red apple on a wooden table, soft light",
            width=768,
            height=768,
            config=config,
        )
    finally:
        cond.rebalance_krea2_conditioning = original
        turbo.rebalance_krea2_conditioning = original

    print(f"images produced: {len(images)} ({images[0].size if images else 'none'})")
    print(f"rebalance calls: {len(calls)}")
    for i, c in enumerate(calls):
        print(f"--- call {i} ---")
        for k, v in c.items():
            print(f"  {k}: {v}")

    if not calls:
        print("FAIL: rebalance was never called")
        return 1

    c = calls[0]
    expected_weights = tuple(cond.KREA2_REBALANCE_DEFAULT_WEIGHTS)
    ok = True
    if c["multiplier"] != 4.0:
        print(f"FAIL: multiplier={c['multiplier']} != 4.0")
        ok = False
    if c["per_layer_weights"] != expected_weights:
        print(f"FAIL: per_layer_weights={c['per_layer_weights']} != {expected_weights}")
        ok = False
    if not c["changed"]:
        print("FAIL: output tensor identical to input (no-op)")
        ok = False
    if c["output_abs_mean"] is not None and c["input_abs_mean"] is not None:
        ratio = c["output_abs_mean"] / c["input_abs_mean"] if c["input_abs_mean"] else 0
        print(f"abs_mean ratio out/in: {ratio:.4f}")
        # With multiplier=4.0 and per-layer weights averaging ~1.8, the global
        # abs-mean scaling should be meaningfully > 1 and < ~10.
        if not (1.5 < ratio < 12.0):
            print(f"FAIL: abs_mean ratio {ratio} outside expected range")
            ok = False
    if ok:
        print("PASS: rebalance was applied with the expected multiplier and weights")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
