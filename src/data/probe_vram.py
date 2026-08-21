"""Stage 2 Step 14 — empirical VRAM and throughput probe.

Published parameter counts do not tell you what fits in 8 GB. This script measures
it: for each candidate backbone it runs real forward+backward steps on synthetic
batches at the configured input size, doubling the batch until it OOMs, and reports
the largest batch that survives plus its throughput.

Needs torch + torchvision (not required by any other Stage 2 script). It creates
no models on disk and touches no dataset — safe to run before the data arrives.

Run:
    python src/data/probe_vram.py
    python src/data/probe_vram.py --size 224 --no-amp
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = {
    "resnet152": ("torchvision.models", "resnet152"),
    "densenet201": ("torchvision.models", "densenet201"),
    "efficientnet_v2_s": ("torchvision.models", "efficientnet_v2_s"),
    "mobilenet_v3_large": ("torchvision.models", "mobilenet_v3_large"),
}


def build(name: str):
    import importlib

    mod_name, fn_name = MODELS[name]
    fn = getattr(importlib.import_module(mod_name), fn_name)
    return fn(weights=None)  # weights irrelevant for a memory probe


def try_batch(name: str, batch: int, size: int, amp: bool, steps: int = 4) -> dict | None:
    """One trial. Returns metrics, or None if it OOMs."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = build(name).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        x = torch.randn(batch, 3, size, size, device="cuda")
        y = torch.randint(0, 2, (batch, 1), device="cuda").float()
        lossf = torch.nn.BCEWithLogitsLoss()

        for i in range(steps):
            if i == 1:  # exclude the first step (cudnn autotune) from timing
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(x)
                loss = lossf(out[:, :1], y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / (steps - 1)
        peak = torch.cuda.max_memory_allocated() / 2**30
        return {"batch": batch, "peak_gib": round(peak, 2),
                "sec_per_step": round(dt, 4),
                "img_per_sec": round(batch / dt, 1)}
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        # The frame's locals are released when this function returns; collect and
        # release the caching allocator so the next trial starts from a clean card.
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed. See requirements.txt for the cu126 install line.")
        return
    if not torch.cuda.is_available():
        print("CUDA not available — probe requires a GPU.")
        return

    amp = not args.no_amp
    dev = torch.cuda.get_device_properties(0)
    print(f"GPU: {dev.name}  |  {dev.total_memory / 2**30:.1f} GiB  |  "
          f"input {args.size}x{args.size}  |  AMP {'on' if amp else 'off'}\n")

    results = {}
    for name in MODELS:
        best, batch = None, 4
        while batch <= args.max_batch:
            r = try_batch(name, batch, args.size, amp)
            if r is None:
                break
            best, batch = r, batch * 2
        if best:
            results[name] = best
            print(f"{name:>20}  max batch {best['batch']:>4}  "
                  f"peak {best['peak_gib']:>5.2f} GiB  "
                  f"{best['img_per_sec']:>7.1f} img/s")
        else:
            results[name] = {"error": "OOM at batch 4"}
            print(f"{name:>20}  OOM even at batch 4")

    payload = {
        "gpu": dev.name,
        "total_gib": round(dev.total_memory / 2**30, 2),
        "input_size": args.size,
        "amp": amp,
        "results": results,
    }
    out = Path("E:/UROP/artifacts/stage2/reports/vram_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[ok] -> {out}")
    print("\nSet the per-model micro-batch below these maxima and use gradient "
          "accumulation to reach an identical EFFECTIVE batch size across all four "
          "models — otherwise batch size confounds the architecture comparison.")


if __name__ == "__main__":
    main()
