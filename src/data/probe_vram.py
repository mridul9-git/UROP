"""Stage 2 Step 14 — empirical VRAM and throughput probe.

Published parameter counts do not tell you what fits in 8 GB. This measures it on
the actual card: real forward+backward+optimizer steps on synthetic batches.

Needs torch + torchvision. Creates no files on disk beyond its JSON report and
touches no dataset — safe to run before the data arrives.

WHY THIS IS NOT A SIMPLE "GROW UNTIL OOM" LOOP
----------------------------------------------
On Windows, recent NVIDIA drivers enable **CUDA System Memory Fallback** by
default: once VRAM is exhausted, allocations silently spill into host RAM across
PCIe instead of raising OutOfMemoryError. Training still "works" — 10-50x slower.

A naive probe therefore reports a huge max batch that is catastrophically slow,
which is worse than no measurement at all. So this probe sweeps a batch ladder and
records THROUGHPUT at every step. Spill has an unmistakable signature: batch size
keeps rising while images/second collapses. The reported figure is the largest
*efficient* batch, not the largest batch that avoids a crash.

Each model runs in its own subprocess so that a wedged CUDA context — which a hard
OOM can leave behind — cannot take down the rest of the run.

Run:
    python src/data/probe_vram.py                 # all four models, 320x320
    python src/data/probe_vram.py --size 224      # resolution ablation
    python src/data/probe_vram.py --model resnet152 --json    # one model (internal)
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Protocol constant: all four models train at this EFFECTIVE batch size, reached by
# gradient accumulation, so batch size cannot confound the architecture comparison
# (docs/dataset_analysis.md §14.2, D020).
EFFECTIVE_BATCH = 32

# Ladder is denser where the 8 GB limit actually bites.
CANDIDATES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]

# A batch counts as efficient while throughput stays within this fraction of the
# best throughput seen for that model. Spill drops it far below.
EFFICIENCY_FLOOR = 0.90
# Stop the sweep once throughput has clearly collapsed — no point probing further.
COLLAPSE_FLOOR = 0.70

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


def _is_oom(exc: BaseException) -> bool:
    """Recognise an out-of-memory failure across torch versions.

    torch <= 2.x raised torch.cuda.OutOfMemoryError; 2.13 surfaces the same
    condition as torch.AcceleratorError. Matching on the message keeps this working
    across both instead of silently crashing the sweep on a version bump.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "outofmemory" in text or "cuda error: oom" in text


def try_batch(name: str, batch: int, size: int, amp: bool, steps: int = 6) -> dict | None:
    """One trial: real forward+backward+step. Returns metrics, or None on OOM.

    Reports BOTH allocated and reserved peak memory. `max_memory_allocated` counts
    live tensors only; the caching allocator reserves more, and it is the reserved
    figure that must fit in the card.
    """
    import torch

    model = opt = x = y = out = loss = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = build(name).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        x = torch.randn(batch, 3, size, size, device="cuda")
        y = torch.randint(0, 2, (batch, 1), device="cuda").float()
        lossf = torch.nn.BCEWithLogitsLoss()

        t0 = None
        for i in range(steps):
            if i == 2:  # skip steps 0-1: cudnn autotune + allocator warm-up
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

        dt = (time.perf_counter() - t0) / (steps - 2)
        return {
            "batch": batch,
            "alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "sec_per_step": round(dt, 4),
            "img_per_sec": round(batch / dt, 1),
        }
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise
        return None
    finally:
        # Release BEFORE the frame dies: `finally` runs while these locals are still
        # bound, so gc.collect() alone would not free the GPU tensors.
        del model, opt, x, y, out, loss
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — a wedged context must not mask the result
            pass


def probe_model(name: str, size: int, amp: bool, cap: int) -> dict:
    """Sweep the batch ladder for one model, recording throughput at every step."""
    trials: list[dict] = []
    best_ips = 0.0
    oom_at = None

    for b in CANDIDATES:
        if b > cap:
            break
        r = try_batch(name, b, size, amp)
        if r is None:
            oom_at = b
            print(f"      batch {b:>4}  OOM", flush=True)
            break
        trials.append(r)
        best_ips = max(best_ips, r["img_per_sec"])
        ratio = r["img_per_sec"] / best_ips
        print(f"      batch {b:>4}  {r['reserved_gib']:>5.2f} GiB reserved  "
              f"{r['img_per_sec']:>7.1f} img/s  ({ratio:.0%} of peak)", flush=True)
        if ratio < COLLAPSE_FLOOR:
            print(f"      -> throughput collapsed; stopping sweep "
                  f"(system-memory spill)", flush=True)
            break

    if not trials:
        return {"error": f"OOM at the smallest batch ({CANDIDATES[0]})", "trials": []}

    efficient = [t for t in trials if t["img_per_sec"] >= EFFICIENCY_FLOOR * best_ips]
    max_eff = max(t["batch"] for t in efficient)
    eff_metrics = next(t for t in efficient if t["batch"] == max_eff)
    spilled = trials[-1]["img_per_sec"] < COLLAPSE_FLOOR * best_ips

    planned = min(max_eff, EFFECTIVE_BATCH)
    planned_metrics = next((t for t in trials if t["batch"] == planned), None)

    return {
        "max_batch_tested_ok": trials[-1]["batch"],
        "oom_at": oom_at,
        "max_efficient_batch": max_eff,
        "at_max_efficient": eff_metrics,
        "planned_micro_batch": planned,
        "accum_steps": max(1, -(-EFFECTIVE_BATCH // planned)),
        "at_planned": planned_metrics,
        "peak_img_per_sec": best_ips,
        "system_memory_spill_detected": spilled,
        "trials": trials,
    }


def run_child(name: str, size: int, amp: bool, cap: int) -> None:
    """Probe one model and emit JSON on stdout (used by the parent process)."""
    print(f"  {name}", flush=True)
    result = probe_model(name, size, amp, cap)
    print("---JSON---", flush=True)
    print(json.dumps(result), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--model", choices=list(MODELS), help="probe a single model")
    ap.add_argument("--json", action="store_true", help="child mode: emit JSON")
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

    if args.model and args.json:
        run_child(args.model, args.size, amp, args.max_batch)
        return

    dev = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"GPU: {dev.name}  |  {dev.total_memory / 2**30:.2f} GiB  |  "
          f"input {args.size}x{args.size}  |  AMP {'on' if amp else 'off'}")
    print(f"torch {torch.__version__}  |  CUDA {torch.version.cuda}")
    print(f"Free at probe start: {free_b / 2**30:.2f} GiB of {total_b / 2**30:.2f} GiB\n")

    results: dict[str, dict] = {}
    for name in MODELS:
        # Subprocess per model: a hard OOM can leave the CUDA context unusable, and
        # process isolation stops that from poisoning the remaining models.
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--model", name, "--json",
             "--size", str(args.size), "--max-batch", str(args.max_batch)]
            + (["--no-amp"] if args.no_amp else []),
            capture_output=True, text=True,
        )
        sys.stdout.write("\n".join(
            ln for ln in proc.stdout.splitlines() if not ln.startswith("---JSON---")
            and not ln.startswith("{")) + "\n")
        if "---JSON---" in proc.stdout:
            payload = proc.stdout.split("---JSON---", 1)[1].strip()
            results[name] = json.loads(payload)
        else:
            results[name] = {"error": "child process failed",
                             "stderr": proc.stderr[-800:]}
            print(f"    -> child failed: {proc.stderr.strip().splitlines()[-1:]}")

    report = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": dev.name,
        "total_gib": round(dev.total_memory / 2**30, 2),
        "free_at_start_gib": round(free_b / 2**30, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "input_size": args.size,
        "amp": amp,
        "effective_batch_target": EFFECTIVE_BATCH,
        "efficiency_floor": EFFICIENCY_FLOOR,
        "results": results,
    }
    out = Path("E:/UROP/artifacts/stage2/reports/vram_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'model':>20} {'eff.batch':>10} {'GiB':>7} {'img/s':>9} "
          f"{'micro':>6} {'accum':>6} {'spill':>6}")
    print("=" * 78)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:>20}  {r['error']}")
            continue
        e, p = r["at_max_efficient"], r["at_planned"]
        print(f"{name:>20} {r['max_efficient_batch']:>10} {e['reserved_gib']:>7.2f} "
              f"{e['img_per_sec']:>9.1f} {r['planned_micro_batch']:>6} "
              f"{r['accum_steps']:>6} "
              f"{'YES' if r['system_memory_spill_detected'] else 'no':>6}")
        if p:
            print(f"{'':>20} {'@micro':>10} {p['reserved_gib']:>7.2f} "
                  f"{p['img_per_sec']:>9.1f}")
    print("=" * 78)
    print(f"\n[ok] -> {out}")


if __name__ == "__main__":
    main()
