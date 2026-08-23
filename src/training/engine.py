"""Stage 3 training and evaluation loops.

Two behaviours here are worth stating explicitly because getting either wrong
produces plausible-looking but wrong numbers:

Gradient accumulation. The effective batch is micro_batch x accum_steps = 32.
Loss is divided by accum_steps before backward so the gradient magnitude equals
what a true batch-32 step would produce; without that division the learning
rate would silently mean something different at accum_steps=2 than at 1. A
trailing partial accumulation window at the end of an epoch is flushed rather
than dropped, so no sample is silently discarded.

AMP. torch.amp autocast + GradScaler. The scaler must step the optimizer, and
gradient clipping (if enabled) must happen after unscale_. Doing it before
would clip scaled gradients, i.e. clip at an arbitrary threshold.

A note specific to this machine: Windows CUDA System Memory Fallback means an
oversized batch does not raise OOM, it silently spills to host RAM at ~10x
slowdown (docs/dataset_analysis.md section 14.2.1). Never tune micro_batch by
raising it until something crashes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import binary_metrics, metrics_table, select_threshold  # noqa: E402


def _autocast(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type=device_type, enabled=enabled)


def build_optimizer(model, cfg: dict):
    o = cfg["optim"]
    name = o["name"].lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=float(o["lr"]),
            weight_decay=float(o["weight_decay"]),
            betas=tuple(o.get("betas", (0.9, 0.999))),
            eps=float(o.get("eps", 1e-8)),
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=float(o["lr"]),
            momentum=float(o.get("momentum", 0.9)),
            weight_decay=float(o["weight_decay"]), nesterov=True,
        )
    raise ValueError(f"optim.name={o['name']!r} not supported (adamw | sgd)")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Epoch-level scheduler with optional linear warmup.

    Warmup matters here because the classifier head is randomly initialised
    while the backbone is pretrained; a large first step can wreck the
    pretrained features before the head is doing anything useful.
    """
    s = cfg["scheduler"]
    name = str(s["name"]).lower()
    epochs = int(cfg["train"]["epochs"])
    warmup = int(s.get("warmup_epochs", 0))

    if name == "none":
        return None
    if name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup), eta_min=float(s.get("min_lr", 0.0)),
        )
    elif name == "step":
        main = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(s.get("step_size", 5)),
            gamma=float(s.get("gamma", 0.1)),
        )
    else:
        raise ValueError(f"scheduler.name={s['name']!r} not supported")

    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, main], milestones=[warmup],
        )
    return main


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, *,
                    accum_steps: int = 1, amp: bool = True,
                    grad_clip_norm: float = 0.0, epoch: int = 0,
                    log_every: int = 50) -> dict:
    model.train()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    running_loss, n_seen, n_steps = 0.0, 0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    n_batches = len(loader)
    for i, (images, targets, _idx) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).unsqueeze(1)

        with _autocast(device_type, amp):
            logits = model(images)
            loss = criterion(logits, targets)

        # Divide so the accumulated gradient equals a true effective-batch step.
        scaler.scale(loss / accum_steps).backward()

        is_step = ((i + 1) % accum_steps == 0) or (i + 1 == n_batches)
        if is_step:
            if grad_clip_norm and grad_clip_norm > 0:
                # Must unscale before clipping, or the threshold is meaningless.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            n_steps += 1

        bs = images.size(0)
        running_loss += float(loss.detach()) * bs
        n_seen += bs

        if log_every and (i + 1) % log_every == 0:
            print(f"    epoch {epoch} batch {i + 1}/{n_batches} "
                  f"loss {running_loss / max(n_seen, 1):.4f}")

    elapsed = time.time() - t0
    return {
        "train_loss": running_loss / max(n_seen, 1),
        "images": n_seen,
        "optimizer_steps": n_steps,
        "seconds": round(elapsed, 2),
        "images_per_sec": round(n_seen / elapsed, 1) if elapsed > 0 else None,
    }


@torch.no_grad()
def predict(model, loader, device, *, amp: bool = True) -> tuple:
    """Return (targets, probabilities, indices) for a whole split.

    Probabilities, not logits: calibration and threshold selection both need the
    sigmoid applied, and doing it in exactly one place stops one caller
    thresholding logits at 0.5 by mistake.
    """
    model.eval()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    ys, ps, ix = [], [], []

    for images, targets, idx in loader:
        images = images.to(device, non_blocking=True)
        with _autocast(device_type, amp):
            logits = model(images)
        probs = torch.sigmoid(logits.float()).squeeze(1)
        ys.append(targets.numpy())
        ps.append(probs.detach().cpu().numpy())
        ix.append(np.asarray(idx))

    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(ix))


def evaluate(model, loader, device, *, threshold: float | None = None,
             threshold_policy: str = "f1", fixed_threshold: float = 0.5,
             amp: bool = True, split_name: str = "", calibration: bool = True,
             calibration_bins: int = 10,
             fallback_test_caveat: bool = False) -> dict:
    """Score one split.

    If `threshold` is None the threshold is SELECTED on this split - only ever
    correct for validation. For test, pass the threshold chosen on validation.
    """
    y_true, y_prob, idx = predict(model, loader, device, amp=amp)

    if threshold is None:
        threshold, thr_info = select_threshold(
            y_true, y_prob, policy=threshold_policy, fixed=fixed_threshold,
        )
        thr_info["selected_on"] = split_name
    else:
        thr_info = {"policy": "inherited", "threshold": float(threshold),
                    "selected_on": "validation (applied unchanged here)"}

    payload = binary_metrics(
        y_true, y_prob, threshold, split_name=split_name,
        include_calibration=calibration, calibration_bins=calibration_bins,
        fallback_test_caveat=fallback_test_caveat,
    )
    payload["threshold_selection"] = thr_info
    payload["_predictions"] = {"y_true": y_true, "y_prob": y_prob, "index": idx}
    return payload


def fit(model, *, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, device, cfg: dict, ckpt_manager, provenance: dict,
        run_dir: Path, fixture: bool = False) -> dict:
    """Full training loop with early stopping and per-epoch checkpointing."""
    t = cfg["train"]
    epochs = int(t["epochs"])
    accum = int(t["accum_steps"])
    amp = bool(t["amp"])
    clip = float(t.get("grad_clip_norm", 0.0))
    es = t.get("early_stopping", {}) or {}
    es_on = bool(es.get("enabled", False))
    es_metric = es.get("metric", "pr_auc")
    es_patience = int(es.get("patience", 5))
    es_min_delta = float(es.get("min_delta", 0.0))

    ev = cfg["eval"]
    history, bad_epochs = [], 0
    tag = "[FIXTURE] " if fixture else ""

    for epoch in range(1, epochs + 1):
        print(f"{tag}epoch {epoch}/{epochs}")
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accum_steps=accum, amp=amp, grad_clip_norm=clip, epoch=epoch,
        )

        val = evaluate(
            model, val_loader, device, threshold=None,
            threshold_policy=ev["threshold_policy"],
            fixed_threshold=float(ev.get("fixed_threshold", 0.5)),
            amp=amp, split_name="val",
            calibration=bool(ev.get("calibration", True)),
            calibration_bins=int(ev.get("calibration_bins", 10)),
        )
        val.pop("_predictions", None)

        if scheduler is not None:
            scheduler.step()

        print(f"{tag}  train loss {tr['train_loss']:.4f}  "
              f"({tr['images_per_sec']} img/s)")
        print(metrics_table(val))

        ck = ckpt_manager.update(
            metrics=val, epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, provenance=provenance,
            min_delta=es_min_delta,
        )

        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **tr,
            "val": {k: v for k, v in val.items() if not k.startswith("_")},
            "checkpoint": ck,
        })

        if es_on:
            if ck["improved"] and ckpt_manager.monitor == es_metric:
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= es_patience:
                    print(f"{tag}early stopping at epoch {epoch}: "
                          f"{es_metric} has not improved in {es_patience} epochs")
                    break

    return {
        "history": history,
        "best_metric": ckpt_manager.best_metric,
        "best_epoch": ckpt_manager.best_epoch,
        "monitor": ckpt_manager.monitor,
        "epochs_run": len(history),
    }
