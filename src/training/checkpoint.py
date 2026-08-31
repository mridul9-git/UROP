"""Stage 3 checkpointing - save, load and resume.

A checkpoint carries enough state to resume a run exactly: model, optimizer,
scheduler, AMP scaler, epoch counter and the best-metric bookkeeping. It also
carries provenance (config, git SHA, pos_weight, split hashes) so a checkpoint
found on disk months later can still be explained.

*.pt and *.pth are excluded from git (.git/info/exclude). Checkpoints stay
local; only the manifest and metrics are ever committable.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state() -> dict:
    """Snapshot every RNG stream set_seed() touches.

    Without this a resumed run restarts the augmentation and shuffle streams
    from the seed rather than continuing them, which is not the same run.
    Checkpoints written before this existed simply have no `rng_state` key and
    are still loadable - see load_checkpoint(restore_rng=...).
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> list[str]:
    """Restore what capture_rng_state() saved. Returns the streams restored."""
    restored = []
    if not state:
        return restored
    if "python" in state:
        random.setstate(state["python"])
        restored.append("python")
    if "numpy" in state:
        np.random.set_state(state["numpy"])
        restored.append("numpy")
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
        restored.append("torch")
    if "torch_cuda_all" in state and torch.cuda.is_available():
        saved = state["torch_cuda_all"]
        # Only restore when the device count matches; a mismatched restore
        # would be silently wrong rather than loudly absent.
        if len(saved) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([t.cpu().to(torch.uint8) for t in saved])
            restored.append("torch_cuda")
    return restored


def save_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None,
                    scaler=None, epoch: int = 0, best_metric: float | None = None,
                    best_epoch: int | None = None, provenance: dict | None = None,
                    extra: dict | None = None) -> Path:
    """Write one checkpoint atomically.

    Atomic because a training run interrupted mid-write would otherwise leave a
    truncated file that loads as a mysterious error much later.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "provenance": provenance or {},
        "model_meta": dict(getattr(model, "uro_meta", {})),
        "rng_state": capture_rng_state(),
    }
    if extra:
        payload.update(extra)

    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(p)
    return p


def load_checkpoint(path: str | Path, *, model=None, optimizer=None,
                    scheduler=None, scaler=None, restore_rng: bool = False,
                    map_location: str = "cpu", strict: bool = True) -> dict:
    """Load a checkpoint, optionally restoring into live objects.

    Returns the raw payload so the caller can inspect provenance even when it
    restores nothing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    # weights_only=False: our payload holds provenance dicts, not just tensors.
    # Only ever load checkpoints this project produced.
    payload = torch.load(p, map_location=map_location, weights_only=False)

    if model is not None:
        saved_meta = payload.get("model_meta") or {}
        live_meta = getattr(model, "uro_meta", {}) or {}
        if saved_meta and live_meta and saved_meta.get("name") != live_meta.get("name"):
            raise ValueError(
                f"Checkpoint architecture mismatch: checkpoint holds "
                f"{saved_meta.get('name')!r} but the live model is "
                f"{live_meta.get('name')!r}. Loading anyway would silently "
                f"produce a model that is neither."
            )
        model.load_state_dict(payload["model_state"], strict=strict)

    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])

    # Reported, never assumed: a caller that asked for RNG restore and got
    # nothing must be able to see that in the returned payload.
    payload["rng_restored"] = (
        restore_rng_state(payload.get("rng_state")) if restore_rng else []
    )
    payload["rng_state_present"] = bool(payload.get("rng_state"))

    return payload


class CheckpointManager:
    """Tracks the monitored metric and writes best/last checkpoints."""

    def __init__(self, run_dir: str | Path, monitor: str = "pr_auc",
                 mode: str = "max", save_best: bool = True, save_last: bool = True):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.best_metric: float | None = None
        self.best_epoch: int | None = None

    @property
    def best_path(self) -> Path:
        return self.dir / "best.pt"

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pt"

    def load_state(self, best_metric: float | None,
                   best_epoch: int | None) -> None:
        """Adopt a previous run's best-so-far, so a resume cannot re-crown a
        worse epoch as 'best' and cannot reset early-stopping patience."""
        self.best_metric = None if best_metric is None else float(best_metric)
        self.best_epoch = None if best_epoch is None else int(best_epoch)

    def is_improvement(self, value: float, min_delta: float = 0.0) -> bool:
        if value is None:
            return False
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return value > self.best_metric + min_delta
        return value < self.best_metric - min_delta

    def update(self, *, metrics: dict, epoch: int, model, optimizer=None,
               scheduler=None, scaler=None, provenance: dict | None = None,
               min_delta: float = 0.0) -> dict:
        """Record an epoch. Returns what was written and whether it improved."""
        value = metrics.get(self.monitor)
        if value is None:
            raise KeyError(
                f"checkpoint.monitor={self.monitor!r} is not present in the "
                f"epoch metrics (available: {sorted(metrics)}). A monitor that "
                f"silently never fires would leave 'best' meaningless."
            )

        improved = self.is_improvement(value, min_delta)
        written = []

        if improved:
            self.best_metric, self.best_epoch = float(value), int(epoch)
            if self.save_best:
                save_checkpoint(self.best_path, model=model, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, epoch=epoch,
                                best_metric=self.best_metric,
                                best_epoch=self.best_epoch, provenance=provenance)
                written.append(str(self.best_path))

        if self.save_last:
            save_checkpoint(self.last_path, model=model, optimizer=optimizer,
                            scheduler=scheduler, scaler=scaler, epoch=epoch,
                            best_metric=self.best_metric,
                            best_epoch=self.best_epoch, provenance=provenance)
            written.append(str(self.last_path))

        return {
            "improved": improved,
            "monitor": self.monitor,
            "value": value,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "written": written,
        }
