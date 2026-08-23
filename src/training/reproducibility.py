"""Stage 3 reproducibility control — seeds, provenance capture and run manifests.

Every number this project reports must be traceable to the exact data, code and
configuration that produced it. That chain is:

    config -> seed -> split identity -> model -> preprocessing -> training -> metrics

This module owns the ends of that chain that are not owned elsewhere: seeding,
git SHA capture, the run directory convention, and the run manifest that records
all of it.

Nothing here writes into the repository tree except under
`experiment.output_root`, which is a git-ignored artifacts directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# Stamped into every fixture run so dry-run output can never be mistaken for a
# real experiment. Asserted by the smoke tests.
FIXTURE_BANNER = "FIXTURE/DRY-RUN - SYNTHETIC DATA - NOT A REAL EXPERIMENT"


def utc_now_iso() -> str:
    """UTC timestamp in the format make_splits.py writes to split_manifest.json."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """Seed every RNG this pipeline touches.

    `deterministic` trades throughput for repeatability. It is ON by default
    because a baseline comparison whose ranking moves between runs is not a
    result. torch is imported lazily so these utilities stay importable in an
    environment without torch (the metric tests do not need it).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch required for training, not utils
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        # CUBLAS needs this for deterministic matmul. Must be set before the first
        # CUDA context is created to have any effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            # Older torch, or an op with no deterministic kernel. Say so rather
            # than letting the run claim a determinism it does not have.
            print("[warn] torch.use_deterministic_algorithms unavailable - "
                  "determinism is BEST-EFFORT for this run")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding.

    Without this, workers inherit correlated RNG streams and augmentation
    diversity silently drops. Nothing raises; the only symptom is a worse model.
    """
    import torch

    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def git_sha(short: bool = False) -> str:
    """Current commit SHA, or an explicit marker when unavailable.

    Never returns a fabricated or empty value: a run whose code version is
    unknown must record UNKNOWN, not ''.
    """
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        out = subprocess.run(args, cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN(git-unavailable)"
    if out.returncode != 0:
        return "UNKNOWN(not-a-git-repo)"
    return out.stdout.strip() or "UNKNOWN(empty)"


def git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes; None if undeterminable.

    A run made from a dirty tree is not reproducible from its SHA alone. Record
    the fact instead of ignoring it.
    """
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def sha1_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-1 of a file, or an explicit marker when it cannot be read."""
    p = Path(path)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha1()
    with open(p, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def environment_snapshot() -> dict:
    """Versions that can change a numeric result if they change."""
    snap = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch
        snap["torch"] = torch.__version__
        snap["cuda_available"] = torch.cuda.is_available()
        snap["cuda"] = torch.version.cuda
        snap["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        snap["torch"] = None
    try:
        import torchvision
        snap["torchvision"] = torchvision.__version__
    except ImportError:
        snap["torchvision"] = None
    try:
        import sklearn
        snap["scikit_learn"] = sklearn.__version__
    except ImportError:
        snap["scikit_learn"] = None
    return snap


def make_run_dir(output_root: str | Path, name: str, model: str, seed: int,
                 fixture: bool = False) -> Path:
    """Create and return this run's output directory.

    Convention:
        <output_root>/[FIXTURE__]<name>__<model>__seed<seed>__<utc-compact>/

    The FIXTURE prefix is part of the directory NAME, not just the log, so a dry
    run is identifiable from a directory listing alone.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = f"{name}__{model}__seed{seed}__{stamp}"
    if fixture:
        parts = f"FIXTURE__{parts}"
    run_dir = Path(output_root) / parts
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_manifest(train_cfg: dict, data_cfg: dict, *, run_dir: Path,
                       pos_weight: float, pos_weight_provenance: dict,
                       split_files: dict, dataset_stats: dict,
                       fixture: bool = False) -> dict:
    """Assemble the full provenance record for one run.

    Mirrors split_manifest.json's discipline: everything needed to explain where
    a number came from, in one JSON file sitting next to the number.
    """
    exp = train_cfg["experiment"]
    manifest = {
        "kind": "FIXTURE_DRY_RUN" if fixture else "experiment",
        "fixture": bool(fixture),
        "created_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "experiment_name": exp["name"],
        "seed": exp["seed"],

        "code": {
            "git_sha": git_sha(),
            "git_sha_short": git_sha(short=True),
            "git_dirty": git_dirty(),
        },
        "environment": environment_snapshot(),

        "model": dict(train_cfg["model"]),
        "optim": dict(train_cfg["optim"]),
        "scheduler": dict(train_cfg["scheduler"]),
        "train": dict(train_cfg["train"]),
        "eval": dict(train_cfg["eval"]),
        "augmentation": dict(train_cfg["augmentation"]),

        # Read from config/data.yaml rather than restated in train.yaml, so the
        # two cannot drift. Recorded here because it changes results.
        "preprocessing": dict(data_cfg["preprocessing"]),
        "target": {
            "column": data_cfg["target"]["column"],
            "uncertainty_policy": data_cfg["target"]["policy"],
            "blank_is_negative": data_cfg["target"]["blank_is_negative"],
        },
        "view": {"keep": data_cfg["view"]["keep"]},

        "loss": {
            "name": train_cfg["loss"]["name"],
            "pos_weight": pos_weight,
            "pos_weight_provenance": pos_weight_provenance,
        },

        "splits": split_files,
        "dataset_stats": dataset_stats,
        "effective_batch": (int(train_cfg["train"]["micro_batch"])
                            * int(train_cfg["train"]["accum_steps"])),
    }

    if fixture:
        manifest["WARNING"] = FIXTURE_BANNER
    return manifest


def write_json(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return p
