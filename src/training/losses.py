"""Stage 3 loss construction, with auditable pos_weight provenance.

D209 selects `pos_weight` in BCEWithLogitsLoss over focal loss and resampling
because it adds zero tunable hyperparameters and keeps the data pipeline
byte-identical across all four architectures. The measured value is

    pos_weight = 6.246739

computed by make_splits.py from the TRAIN split alone: 34,482 negative /
5,520 positive over 40,002 images at 13.80% prevalence.

The word "alone" carries the weight. Computing pos_weight over the full pool -
or over train+val - would leak validation/test prevalence into the training
objective. This module therefore never reads the val or test CSV, and says so
in the provenance record it returns.

Three sources are offered, and whichever is used is cross-checked against a
recomputation from the train CSV when that file is available. A stale manifest
paired with a regenerated split is exactly the kind of silent drift that would
otherwise never surface.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn


def recompute_pos_weight(train_csv: str | Path) -> tuple[float, dict]:
    """neg/pos over the TRAIN split only.

    Returns the value and the counts it came from, so a caller can put both in
    the manifest rather than an unexplained float.
    """
    p = Path(train_csv)
    if not p.exists():
        raise FileNotFoundError(f"Train split CSV not found for pos_weight: {p}")
    frame = pd.read_csv(p, usecols=["target"])
    pos = int((frame["target"] == 1).sum())
    neg = int((frame["target"] == 0).sum())
    other = len(frame) - pos - neg
    if other:
        raise ValueError(
            f"{p.name}: {other} row(s) have a target that is neither 0 nor 1. "
            f"The binary target must be fully resolved by the uncertainty "
            f"policy before training."
        )
    if pos == 0:
        raise ValueError(
            f"{p.name}: zero positive rows - pos_weight is undefined. "
            f"Refusing to train on a single-class split."
        )
    return neg / pos, {"positive": pos, "negative": neg, "images": len(frame)}


def resolve_pos_weight(train_cfg: dict, manifest: dict | None = None) -> tuple[float, dict]:
    """Return (pos_weight, provenance) according to loss.pos_weight_source.

    provenance is a dict recorded verbatim in run_manifest.json so that any
    reported number can be traced to how its class weighting was obtained.
    """
    loss_cfg = train_cfg["loss"]
    source = loss_cfg["pos_weight_source"]
    tol = float(loss_cfg.get("pos_weight_tolerance", 1e-4))

    splits_dir = Path(train_cfg["data"]["splits_dir"])
    train_csv = splits_dir / train_cfg["data"]["train_csv"]

    provenance: dict = {
        "source": source,
        "computed_from": "train split only (never val/test) - D209",
        "tolerance": tol,
    }

    if source == "manifest":
        if manifest is None or "pos_weight_from_train" not in manifest:
            raise KeyError(
                "loss.pos_weight_source='manifest' but split_manifest.json has "
                "no 'pos_weight_from_train' key. Regenerate the splits "
                "(python src/data/make_splits.py) or switch the source to "
                "'recompute'."
            )
        value = float(manifest["pos_weight_from_train"])
        provenance["manifest_value"] = value
    elif source == "recompute":
        value, counts = recompute_pos_weight(train_csv)
        provenance["counts"] = counts
    elif source == "explicit":
        value = float(loss_cfg["pos_weight_value"])
        provenance["explicit_value"] = value
    else:  # pragma: no cover - validated in config.validate_train_config
        raise ValueError(f"unknown pos_weight_source {source!r}")

    # Cross-check against the train CSV whenever it exists. This is the check
    # that catches a manifest left behind by a previous split generation.
    if train_csv.exists():
        recomputed, counts = recompute_pos_weight(train_csv)
        provenance["recomputed_from_train_csv"] = recomputed
        provenance["train_counts"] = counts
        provenance["agrees_with_train_csv"] = abs(recomputed - value) <= tol
        if abs(recomputed - value) > tol:
            raise ValueError(
                f"pos_weight disagreement beyond tolerance {tol}:\n"
                f"  source ({source}) : {value!r}\n"
                f"  recomputed from   : {train_csv}\n"
                f"  recomputed value  : {recomputed!r}\n"
                f"  train counts      : {counts}\n"
                f"The configured/most-recent split and the class weighting are "
                f"out of sync. Regenerate one or the other - do not widen the "
                f"tolerance to make this pass."
            )
    else:
        provenance["recomputed_from_train_csv"] = None
        provenance["agrees_with_train_csv"] = None
        print(f"[warn] {train_csv} absent - pos_weight could not be "
              f"independently verified against the train split")

    provenance["value"] = value
    return value, provenance


def build_loss(train_cfg: dict, pos_weight: float,
               device: str | torch.device = "cpu") -> nn.Module:
    """BCEWithLogitsLoss with the resolved pos_weight.

    BCEWithLogits (not BCE + a separate sigmoid) because it is numerically
    stable under AMP: the sigmoid is fused into the loss in log-space, so fp16
    logits do not saturate before the loss sees them.
    """
    name = train_cfg["loss"]["name"]
    if name != "bce_with_logits":
        raise ValueError(
            f"loss.name={name!r} is not supported. D209 selects "
            f"BCEWithLogitsLoss + pos_weight and rejects focal loss "
            f"(two extra tunable hyperparameters, which would confound RQ1) "
            f"and resampling. Changing this is a documentation decision."
        )
    weight = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight)


def prevalence_baseline(targets) -> float:
    """PR-AUC's no-skill baseline: the positive prevalence of the scored set.

    Section 12.3 requires PR-AUC always be reported beside this number, because
    PR-AUC is prevalence-dependent and meaningless without it. Computed from the
    set being scored, never hard-coded - the documented 0.123 predates the
    measurement and would understate the real baseline.
    """
    import numpy as np
    arr = np.asarray(targets, dtype=float)
    return float(arr.mean()) if arr.size else float("nan")
