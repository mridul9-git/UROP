"""Stage 3 configuration loading and validation.

One rule: every training hyperparameter comes from config/train.yaml, and every
data parameter comes from config/data.yaml. Nothing is defaulted silently inside
Python. If a required key is absent this module raises and names the key -- a run
that silently substitutes a default is a run whose manifest is a lie.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "config" / "train.yaml"

# The compute-validated baseline set. Keys match probe_vram.py MODELS and the
# per-model keys in artifacts/stage2/reports/vram_probe.json, so a training run
# can be compared against its own VRAM measurement by name.
SUPPORTED_MODELS = (
    "resnet152",
    "densenet201",
    "efficientnet_v2_s",
    "mobilenet_v3_large",
)

# Every key that must be present. Nested via dotted paths.
REQUIRED_KEYS = (
    "experiment.name", "experiment.seed", "experiment.output_root",
    "data.data_config", "data.splits_dir", "data.train_csv", "data.val_csv",
    "data.test_csv", "data.split_manifest", "data.on_missing_image",
    "augmentation.enabled",
    "model.name", "model.pretrained", "model.num_outputs",
    "loss.name", "loss.pos_weight_source", "loss.pos_weight_value",
    "optim.name", "optim.lr", "optim.weight_decay",
    "scheduler.name",
    "train.epochs", "train.micro_batch", "train.accum_steps", "train.amp",
    "train.deterministic",
    "eval.threshold_policy",
    "checkpoint.monitor", "checkpoint.mode",
)


def _get_dotted(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found: {p}\n"
            f"Stage 3 needs both config/train.yaml (hyperparameters) and "
            f"config/data.yaml (data parameters)."
        )
    with open(p, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{p} did not parse to a mapping")
    return loaded


def load_train_config(path: str | Path | None = None) -> dict:
    """Load config/train.yaml.

    Resolution order: explicit path > $UROP_TRAIN_CONFIG > repo config/train.yaml.
    The env var mirrors $UROP_DATA_CONFIG in chexpert_metadata.load_config, so a
    fixture run can point at a fixture config without editing the real one.
    """
    resolved = Path(path or os.environ.get("UROP_TRAIN_CONFIG") or DEFAULT_TRAIN_CONFIG)
    cfg = load_yaml(resolved)
    cfg["_config_path"] = str(resolved)
    return cfg


def validate_train_config(cfg: dict) -> None:
    """Fail loudly on a missing or incoherent setting, before any GPU work."""
    missing = []
    for key in REQUIRED_KEYS:
        try:
            _get_dotted(cfg, key)
        except KeyError:
            missing.append(key)
    if missing:
        raise KeyError(
            f"config/train.yaml is missing required key(s): {missing}\n"
            f"Every training parameter must be explicit - no silent defaults."
        )

    model = cfg["model"]["name"]
    if model not in SUPPORTED_MODELS:
        raise ValueError(
            f"model.name={model!r} is not one of the four compute-validated "
            f"baselines {SUPPORTED_MODELS}.\n"
            f"Adding an architecture requires a documentation decision first "
            f"(docs/dataset_analysis.md section 14.2), not a config edit."
        )

    if int(cfg["model"]["num_outputs"]) != 1:
        raise ValueError(
            "model.num_outputs must be 1: the target is binary Cardiomegaly "
            "scored with BCEWithLogitsLoss and a single logit. A 2-logit head "
            "would silently change the loss, the threshold semantics and "
            "pos_weight (docs/dataset_analysis.md D202/D209)."
        )

    micro = int(cfg["train"]["micro_batch"])
    accum = int(cfg["train"]["accum_steps"])
    if micro < 1 or accum < 1:
        raise ValueError("train.micro_batch and train.accum_steps must both be >= 1")

    if cfg["train"]["deterministic"] and cfg["train"].get("cudnn_benchmark"):
        raise ValueError(
            "train.deterministic=true is incompatible with "
            "train.cudnn_benchmark=true - benchmark mode picks algorithms "
            "non-deterministically."
        )

    src = cfg["loss"]["pos_weight_source"]
    if src not in ("manifest", "recompute", "explicit"):
        raise ValueError(
            f"loss.pos_weight_source={src!r} must be one of "
            f"manifest | recompute | explicit"
        )

    policy = cfg["eval"]["threshold_policy"]
    if policy not in ("f1", "youden", "prevalence", "fixed"):
        raise ValueError(
            f"eval.threshold_policy={policy!r} must be one of "
            f"f1 | youden | prevalence | fixed"
        )

    if cfg["checkpoint"]["mode"] not in ("max", "min"):
        raise ValueError("checkpoint.mode must be 'max' or 'min'")

    _validate_augmentation(cfg["augmentation"])


def _validate_augmentation(aug: dict) -> None:
    """Guard the augmentation decisions D208 rests on.

    docs/dataset_analysis.md section 11 forbids several augmentations for reasons
    specific to cardiomegaly -- they alter the cardiothoracic ratio, i.e. they
    alter the label. Enabling one via config would silently invalidate D208, so
    the code refuses rather than trusting the operator to have read section 11.
    """
    forbidden = {
        "horizontal_flip": "11.1 - fabricates a ~50% dextrocardia population",
        "vertical_flip": "11 - no radiograph is ever upside-down",
        "shear": "11 - deforms the cardiac silhouette, altering the measurement",
        "random_erasing": "11 - can erase the cardiac border, keeping the label",
        "mixup": "11 - blends two chests and their labels",
        "cutmix": "11 - blends two chests and their labels",
        "clahe": "10.3 - excluded from core; best-model ablation only",
    }
    enabled = [f"{k} (section {why})" for k, why in forbidden.items() if aug.get(k)]
    if enabled:
        raise ValueError(
            "config/train.yaml enables augmentation(s) that "
            "docs/dataset_analysis.md section 11 excludes from the core "
            "protocol:\n  "
            + "\n  ".join(enabled)
            + "\n\nThese are not style preferences - each one changes the "
              "cardiothoracic ratio or the label. Amend section 11 and D208 "
              "first if this is a deliberate methodology change."
        )

    if aug.get("enabled"):
        zmin, zmax = float(aug["zoom_min"]), float(aug["zoom_max"])
        if not (0 < zmin <= zmax):
            raise ValueError(f"augmentation zoom range invalid: {zmin}..{zmax}")
        if float(aug["rotation_deg"]) > 15.0:
            raise ValueError(
                f"augmentation.rotation_deg={aug['rotation_deg']} exceeds the "
                f"15 degree ceiling section 11 gives, beyond which the cardiac "
                f"silhouette stops resembling a real radiograph."
            )


def load_data_config(train_cfg: dict) -> dict:
    """Load the Stage 2 data config named by train.yaml.

    Uses chexpert_metadata.load_config so Stage 3 reads the data config through
    exactly the same code path as Stage 2 -- no second, subtly different loader.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
    from chexpert_metadata import load_config  # noqa: E402

    return load_config(train_cfg["data"]["data_config"])


def effective_batch(cfg: dict) -> int:
    return int(cfg["train"]["micro_batch"]) * int(cfg["train"]["accum_steps"])


def snapshot_config(cfg: dict, dest: str | Path) -> Path:
    """Write a verbatim copy of the resolved config next to the run's outputs.

    The manifest records values; this records the file. Together they let a run
    be re-executed even if config/train.yaml has since changed.
    """
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    return p
