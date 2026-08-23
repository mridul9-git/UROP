"""Stage 3 model factory - the four compute-validated baselines, one interface.

RQ1 is a controlled comparison. Anything that differs between architectures
other than the architecture itself is a confound, so every model is built the
same way here:

    torchvision backbone, ImageNet weights
    -> final classifier replaced with a single-logit Linear
    -> everything else untouched

Single logit + BCEWithLogitsLoss (not 2 logits + CrossEntropy) because the
project's pos_weight (D209), threshold policy and PR-AUC reporting are all
defined for a one-dimensional score.

The four keys match probe_vram.py MODELS and the per-model keys in
artifacts/stage2/reports/vram_probe.json, so a run can be compared against its
own measured VRAM/throughput by name. Do not add a fifth without a
documentation decision - docs/dataset_analysis.md section 14.2 lists exactly
these four as measured.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

# name -> (constructor, default-weights enum)
_BACKBONES = {
    "resnet152": (tvm.resnet152, tvm.ResNet152_Weights.IMAGENET1K_V1),
    "densenet201": (tvm.densenet201, tvm.DenseNet201_Weights.IMAGENET1K_V1),
    "efficientnet_v2_s": (tvm.efficientnet_v2_s,
                          tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large,
                           tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
}

SUPPORTED_MODELS = tuple(_BACKBONES)


def _head(in_features: int, num_outputs: int, dropout: float) -> nn.Module:
    """The classifier head. Identical construction for all four backbones."""
    if dropout and dropout > 0:
        return nn.Sequential(nn.Dropout(p=float(dropout)),
                             nn.Linear(in_features, num_outputs))
    return nn.Linear(in_features, num_outputs)


def _replace_head(model: nn.Module, name: str, num_outputs: int,
                  dropout: float) -> tuple[nn.Module, int]:
    """Swap each backbone's classifier for a fresh `num_outputs` head.

    Each family exposes its head differently, which is exactly why this lives in
    one function: doing it ad hoc per experiment is how an architecture-specific
    difference creeps into a controlled comparison. Returns the model and the
    backbone's feature width (recorded in the manifest, and the hook Stage 4
    feature fusion will need).
    """
    if name == "resnet152":
        in_features = model.fc.in_features
        model.fc = _head(in_features, num_outputs, dropout)
    elif name == "densenet201":
        in_features = model.classifier.in_features
        model.classifier = _head(in_features, num_outputs, dropout)
    elif name in ("efficientnet_v2_s", "mobilenet_v3_large"):
        # Both use a Sequential classifier whose final Linear is the head.
        # Read in_features off the module - never assume a literal width.
        last_linear_idx = max(
            i for i, m in enumerate(model.classifier) if isinstance(m, nn.Linear)
        )
        in_features = model.classifier[last_linear_idx].in_features
        model.classifier[last_linear_idx] = _head(in_features, num_outputs, dropout)
    else:  # pragma: no cover - guarded by build_model
        raise ValueError(f"no head-replacement rule for {name!r}")
    return model, in_features


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def build_model(name: str, *, pretrained: bool = True, num_outputs: int = 1,
                dropout: float = 0.0) -> nn.Module:
    """Construct one baseline.

    `pretrained=True` loads ImageNet weights, which section 10.2 depends on: the
    grayscale-replication preprocessing is chosen precisely so the pretrained
    first-conv statistics apply unchanged.
    """
    if name not in _BACKBONES:
        raise ValueError(
            f"Unknown model {name!r}. The compute-validated baseline set is "
            f"{SUPPORTED_MODELS} (artifacts/stage2/reports/vram_probe.json). "
            f"Adding an architecture is a documentation decision, not a config "
            f"edit."
        )
    if int(num_outputs) != 1:
        raise ValueError(
            "num_outputs must be 1: binary Cardiomegaly is scored as a single "
            "logit with BCEWithLogitsLoss + pos_weight (D209)."
        )

    ctor, weights_enum = _BACKBONES[name]
    weights = weights_enum if pretrained else None
    model = ctor(weights=weights)
    model, in_features = _replace_head(model, name, int(num_outputs), float(dropout))

    # Attached rather than printed, so the manifest can carry it verbatim.
    model.uro_meta = {
        "name": name,
        "pretrained": bool(pretrained),
        "weights": str(weights) if weights is not None else None,
        "num_outputs": int(num_outputs),
        "dropout": float(dropout),
        "feature_dim": int(in_features),
        "trainable_params": count_parameters(model, trainable_only=True),
        "total_params": count_parameters(model, trainable_only=False),
    }
    return model


def model_summary(model: nn.Module) -> dict:
    """Provenance block for the run manifest."""
    meta = dict(getattr(model, "uro_meta", {}))
    meta["total_params_millions"] = round(
        count_parameters(model, trainable_only=False) / 1e6, 1
    )
    return meta


def build_from_config(train_cfg: dict) -> nn.Module:
    m = train_cfg["model"]
    return build_model(
        m["name"],
        pretrained=bool(m["pretrained"]),
        num_outputs=int(m["num_outputs"]),
        dropout=float(m.get("dropout", 0.0)),
    )


@torch.no_grad()
def check_output_shape(model: nn.Module, input_size: int = 320,
                       batch: int = 2, device: str = "cpu") -> tuple:
    """Forward one synthetic batch and return the output shape.

    Used by the smoke tests and by train.py before any real data is touched: a
    head-replacement mistake should surface in milliseconds, not after an epoch.
    """
    model = model.to(device).eval()
    x = torch.zeros(batch, 3, input_size, input_size, device=device)
    out = model(x)
    return tuple(out.shape)
