"""Stage 3 preprocessing and augmentation transforms.

Two builders, two objects, no shared flag.

docs/dataset_analysis.md section 11 is explicit: "Enforced by constructing two
separate transform objects -- never one transform with an `is_train` flag that
can be passed wrongly." That is honoured literally here. There is no parameter
anywhere in this module that turns a train transform into an eval transform;
`build_eval_transform` is physically incapable of producing augmentation.

Preprocessing (section 10), identical across all four baselines and across all
three splits:

    PIL load as 'L'  ->  resize 320x320  ->  replicate gray to 3 channels
    ->  /255  ->  ImageNet normalize

No crop (section 10.1: cropping can delete a chest wall, which is the
denominator of the cardiothoracic ratio). No CLAHE (section 10.3).

Augmentation (section 11), TRAIN SPLIT ONLY:

    rotation +/-10 deg, isotropic zoom 0.9-1.0, translation +/-5% (pad, never
    crop), brightness +/-10%, contrast +/-10%

Deliberately absent, each for a cardiomegaly-specific reason given in section 11:
horizontal flip, vertical flip, shear/elastic/perspective, cutout/random
erasing, MixUp/CutMix.
"""

from __future__ import annotations

from PIL import Image
from torchvision import transforms

# Bilinear, matching "direct bilinear to 320 x 320" in section 10.
_INTERP = transforms.InterpolationMode.BILINEAR


class AspectPadResize:
    """Aspect-preserving resize + zero pad to a square.

    The alternative to direct resize that config/data.yaml exposes as
    `preprocessing.resize_mode: aspect_pad`. Section 10.1 recommends `direct`
    and explains why (padding introduces borders whose size correlates with the
    original aspect ratio -- a spurious feature). This exists so the alternative
    is a one-line ablation rather than a rewrite, exactly as section 10.1 says.
    """

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new(img.mode, (self.size, self.size), color=0)
        canvas.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas

    def __repr__(self) -> str:
        return f"AspectPadResize(size={self.size})"


def _resize_step(input_size: int, resize_mode: str):
    if resize_mode == "direct":
        return transforms.Resize((input_size, input_size), interpolation=_INTERP)
    if resize_mode == "aspect_pad":
        return AspectPadResize(input_size)
    raise ValueError(
        f"preprocessing.resize_mode={resize_mode!r} must be 'direct' or "
        f"'aspect_pad' (config/data.yaml; docs/dataset_analysis.md section 10.1)"
    )


def _to_rgb_step(to_rgb: str):
    if to_rgb != "replicate":
        raise ValueError(
            f"preprocessing.to_rgb={to_rgb!r} is not supported. Section 10.2 "
            f"selects 'replicate' on experimental-control grounds: it uses the "
            f"pretrained first-conv weights exactly as trained, identically "
            f"across all four architectures. A 1-channel stem surgery would "
            f"have to be re-derived per backbone."
        )
    # Input images are opened as mode 'L'; this replicates that channel 3x.
    return transforms.Grayscale(num_output_channels=3)


def _tail(data_cfg: dict) -> list:
    """The shared, non-random tail: to-RGB -> tensor(/255) -> ImageNet normalize."""
    pre = data_cfg["preprocessing"]
    norm = pre["normalize"]
    return [
        _to_rgb_step(pre["to_rgb"]),
        transforms.ToTensor(),                       # -> CHW float in [0,1]
        transforms.Normalize(mean=list(norm["mean"]), std=list(norm["std"])),
    ]


def build_eval_transform(data_cfg: dict):
    """Validation / test transform. Deterministic. No augmentation, ever.

    This function takes no augmentation configuration at all, so no caller can
    accidentally introduce it. Also used for the training set when computing
    dataset statistics, where augmentation would distort the numbers.
    """
    pre = data_cfg["preprocessing"]
    steps = [_resize_step(int(pre["input_size"]), pre["resize_mode"])]
    steps += _tail(data_cfg)
    return transforms.Compose(steps)


def build_train_transform(data_cfg: dict, aug_cfg: dict):
    """Training transform: the eval pipeline plus the section 11 augmentations.

    Resize happens FIRST so the geometric augmentations operate in the same
    320x320 frame for every image regardless of its native dimensions. That
    keeps augmentation magnitude (a +/-5% translation, a 0.9-1.0 zoom) uniform
    across the corpus instead of varying with source resolution.

    `augmentation.enabled: false` yields a transform equal to the eval pipeline,
    which is what an augmentation ablation needs.
    """
    pre = data_cfg["preprocessing"]
    steps = [_resize_step(int(pre["input_size"]), pre["resize_mode"])]

    if aug_cfg.get("enabled"):
        # RandomAffine's `scale` samples ONE factor applied to both axes, i.e. it
        # is isotropic. Section 11 requires that: anisotropic aspect jitter would
        # change the cardiothoracic ratio directly, which changes the label.
        # `shear` is left at None. Translation fills with 0 (pads) rather than
        # cropping content away -- required by section 11 / section 10.1.
        steps.append(
            transforms.RandomAffine(
                degrees=float(aug_cfg["rotation_deg"]),
                translate=(float(aug_cfg["translate_frac"]),
                           float(aug_cfg["translate_frac"])),
                scale=(float(aug_cfg["zoom_min"]), float(aug_cfg["zoom_max"])),
                shear=None,
                interpolation=_INTERP,
                fill=0,
            )
        )
        # Photometric only. saturation/hue are meaningless on grayscale and are
        # not in the section 11 table, so they are not passed.
        steps.append(
            transforms.ColorJitter(
                brightness=float(aug_cfg["brightness"]),
                contrast=float(aug_cfg["contrast"]),
            )
        )

    steps += _tail(data_cfg)
    return transforms.Compose(steps)


def describe_transform(t) -> list[str]:
    """Flat, loggable description of a transform pipeline.

    Written into the run manifest so the exact preprocessing of a run is
    recoverable from its artifacts rather than from this source file.
    """
    inner = getattr(t, "transforms", [t])
    return [repr(step) for step in inner]
