"""Shared CheXpert metadata loading, label processing and view selection.

Every Stage 2 script imports from here so that the definition of "a usable
frontal Cardiomegaly example" exists in exactly one place.

Raw CheXpert label encoding (Irvin et al. 2019; CheXpert datasheet arXiv:2105.03020):

    1.0   positive
    0.0   negative
   -1.0   uncertain
   blank  observation not mentioned in the report

Nothing in this module deletes, moves or rewrites any file on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# --- CheXpert schema constants -------------------------------------------------

PATH_COL = "Path"
SEX_COL = "Sex"
AGE_COL = "Age"
VIEW_COL = "Frontal/Lateral"
PROJECTION_COL = "AP/PA"

OBSERVATIONS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

POSITIVE, NEGATIVE, UNCERTAIN = 1.0, 0.0, -1.0

_PATIENT_RE = re.compile(r"patient(\d+)", re.IGNORECASE)
_STUDY_RE = re.compile(r"study(\d+)", re.IGNORECASE)
_VIEW_RE = re.compile(r"view(\d+)_(frontal|lateral)", re.IGNORECASE)


# --- config --------------------------------------------------------------------


DEFAULT_CONFIG = "E:/UROP/config/data.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load the Stage 2 config.

    Resolution order: explicit `path` > $UROP_DATA_CONFIG > DEFAULT_CONFIG. The
    env var exists so the scripts can be pointed at a fixture config for testing
    without editing the real one.
    """
    import os

    resolved = Path(path or os.environ.get("UROP_DATA_CONFIG") or DEFAULT_CONFIG)
    with open(resolved, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs(cfg: dict) -> None:
    for key in ("artifacts", "figures", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


# --- loading -------------------------------------------------------------------


def load_split_csv(cfg: dict, which: str = "train") -> pd.DataFrame:
    """Load train.csv or valid.csv and attach parsed identifier columns.

    `which` is "train" or "valid". Raises a clear error if the file is absent so
    that a missing download never looks like an empty dataset.
    """
    root = Path(cfg["dataset"]["root"])
    fname = cfg["dataset"]["train_csv"] if which == "train" else cfg["dataset"]["valid_csv"]
    csv_path = root / fname
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\n"
            f"Set dataset.root in config/data.yaml to the directory containing "
            f"train.csv, valid.csv, train/ and valid/."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in (PATH_COL, VIEW_COL, PROJECTION_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {missing}")
    df["source_csv"] = which
    return attach_identifiers(df)


def attach_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Parse patient / study / view identifiers out of the Path column.

    CheXpert paths look like:
        CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg

    Parsing is regex-based rather than positional so that a changed root
    directory name in a future release does not silently shift every field.
    """
    df = df.copy()
    paths = df[PATH_COL].astype(str)

    df["patient_id"] = paths.str.extract(_PATIENT_RE, expand=False)
    df["study_num"] = paths.str.extract(_STUDY_RE, expand=False)
    view_parts = paths.str.extract(_VIEW_RE)
    df["view_num"] = view_parts[0]
    df["view_kind"] = view_parts[1].str.lower()

    unparsed = df["patient_id"].isna() | df["study_num"].isna()
    if unparsed.any():
        raise ValueError(
            f"{int(unparsed.sum())} rows have Path values that do not match the "
            f"expected patient*/study*/view* layout. First offender: "
            f"{df.loc[unparsed, PATH_COL].iloc[0]}"
        )

    df["patient_id"] = "patient" + df["patient_id"]
    df["study_id"] = df["patient_id"] + "/study" + df["study_num"]
    return df


# --- label processing ----------------------------------------------------------


@dataclass(frozen=True)
class LabelCounts:
    positive: int
    negative: int
    uncertain: int
    blank: int

    @property
    def total(self) -> int:
        return self.positive + self.negative + self.uncertain + self.blank


def raw_label_counts(df: pd.DataFrame, column: str) -> LabelCounts:
    """Count the four raw states of a CheXpert observation column."""
    col = df[column]
    return LabelCounts(
        positive=int((col == POSITIVE).sum()),
        negative=int((col == NEGATIVE).sum()),
        uncertain=int((col == UNCERTAIN).sum()),
        blank=int(col.isna().sum()),
    )


def apply_uncertainty_policy(
    df: pd.DataFrame,
    column: str,
    policy: str,
    blank_is_negative: bool = True,
) -> pd.DataFrame:
    """Return a copy with a binary `target` column and the policy applied.

    policy:
        "u_zeros" — uncertain (-1) becomes 0
        "u_ones"  — uncertain (-1) becomes 1
        "exclude" — uncertain rows are DROPPED (rows are dropped from the returned
                    frame only; nothing is deleted on disk)

    Blank/no-mention becomes 0 when `blank_is_negative`, otherwise those rows are
    dropped as well.
    """
    if policy not in {"u_zeros", "u_ones", "exclude"}:
        raise ValueError(f"unknown uncertainty policy: {policy!r}")

    out = df.copy()
    col = out[column]

    if blank_is_negative:
        target = col.fillna(NEGATIVE)
    else:
        out = out[col.notna()].copy()
        target = out[column]

    if policy == "u_zeros":
        target = target.replace(UNCERTAIN, NEGATIVE)
    elif policy == "u_ones":
        target = target.replace(UNCERTAIN, POSITIVE)
    else:  # exclude
        keep = target != UNCERTAIN
        out = out[keep].copy()
        target = target[keep]

    out["target"] = target.astype(int)
    out["uncertainty_policy"] = policy
    return out


# --- view selection ------------------------------------------------------------


def select_frontal(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the view types listed in config (`view.keep`).

    The AP/PA projection column is retained regardless, so that test-set metrics
    can be reported stratified by projection (docs/dataset_analysis.md §4).
    """
    keep = set(cfg["view"]["keep"])
    out = df[df[VIEW_COL].isin(keep)].copy()
    out[PROJECTION_COL] = out[PROJECTION_COL].fillna("Unknown")
    return out


# --- patient-level helpers -----------------------------------------------------


# --- image path resolution -----------------------------------------------------
# The metadata CSV and the image release disagree about the top-level directory
# name. train_cheXbert.csv ships full-release paths:
#     CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg
# while the downsampled release unpacks as CheXpert-v1.0-small/. Joining the two
# naively resolves to nothing, silently, for every image in the dataset.

_RELEASE_PREFIX_RE = re.compile(r"^chexpert-v1\.0(-small)?$", re.IGNORECASE)


def strip_release_prefix(rel_path: str) -> str:
    """Drop the leading release-directory segment from a metadata Path.

    `CheXpert-v1.0/train/patientX/...`       -> `train/patientX/...`
    `CheXpert-v1.0-small/train/patientX/...` -> `train/patientX/...`
    `train/patientX/...`                     -> unchanged

    Only a segment that actually looks like a CheXpert release directory is
    removed, so a path that is already release-relative is never truncated.
    """
    parts = str(rel_path).replace("\\", "/").split("/")
    if parts and _RELEASE_PREFIX_RE.match(parts[0]):
        parts = parts[1:]
    return "/".join(parts)


def _images_cfg(cfg: dict) -> dict:
    """Fetch the `images` config block, or explain what is missing.

    A bare KeyError here is loud but useless — this names the file and the exact
    keys required.
    """
    img = cfg.get("images")
    if not img:
        raise KeyError(
            "config/data.yaml has no `images:` block. Image-dependent scripts need:\n"
            "  images:\n"
            "    root: <directory CONTAINING the release directory>\n"
            "    release_dir: CheXpert-v1.0-small\n"
            "    expect_subdir: train"
        )
    missing = [k for k in ("root", "release_dir", "expect_subdir") if k not in img]
    if missing:
        raise KeyError(f"config/data.yaml `images:` block is missing key(s): {missing}")
    return img


def image_release_root(cfg: dict) -> Path:
    """The directory the images are expected to live under: root / release_dir."""
    img = _images_cfg(cfg)
    return Path(img["root"]) / img["release_dir"]


def verify_image_root(cfg: dict) -> Path:
    """Fail loudly unless the expected image release layout is present.

    Called once at the start of every image-dependent script. A missing or
    mis-shaped image root must stop the run immediately — the alternative is
    hundreds of thousands of per-file 'missing' records that look like a corrupt
    dataset rather than an un-downloaded one.
    """
    img = _images_cfg(cfg)
    root = Path(img["root"])
    release = image_release_root(cfg)
    expect = release / img["expect_subdir"]

    if not root.exists():
        raise FileNotFoundError(
            f"Image root does not exist: {root}\n"
            f"Set `images.root` in config/data.yaml to the directory that CONTAINS "
            f"the unpacked release directory '{img['release_dir']}'."
        )
    if not release.exists():
        siblings = sorted(p.name for p in root.iterdir() if p.is_dir())[:10]
        raise FileNotFoundError(
            f"Release directory not found: {release}\n"
            f"Directories present in {root}: {siblings or '(none)'}\n"
            f"Either unpack the release there, or set `images.release_dir` to the "
            f"actual directory name."
        )
    if not expect.exists():
        present = sorted(p.name for p in release.iterdir() if p.is_dir())[:10]
        raise FileNotFoundError(
            f"Expected subdirectory '{img['expect_subdir']}' not found under {release}\n"
            f"Subdirectories present: {present or '(none)'}\n"
            f"The release may have unpacked one level deeper (a nested "
            f"'{img['release_dir']}/{img['release_dir']}/' is a common archive quirk)."
        )
    return release


def resolve_image_path(cfg: dict, rel_path: str) -> Path:
    """Map one metadata Path to its location in the image release.

    Pure path arithmetic — does not touch the filesystem, so it is cheap enough to
    call once per row. Existence is checked by the caller (validate_images records
    a `missing` status rather than raising, so one absent file does not abort a
    200k-image sweep).
    """
    return image_release_root(cfg) / strip_release_prefix(rel_path)


def patient_level_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per patient for stratification and leakage checks.

    A patient is positive if ANY of their retained images is positive. This is the
    grouping key for stratified patient-level splitting: it keeps every image of a
    patient in one split while still balancing prevalence across splits.
    """
    grouped = df.groupby("patient_id", sort=True)
    return pd.DataFrame(
        {
            "patient_id": grouped.size().index,
            "n_images": grouped.size().to_numpy(),
            "n_studies": grouped["study_id"].nunique().to_numpy(),
            "any_positive": grouped["target"].max().to_numpy().astype(int),
            "n_positive_images": grouped["target"].sum().to_numpy().astype(int),
        }
    ).reset_index(drop=True)


def assert_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    """Hard assertion that no patient appears in more than one split.

    This is the five-line check the whole project's validity rests on. Call it
    every time a split is created or loaded — never assume.
    """
    names = list(splits)
    sets = {name: set(splits[name]["patient_id"].unique()) for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(
                    f"PATIENT LEAKAGE: {len(overlap)} patient(s) in both "
                    f"{a!r} and {b!r}. Examples: {sorted(overlap)[:5]}"
                )


def imbalance_summary(df: pd.DataFrame) -> dict:
    """Positive/negative counts, prevalence and the pos_weight for BCE loss."""
    pos = int((df["target"] == 1).sum())
    neg = int((df["target"] == 0).sum())
    total = pos + neg
    return {
        "positive": pos,
        "negative": neg,
        "total": total,
        "prevalence": pos / total if total else float("nan"),
        "neg_per_pos": neg / pos if pos else float("nan"),
        # pos_weight for torch.nn.BCEWithLogitsLoss — computed from TRAIN only.
        "pos_weight": neg / pos if pos else float("nan"),
    }


# --- plotting style ------------------------------------------------------------
# Palette and chrome from the project's data-viz reference (light surface).

PALETTE = {
    "series_1": "#2a78d6",   # blue    — negative / primary
    "series_2": "#eb6834",   # orange  — positive
    "series_3": "#1baf7a",   # aqua    — uncertain
    "series_4": "#eda100",   # yellow
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}


def compact_formatter(vmax: float):
    """Axis tick formatter that adapts to magnitude.

    A fixed `/1000 -> k` formatter renders every tick as "0k" whenever the axis
    tops out below ~10k (which happens on subsets and fixtures), so choose the
    unit from the actual data range instead.
    """
    if vmax >= 1_000_000:
        return lambda v, _: f"{v / 1e6:.1f}M"
    if vmax >= 10_000:
        return lambda v, _: f"{v / 1e3:.0f}k"
    return lambda v, _: f"{v:,.0f}"


def apply_plot_style() -> None:
    """Recessive grid, hairline axes, system sans — applied once per script."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "axes.edgecolor": PALETTE["axis"],
            "axes.labelcolor": PALETTE["ink_secondary"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelcolor": PALETTE["ink_secondary"],
            "ytick.labelcolor": PALETTE["ink_secondary"],
            "legend.frameon": False,
            "figure.dpi": 130,
        }
    )
