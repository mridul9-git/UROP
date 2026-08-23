"""Build a tiny synthetic CheXpert-shaped fixture for the Stage 3 dry run.

The point is that the entire pipeline - CSV -> dataset -> transforms -> model ->
loss -> backward -> optimizer -> checkpoint -> evaluation -> metrics - can be
exercised end to end with no CheXpert image on disk. Image acquisition is
blocked; the code must still be provably runnable.

Everything produced here is SYNTHETIC NOISE. It carries no diagnostic signal
and must never be presented as data. Three guards make that hard to forget:

* the release directory is named `FIXTURE-v0.0-synthetic`, not a CheXpert name
* every generated config, CSV and run is stamped FIXTURE
* the fixture writes only under a caller-supplied directory (default: the
  system temp dir), never into artifacts/stage2/

Patient/study/view path structure mirrors the real release exactly, so the
centralized resolver and the patient-disjointness logic are genuinely
exercised rather than bypassed.

Run:  python src/training/make_fixture.py --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

FIXTURE_RELEASE = "FIXTURE-v0.0-synthetic"
FIXTURE_PREFIX = "CheXpert-v1.0"  # metadata prefix, as the real CSVs ship

SPLIT_COLUMNS = [
    "Path", "patient_id", "study_id", "view_num", "Sex", "Age",
    "Frontal/Lateral", "AP/PA", "Cardiomegaly", "Support Devices",
    "target", "uncertainty_policy",
]


def _synthetic_image(rng: random.Random, size: int, positive: bool) -> Image.Image:
    """A small grayscale image. Faint class-dependent bias only.

    A tiny brightness offset for positives lets a smoke run show a loss that
    moves, which is what proves the backward pass is wired up. It is NOT a
    finding and any metric computed on it is meaningless.
    """
    base = 110 + (18 if positive else 0)
    arr = np.clip(
        np.random.default_rng(rng.randint(0, 2 ** 31)).normal(base, 25, (size, size)),
        0, 255,
    ).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def build_fixture(out_dir: str | Path, *, patients: int = 24, image_size: int = 64,
                  prevalence: float = 0.25, seed: int = 42,
                  uncertainty_policy: str = "u_zeros") -> dict:
    """Create images, split CSVs, a split manifest and a fixture data config.

    Patients are partitioned 50/25/25 and never shared across splits, so the
    disjointness assertion is a real check rather than a formality.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    images_root = out / "images"
    release = images_root / FIXTURE_RELEASE
    splits_dir = out / "splits"
    (release / "train").mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    n_train = patients // 2
    n_val = (patients - n_train) // 2
    assignment = (["train"] * n_train + ["val"] * n_val
                  + ["test"] * (patients - n_train - n_val))

    rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    # Assign patient-level labels PER SPLIT, guaranteeing at least one positive
    # and one negative in each. Independent coin flips can leave a small split
    # single-class, which makes ROC-AUC/PR-AUC undefined and means the dry run
    # never exercises the real metric path.
    labels: dict[int, bool] = {}
    for split in ("train", "val", "test"):
        members = [i + 1 for i, s in enumerate(assignment) if s == split]
        if len(members) < 2:
            raise RuntimeError(
                f"fixture split {split!r} has {len(members)} patient(s); "
                f"at least 2 are needed to hold both classes. Raise --patients."
            )
        flags = [rng.random() < prevalence for _ in members]
        if not any(flags):
            flags[rng.randrange(len(flags))] = True
        if all(flags):
            flags[rng.randrange(len(flags))] = False
        labels.update(dict(zip(members, flags)))

    for pi in range(1, patients + 1):
        split = assignment[pi - 1]
        pid = f"patient{pi:05d}"
        # Patient-level label, so every image of a patient agrees - matching the
        # real stratification key (any-positive at patient level).
        positive = labels[pi]
        n_studies = rng.randint(1, 2)

        for si in range(1, n_studies + 1):
            sid = f"{pid}/study{si}"
            for vi in range(1, rng.randint(1, 2) + 1):
                rel = f"{FIXTURE_PREFIX}/train/{pid}/study{si}/view{vi}_frontal.jpg"
                dest = release / "train" / pid / f"study{si}" / f"view{vi}_frontal.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                _synthetic_image(rng, image_size, positive).save(dest, "JPEG", quality=92)

                rows[split].append({
                    "Path": rel,
                    "patient_id": pid,
                    "study_id": sid,
                    "view_num": vi,
                    "Sex": rng.choice(["Male", "Female"]),
                    "Age": rng.randint(20, 90),
                    "Frontal/Lateral": "Frontal",
                    "AP/PA": rng.choice(["AP", "PA"]),
                    # Raw 4-state value consistent with the derived target under
                    # u_zeros, so the dataset's re-derivation check passes.
                    "Cardiomegaly": 1.0 if positive else "",
                    "Support Devices": 0,
                    "target": 1 if positive else 0,
                    "uncertainty_policy": uncertainty_policy,
                })

    for split, recs in rows.items():
        if not recs:
            raise RuntimeError(f"fixture produced no rows for split {split!r}")
        with open(splits_dir / f"{split}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=SPLIT_COLUMNS)
            w.writeheader()
            w.writerows(recs)

    # Every split must hold both classes, not just train: a single-class val or
    # test split silently reduces ROC-AUC/PR-AUC to null.
    for split, recs in rows.items():
        pos = sum(r["target"] for r in recs)
        if pos == 0 or pos == len(recs):
            raise RuntimeError(
                f"fixture {split} split is single-class (pos={pos} of "
                f"{len(recs)}); raise --patients or adjust --prevalence"
            )

    n_pos = sum(r["target"] for r in rows["train"])
    n_neg = len(rows["train"]) - n_pos

    manifest = {
        "FIXTURE": True,
        "WARNING": "SYNTHETIC FIXTURE - NOT CHEXPERT - NOT EXPERIMENTAL DATA",
        "release": FIXTURE_RELEASE,
        "target": "Cardiomegaly",
        "uncertainty_policy": uncertainty_policy,
        "seed": seed,
        "mode": "fixture",
        "pos_weight_from_train": n_neg / n_pos,
        "summary": [
            {"split": s, "images": len(r),
             "patients": len({x["patient_id"] for x in r}),
             "positive": sum(x["target"] for x in r)}
            for s, r in rows.items()
        ],
    }
    with open(splits_dir / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # A data config shaped exactly like config/data.yaml so the real loader,
    # resolver and preprocessing code paths run unmodified.
    data_cfg = {
        "dataset": {"name": "FIXTURE", "release": FIXTURE_RELEASE,
                    "root": str(out), "train_csv": "train.csv",
                    "valid_csv": "valid.csv"},
        "images": {"root": str(images_root), "release_dir": FIXTURE_RELEASE,
                   "expect_subdir": "train"},
        "test_set": {"enabled": False, "root": str(out / "none"),
                     "labels_csv": "groundtruth.csv"},
        "target": {"column": "Cardiomegaly", "uncertain_value": -1.0,
                   "policy": uncertainty_policy, "blank_is_negative": True},
        "view": {"frontal_lateral_column": "Frontal/Lateral",
                 "projection_column": "AP/PA", "keep": ["Frontal"],
                 "combine_ap_pa": True},
        "split": {"unit": "patient", "seed": seed, "train_frac": 0.85,
                  "val_frac": 0.15, "stratify": True,
                  "out_dir": str(splits_dir)},
        "preprocessing": {"input_size": image_size, "resize_mode": "direct",
                          "to_rgb": "replicate",
                          "normalize": {"mean": [0.485, 0.456, 0.406],
                                        "std": [0.229, 0.224, 0.225]}},
        "subset": {"enabled": False, "target_frontal_images": 0, "seed": seed},
        "paths": {"artifacts": str(out / "artifacts"),
                  "figures": str(out / "figures"),
                  "reports": str(out / "reports")},
    }
    data_cfg_path = out / "fixture_data.yaml"
    with open(data_cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data_cfg, fh, sort_keys=False)

    return {
        "out_dir": str(out),
        "data_config": str(data_cfg_path),
        "splits_dir": str(splits_dir),
        "images_root": str(images_root),
        "release_dir": FIXTURE_RELEASE,
        "counts": {s: len(r) for s, r in rows.items()},
        "pos_weight_from_train": manifest["pos_weight_from_train"],
        "image_size": image_size,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="output directory (default: a temp dir)")
    ap.add_argument("--patients", type=int, default=24)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--prevalence", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = args.out or tempfile.mkdtemp(prefix="urop_fixture_")
    info = build_fixture(out, patients=args.patients, image_size=args.image_size,
                         prevalence=args.prevalence, seed=args.seed)
    print("FIXTURE/DRY-RUN - SYNTHETIC DATA - NOT A REAL EXPERIMENT")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
