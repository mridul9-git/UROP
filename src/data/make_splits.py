"""Stage 2 Steps 5 + 13 + 15 — reproducible patient-level splits.

Split unit is the PATIENT, never the image and never the study. Stratification is
applied at the patient level (a patient counts as positive if any of their
retained frontal studies is Cardiomegaly-positive), so prevalence is balanced
across splits without ever putting one person's images on both sides.

Two modes:

  official-test (default, `test_set.enabled: true` in config)
      train / val  carved from the official CheXpert training patients
      test         the official 500-patient, 5-radiologist-consensus test set

  fallback (`test_set.enabled: false`)
      train / val / test all carved from the official training patients using the
      config's fallback fractions. Used only if the official test set cannot be
      obtained; the report must say so explicitly.

The optional Tier-1 subset (config `subset`) is a patient-level stratified
subsample of the TRAIN split only. Validation and test are never subsampled.

Outputs to artifacts/stage2/splits/:
    train.csv val.csv test.csv     one row per image, with target and metadata
    split_manifest.json            seed, fractions, counts, hashes — the audit trail

Run:  python src/data/make_splits.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chexpert_metadata import (  # noqa: E402
    PROJECTION_COL,
    apply_uncertainty_policy,
    assert_disjoint,
    ensure_dirs,
    imbalance_summary,
    load_config,
    load_split_csv,
    patient_level_labels,
    select_frontal,
)

KEEP_COLS = [
    "Path", "patient_id", "study_id", "view_num", "Sex", "Age",
    "Frontal/Lateral", "AP/PA", "Cardiomegaly", "Support Devices",
    "target", "uncertainty_policy",
]


GRANULARITY = 20  # folds; 20 gives 5% resolution on the requested fraction


def stratified_patient_split(
    patients: pd.DataFrame, frac_b: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split patient IDs into (A, B) with B ≈ frac_b, balancing `any_positive`.

    StratifiedGroupKFold with n_splits = round(1/frac_b) only lands on the
    requested fraction when 1/frac_b happens to be an integer — asking for 15%
    that way actually yields 1/7 ≈ 14.3%. Instead this cuts a fixed 20 folds and
    unions round(frac_b * 20) of them, so 0.15 means exactly 3/20 of the patients.

    Groups are patient IDs, so no patient can straddle the boundary.
    """
    if not 0.0 < frac_b < 1.0:
        raise ValueError(f"frac_b must be in (0,1), got {frac_b}")
    k = max(1, min(GRANULARITY - 1, round(frac_b * GRANULARITY)))
    sgkf = StratifiedGroupKFold(n_splits=GRANULARITY, shuffle=True, random_state=seed)
    folds = [
        test_idx for _, test_idx in
        sgkf.split(patients, patients["any_positive"], groups=patients["patient_id"])
    ]
    idx_b = np.concatenate(folds[:k])
    mask_b = np.zeros(len(patients), dtype=bool)
    mask_b[idx_b] = True
    ids = patients["patient_id"].to_numpy()
    return ids[~mask_b], ids[mask_b]


def subsample_patients(
    patients: pd.DataFrame, target_images: int, seed: int
) -> np.ndarray:
    """Pick patients until ~target_images frontal images are covered, preserving
    class prevalence. Sampling is at the patient level so no patient is split."""
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (1, 0):
        pool = patients[patients["any_positive"] == label]
        share = target_images * (pool["n_images"].sum() / patients["n_images"].sum())
        order = rng.permutation(len(pool))
        cum = pool.iloc[order]["n_images"].cumsum().to_numpy()
        take = int(np.searchsorted(cum, share) + 1)
        chosen.append(pool.iloc[order[:take]]["patient_id"].to_numpy())
    return np.concatenate(chosen)


def summarise(name: str, df: pd.DataFrame) -> dict:
    s = imbalance_summary(df)
    proj = df[PROJECTION_COL].value_counts().to_dict()
    return {
        "split": name,
        "patients": int(df["patient_id"].nunique()),
        "studies": int(df["study_id"].nunique()),
        "images": int(len(df)),
        "positive": s["positive"],
        "negative": s["negative"],
        "prevalence": round(s["prevalence"], 5),
        "projection_counts": {str(k): int(v) for k, v in proj.items()},
    }


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    seed = cfg["split"]["seed"]
    policy = cfg["target"]["policy"]
    target_col = cfg["target"]["column"]
    out_dir = Path(cfg["split"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg["split"]["unit"] != "patient":
        raise ValueError("split.unit must be 'patient'. Image-level splitting leaks.")

    # --- build the modelling frame from the official training CSV --------------
    frame = apply_uncertainty_policy(
        select_frontal(cfg, load_split_csv(cfg, "train")),
        target_col, policy, cfg["target"]["blank_is_negative"],
    )
    patients = patient_level_labels(frame)
    print(f"[info] pool: {len(frame):,} frontal images / {len(patients):,} patients "
          f"(policy={policy})")

    use_official_test = bool(cfg["test_set"]["enabled"])
    manifest: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "release": cfg["dataset"]["release"],
        "target": target_col,
        "uncertainty_policy": policy,
        "blank_is_negative": cfg["target"]["blank_is_negative"],
        "views_kept": cfg["view"]["keep"],
        "ap_pa_combined": cfg["view"]["combine_ap_pa"],
        "split_unit": "patient",
        "seed": seed,
        "stratified_by": "patient-level any-positive Cardiomegaly",
        "mode": "official-test" if use_official_test else "fallback-carved-test",
    }

    if use_official_test:
        train_ids, val_ids = stratified_patient_split(
            patients, cfg["split"]["val_frac"], seed
        )
        test_frame = _load_official_test(cfg, policy, target_col)
        manifest["fractions"] = {
            "train": cfg["split"]["train_frac"], "val": cfg["split"]["val_frac"],
            "test": "official CheXpert test set (500 patients, 5-radiologist consensus)",
        }
    else:
        f = cfg["split"]["fallback"]
        holdout = f["val_frac"] + f["test_frac"]
        train_ids, rest_ids = stratified_patient_split(patients, holdout, seed)
        rest = patients[patients["patient_id"].isin(rest_ids)].reset_index(drop=True)
        val_ids, test_ids = stratified_patient_split(
            rest, f["test_frac"] / holdout, seed
        )
        test_frame = frame[frame["patient_id"].isin(test_ids)].copy()
        manifest["fractions"] = f
        print("[warn] official test set disabled — carving test from training patients")

    train_frame = frame[frame["patient_id"].isin(train_ids)].copy()
    val_frame = frame[frame["patient_id"].isin(val_ids)].copy()

    # --- optional Tier-1 subset, TRAIN ONLY ------------------------------------
    if cfg["subset"]["enabled"]:
        tp = patient_level_labels(train_frame)
        keep = subsample_patients(tp, cfg["subset"]["target_frontal_images"],
                                  cfg["subset"]["seed"])
        before = len(train_frame)
        train_frame = train_frame[train_frame["patient_id"].isin(keep)].copy()
        manifest["subset"] = {
            "enabled": True,
            "seed": cfg["subset"]["seed"],
            "target_frontal_images": cfg["subset"]["target_frontal_images"],
            "train_images_before": before,
            "train_images_after": len(train_frame),
            "note": "patient-level stratified subsample of TRAIN only; "
                    "val and test are never subsampled",
        }
        print(f"[info] subset: train {before:,} -> {len(train_frame):,} images")
    else:
        manifest["subset"] = {"enabled": False}

    # --- the check the whole project rests on ----------------------------------
    splits = {"train": train_frame, "val": val_frame, "test": test_frame}
    assert_disjoint(splits)
    print("[ok] patient-disjointness verified for train / val / test")

    manifest["summary"] = [summarise(k, v) for k, v in splits.items()]
    manifest["files"] = {}
    for name, d in splits.items():
        cols = [c for c in KEEP_COLS if c in d.columns]
        path = out_dir / f"{name}.csv"
        d[cols].sort_values("Path").to_csv(path, index=False)
        manifest["files"][name] = {
            "path": str(path),
            "rows": int(len(d)),
            "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
        }

    # pos_weight is computed from TRAIN ONLY — never from val or test.
    manifest["pos_weight_from_train"] = round(imbalance_summary(train_frame)["pos_weight"], 6)

    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n" + "-" * 62)
    for s in manifest["summary"]:
        print(f"{s['split']:>5} | {s['patients']:>6,} pts | {s['images']:>7,} imgs "
              f"| pos {s['positive']:>6,} | prev {100*s['prevalence']:5.2f}%")
    print("-" * 62)
    print(f"pos_weight (train) = {manifest['pos_weight_from_train']}")
    print(f"[ok] splits -> {out_dir}")


def _load_official_test(cfg: dict, policy: str, target_col: str) -> pd.DataFrame:
    """Load the official expert-annotated test set (labels + image paths)."""
    from chexpert_metadata import attach_identifiers

    tdir = Path(cfg["test_set"]["root"])
    csv = tdir / cfg["test_set"]["labels_csv"]
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Either place the official test labels there "
            f"(github.com/rajpurkarlab/cheXpert-test-set-labels) with the matching "
            f"images, or set test_set.enabled: false in config/data.yaml to carve a "
            f"fallback test split from the training patients."
        )
    df = attach_identifiers(pd.read_csv(csv))
    df["source_csv"] = "official_test"
    for col in ("Frontal/Lateral", "AP/PA"):
        if col not in df.columns:
            # The ground-truth CSV carries labels only; view is recoverable from Path.
            df[col] = np.where(
                df["view_kind"].eq("frontal"), "Frontal", "Lateral"
            ) if col == "Frontal/Lateral" else "Unknown"
    df = select_frontal(cfg, df)
    return apply_uncertainty_policy(
        df, target_col, policy, cfg["target"]["blank_is_negative"]
    )


if __name__ == "__main__":
    main()
