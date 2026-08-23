"""Stage 3 dataset - reads the Stage 2 split CSVs and serves image/target pairs.

Design rules, each traceable to a project invariant:

* Splits are READ, never recomputed. make_splits.py owns patient-level
  splitting and its audit trail (split_manifest.json). Recomputing here would
  create a second, silently divergent definition of "the test set".
* Image paths go through chexpert_metadata.resolve_image_path only. No ad-hoc
  joins - the naive `root / Path` join is the exact bug that was fixed on
  21/08/2026 and is regression-tested in test_path_resolution.py.
* A missing image is a hard error, never a dropped row. Dropping would change
  the denominator of every metric with nothing in the logs to say so.
* The stored `target` column is re-derived from the raw `Cardiomegaly` column
  and the configured uncertainty policy, and the two must agree. That catches a
  split CSV generated under a different policy being fed to a differently
  configured run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))

from chexpert_metadata import (  # noqa: E402
    apply_uncertainty_policy,
    resolve_image_path,
    verify_image_root,
)
from transforms import build_eval_transform, build_train_transform  # noqa: E402

REQUIRED_COLUMNS = ("Path", "patient_id", "study_id", "target", "uncertainty_policy")


class CardiomegalyDataset(Dataset):
    """One split of the binary Cardiomegaly task.

    Returns `(image_tensor, target_float, index)`. The index is returned so
    predictions can be joined back to `self.frame` rows for per-row error
    analysis without relying on DataLoader ordering.
    """

    def __init__(self, split_csv: str | Path, data_cfg: dict, transform,
                 *, split_name: str, on_missing: str = "error",
                 exclusions_csv: str | Path | None = None,
                 verify_images: bool = True):
        self.split_name = split_name
        self.split_csv = Path(split_csv)
        self.data_cfg = data_cfg
        self.transform = transform
        self.on_missing = on_missing

        if not self.split_csv.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {self.split_csv}\n"
                f"Generate the splits first:  python src/data/make_splits.py"
            )

        frame = pd.read_csv(self.split_csv)
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing_cols:
            raise ValueError(
                f"{self.split_csv} is missing column(s) {missing_cols}. "
                f"Expected the schema written by make_splits.py."
            )
        if frame.empty:
            raise ValueError(f"{self.split_csv} contains no rows.")

        self._check_policy_consistency(frame)
        self.excluded_count = 0
        if exclusions_csv:
            frame = self._apply_exclusions(frame, exclusions_csv)

        self.frame = frame.reset_index(drop=True)
        self.targets = self.frame["target"].to_numpy(dtype=np.float32)
        self.patient_ids = self.frame["patient_id"].to_numpy()

        # Resolve once, up front. resolve_image_path is pure arithmetic (no
        # filesystem access), so this is cheap even at ~200k rows.
        self.image_paths = [
            resolve_image_path(data_cfg, rel) for rel in self.frame["Path"]
        ]

        if verify_images:
            self._verify_images_present()

    # --- integrity -------------------------------------------------------

    def _check_policy_consistency(self, frame: pd.DataFrame) -> None:
        """The split's policy, the config's policy and the stored target must agree.

        Three independent statements of the same fact. If they diverge, the run
        would train on labels that are not the ones the config claims.
        """
        cfg_policy = self.data_cfg["target"]["policy"]
        csv_policies = set(frame["uncertainty_policy"].dropna().unique())
        if csv_policies != {cfg_policy}:
            raise ValueError(
                f"Uncertainty-policy mismatch in {self.split_csv.name}.\n"
                f"  config/data.yaml target.policy : {cfg_policy!r}\n"
                f"  policies stamped in the CSV     : {sorted(csv_policies)}\n"
                f"Training on a split built under a different policy would "
                f"silently change the label definition (D203). Regenerate the "
                f"splits or fix the config - do not override this."
            )

        target_col = self.data_cfg["target"]["column"]
        if target_col not in frame.columns:
            # The raw observation column is not strictly required to train, but
            # without it the stored target cannot be independently verified.
            print(f"[warn] {self.split_csv.name}: raw {target_col!r} column absent; "
                  f"stored `target` cannot be re-derived and is trusted as-is")
            return

        rederived = apply_uncertainty_policy(
            frame.copy(), target_col, cfg_policy,
            self.data_cfg["target"]["blank_is_negative"],
        )["target"].to_numpy(dtype=np.float32)
        stored = frame["target"].to_numpy(dtype=np.float32)
        n_bad = int((rederived != stored).sum())
        if n_bad:
            raise ValueError(
                f"{self.split_csv.name}: {n_bad} of {len(frame)} rows have a "
                f"stored `target` that does not match re-applying policy "
                f"{cfg_policy!r} to the raw {target_col!r} column. The split CSV "
                f"and the configured label definition disagree."
            )

    def _apply_exclusions(self, frame: pd.DataFrame,
                          exclusions_csv: str | Path) -> pd.DataFrame:
        """Skip rows whose image failed integrity checking (section 8.4).

        These are SKIPPED, never deleted from disk. A missing exclusions file is
        tolerated because validate_images.py has not run yet - but the fact is
        logged, never silent.
        """
        p = Path(exclusions_csv)
        if not p.exists():
            print(f"[info] {self.split_name}: no exclusions file at {p} "
                  f"(validate_images.py has not run) - using all rows")
            return frame
        excl = pd.read_csv(p)
        if "Path" not in excl.columns:
            raise ValueError(f"{p} has no 'Path' column")
        bad = set(excl["Path"].astype(str))
        before = len(frame)
        frame = frame[~frame["Path"].astype(str).isin(bad)]
        self.excluded_count = before - len(frame)
        print(f"[info] {self.split_name}: excluded {self.excluded_count:,} of "
              f"{before:,} rows via {p.name}")
        if frame.empty:
            raise ValueError(
                f"{self.split_name}: every row was excluded by {p.name}. "
                f"Refusing to build an empty dataset."
            )
        return frame

    def _verify_images_present(self) -> None:
        """Fail loudly if any referenced image is absent.

        `on_missing='error'` stops at the first offender; `'report'` collects
        them all and then raises with the full list. Neither drops a row. Both
        run verify_image_root first, so an un-downloaded release produces one
        actionable error instead of N per-file 'missing' records.
        """
        verify_image_root(self.data_cfg)

        missing: list[str] = []
        for rel, resolved in zip(self.frame["Path"], self.image_paths):
            if not resolved.exists():
                if self.on_missing == "error":
                    raise FileNotFoundError(
                        f"{self.split_name}: image referenced by the split CSV "
                        f"does not exist on disk.\n"
                        f"  CSV Path : {rel}\n"
                        f"  resolved : {resolved}\n"
                        f"Set data.on_missing_image: report in config/train.yaml "
                        f"to list every missing file instead of stopping here. "
                        f"Rows are never silently dropped."
                    )
                missing.append(f"{rel}  ->  {resolved}")
        if missing:
            head = "\n  ".join(missing[:20])
            raise FileNotFoundError(
                f"{self.split_name}: {len(missing):,} of {len(self.frame):,} "
                f"images referenced by the split CSV are absent.\n  {head}"
                + (f"\n  ... and {len(missing) - 20:,} more" if len(missing) > 20 else "")
            )

    # --- torch API -------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            with Image.open(path) as im:
                # 'L' matches section 10: the release is already 8-bit grayscale.
                img = im.convert("L")
                img.load()
        except Exception as exc:  # noqa: BLE001 - re-raised with context, not swallowed
            raise RuntimeError(
                f"{self.split_name}: failed to read image at row {idx}: {path} "
                f"({type(exc).__name__}: {exc}). Run validate_images.py to "
                f"quarantine unreadable files into excluded_images.csv rather "
                f"than skipping them here."
            ) from exc

        tensor = self.transform(img)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return tensor, target, idx

    # --- reporting -------------------------------------------------------

    def stats(self) -> dict:
        """Summary suitable for the run manifest and the console."""
        n = len(self.frame)
        pos = int(self.targets.sum())
        neg = n - pos
        out = {
            "split": self.split_name,
            "csv": str(self.split_csv),
            "images": n,
            "patients": int(self.frame["patient_id"].nunique()),
            "studies": int(self.frame["study_id"].nunique()),
            "positive": pos,
            "negative": neg,
            "prevalence": round(pos / n, 5) if n else None,
            "excluded_rows": self.excluded_count,
            "uncertainty_policy": self.data_cfg["target"]["policy"],
        }
        if "AP/PA" in self.frame.columns:
            out["projection_counts"] = (
                self.frame["AP/PA"].value_counts().to_dict()
            )
        return out


def assert_no_patient_overlap(**named: CardiomegalyDataset) -> None:
    """Hard patient-disjointness gate across every split about to be used.

    make_splits.assert_disjoint already guarantees this when the splits are
    written. This re-checks it at training time, because the CSVs are files on
    disk that can be regenerated, hand-edited or mixed between runs. Cardiomegaly
    is chronic, so a shared patient means memorising the patient is memorising
    the label.
    """
    names = list(named)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(named[a].patient_ids) & set(named[b].patient_ids)
            if overlap:
                sample = sorted(overlap)[:10]
                raise AssertionError(
                    f"PATIENT LEAKAGE: {len(overlap):,} patient(s) appear in "
                    f"both '{a}' and '{b}'. Examples: {sample}\n"
                    f"Splitting is patient-level and non-negotiable "
                    f"(docs/dataset_analysis.md section 5, D205)."
                )


def load_split_manifest(splits_dir: str | Path, filename: str) -> dict:
    p = Path(splits_dir) / filename
    if not p.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {p}\n"
            f"It is the audit trail for the splits and the source of "
            f"pos_weight. Run: python src/data/make_splits.py"
        )
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_datasets(train_cfg: dict, data_cfg: dict, *, verify_images: bool = True):
    """Construct the three splits with the correct transform on each.

    The train split gets the augmenting transform; val and test get the
    deterministic one. They are separate objects built by separate functions -
    there is no flag that could be passed wrongly (section 11).
    """
    d = train_cfg["data"]
    splits_dir = Path(d["splits_dir"])
    on_missing = d["on_missing_image"]
    exclusions = d.get("exclusions_csv")

    train_tf = build_train_transform(data_cfg, train_cfg["augmentation"])
    eval_tf = build_eval_transform(data_cfg)

    train_ds = CardiomegalyDataset(
        splits_dir / d["train_csv"], data_cfg, train_tf, split_name="train",
        on_missing=on_missing, exclusions_csv=exclusions, verify_images=verify_images,
    )
    val_ds = CardiomegalyDataset(
        splits_dir / d["val_csv"], data_cfg, eval_tf, split_name="val",
        on_missing=on_missing, exclusions_csv=exclusions, verify_images=verify_images,
    )
    test_ds = CardiomegalyDataset(
        splits_dir / d["test_csv"], data_cfg, eval_tf, split_name="test",
        on_missing=on_missing, exclusions_csv=exclusions, verify_images=verify_images,
    )

    assert_no_patient_overlap(train=train_ds, val=val_ds, test=test_ds)
    return train_ds, val_ds, test_ds
