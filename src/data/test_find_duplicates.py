"""Regression tests for duplicate-analysis hash-failure handling.

Guards the defect fixed on 24/08/2026: `sha1_of` and `dhash` used to return an
empty string on any exception, and `main()` dropped those rows with a bare
`if dig:`. The result was a duplicates report that looked complete but had been
computed over a silently reduced corpus - and the files most likely to be
dropped are the damaged ones, which is exactly what the sweep is for.

What is asserted here:

  1. normal hashing works and is correct (SHA-1 matches hashlib directly)
  2. a deliberate failure is returned as data, not as an empty digest
  3. `hash_all` raises HashFailureError rather than returning a partial map
  4. the error text names the affected path
  5. the error text says the report is incomplete
  6. the failure manifest CSV is written and names the path + the exception
  7. running the script end-to-end exits NON-ZERO on a hash failure
  8. NO duplicates report (duplicates.md / duplicates_exact.csv) is written
     after a failure
  9. the same script on an intact corpus exits 0 and DOES write the report

Requires no CheXpert image: everything runs on a throwaway synthetic release.

Run:  python src/data/test_find_duplicates.py
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from find_duplicates import (  # noqa: E402
    FAILURE_CSV, HashFailureError, dhash, hash_all, sha1_of,
)

_FAILURES: list[str] = []
_PASSES = 0

SPLIT_COLUMNS = [
    "Path", "patient_id", "study_id", "view_num", "Sex", "Age",
    "Frontal/Lateral", "AP/PA", "Cardiomegaly", "Support Devices",
    "target", "uncertainty_policy",
]
RELEASE = "FIXTURE-v0.0-synthetic"


def check(label: str, ok: bool, detail: str = "") -> None:
    global _PASSES
    if ok:
        _PASSES += 1
        print(f"  PASS  {label}" + (f"  - {detail}" if detail else ""))
    else:
        _FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"  - {detail}" if detail else ""))


def build_corpus(root: Path, n_patients: int = 4) -> dict:
    """A tiny synthetic release plus the CSV and config the script reads."""
    images_root = root / "images"
    release = images_root / RELEASE
    reports = root / "reports"
    (release / "train").mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    rows, files = [], []
    rng = np.random.default_rng(0)
    for pi in range(1, n_patients + 1):
        pid = f"patient{pi:05d}"
        rel = f"CheXpert-v1.0/train/{pid}/study1/view1_frontal.jpg"
        dest = release / "train" / pid / "study1" / "view1_frontal.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        arr = rng.integers(0, 255, (32, 32), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(dest, "JPEG", quality=90)
        files.append(dest)
        rows.append({
            "Path": rel, "patient_id": pid, "study_id": f"{pid}/study1",
            "view_num": 1, "Sex": "Female", "Age": 50,
            "Frontal/Lateral": "Frontal", "AP/PA": "AP",
            "Cardiomegaly": 1.0, "Support Devices": 0,
            "target": 1, "uncertainty_policy": "u_zeros",
        })

    with open(root / "train.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SPLIT_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    cfg = {
        "dataset": {"name": "FIXTURE", "release": RELEASE, "root": str(root),
                    "train_csv": "train.csv", "valid_csv": "valid.csv"},
        "images": {"root": str(images_root), "release_dir": RELEASE,
                   "expect_subdir": "train"},
        "test_set": {"enabled": False, "root": str(root / "none"),
                     "labels_csv": "groundtruth.csv"},
        "target": {"column": "Cardiomegaly", "uncertain_value": -1.0,
                   "policy": "u_zeros", "blank_is_negative": True},
        "view": {"frontal_lateral_column": "Frontal/Lateral",
                 "projection_column": "AP/PA", "keep": ["Frontal"],
                 "combine_ap_pa": True},
        "split": {"unit": "patient", "seed": 42, "train_frac": 0.85,
                  "val_frac": 0.15, "stratify": True,
                  "out_dir": str(root / "splits")},
        "preprocessing": {"input_size": 32, "resize_mode": "direct",
                          "to_rgb": "replicate",
                          "normalize": {"mean": [0.485, 0.456, 0.406],
                                        "std": [0.229, 0.224, 0.225]}},
        "subset": {"enabled": False, "target_frontal_images": 0, "seed": 42},
        "paths": {"artifacts": str(root / "artifacts"),
                  "figures": str(root / "figures"), "reports": str(reports)},
    }
    cfg_path = root / "fixture_data.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    return {"cfg": cfg, "cfg_path": cfg_path, "reports": reports,
            "files": files, "paths": [r["Path"] for r in rows],
            "release": release}


# --- 1. normal hashing ---------------------------------------------------

def test_normal_hashing(c: dict) -> None:
    print("\nNormal hashing")
    rel = c["paths"][0]
    got_rel, digest, error = sha1_of(c["cfg"], rel)

    check("sha1_of returns a 3-tuple with no error on a readable file",
          got_rel == rel and error is None and digest is not None)

    expected = hashlib.sha1(c["files"][0].read_bytes()).hexdigest()
    check("sha1_of digest matches hashlib over the same bytes",
          digest == expected, digest[:16] + "...")

    _, bits, derr = dhash(c["cfg"], rel)
    check("dhash returns a 64-bit string with no error",
          derr is None and bits is not None and len(bits) == 64,
          f"len={len(bits) if bits else None}")

    digests = hash_all(sha1_of, c["cfg"], c["paths"], 4, "sha1", c["reports"])
    check("hash_all covers every input path",
          len(digests) == len(c["paths"]), f"{len(digests)}/{len(c['paths'])}")
    check("hash_all writes no failure manifest when everything succeeds",
          not (c["reports"] / FAILURE_CSV).exists())


# --- 2. deliberate failure ----------------------------------------------

def test_hash_failure_is_data(c: dict) -> None:
    print("\nDeliberate hash failure is returned as data, never as an empty digest")
    missing = "CheXpert-v1.0/train/patient99999/study1/view1_frontal.jpg"
    rel, digest, error = sha1_of(c["cfg"], missing)

    check("sha1_of reports the failure instead of an empty string",
          digest is None and error is not None, str(error)[:60])
    check("the error names the exception type",
          "FileNotFoundError" in (error or ""), str(error)[:60])
    check("the returned path is the offending one", rel == missing)

    _, bits, derr = dhash(c["cfg"], missing)
    check("dhash also reports the failure rather than an empty string",
          bits is None and derr is not None, str(derr)[:60])


def test_hash_all_raises(c: dict) -> None:
    print("\nhash_all fails loudly and names the file")
    victim = c["files"][1]
    backup = victim.with_suffix(".jpg.bak")
    victim.rename(backup)
    try:
        raised = None
        try:
            hash_all(sha1_of, c["cfg"], c["paths"], 4, "sha1", c["reports"])
        except HashFailureError as exc:
            raised = exc

        check("hash_all raises HashFailureError rather than returning partial",
              raised is not None,
              type(raised).__name__ if raised else "no exception")

        msg = str(raised or "")
        victim_rel = c["paths"][1]
        check("the error names the affected path",
              victim_rel in msg, victim_rel)
        check("the error states the report is incomplete",
              "INCOMPLETE" in msg.upper(), "found 'INCOMPLETE'")
        check("the error carries the underlying exception",
              "FileNotFoundError" in msg)
        check("the error reports how many of how many failed",
              "1 of 4" in msg or "1 of 4," in msg,
              [ln for ln in msg.splitlines() if " of " in ln][:1])

        manifest = c["reports"] / FAILURE_CSV
        check("a failure manifest CSV is written", manifest.exists())
        if manifest.exists():
            recs = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
            check("the manifest has the expected columns",
                  set(recs[0]) == {"Path", "resolved", "stage", "error"},
                  str(sorted(recs[0])))
            check("the manifest names the affected path and its error",
                  recs[0]["Path"] == victim_rel
                  and "FileNotFoundError" in recs[0]["error"],
                  recs[0]["Path"])
    finally:
        backup.rename(victim)
        (c["reports"] / FAILURE_CSV).unlink(missing_ok=True)


def test_corrupt_image_fails_dhash(c: dict) -> None:
    print("\nA corrupt image aborts the perceptual (L5) pass too")
    victim = c["files"][2]
    original = victim.read_bytes()
    victim.write_bytes(b"this is not a JPEG")
    try:
        raised = None
        try:
            hash_all(dhash, c["cfg"], c["paths"], 4, "dhash", c["reports"])
        except HashFailureError as exc:
            raised = exc
        check("hash_all(dhash) raises on an undecodable image",
              raised is not None)
        check("the dhash error names the affected path",
              c["paths"][2] in str(raised or ""), c["paths"][2])
    finally:
        victim.write_bytes(original)
        (c["reports"] / FAILURE_CSV).unlink(missing_ok=True)


# --- 3. end-to-end script behaviour --------------------------------------

def _run_script(cfg_path: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["UROP_DATA_CONFIG"] = str(cfg_path)
    return subprocess.run(
        [sys.executable, str(HERE / "find_duplicates.py")],
        capture_output=True, text=True, env=env, timeout=300,
    )


def test_script_exit_codes(c: dict) -> None:
    print("\nEnd-to-end: exit code and report emission")
    reports = c["reports"]
    for name in ("duplicates.md", "duplicates_exact.csv", FAILURE_CSV):
        (reports / name).unlink(missing_ok=True)

    # --- intact corpus: must succeed and write the report ---
    ok = _run_script(c["cfg_path"])
    check("intact corpus exits 0", ok.returncode == 0,
          f"rc={ok.returncode} {ok.stderr.strip().splitlines()[-1:]}")
    check("intact corpus writes duplicates.md", (reports / "duplicates.md").exists())
    check("intact corpus writes duplicates_exact.csv",
          (reports / "duplicates_exact.csv").exists())

    # Clear the good report so its presence later cannot be a false negative.
    for name in ("duplicates.md", "duplicates_exact.csv"):
        (reports / name).unlink(missing_ok=True)

    # --- one unhashable file: must fail and write NO report ---
    victim = c["files"][0]
    backup = victim.with_suffix(".jpg.bak")
    victim.rename(backup)
    try:
        bad = _run_script(c["cfg_path"])
        combined = bad.stdout + bad.stderr

        check("a hash failure exits NON-ZERO", bad.returncode != 0,
              f"rc={bad.returncode}")
        check("the exit code is 2 (integrity failure, not a generic crash)",
              bad.returncode == 2, f"rc={bad.returncode}")
        check("stderr names the affected path", c["paths"][0] in combined,
              c["paths"][0])
        check("stderr says the report is incomplete",
              "INCOMPLETE" in combined.upper())
        check("NO duplicates.md is written after a failure",
              not (reports / "duplicates.md").exists())
        check("NO duplicates_exact.csv is written after a failure",
              not (reports / "duplicates_exact.csv").exists())
        check("the failure manifest IS written", (reports / FAILURE_CSV).exists())
    finally:
        backup.rename(victim)


def main() -> int:
    print("=" * 72)
    print("find_duplicates hash-failure regression tests (synthetic fixture)")
    print("=" * 72)

    tmp = Path(tempfile.mkdtemp(prefix="urop_dupfail_"))
    try:
        c = build_corpus(tmp / "corpus")
        print(f"\nfixture: {len(c['paths'])} images at {c['release']}")
        test_normal_hashing(c)
        test_hash_failure_is_data(c)
        test_hash_all_raises(c)
        test_corrupt_image_fails_dhash(c)
        test_script_exit_codes(c)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 72)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED, {_PASSES} passed")
        for f in _FAILURES:
            print(f"  FAILED: {f}")
        return 1
    print(f"ALL {_PASSES} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
