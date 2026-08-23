"""Smoke test for image-path resolution (Stage 2).

The metadata CSV and the image release disagree about the top-level directory:
`train_cheXbert.csv` ships full-release paths (`CheXpert-v1.0/...`) while the
downsampled release unpacks as `CheXpert-v1.0-small/`. Getting this wrong fails
silently on *every* image, so it is worth a test rather than a manual check.

Builds a throwaway release tree in a temp directory, resolves representative
paths against it, and asserts the loud-failure behaviour. Touches no real data
and downloads nothing.

Run:  python src/data/test_path_resolution.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chexpert_metadata import (  # noqa: E402
    image_release_root,
    resolve_image_path,
    strip_release_prefix,
    verify_image_root,
)

# Representative rows taken verbatim from train_cheXbert.csv.
CASES = [
    ("frontal AP", "CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg"),
    ("frontal PA", "CheXpert-v1.0/train/patient00002/study2/view1_frontal.jpg"),
    ("lateral",    "CheXpert-v1.0/train/patient00002/study1/view2_lateral.jpg"),
]

# Paths that must survive resolution unchanged or near-unchanged.
EDGE_CASES = [
    ("already small-release prefix", "CheXpert-v1.0-small/train/patient1/study1/view1_frontal.jpg",
     "train/patient1/study1/view1_frontal.jpg"),
    ("no release prefix at all",     "train/patient1/study1/view1_frontal.jpg",
     "train/patient1/study1/view1_frontal.jpg"),
    ("windows separators",           r"CheXpert-v1.0\train\patient1\study1\view1_frontal.jpg",
     "train/patient1/study1/view1_frontal.jpg"),
    ("valid split, not train",       "CheXpert-v1.0/valid/patient64541/study1/view1_frontal.jpg",
     "valid/patient64541/study1/view1_frontal.jpg"),
]

PASS, FAIL = "  PASS", "  FAIL"
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if ok else FAIL}  {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        failures += 1


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="chexpert_pathtest_"))
    try:
        images_root = tmp / "raw"
        release = images_root / "CheXpert-v1.0-small"
        cfg = {"images": {"root": str(images_root),
                          "release_dir": "CheXpert-v1.0-small",
                          "expect_subdir": "train"}}

        # ---------------------------------------------- loud-failure behaviour --
        print("\nLoud failure when the release is absent or mis-shaped")
        for label, setup in [
            ("missing images.root", lambda: None),
            ("missing release dir", lambda: images_root.mkdir(parents=True, exist_ok=True)),
            ("missing train/ subdir", lambda: release.mkdir(parents=True, exist_ok=True)),
        ]:
            setup()
            try:
                verify_image_root(cfg)
                check(label, False, "expected FileNotFoundError, got none")
            except FileNotFoundError as exc:
                first = str(exc).splitlines()[0]
                check(label, True, first[:64])

        # ------------------------------------------------ prefix stripping ------
        print("\nPrefix stripping")
        for label, raw, expected in EDGE_CASES:
            got = strip_release_prefix(raw)
            check(label, got == expected, f"{got!r}")

        # ------------------------------------------- resolution against a tree --
        print("\nResolution against a real directory tree")
        for _, rel in CASES:
            target = release / strip_release_prefix(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\xff\xd8\xff")  # 3-byte JPEG magic; content irrelevant

        release_ok = verify_image_root(cfg)
        check("verify_image_root passes on a well-formed tree",
              release_ok == release, str(release_ok))

        for label, rel in CASES:
            resolved = resolve_image_path(cfg, rel)
            expected = release / strip_release_prefix(rel)
            check(f"{label:<12} resolves", resolved == expected, str(resolved))
            check(f"{label:<12} exists on disk", resolved.exists())

        # The defect this test exists to prevent: the old naive join.
        print("\nRegression — the naive join this replaced")
        naive = release / CASES[0][1]
        check("naive root/Path join does NOT exist (bug reproduced)",
              not naive.exists(), str(naive))

        print(f"\nimage_release_root() -> {image_release_root(cfg)}")
        print("\n" + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
        sys.exit(1 if failures else 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
