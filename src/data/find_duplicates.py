"""Stage 2 Step 7 — duplicate risk analysis.

Distinguishes five separate things that get carelessly lumped together as
"duplicates". Only levels 1 and 2 are defects. Levels 3 and 4 are legitimate
clinical data and are NEVER removed.

  L1  exact duplicate file        -> SHA-1 over file bytes         (defect)
  L2  same image, several paths   -> same SHA-1, different Path    (defect)
  L3  same study, several views   -> group by study_id             (legitimate)
  L4  same patient, many studies  -> group by patient_id           (legitimate)
  L5  near-duplicate images       -> dHash + Hamming distance      (see below)

On L5: near-duplicates WITHIN a patient are expected (serial follow-up films of
the same person) and are already neutralised by patient-level splitting. The only
near-duplicates that threaten the experiment are those spanning DIFFERENT patient
IDs, so this script reports only cross-patient collisions. That reduces the
problem from ~1.8e10 naive pairwise comparisons to a hash-bucket group-by.

NOTHING IS DELETED. Findings go to reports/duplicates_*.csv for a human decision.

Run:
    python src/data/find_duplicates.py                # L1-L4 (exact hashing)
    python src/data/find_duplicates.py --near         # also L5 (perceptual)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chexpert_metadata import (  # noqa: E402
    PATH_COL,
    ensure_dirs,
    load_config,
    load_split_csv,
    resolve_image_path,
    select_frontal,
    verify_image_root,
)

CHUNK = 1 << 20  # 1 MiB

# Written when hashing fails. Named so it can never be mistaken for a result.
FAILURE_CSV = "duplicates_hash_failures.csv"


class HashFailureError(RuntimeError):
    """At least one file could not be hashed, so the analysis is incomplete.

    Raised instead of returning a partial result. A duplicate report computed
    over a silently reduced corpus understates every count it prints — and the
    files most likely to be dropped are the damaged ones, which are exactly the
    ones a duplicate/integrity sweep exists to find.
    """


def sha1_of(cfg: dict, rel: str) -> tuple[str, str | None, str | None]:
    """Content hash of one file.

    Returns `(rel, hexdigest, None)` on success and `(rel, None, reason)` on
    failure. The failure is DATA, not an empty string: an empty digest was
    previously indistinguishable from "not in the corpus" and got dropped by a
    truthiness test in the caller. Callers must treat a non-None reason as
    fatal — see `hash_all`.
    """
    try:
        h = hashlib.sha1()
        with open(resolve_image_path(cfg, rel), "rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
        return rel, h.hexdigest(), None
    except Exception as exc:  # noqa: BLE001 — reported, never swallowed
        return rel, None, f"{type(exc).__name__}: {exc}"


def dhash(cfg: dict, rel: str, size: int = 8) -> tuple[str, str | None, str | None]:
    """64-bit difference hash — robust to resize/compression, sensitive to content.

    Same contract as `sha1_of`: `(rel, bitstring, None)` or `(rel, None, reason)`.
    """
    try:
        with Image.open(resolve_image_path(cfg, rel)) as im:
            g = np.asarray(im.convert("L").resize((size + 1, size)), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        return rel, "".join("1" if b else "0" for b in bits), None
    except Exception as exc:  # noqa: BLE001 — reported, never swallowed
        return rel, None, f"{type(exc).__name__}: {exc}"


def hash_all(fn, cfg: dict, paths: list[str], workers: int, label: str,
             rep_dir: Path) -> dict[str, str]:
    """Hash every path with `fn`, or fail loudly naming every file that could not be.

    Returns `{rel: digest}` covering ALL of `paths`, or raises. There is
    deliberately no partial-success path: the previous behaviour dropped failed
    files from the digest map and then reported "Files hashed: N" against a
    larger scope figure, with nothing tying the two numbers together.

    On failure the offending paths and their exceptions are written to
    `reports/duplicates_hash_failures.csv` — a diagnostic, not a result — and
    HashFailureError is raised before any duplicate report is written.
    """
    digests: dict[str, str] = {}
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, cfg, p) for p in paths]
        for f in tqdm(as_completed(futs), total=len(futs), unit="file"):
            rel, digest, error = f.result()
            if error is not None:
                failures.append({"Path": rel,
                                 "resolved": str(resolve_image_path(cfg, rel)),
                                 "stage": label, "error": error})
            else:
                digests[rel] = digest

    if failures:
        rep_dir.mkdir(parents=True, exist_ok=True)
        out = rep_dir / FAILURE_CSV
        pd.DataFrame(failures, columns=["Path", "resolved", "stage", "error"]) \
            .to_csv(out, index=False)

        shown = failures[:10]
        detail = "\n".join(f"    {r['Path']}\n      -> {r['resolved']}\n"
                           f"      {r['error']}" for r in shown)
        more = (f"\n    ... and {len(failures) - len(shown):,} more"
                if len(failures) > len(shown) else "")
        raise HashFailureError(
            f"{label}: {len(failures):,} of {len(paths):,} files could not be "
            f"hashed.\n"
            f"THE DUPLICATE REPORT IS INCOMPLETE AND HAS NOT BEEN WRITTEN.\n"
            f"Affected files:\n{detail}{more}\n"
            f"  Full list: {out}\n"
            f"  Duplicate counts computed without these files would understate "
            f"every total, and damaged files are precisely what this sweep "
            f"looks for. Run validate_images.py to quarantine unreadable files "
            f"into excluded_images.csv, then re-run."
        )

    # Every input path is accounted for; assert rather than trust the loop.
    if len(digests) != len(set(paths)):
        raise HashFailureError(
            f"{label}: hashed {len(digests):,} distinct paths but was given "
            f"{len(set(paths)):,}. Refusing to report on a partial corpus."
        )
    return digests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", action="store_true", help="also run L5 perceptual hashing")
    ap.add_argument("--max-hamming", type=int, default=4,
                    help="dHash Hamming distance treated as near-duplicate")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    # Stop immediately if the image release is absent or mis-shaped, rather than
    # emitting one 'unreadable' record per image.
    release = verify_image_root(cfg)
    print(f"[info] image release: {release}")
    rep_dir = Path(cfg["paths"]["reports"])

    df = select_frontal(cfg, load_split_csv(cfg, "train"))
    paths = df[PATH_COL].astype(str).tolist()
    meta = df.set_index(PATH_COL)[["patient_id", "study_id"]]

    lines = ["# Stage 2 — duplicate analysis (generated)\n"]
    lines.append(f"Scope: `{len(paths):,}` frontal training images.\n")

    # ------------------------------------------------- L3 / L4: legitimate ------
    n_studies = df["study_id"].nunique()
    n_patients = df["patient_id"].nunique()
    multi_view = df.groupby("study_id").size()
    multi_study = df.groupby("patient_id")["study_id"].nunique()

    lines.append("\n## L3/L4 — legitimate repetition (NOT removed)\n")
    lines.append("| Level | Statistic | Value |\n|---|---|---:|")
    lines.append(f"| L3 | Studies with >1 frontal image | "
                 f"{int((multi_view > 1).sum()):,} of {n_studies:,} |")
    lines.append(f"| L3 | Max frontal images in one study | {int(multi_view.max())} |")
    lines.append(f"| L4 | Patients with >1 study | "
                 f"{int((multi_study > 1).sum()):,} of {n_patients:,} |")
    lines.append(f"| L4 | Max studies for one patient | {int(multi_study.max())} |")
    lines.append(
        "\nL3 rows are separate radiographs acquired in the same visit and L4 rows are "
        "genuine follow-up imaging. Both are kept. Patient-level splitting — not "
        "deletion — is what stops them leaking.\n"
    )

    # ------------------------------------------------- L1 / L2: exact ----------
    print(f"[info] SHA-1 hashing {len(paths):,} files")
    # Raises HashFailureError naming every unhashable file rather than dropping
    # it. Nothing below this line runs on a partial corpus.
    digests = hash_all(sha1_of, cfg, paths, args.workers, "sha1", rep_dir)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for rel, dig in digests.items():
        by_hash[dig].append(rel)
    dup_groups = {d: ps for d, ps in by_hash.items() if len(ps) > 1}

    rows = []
    for dig, group in dup_groups.items():
        pats = sorted({meta.loc[p, "patient_id"] for p in group})
        studies = sorted({meta.loc[p, "study_id"] for p in group})
        for p in group:
            rows.append({
                "sha1": dig, "Path": p,
                "patient_id": meta.loc[p, "patient_id"],
                "study_id": meta.loc[p, "study_id"],
                "group_size": len(group),
                "spans_patients": len(pats) > 1,
                "spans_studies": len(studies) > 1,
            })
    exact = pd.DataFrame(rows)
    exact.to_csv(rep_dir / "duplicates_exact.csv", index=False)

    n_extra = sum(len(g) - 1 for g in dup_groups.values())
    cross_pat = int(exact["spans_patients"].sum()) if len(exact) else 0
    lines.append("\n## L1/L2 — exact duplicates (SHA-1 over file bytes)\n")
    lines.append("| Finding | Value |\n|---|---:|")
    lines.append(f"| Files hashed | {len(digests):,} |")
    lines.append(f"| Distinct content hashes | {len(by_hash):,} |")
    lines.append(f"| Duplicate groups | {len(dup_groups):,} |")
    lines.append(f"| Redundant files (group size − 1) | {n_extra:,} |")
    lines.append(f"| Duplicate rows spanning >1 patient | {cross_pat:,} |")
    lines.append("\nDetail: `reports/duplicates_exact.csv`. Nothing deleted.\n")
    if cross_pat:
        lines.append(
            "> **Cross-patient exact duplicates found.** These are the dangerous kind: "
            "identical pixels under two patient IDs defeat patient-level splitting. "
            "Resolve before training — keep one copy and record the decision.\n"
        )

    # ------------------------------------------------- L5: near-duplicate ------
    if args.near:
        print(f"[info] dHash over {len(paths):,} files")
        # Same contract as the SHA-1 pass: any unreadable image aborts the L5
        # analysis rather than quietly shrinking the comparison set.
        hashes = hash_all(dhash, cfg, paths, args.workers, "dhash", rep_dir)

        # Bucket by the first 16 bits, then compare only within buckets. Exact
        # matches and small-Hamming pairs almost always share a prefix, which
        # turns an O(n^2) sweep into something that finishes in minutes.
        buckets: dict[str, list[str]] = defaultdict(list)
        for rel, hh in hashes.items():
            buckets[hh[:16]].append(rel)

        arr = {rel: np.frombuffer(hh.encode(), dtype=np.uint8) for rel, hh in hashes.items()}
        near_rows = []
        for bucket in tqdm(buckets.values(), desc="buckets", unit="bucket"):
            if len(bucket) < 2:
                continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    pa, pb = meta.loc[a, "patient_id"], meta.loc[b, "patient_id"]
                    if pa == pb:
                        continue  # within-patient — handled by the split, not a defect
                    dist = int(np.count_nonzero(arr[a] != arr[b]))
                    if dist <= args.max_hamming:
                        near_rows.append({
                            "path_a": a, "path_b": b,
                            "patient_a": pa, "patient_b": pb,
                            "hamming": dist,
                        })
        near = pd.DataFrame(near_rows)
        near.to_csv(rep_dir / "duplicates_near_crosspatient.csv", index=False)
        lines.append(f"\n## L5 — near-duplicates across different patients\n")
        lines.append(f"dHash 64-bit, Hamming ≤ {args.max_hamming}, "
                     f"cross-patient pairs only.\n")
        lines.append(f"- Cross-patient near-duplicate pairs: **{len(near):,}**")
        lines.append(f"- Distinct patients involved: "
                     f"**{len(set(near['patient_a']) | set(near['patient_b'])) if len(near) else 0:,}**")
        lines.append("\nDetail: `reports/duplicates_near_crosspatient.csv`.\n")
    else:
        lines.append("\n## L5 — near-duplicates\n")
        lines.append(
            "Not run in this invocation. Re-run with `--near` to enable. Deferring it "
            "is defensible only because within-patient near-duplicates are already "
            "neutralised by patient-level splitting; the cross-patient case is the "
            "one that matters and `--near` is the check for it.\n"
        )

    out = rep_dir / "duplicates.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] report -> {out}")


if __name__ == "__main__":
    try:
        main()
    except HashFailureError as exc:
        # Exit 2 (not 1) so an automated caller can distinguish "some files are
        # unhashable, the report is incomplete" from an ordinary crash.
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(2)
