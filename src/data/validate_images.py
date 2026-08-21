"""Stage 2 Steps 8 + 9 — image integrity validation and dimension analysis.

One pass over every retained frontal image. For each file it records existence,
whether the JPEG decodes, its dimensions, aspect ratio and basic pixel sanity.

NOTHING IS DELETED. Failures are written to an exclusion manifest with a reason
column; the training pipeline later reads that manifest and skips those rows.

Two-tier check:
  * cheap  — PIL lazy open: header parse + dimensions only (fast, catches truncation
             of the header and non-images)
  * full   — force a full decode and compute min/max/mean pixel values (catches
             truncated scan data and blank/constant images)

Run:
    python src/data/validate_images.py                 # full check, all frontal images
    python src/data/validate_images.py --sample 5000   # quick reconnaissance pass
    python src/data/validate_images.py --dims-only     # skip full decode (much faster)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chexpert_metadata import (  # noqa: E402
    PALETTE,
    PATH_COL,
    apply_plot_style,
    compact_formatter,
    ensure_dirs,
    load_config,
    load_split_csv,
    select_frontal,
)

# Surface truncated files as errors instead of silently padding them with grey.
ImageFile.LOAD_TRUNCATED_IMAGES = False
warnings.simplefilter("error", Image.DecompressionBombWarning)


def check_one(rel_path: str, root: Path, full_decode: bool) -> dict:
    """Return a record describing one image. Never raises; never writes."""
    rec = {
        "Path": rel_path,
        "status": "valid",
        "reason": "",
        "width": np.nan,
        "height": np.nan,
        "mode": "",
        "pmin": np.nan,
        "pmax": np.nan,
        "pmean": np.nan,
    }
    # CheXpert Path values already include the release directory as their first
    # segment, so resolve against the release root's PARENT.
    fpath = (root.parent / rel_path) if not (root / rel_path).exists() else (root / rel_path)

    if not fpath.exists():
        rec["status"] = "missing"
        rec["reason"] = "file not found on disk"
        return rec

    try:
        with Image.open(fpath) as im:
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
            if im.width <= 0 or im.height <= 0:
                rec["status"] = "invalid_dims"
                rec["reason"] = f"non-positive dimensions {im.size}"
                return rec
            if full_decode:
                arr = np.asarray(im.convert("L"), dtype=np.uint8)
                rec["pmin"] = int(arr.min())
                rec["pmax"] = int(arr.max())
                rec["pmean"] = float(arr.mean())
                if rec["pmin"] == rec["pmax"]:
                    rec["status"] = "constant"
                    rec["reason"] = f"all pixels = {rec['pmin']} (blank image)"
    except Exception as exc:  # noqa: BLE001 — we want every failure mode recorded
        rec["status"] = "unreadable"
        rec["reason"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="check only N randomly sampled images (seeded)")
    ap.add_argument("--dims-only", action="store_true",
                    help="skip full pixel decode (dimensions + header only)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    apply_plot_style()
    root = Path(cfg["dataset"]["root"])
    rep_dir = Path(cfg["paths"]["reports"])
    fig_dir = Path(cfg["paths"]["figures"])

    df = select_frontal(cfg, load_split_csv(cfg, "train"))
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=cfg["split"]["seed"])
    paths = df[PATH_COL].astype(str).tolist()
    print(f"[info] checking {len(paths):,} frontal images "
          f"({'dims only' if args.dims_only else 'full decode'})")

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_one, p, root, not args.dims_only) for p in paths]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            records.append(fut.result())

    res = pd.DataFrame(records)
    res.to_csv(rep_dir / "image_validation_full.csv", index=False)

    # ------------------------------------------------------------ Step 8 -------
    counts = res["status"].value_counts()
    valid = res[res["status"] == "valid"]
    excluded = res[res["status"] != "valid"]
    excluded.to_csv(rep_dir / "excluded_images.csv", index=False)

    lines = ["# Stage 2 — image integrity and dimension report (generated)\n"]
    lines.append(f"Checked `{len(res):,}` frontal images from "
                 f"`{cfg['dataset']['release']}`.\n")
    lines.append("\n## Step 8 — integrity\n")
    lines.append("| Status | Count | Percentage |\n|---|---:|---:|")
    for status in ["valid", "missing", "unreadable", "invalid_dims", "constant"]:
        n = int(counts.get(status, 0))
        lines.append(f"| {status} | {n:,} | {100 * n / len(res):.4f}% |")
    lines.append(f"| **total checked** | **{len(res):,}** | **100.0000%** |")
    lines.append(
        f"\nExclusion manifest: `reports/excluded_images.csv` "
        f"({len(excluded):,} rows). No file was deleted or modified.\n"
    )
    if len(excluded):
        lines.append("\n### Failure reasons\n")
        lines.append("| Reason | Count |\n|---|---:|")
        for reason, n in excluded["reason"].value_counts().head(20).items():
            lines.append(f"| {reason} | {n:,} |")

    # ------------------------------------------------------------ Step 9 -------
    if len(valid):
        w, h = valid["width"].to_numpy(), valid["height"].to_numpy()
        ar = w / h
        lines.append("\n## Step 9 — dimensions\n")
        lines.append("| Statistic | Width | Height | Aspect ratio (W/H) |\n|---|---:|---:|---:|")
        for name, fn in (("min", np.min), ("p05", lambda a: np.percentile(a, 5)),
                         ("median", np.median), ("mean", np.mean),
                         ("p95", lambda a: np.percentile(a, 95)), ("max", np.max)):
            lines.append(f"| {name} | {fn(w):,.0f} | {fn(h):,.0f} | {fn(ar):.3f} |")

        lines.append("\n### Most common exact dimensions\n")
        lines.append("| Width × Height | Images | Percentage |\n|---|---:|---:|")
        common = valid.groupby(["width", "height"]).size().sort_values(ascending=False).head(15)
        for (cw, ch), n in common.items():
            lines.append(f"| {int(cw)} × {int(ch)} | {n:,} | {100 * n / len(valid):.2f}% |")

        lines.append("\n### Implication for model input size\n")
        short = np.minimum(w, h)
        for size in (224, 320, 384):
            up = int((short < size).sum())
            lines.append(
                f"- **{size}×{size}** — {up:,} images ({100 * up / len(valid):.1f}%) "
                f"have a short side below {size} and would be UPSAMPLED "
                f"(no information gained)."
            )
        lines.append("| Colour mode | Images |\n|---|---:|")
        for mode, n in valid["mode"].value_counts().items():
            lines.append(f"| {mode} | {n:,} |")

        _fig_dimensions(valid, fig_dir)
        lines.append("\nFigure: `figures/image_dimensions.png`\n")

    out = rep_dir / "image_validation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] report -> {out}")
    print(f"[ok] valid {len(valid):,} | excluded {len(excluded):,}")


def _fig_dimensions(valid: pd.DataFrame, fig_dir: Path) -> None:
    """Two independent distributions -> small multiples, never a dual axis."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    axes[0].hist(valid["width"], bins=60, color=PALETTE["series_1"])
    axes[0].set_title("Image width (px)")
    axes[0].set_ylabel("images")

    ar = valid["width"] / valid["height"]
    axes[1].hist(ar, bins=60, color=PALETTE["series_2"])
    axes[1].set_title("Aspect ratio (width / height)")
    axes[1].axvline(1.0, color=PALETTE["muted"], linewidth=1.5, linestyle="--")
    axes[1].text(1.02, axes[1].get_ylim()[1] * 0.9, "square",
                 fontsize=9, color=PALETTE["muted"])

    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(compact_formatter(ax.get_ylim()[1]))
    fig.tight_layout()
    fig.savefig(fig_dir / "image_dimensions.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
