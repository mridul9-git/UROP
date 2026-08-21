"""Stage 2 Steps 3, 4, 6, 12 — CheXpert metadata analysis.

Answers, from train.csv/valid.csv alone (no image files needed):
  * how the Cardiomegaly label is actually distributed across its four raw states
  * how many patients / studies / images / frontal / lateral there are
  * what each uncertainty policy does to the usable class balance
  * how much patient repetition exists (the leakage argument)

Writes Markdown tables to artifacts/stage2/reports/ and figures to
artifacts/stage2/figures/. Reads nothing but the CSVs; deletes nothing.

Run:  python src/data/analyze_metadata.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chexpert_metadata import (  # noqa: E402
    OBSERVATIONS,
    PALETTE,
    PROJECTION_COL,
    VIEW_COL,
    apply_plot_style,
    apply_uncertainty_policy,
    compact_formatter,
    ensure_dirs,
    imbalance_summary,
    load_config,
    load_split_csv,
    patient_level_labels,
    raw_label_counts,
    select_frontal,
)


def _md_table(rows: list[list], header: list[str], aligns: list[str] | None = None) -> str:
    aligns = aligns or (["---"] + ["---:"] * (len(header) - 1))
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "—"


def _bar_labels(ax, bars, values, total=None, horizontal=True):
    """Direct-label every bar; no value axis clutter."""
    for bar, v in zip(bars, values):
        txt = f"{v:,}" + (f"  ({100 * v / total:.1f}%)" if total else "")
        if horizontal:
            ax.text(
                bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2,
                txt,
                va="center",
                ha="left",
                fontsize=9,
                color=PALETTE["ink_secondary"],
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                txt,
                va="bottom",
                ha="center",
                fontsize=9,
                color=PALETTE["ink_secondary"],
            )


def main() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    apply_plot_style()

    fig_dir = Path(cfg["paths"]["figures"])
    rep_dir = Path(cfg["paths"]["reports"])
    target = cfg["target"]["column"]

    train = load_split_csv(cfg, "train")
    try:
        valid = load_split_csv(cfg, "valid")
    except FileNotFoundError:
        valid = None
        print("[warn] valid.csv not found — official validation set stats skipped.")

    md: list[str] = ["# Stage 2 — CheXpert metadata analysis (generated)\n"]
    md.append(f"Release: `{cfg['dataset']['release']}`  ·  target: `{target}`\n")

    # ---------------------------------------------------------------- Step 6 ---
    n_images = len(train)
    n_patients = train["patient_id"].nunique()
    n_studies = train["study_id"].nunique()
    frontal = train[train[VIEW_COL] == "Frontal"]
    lateral = train[train[VIEW_COL] == "Lateral"]

    md.append("\n## Corpus totals — official training CSV\n")
    md.append(
        _md_table(
            [
                ["Total patients", f"{n_patients:,}", "100.00%"],
                ["Total studies", f"{n_studies:,}", "—"],
                ["Total images", f"{n_images:,}", "100.00%"],
                ["Frontal images", f"{len(frontal):,}", _pct(len(frontal), n_images)],
                ["Lateral images", f"{len(lateral):,}", _pct(len(lateral), n_images)],
            ],
            ["Category", "Count", "Percentage"],
        )
    )

    if valid is not None:
        md.append("\n### Official validation CSV\n")
        md.append(
            _md_table(
                [
                    ["Patients", f"{valid['patient_id'].nunique():,}"],
                    ["Studies", f"{valid['study_id'].nunique():,}"],
                    ["Images", f"{len(valid):,}"],
                    ["Frontal images", f"{(valid[VIEW_COL] == 'Frontal').sum():,}"],
                ],
                ["Category", "Count"],
                ["---", "---:"],
            )
        )

    # -------------------------------------------------- Step 4: view breakdown --
    md.append("\n## View / projection breakdown (training CSV)\n")
    proj = (
        train.assign(**{PROJECTION_COL: train[PROJECTION_COL].fillna("(blank)")})
        .groupby([VIEW_COL, PROJECTION_COL])
        .size()
        .reset_index(name="images")
        .sort_values("images", ascending=False)
    )
    md.append(
        _md_table(
            [[r[VIEW_COL], r[PROJECTION_COL], f"{r['images']:,}", _pct(r["images"], n_images)]
             for _, r in proj.iterrows()],
            ["Frontal/Lateral", "AP/PA", "Images", "% of all images"],
            ["---", "---", "---:", "---:"],
        )
    )

    # ------------------------------------------- Step 3: raw label distribution --
    md.append(f"\n## `{target}` raw label states\n")
    md.append("Counted at IMAGE level (train.csv rows) and at STUDY level.\n")

    img_counts = raw_label_counts(train, target)
    study_first = train.drop_duplicates("study_id")
    std_counts = raw_label_counts(study_first, target)

    md.append(
        _md_table(
            [
                ["Positive (1.0)", f"{img_counts.positive:,}", _pct(img_counts.positive, img_counts.total),
                 f"{std_counts.positive:,}", _pct(std_counts.positive, std_counts.total)],
                ["Negative (0.0)", f"{img_counts.negative:,}", _pct(img_counts.negative, img_counts.total),
                 f"{std_counts.negative:,}", _pct(std_counts.negative, std_counts.total)],
                ["Uncertain (-1.0)", f"{img_counts.uncertain:,}", _pct(img_counts.uncertain, img_counts.total),
                 f"{std_counts.uncertain:,}", _pct(std_counts.uncertain, std_counts.total)],
                ["Blank (no mention)", f"{img_counts.blank:,}", _pct(img_counts.blank, img_counts.total),
                 f"{std_counts.blank:,}", _pct(std_counts.blank, std_counts.total)],
            ],
            ["State", "Images", "% images", "Studies", "% studies"],
            ["---", "---:", "---:", "---:", "---:"],
        )
    )

    # All 14 observations, for context and for the Support Devices confound probe.
    md.append("\n### All 14 observations (study level, for context)\n")
    rows = []
    for obs in OBSERVATIONS:
        c = raw_label_counts(study_first, obs)
        rows.append([obs, f"{c.positive:,}", f"{c.uncertain:,}", f"{c.negative + c.blank:,}"])
    md.append(_md_table(rows, ["Observation", "Positive", "Uncertain", "Negative + blank"]))

    # ------------------------------- Steps 4 + 12: frontal-only usable examples --
    frontal_sel = select_frontal(cfg, train)
    md.append("\n## Usable frontal examples under each uncertainty policy\n")
    md.append(
        "Frontal images only. `exclude` drops uncertain rows from the modelling "
        "frame; no file is removed from disk.\n"
    )
    policy_rows = []
    policy_stats = {}
    for policy in ("u_zeros", "u_ones", "exclude"):
        d = apply_uncertainty_policy(
            frontal_sel, target, policy, cfg["target"]["blank_is_negative"]
        )
        s = imbalance_summary(d)
        policy_stats[policy] = s
        policy_rows.append(
            [
                policy,
                f"{s['total']:,}",
                f"{s['positive']:,}",
                f"{s['negative']:,}",
                f"{100 * s['prevalence']:.2f}%",
                f"1 : {s['neg_per_pos']:.2f}",
                f"{s['pos_weight']:.3f}",
            ]
        )
    md.append(
        _md_table(
            policy_rows,
            ["Policy", "Usable images", "Positive", "Negative", "Prevalence",
             "Pos : Neg", "pos_weight"],
            ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
        )
    )

    # --------------------------------------------- Step 5: leakage evidence -----
    primary = apply_uncertainty_policy(
        frontal_sel, target, cfg["target"]["policy"], cfg["target"]["blank_is_negative"]
    )
    plab = patient_level_labels(primary)

    md.append("\n## Patient repetition — the leakage argument\n")
    md.append(
        _md_table(
            [
                ["Patients with frontal images", f"{len(plab):,}"],
                ["Mean images per patient", f"{plab['n_images'].mean():.2f}"],
                ["Median images per patient", f"{plab['n_images'].median():.0f}"],
                ["Max images per patient", f"{plab['n_images'].max():,}"],
                ["Patients with >1 image", f"{(plab['n_images'] > 1).sum():,} "
                 f"({_pct(int((plab['n_images'] > 1).sum()), len(plab))})"],
                ["Mean studies per patient", f"{plab['n_studies'].mean():.2f}"],
                ["Patients with >1 study", f"{(plab['n_studies'] > 1).sum():,} "
                 f"({_pct(int((plab['n_studies'] > 1).sum()), len(plab))})"],
                ["Patients positive (any study)", f"{plab['any_positive'].sum():,} "
                 f"({_pct(int(plab['any_positive'].sum()), len(plab))})"],
            ],
            ["Statistic", "Value"],
            ["---", "---:"],
        )
    )
    md.append(
        f"\n> {(plab['n_images'] > 1).sum():,} of {len(plab):,} patients contribute more "
        f"than one frontal image. Under a random image-level split those patients' "
        f"images land on both sides of the train/test boundary.\n"
    )

    # ------------------------------------------------------------- figures ------
    _fig_label_states(img_counts, target, fig_dir)
    _fig_view_breakdown(proj, fig_dir)
    _fig_policy_balance(policy_stats, fig_dir)
    _fig_images_per_patient(plab, fig_dir)

    md.append("\n## Figures\n")
    for name in (
        "cardiomegaly_label_states.png",
        "view_breakdown.png",
        "uncertainty_policy_balance.png",
        "images_per_patient.png",
    ):
        md.append(f"- `figures/{name}`")

    out = rep_dir / "metadata_analysis.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] report  -> {out}")
    print(f"[ok] figures -> {fig_dir}")

    # Machine-readable companion for downstream scripts / the written report.
    pd.DataFrame(policy_stats).T.to_csv(rep_dir / "policy_class_balance.csv")
    plab.to_csv(rep_dir / "patient_level_labels.csv", index=False)


# --- figures -------------------------------------------------------------------


def _fig_label_states(counts, target: str, fig_dir: Path) -> None:
    """Magnitude across four named states -> horizontal bar, direct-labelled."""
    labels = ["Positive", "Negative", "Uncertain", "Blank (no mention)"]
    values = [counts.positive, counts.negative, counts.uncertain, counts.blank]
    colors = [PALETTE["series_2"], PALETTE["series_1"], PALETTE["series_3"], PALETTE["muted"]]

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    for b in bars:
        b.set_linewidth(0)
    ax.set_title(f"{target} label states — all training images")
    ax.set_xlim(0, max(values) * 1.28)
    ax.xaxis.set_visible(False)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    _bar_labels(ax, bars, values[::-1], total=counts.total)
    fig.tight_layout()
    fig.savefig(fig_dir / "cardiomegaly_label_states.png", bbox_inches="tight")
    plt.close(fig)


def _fig_view_breakdown(proj: pd.DataFrame, fig_dir: Path) -> None:
    labels = [f"{r[VIEW_COL]} · {r[PROJECTION_COL]}" for _, r in proj.iterrows()]
    values = proj["images"].tolist()
    total = sum(values)
    colors = [
        PALETTE["series_1"] if v == "Frontal" else PALETTE["series_4"]
        for v in proj[VIEW_COL]
    ]

    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(labels) + 1.7))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    ax.set_title("Images by view and projection")
    ax.set_xlim(0, max(values) * 1.30)
    ax.xaxis.set_visible(False)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    _bar_labels(ax, bars, values[::-1], total=total)
    fig.tight_layout()
    fig.savefig(fig_dir / "view_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def _fig_policy_balance(stats: dict, fig_dir: Path) -> None:
    """Two series (positive / negative) across three policies -> grouped bars + legend."""
    policies = list(stats)
    pos = [stats[p]["positive"] for p in policies]
    neg = [stats[p]["negative"] for p in policies]
    x = np.arange(len(policies))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    b1 = ax.bar(x - w / 2, pos, w, label="Positive", color=PALETTE["series_2"])
    b2 = ax.bar(x + w / 2, neg, w, label="Negative", color=PALETTE["series_1"])
    ax.set_xticks(x, policies)
    ax.set_title("Usable frontal images by uncertainty policy")
    ax.set_ylim(0, max(neg) * 1.18)
    # Bars are direct-labelled, so the value axis would only repeat them.
    ax.yaxis.set_visible(False)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", fontsize=9)
    for bars, vals in ((b1, pos), (b2, neg)):
        _bar_labels(ax, bars, vals, horizontal=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "uncertainty_policy_balance.png", bbox_inches="tight")
    plt.close(fig)


def _fig_images_per_patient(plab: pd.DataFrame, fig_dir: Path) -> None:
    """Distribution of a count -> histogram, clipped tail with an honest label."""
    clip = 12
    vals = plab["n_images"].clip(upper=clip)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.hist(vals, bins=np.arange(0.5, clip + 1.5, 1), color=PALETTE["series_1"], rwidth=0.86)
    ax.set_title("Frontal images per patient (leakage exposure)")
    ax.set_xlabel(f"images per patient  (≥{clip} pooled in the last bar)")
    ax.set_ylabel("patients")
    ax.set_xticks(range(1, clip + 1))
    ax.yaxis.set_major_formatter(compact_formatter(ax.get_ylim()[1]))
    ax.grid(axis="x", visible=False)
    median = plab["n_images"].median()
    ax.axvline(median, color=PALETTE["series_2"], linewidth=2, zorder=3)
    ax.text(median + 0.15, ax.get_ylim()[1] * 0.92, f"median {median:.0f}",
            color=PALETTE["series_2"], fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "images_per_patient.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
