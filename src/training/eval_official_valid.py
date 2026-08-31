"""Stage 3 expert check - score a trained checkpoint on the OFFICIAL CheXpert
200-patient validation set.

    python src/training/eval_official_valid.py --checkpoint <run>/checkpoints/best.pt \
        --threshold 0.44209548389690995

This is a SECONDARY expert-labelled check, never a model-selection surface
(docs/dataset_analysis.md D012, section 13.2: "Official 200-patient valid set
kept as a secondary expert-labelled check, never tuned on").

Two rules make that binding rather than aspirational:

* The threshold is REQUIRED on the command line and is applied unchanged. This
  script has no threshold-selection path at all, so it cannot tune on the
  official set even by mistake. Pass the threshold that was selected on the
  carved validation split, and it is recorded here with its provenance.
* Nothing is written outside the new evaluation directory. The training run,
  config, splits and `data/` are read-only to this script.

The frontal-only filter, identifier parsing and uncertainty policy all come from
chexpert_metadata - the same functions make_splits.py used to build the frozen
splits - so the official set is prepared under exactly the project's view and
label policy rather than a second, subtly different one.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

from checkpoint import load_checkpoint  # noqa: E402
from chexpert_metadata import (  # noqa: E402
    apply_uncertainty_policy, attach_identifiers, load_config,
    resolve_image_path, select_frontal,
)
from config import load_train_config, validate_train_config  # noqa: E402
from dataset import CardiomegalyDataset  # noqa: E402
from engine import evaluate  # noqa: E402
from metrics import metrics_table  # noqa: E402
from models import build_from_config, model_summary  # noqa: E402
from reproducibility import (  # noqa: E402
    environment_snapshot, git_dirty, git_sha, sha1_of_file, utc_now_iso,
    write_json,
)
from transforms import build_eval_transform, describe_transform  # noqa: E402

# The column contract make_splits.py writes. Matched exactly so the official set
# is served by the same CardiomegalyDataset that serves the frozen splits.
KEEP_COLS = [
    "Path", "patient_id", "study_id", "view_num", "Sex", "Age",
    "Frontal/Lateral", "AP/PA", "Cardiomegaly", "Support Devices",
    "target", "uncertainty_policy",
]


def build_official_frame(data_cfg: dict, valid_csv: Path) -> tuple[pd.DataFrame, dict]:
    """Read the official valid.csv and prepare it under the project's policy.

    Uses the same three chexpert_metadata functions make_splits.py uses. Returns
    the frontal-only frame plus a record of exactly what was dropped, so the
    view filter is auditable rather than implicit.
    """
    raw = pd.read_csv(valid_csv)
    with_ids = attach_identifiers(raw)
    frontal = select_frontal(data_cfg, with_ids)

    policy = data_cfg["target"]["policy"]
    frame = apply_uncertainty_policy(
        frontal, data_cfg["target"]["column"], policy,
        data_cfg["target"]["blank_is_negative"],
    )
    frame["uncertainty_policy"] = policy

    view_col = data_cfg["view"]["frontal_lateral_column"]
    counts = raw[view_col].value_counts().to_dict()
    audit = {
        "rows_in_csv": int(len(raw)),
        "view_counts_in_csv": {str(k): int(v) for k, v in counts.items()},
        "views_kept": list(data_cfg["view"]["keep"]),
        "rows_kept_frontal": int(len(frame)),
        "rows_excluded_non_frontal": int(len(raw) - len(frame)),
        "uncertainty_policy_applied": policy,
        # The official set is radiologist-adjudicated: there are no uncertain
        # labels for the policy to act on. Recorded, not assumed.
        "uncertain_labels_in_csv": int(
            (raw[data_cfg["target"]["column"]]
             == data_cfg["target"]["uncertain_value"]).sum()
        ),
        "blank_labels_in_csv": int(raw[data_cfg["target"]["column"]].isna().sum()),
    }
    return frame[KEEP_COLS].sort_values("Path").reset_index(drop=True), audit


def load_data_cfg(train_cfg: dict) -> dict:
    return load_config(train_cfg["data"]["data_config"])


def _official_valid_csv(data_cfg: dict) -> str:
    """dataset.valid_csv from config/data.yaml, rooted at the repo if relative."""
    v = data_cfg["dataset"]["valid_csv"]
    p = Path(v)
    return str(p if p.is_absolute() else REPO_ROOT / p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="trained checkpoint to score (e.g. <run>/checkpoints/best.pt)")
    ap.add_argument("--threshold", type=float, required=True,
                    help="threshold selected ELSEWHERE and applied unchanged here. "
                         "This script never selects one.")
    ap.add_argument("--threshold-source", default="carved-val-f1",
                    help="provenance label for --threshold, recorded in the manifest")
    ap.add_argument("--valid-csv", default=None,
                    help="official validation CSV (default: dataset.valid_csv "
                         "from config/data.yaml)")
    ap.add_argument("--config", default=None,
                    help="train config; default is the checkpoint's own run "
                         "config_snapshot.yaml so preprocessing matches training")
    ap.add_argument("--out-root", default=None,
                    help="evaluation root (default: <output_root>/../eval)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: the run's micro_batch")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    run_dir = ckpt_path.parent.parent

    # Preprocessing must be the run's own, or the numbers describe a different
    # pipeline than the one that produced the weights.
    cfg_path = args.config or (run_dir / "config_snapshot.yaml")
    if not Path(cfg_path).exists():
        raise FileNotFoundError(
            f"No config for this checkpoint at {cfg_path}. Pass --config."
        )
    cfg = load_train_config(cfg_path)
    validate_train_config(cfg)
    data_cfg = load_data_cfg(cfg)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    valid_csv = Path(args.valid_csv or _official_valid_csv(data_cfg)).resolve()
    if not valid_csv.is_file():
        raise FileNotFoundError(f"Official validation CSV not found: {valid_csv}")

    print("=" * 72)
    print("OFFICIAL VALIDATION EXPERT CHECK - secondary, never tuned on (D012)")
    print("=" * 72)
    print(f"checkpoint : {ckpt_path}")
    print(f"valid_csv  : {valid_csv}")
    print(f"threshold  : {args.threshold!r}  (source: {args.threshold_source})")
    print(f"device     : {device}")

    # --- prepare the official set under the project's view + label policy ---
    frame, audit = build_official_frame(data_cfg, valid_csv)
    print(f"\nview filter: {audit['view_counts_in_csv']} -> kept "
          f"{audit['rows_kept_frontal']} frontal, excluded "
          f"{audit['rows_excluded_non_frontal']} non-frontal")
    print(f"uncertain labels in official CSV: {audit['uncertain_labels_in_csv']} "
          f"(expert-adjudicated set carries none)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.out_root
                    or (Path(cfg["experiment"]["output_root"]).parent / "eval"))
    out_dir = out_root / (f"official_valid__{cfg['model']['name']}"
                          f"__seed{cfg['experiment']['seed']}__{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Materialised INSIDE the evaluation directory. data/ is never written.
    eval_csv = out_dir / "official_valid_frontal.csv"
    frame.to_csv(eval_csv, index=False)
    print(f"prepared eval CSV: {eval_csv}")

    # --- pre-flight: every referenced image must exist ---------------------
    resolved = [resolve_image_path(data_cfg, p) for p in frame["Path"]]
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(resolved)} official frontal images are "
            f"absent. First: {missing[0]}"
        )
    print(f"path check : {len(resolved)}/{len(frame)} frontal images resolve and exist")

    # --- dataset via the SAME class and transform the frozen splits use ----
    eval_tf = build_eval_transform(data_cfg)
    ds = CardiomegalyDataset(
        eval_csv, data_cfg, eval_tf, split_name="official_valid",
        on_missing="error", exclusions_csv=None, verify_images=True,
    )
    stats = ds.stats()
    print(f"dataset    : {stats['images']} images  {stats['patients']} patients  "
          f"prevalence {stats['prevalence']}")

    # --- model -------------------------------------------------------------
    model = build_from_config(cfg)
    payload_ck = load_checkpoint(ckpt_path, model=model, map_location="cpu",
                                 strict=True)
    model = model.to(device).eval()
    ck_epoch = int(payload_ck["epoch"])
    print(f"model      : {cfg['model']['name']} restored from epoch {ck_epoch} "
          f"(best_epoch={payload_ck.get('best_epoch')}, "
          f"best_metric={payload_ck.get('best_metric')})")

    batch = int(args.batch_size or cfg["train"]["micro_batch"])
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                        pin_memory=device.startswith("cuda"), drop_last=False)

    # --- score at the INHERITED threshold ---------------------------------
    # threshold is passed, so engine.evaluate takes its inherited branch and no
    # selection happens on this set.
    result = evaluate(
        model, loader, device, threshold=float(args.threshold),
        amp=bool(cfg["train"]["amp"]) and device.startswith("cuda"),
        split_name="official_valid",
        calibration=bool(cfg["eval"].get("calibration", True)),
        calibration_bins=int(cfg["eval"].get("calibration_bins", 10)),
        fallback_test_caveat=False,   # this set IS expert ground truth
    )
    preds = result.pop("_predictions")
    if result["threshold_selection"]["policy"] != "inherited":
        raise RuntimeError(
            "threshold was SELECTED on the official set - refusing to report. "
            "This set must never be tuned on."
        )

    print()
    print(metrics_table(result))

    # --- per-image predictions --------------------------------------------
    order = np.asarray(preds["index"])
    pred_df = frame.iloc[order][["Path", "patient_id", "study_id", "AP/PA"]].copy()
    pred_df["y_true"] = preds["y_true"]
    pred_df["y_prob"] = preds["y_prob"]
    pred_df["y_pred"] = (preds["y_prob"] >= float(args.threshold)).astype(int)
    pred_path = out_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    # --- provenance --------------------------------------------------------
    manifest = {
        "kind": "official_validation_expert_check",
        "role": ("SECONDARY expert-labelled check (docs/dataset_analysis.md D012, "
                 "section 13.2). Never used for model selection or threshold "
                 "tuning."),
        "created_utc": utc_now_iso(),
        "out_dir": str(out_dir),

        "checkpoint": {
            "path": str(ckpt_path),
            "sha1": sha1_of_file(ckpt_path),
            "epoch": ck_epoch,
            "best_epoch": payload_ck.get("best_epoch"),
            "best_metric": payload_ck.get("best_metric"),
            "provenance": payload_ck.get("provenance"),
            "model_meta": payload_ck.get("model_meta"),
            "source_run_dir": str(run_dir),
        },
        "threshold": {
            "value": float(args.threshold),
            "source": args.threshold_source,
            "selected_on": ("carved CheXbert validation split "
                            "(artifacts/stage2/splits/val.csv)"),
            "applied_policy": result["threshold_selection"]["policy"],
            "note": ("Selected on the carved validation split at 13.65% "
                     "prevalence and applied UNCHANGED here at a different "
                     "prevalence. Not re-optimised on this set."),
        },
        "dataset": {
            "official_valid_csv": str(valid_csv),
            "official_valid_csv_sha1": sha1_of_file(valid_csv),
            "image_root": str(
                resolve_image_path(data_cfg, frame["Path"].iloc[0]).parents[2]
            ),
            "prepared_eval_csv": str(eval_csv),
            "prepared_eval_csv_sha1": sha1_of_file(eval_csv),
            "view_filter_audit": audit,
            "stats": stats,
            "label_source": ("official CheXpert validation annotations "
                             "(radiologist-adjudicated), NOT CheXbert"),
        },
        "comparison_note": {
            "carved_val": {
                "csv": str(Path(cfg["data"]["splits_dir"]) / cfg["data"]["val_csv"]),
                "label_source": "CheXbert (train_cheXbert.csv), policy u_zeros",
                "prevalence": 0.136521,
                "role": "model selection + threshold selection",
            },
            "official_valid": {
                "csv": str(valid_csv),
                "label_source": "radiologist-adjudicated",
                "prevalence": stats["prevalence"],
                "role": "secondary expert check only",
            },
            "warning": ("PR-AUC is not comparable across these two sets: the "
                        "no-skill baseline equals prevalence, which differs."),
        },
        "code": {
            "git_sha": git_sha(),
            "git_sha_short": git_sha(short=True),
            "git_dirty": git_dirty(),
            "entry_point": str(Path(__file__).resolve()),
        },
        "environment": environment_snapshot(),
        "config": {
            "train_config_source": cfg.get("_config_path"),
            "data_config": cfg["data"]["data_config"],
            "preprocessing": dict(data_cfg["preprocessing"]),
            "view": dict(data_cfg["view"]),
            "target": dict(data_cfg["target"]),
            "amp": bool(cfg["train"]["amp"]),
            "batch_size": batch,
            "device": device,
        },
        "model_summary": model_summary(model),
        "transform": describe_transform(eval_tf),
        "confidence_intervals": ("NOT COMPUTED - the Stage 3 metrics module "
                                 "provides no interval estimator, and none was "
                                 "invented for this run."),
        "test_split": "NOT EVALUATED - the fallback carved test split is untouched.",
    }

    write_json(out_dir / "eval_manifest.json", manifest)
    write_json(out_dir / "metrics.json", {
        "split": "official_valid",
        "threshold": float(args.threshold),
        "threshold_source": args.threshold_source,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ck_epoch,
        "metrics": result,
        "dataset_stats": stats,
        "view_filter_audit": audit,
    })

    print(f"\nwrote {out_dir / 'metrics.json'}")
    print(f"wrote {out_dir / 'eval_manifest.json'}")
    print(f"wrote {pred_path}")
    print(f"wrote {eval_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
