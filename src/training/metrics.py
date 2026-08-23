"""Stage 3 evaluation metrics.

Section 12.3 sets the reporting rule this module implements:

    ROC-AUC and PR-AUC together, with the PR-AUC baseline stated, plus recall,
    specificity, precision and F1 at the pre-declared validation-selected
    threshold, plus the confusion matrix.

Two properties are deliberate:

1. `pr_auc_baseline` is computed from the set being scored, never hard-coded.
   PR-AUC is prevalence-dependent; quoting it without its no-skill baseline is
   meaningless, and a stale constant would misstate it.
2. Threshold selection is a separate function from metric computation, so a
   threshold chosen on validation can be applied unchanged to test. Selecting a
   threshold on test is tuning on test.

This module computes numbers. It makes no claim about what they mean. In
particular the current test arm is a CheXbert-labelled fallback carve, not an
expert-adjudicated set - see `GROUND_TRUTH_CAVEAT`.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Attached to every metrics payload for the fallback split so no downstream
# report can quietly present these numbers as clinical ground truth.
GROUND_TRUTH_CAVEAT = (
    "Labels are CheXbert-derived from radiology reports (D211). The current "
    "test arm is a fallback patient-level carve from the training pool "
    "(mode=fallback-carved-test), NOT the official 5-radiologist consensus "
    "test set, which has not been acquired. Performance is agreement with a "
    "report-derived label, not clinical diagnostic accuracy."
)


def _as_arrays(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.shape != ys.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_score {ys.shape}")
    if yt.size == 0:
        raise ValueError("empty arrays - nothing to score")
    bad = set(np.unique(yt)) - {0.0, 1.0}
    if bad:
        raise ValueError(f"y_true must be binary 0/1; found {sorted(bad)}")
    return yt, ys


def select_threshold(y_true, y_score, policy: str = "f1",
                     fixed: float = 0.5) -> tuple[float, dict]:
    """Choose an operating threshold. Call this on VALIDATION ONLY.

    Returns (threshold, diagnostics). The diagnostics say which policy was used
    and on how many samples, so the run manifest can record that the threshold
    was not derived from test.
    """
    yt, ys = _as_arrays(y_true, y_score)
    info = {"policy": policy, "n": int(yt.size), "positives": int(yt.sum())}

    if policy == "fixed":
        thr = float(fixed)
    elif policy == "prevalence":
        # Threshold at the base rate - a defensible prior-matched operating point.
        thr = float(yt.mean())
    elif policy == "youden":
        fpr, tpr, thresholds = roc_curve(yt, ys)
        j = tpr - fpr
        thr = float(thresholds[int(np.argmax(j))])
        info["youden_j"] = float(np.max(j))
    elif policy == "f1":
        # Sweep the observed score values; picking the best F1 on validation.
        candidates = np.unique(ys)
        if candidates.size > 2000:
            candidates = np.quantile(ys, np.linspace(0, 1, 2000))
        best_f1, thr = -1.0, 0.5
        for c in candidates:
            pred = (ys >= c).astype(int)
            tp = float(((pred == 1) & (yt == 1)).sum())
            fp = float(((pred == 1) & (yt == 0)).sum())
            fn = float(((pred == 0) & (yt == 1)).sum())
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1, thr = f1, float(c)
        info["best_f1_on_selection_set"] = best_f1
    else:
        raise ValueError(
            f"threshold_policy={policy!r} must be f1 | youden | prevalence | fixed"
        )

    info["threshold"] = float(thr)
    return float(thr), info


def binary_metrics(y_true, y_score, threshold: float,
                   *, calibration_bins: int = 10,
                   include_calibration: bool = True,
                   split_name: str = "",
                   fallback_test_caveat: bool = False) -> dict:
    """Full metric payload at a given threshold.

    `threshold` must have been selected on validation. This function does not
    choose one - passing it in is what keeps test untuned.
    """
    yt, ys = _as_arrays(y_true, y_score)
    pred = (ys >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    tp, tn, fp, fn = int(tp), int(tn), int(fp), int(fn)

    recall = tp / (tp + fn) if (tp + fn) else float("nan")        # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")     # PPV
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and not np.isnan(precision + recall) else float("nan"))
    accuracy = (tp + tn) / yt.size

    prevalence = float(yt.mean())
    single_class = len(np.unique(yt)) < 2

    out = {
        "split": split_name,
        "n": int(yt.size),
        "threshold": float(threshold),
        "prevalence": round(prevalence, 6),

        # Discrimination. Both, always - section 12.3.
        "roc_auc": float(roc_auc_score(yt, ys)) if not single_class else None,
        "pr_auc": float(average_precision_score(yt, ys)) if not single_class else None,
        # The no-skill line for PR-AUC, computed here rather than quoted.
        "pr_auc_baseline": round(prevalence, 6),

        # Threshold-dependent.
        "recall_sensitivity": _r(recall),
        "specificity": _r(specificity),
        "precision_ppv": _r(precision),
        "npv": _r(npv),
        "f1": _r(f1),
        # Reported only alongside the rest: a constant-negative model scores
        # ~86% here, so accuracy alone is close to meaningless (section 12.3).
        "accuracy": _r(accuracy),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }

    if single_class:
        out["warning"] = (
            "y_true contains a single class - ROC-AUC and PR-AUC are undefined "
            "and are reported as null rather than as a number."
        )

    if include_calibration:
        out["calibration"] = calibration(yt, ys, bins=calibration_bins)

    if fallback_test_caveat:
        out["ground_truth_caveat"] = GROUND_TRUTH_CAVEAT

    return out


def _r(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def calibration(y_true, y_prob, bins: int = 10) -> dict:
    """Brier score plus a reliability table.

    Scores must already be probabilities (post-sigmoid). Included because a
    model can discriminate well and still be badly calibrated, which matters
    the moment a threshold is chosen.
    """
    yt, yp = _as_arrays(y_true, y_prob)
    if yp.min() < 0 or yp.max() > 1:
        return {
            "error": "scores are not in [0,1]; apply sigmoid before calibration",
            "min": float(yp.min()), "max": float(yp.max()),
        }

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    idx = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, bins - 1)
    table = []
    ece = 0.0
    for b in range(int(bins)):
        m = idx == b
        count = int(m.sum())
        if not count:
            continue
        conf = float(yp[m].mean())
        acc = float(yt[m].mean())
        ece += (count / yt.size) * abs(acc - conf)
        table.append({
            "bin": b,
            "range": [round(float(edges[b]), 3), round(float(edges[b + 1]), 3)],
            "count": count,
            "mean_predicted": round(conf, 6),
            "observed_frequency": round(acc, 6),
        })

    return {
        "brier": round(float(brier_score_loss(yt, yp)), 6),
        "ece": round(float(ece), 6),
        "bins": table,
    }


def metrics_table(payload: dict) -> str:
    """Console rendering that always shows PR-AUC next to its baseline."""
    lines = [
        f"  n={payload['n']:,}  prevalence={payload['prevalence']:.4f}  "
        f"threshold={payload['threshold']:.4f}",
        f"  ROC-AUC : {_fmt(payload['roc_auc'])}",
        f"  PR-AUC  : {_fmt(payload['pr_auc'])}   "
        f"(no-skill baseline {payload['pr_auc_baseline']:.4f})",
        f"  recall/sens {_fmt(payload['recall_sensitivity'])}   "
        f"specificity {_fmt(payload['specificity'])}",
        f"  precision   {_fmt(payload['precision_ppv'])}   "
        f"F1          {_fmt(payload['f1'])}",
        f"  accuracy    {_fmt(payload['accuracy'])}  "
        f"(not meaningful alone at this prevalence)",
    ]
    cm = payload["confusion_matrix"]
    lines.append(f"  confusion  tn={cm['tn']:,} fp={cm['fp']:,} "
                 f"fn={cm['fn']:,} tp={cm['tp']:,}")
    if "calibration" in payload and "brier" in payload["calibration"]:
        lines.append(f"  Brier {payload['calibration']['brier']:.4f}   "
                     f"ECE {payload['calibration']['ece']:.4f}")
    return "\n".join(lines)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"
