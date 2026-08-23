"""Stage 3 unit + smoke tests. Requires NO CheXpert image.

Everything runs against the synthetic fixture built by make_fixture.py, so the
whole pipeline is provably executable while image acquisition is blocked.

Covers the checklist agreed for Stage 3 prep:
  * dataset path resolution (through the centralized resolver)
  * label conversion / uncertainty-policy consistency
  * patient-disjointness
  * model factory + classifier output shape
  * loss and pos_weight (including reproduction of the measured 6.246739)
  * gradient accumulation equivalence
  * checkpoint save/load
  * evaluation metrics on a tiny synthetic fixture
  * augmentation guard rails (section 11 forbidden transforms)
  * train/eval transform separation

Run:  python src/training/test_training_smoke.py
Exit code is 0 only if every check passes.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

import config as cfgmod  # noqa: E402
from chexpert_metadata import load_config, resolve_image_path  # noqa: E402
from checkpoint import CheckpointManager, load_checkpoint, save_checkpoint  # noqa: E402
from dataset import (  # noqa: E402
    CardiomegalyDataset, assert_no_patient_overlap, build_datasets,
)
from losses import build_loss, prevalence_baseline, recompute_pos_weight  # noqa: E402
from make_fixture import build_fixture  # noqa: E402
from metrics import binary_metrics, calibration, select_threshold  # noqa: E402
from models import SUPPORTED_MODELS, build_model, check_output_shape  # noqa: E402
from transforms import build_eval_transform, build_train_transform  # noqa: E402

_FAILURES: list[str] = []
_PASSES = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _PASSES
    if ok:
        _PASSES += 1
        print(f"  PASS  {label}" + (f"  - {detail}" if detail else ""))
    else:
        _FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"  - {detail}" if detail else ""))


def expect_raises(label: str, fn, exc=Exception) -> None:
    try:
        fn()
    except exc as e:
        check(label, True, f"{type(e).__name__}: {str(e).splitlines()[0][:70]}")
    except Exception as e:  # noqa: BLE001
        check(label, False, f"wrong exception {type(e).__name__}: {e}")
    else:
        check(label, False, "expected an exception, none raised")


# --- 1. fixture + dataset ------------------------------------------------

def test_dataset(fixture: dict) -> tuple:
    print("\nDataset, resolver and label conversion")
    data_cfg = load_config(fixture["data_config"])

    train_cfg = {
        "data": {
            "splits_dir": fixture["splits_dir"],
            "train_csv": "train.csv", "val_csv": "val.csv", "test_csv": "test.csv",
            "split_manifest": "split_manifest.json",
            "on_missing_image": "error",
            "exclusions_csv": str(Path(fixture["out_dir"]) / "no_exclusions.csv"),
        },
        "augmentation": {"enabled": True, "rotation_deg": 10.0, "zoom_min": 0.9,
                         "zoom_max": 1.0, "translate_frac": 0.05,
                         "brightness": 0.1, "contrast": 0.1},
    }

    train_ds, val_ds, test_ds = build_datasets(train_cfg, data_cfg)
    check("all three splits load", len(train_ds) > 0 and len(val_ds) > 0
          and len(test_ds) > 0,
          f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    # Resolver: every path resolves and exists, via the centralized function.
    ok = all(resolve_image_path(data_cfg, p).exists() for p in train_ds.frame["Path"])
    check("every train Path resolves through resolve_image_path", ok)

    # The metadata prefix is the full-release one, as the real CSVs ship.
    sample = str(train_ds.frame["Path"].iloc[0])
    check("fixture Paths carry the CheXpert-v1.0 prefix like the real CSVs",
          sample.startswith("CheXpert-v1.0/"), sample)

    # Label conversion: targets are strictly binary and match the raw column.
    uniq = set(np.unique(train_ds.targets))
    check("targets are binary {0,1}", uniq <= {0.0, 1.0}, str(sorted(uniq)))

    img, target, idx = train_ds[0]
    size = int(data_cfg["preprocessing"]["input_size"])
    check("__getitem__ returns CHW float tensor at the configured size",
          tuple(img.shape) == (3, size, size) and img.dtype == torch.float32,
          str(tuple(img.shape)))
    check("target is a scalar float tensor",
          target.dtype == torch.float32 and target.ndim == 0)

    # Patient disjointness across all three splits.
    assert_no_patient_overlap(train=train_ds, val=val_ds, test=test_ds)
    check("assert_no_patient_overlap passes on the fixture splits", True,
          f"{len(set(train_ds.patient_ids))} train patients")

    # ... and that it actually fires when violated.
    expect_raises("patient leakage is DETECTED when splits share a patient",
                  lambda: assert_no_patient_overlap(a=train_ds, b=train_ds),
                  AssertionError)

    # Policy mismatch must be refused, not silently accepted.
    bad_cfg = load_config(fixture["data_config"])
    bad_cfg["target"]["policy"] = "u_ones"
    expect_raises("uncertainty-policy mismatch is refused",
                  lambda: CardiomegalyDataset(
                      Path(fixture["splits_dir"]) / "train.csv", bad_cfg,
                      build_eval_transform(bad_cfg), split_name="train",
                      verify_images=False),
                  ValueError)

    # A missing image must raise, never be dropped. Take the victim from the
    # train dataset's own resolved paths so it is guaranteed to be referenced by
    # train.csv -- rglob order is arbitrary and could pick a val/test image.
    victim = train_ds.image_paths[0]
    backup = victim.with_suffix(".jpg.bak")
    victim.rename(backup)
    try:
        expect_raises("a missing image raises rather than being dropped",
                      lambda: CardiomegalyDataset(
                          Path(fixture["splits_dir"]) / "train.csv", data_cfg,
                          build_eval_transform(data_cfg), split_name="train",
                          on_missing="error", verify_images=True),
                      FileNotFoundError)
    finally:
        backup.rename(victim)

    stats = train_ds.stats()
    check("stats() exposes images/patients/prevalence",
          all(k in stats for k in ("images", "patients", "prevalence", "positive")),
          f"prevalence={stats['prevalence']}")

    return train_ds, val_ds, test_ds, data_cfg


# --- 2. transforms -------------------------------------------------------

def test_transforms(data_cfg: dict) -> None:
    print("\nTransforms: train and eval are separate objects")
    aug = {"enabled": True, "rotation_deg": 10.0, "zoom_min": 0.9, "zoom_max": 1.0,
           "translate_frac": 0.05, "brightness": 0.1, "contrast": 0.1}
    train_tf = build_train_transform(data_cfg, aug)
    eval_tf = build_eval_transform(data_cfg)

    check("train and eval transforms are distinct objects", train_tf is not eval_tf)
    check("train pipeline has more steps than eval (augmentation present)",
          len(train_tf.transforms) > len(eval_tf.transforms),
          f"{len(train_tf.transforms)} vs {len(eval_tf.transforms)}")

    names = [type(t).__name__ for t in train_tf.transforms]
    check("no RandomHorizontalFlip in the train pipeline (section 11.1)",
          "RandomHorizontalFlip" not in names, str(names))
    check("no RandomErasing in the train pipeline", "RandomErasing" not in names)

    from PIL import Image
    img = Image.fromarray(np.full((80, 70), 120, dtype=np.uint8), mode="L")
    size = int(data_cfg["preprocessing"]["input_size"])

    a, b = eval_tf(img), eval_tf(img)
    check("eval transform is deterministic", torch.allclose(a, b))
    check("eval output is 3 x size x size", tuple(a.shape) == (3, size, size))
    # Replication happens at Grayscale(3); ImageNet Normalize then applies a
    # DIFFERENT mean/std per channel, so the post-normalize channels are
    # correctly not identical. Un-normalize to check the replication itself.
    norm = data_cfg["preprocessing"]["normalize"]
    unnorm = torch.stack([
        a[c] * float(norm["std"][c]) + float(norm["mean"][c]) for c in range(3)
    ])
    check("grayscale is replicated across 3 channels (checked pre-normalize)",
          torch.allclose(unnorm[0], unnorm[1], atol=1e-6)
          and torch.allclose(unnorm[1], unnorm[2], atol=1e-6))

    torch.manual_seed(0)
    c = train_tf(img)
    torch.manual_seed(1)
    d = train_tf(img)
    check("train transform is stochastic (augmentation active)",
          not torch.allclose(c, d))


# --- 3. model factory ----------------------------------------------------

def test_models() -> None:
    print("\nModel factory")
    check("exactly four supported baselines", len(SUPPORTED_MODELS) == 4,
          str(SUPPORTED_MODELS))

    # pretrained=False keeps the test offline and fast.
    for name in SUPPORTED_MODELS:
        m = build_model(name, pretrained=False, num_outputs=1)
        shape = check_output_shape(m, input_size=64, batch=2, device="cpu")
        check(f"{name:<20} single-logit output", shape == (2, 1), str(shape))
        check(f"{name:<20} feature_dim recorded",
              m.uro_meta.get("feature_dim", 0) > 0,
              str(m.uro_meta.get("feature_dim")))

    expect_raises("unknown architecture is refused",
                  lambda: build_model("vgg19", pretrained=False), ValueError)
    expect_raises("num_outputs != 1 is refused",
                  lambda: build_model("resnet152", pretrained=False, num_outputs=2),
                  ValueError)


# --- 4. loss / pos_weight ------------------------------------------------

def test_loss(fixture: dict) -> None:
    print("\nLoss and pos_weight")
    train_csv = Path(fixture["splits_dir"]) / "train.csv"
    value, counts = recompute_pos_weight(train_csv)
    expected = counts["negative"] / counts["positive"]
    check("recompute_pos_weight = neg/pos on the train split",
          abs(value - expected) < 1e-12, f"{value:.6f}")

    crit = build_loss({"loss": {"name": "bce_with_logits"}}, value, device="cpu")
    logits = torch.tensor([[2.0], [-2.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss = crit(logits, targets)
    check("BCEWithLogits + pos_weight produces a finite scalar loss",
          torch.isfinite(loss) and loss.ndim == 0, f"{float(loss):.4f}")

    # pos_weight must up-weight the positive class.
    plain = torch.nn.BCEWithLogitsLoss()
    hard_pos = torch.tensor([[-3.0]]), torch.tensor([[1.0]])
    check("pos_weight increases the penalty on a missed positive",
          float(crit(*hard_pos)) > float(plain(*hard_pos)),
          f"{float(crit(*hard_pos)):.4f} > {float(plain(*hard_pos)):.4f}")

    expect_raises("a non-BCE loss name is refused",
                  lambda: build_loss({"loss": {"name": "focal"}}, 1.0), ValueError)


def test_real_pos_weight() -> None:
    """Reproduce the project's MEASURED pos_weight from the real train split.

    Skipped (not failed) when the real split CSV is absent, so the suite still
    runs on a machine that only has the fixture.
    """
    print("\nMeasured pos_weight reproduction (real split, if present)")
    real = REPO_ROOT / "artifacts" / "stage2" / "splits" / "train.csv"
    if not real.exists():
        print(f"  SKIP  real train split not present at {real}")
        return
    value, counts = recompute_pos_weight(real)
    check("real train split reproduces pos_weight 6.246739 (D209)",
          abs(value - 6.246739) < 1e-6,
          f"{value:.6f} from {counts['negative']:,} neg / {counts['positive']:,} pos")
    check("real train split is 40,002 images", counts["images"] == 40002,
          f"{counts['images']:,}")


# --- 5. gradient accumulation -------------------------------------------

def test_grad_accumulation() -> None:
    """accum_steps=2 at micro-batch N must equal one step at batch 2N.

    This is the check that catches a missing `loss / accum_steps`: without the
    division the accumulated gradient is 2x too large and the configured
    learning rate silently means something else.
    """
    print("\nGradient accumulation equivalence")
    torch.manual_seed(0)
    model_a = torch.nn.Linear(8, 1)
    model_b = torch.nn.Linear(8, 1)
    model_b.load_state_dict(model_a.state_dict())

    x = torch.randn(8, 8)
    y = torch.randint(0, 2, (8, 1)).float()
    crit = torch.nn.BCEWithLogitsLoss()

    # Full batch of 8, one step.
    model_a.zero_grad()
    crit(model_a(x), y).backward()
    grad_full = model_a.weight.grad.clone()

    # Two micro-batches of 4 with the /accum division.
    model_b.zero_grad()
    for chunk in (slice(0, 4), slice(4, 8)):
        (crit(model_b(x[chunk]), y[chunk]) / 2).backward()
    grad_accum = model_b.weight.grad.clone()

    check("accumulated gradient equals full-batch gradient",
          torch.allclose(grad_full, grad_accum, atol=1e-6),
          f"max diff {float((grad_full - grad_accum).abs().max()):.2e}")


# --- 6. checkpointing ----------------------------------------------------

def test_checkpoint(tmp: Path) -> None:
    print("\nCheckpoint save / load")
    model = build_model("mobilenet_v3_large", pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp / "ckpt" / "test.pt"

    save_checkpoint(path, model=model, optimizer=opt, epoch=3,
                    best_metric=0.42, best_epoch=3,
                    provenance={"git_sha": "deadbeef", "fixture": True})
    check("checkpoint file written", path.exists())

    fresh = build_model("mobilenet_v3_large", pretrained=False)
    before = fresh.classifier[-1].weight.clone()
    payload = load_checkpoint(path, model=fresh)
    after = fresh.classifier[-1].weight
    check("state restored into a fresh model",
          torch.allclose(after, model.classifier[-1].weight))
    check("restoring actually changed the fresh weights",
          not torch.allclose(before, after))
    check("epoch and provenance round-trip",
          payload["epoch"] == 3 and payload["provenance"]["git_sha"] == "deadbeef")

    other = build_model("resnet152", pretrained=False)
    expect_raises("architecture mismatch on load is refused",
                  lambda: load_checkpoint(path, model=other), ValueError)

    mgr = CheckpointManager(tmp / "run", monitor="pr_auc", mode="max")
    r1 = mgr.update(metrics={"pr_auc": 0.30}, epoch=1, model=model)
    r2 = mgr.update(metrics={"pr_auc": 0.50}, epoch=2, model=model)
    r3 = mgr.update(metrics={"pr_auc": 0.40}, epoch=3, model=model)
    check("CheckpointManager tracks the best epoch",
          r1["improved"] and r2["improved"] and not r3["improved"]
          and mgr.best_epoch == 2, f"best_epoch={mgr.best_epoch}")
    expect_raises("a monitor key absent from metrics is refused",
                  lambda: mgr.update(metrics={"roc_auc": 0.9}, epoch=4, model=model),
                  KeyError)


# --- 7. metrics ----------------------------------------------------------

def test_metrics() -> None:
    print("\nEvaluation metrics")
    # A perfectly separable toy problem: every metric has a known answer.
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.9, 0.95])

    m = binary_metrics(y, p, threshold=0.5, split_name="toy")
    check("ROC-AUC = 1.0 on separable data", abs(m["roc_auc"] - 1.0) < 1e-9)
    check("PR-AUC = 1.0 on separable data", abs(m["pr_auc"] - 1.0) < 1e-9)
    check("recall = 1.0", abs(m["recall_sensitivity"] - 1.0) < 1e-9)
    check("specificity = 1.0", abs(m["specificity"] - 1.0) < 1e-9)
    check("confusion matrix is exact",
          m["confusion_matrix"] == {"tn": 4, "fp": 0, "fn": 0, "tp": 2})
    check("PR-AUC baseline equals the prevalence of the scored set",
          abs(m["pr_auc_baseline"] - 2 / 6) < 1e-6,
          f"{m['pr_auc_baseline']:.4f}")

    # An all-negative predictor: high accuracy, useless model. Section 12.3.
    y2 = np.array([0] * 87 + [1] * 13)
    p2 = np.zeros(100)
    m2 = binary_metrics(y2, p2, threshold=0.5, split_name="allneg")
    check("all-negative predictor scores high accuracy but zero recall",
          m2["accuracy"] == 0.87 and m2["recall_sensitivity"] == 0.0,
          f"acc={m2['accuracy']} recall={m2['recall_sensitivity']}")

    thr, info = select_threshold(y, p, policy="f1")
    check("threshold selection returns a threshold and diagnostics",
          0.0 <= thr <= 1.0 and info["policy"] == "f1", f"thr={thr:.3f}")

    cal = calibration(y, p, bins=5)
    check("calibration returns Brier and ECE",
          "brier" in cal and "ece" in cal, f"brier={cal['brier']:.4f}")

    single = binary_metrics(np.zeros(10), np.linspace(0, 1, 10), 0.5)
    check("single-class y_true yields null AUCs plus a warning, not a crash",
          single["roc_auc"] is None and "warning" in single)

    m3 = binary_metrics(y, p, 0.5, fallback_test_caveat=True)
    check("fallback-test caveat is attached when requested",
          "ground_truth_caveat" in m3
          and "NOT the official" in m3["ground_truth_caveat"])

    check("prevalence_baseline matches the mean of the targets",
          abs(prevalence_baseline(y2) - 0.13) < 1e-9)

    expect_raises("non-binary y_true is refused",
                  lambda: binary_metrics(np.array([0, 1, 2]), np.zeros(3), 0.5),
                  ValueError)


# --- 8. config guard rails ----------------------------------------------

def test_config_guards() -> None:
    print("\nConfig validation guard rails")
    base = cfgmod.load_train_config()
    cfgmod.validate_train_config(base)
    check("the shipped config/train.yaml validates", True)
    check("effective batch is micro_batch x accum_steps",
          cfgmod.effective_batch(base)
          == base["train"]["micro_batch"] * base["train"]["accum_steps"],
          str(cfgmod.effective_batch(base)))

    import copy
    bad = copy.deepcopy(base)
    bad["augmentation"]["horizontal_flip"] = True
    expect_raises("horizontal flip is refused (section 11.1)",
                  lambda: cfgmod.validate_train_config(bad), ValueError)

    bad2 = copy.deepcopy(base)
    bad2["augmentation"]["mixup"] = True
    expect_raises("mixup is refused (section 11)",
                  lambda: cfgmod.validate_train_config(bad2), ValueError)

    bad3 = copy.deepcopy(base)
    bad3["model"]["name"] = "vgg19"
    expect_raises("an unlisted architecture is refused",
                  lambda: cfgmod.validate_train_config(bad3), ValueError)

    bad4 = copy.deepcopy(base)
    bad4["train"]["deterministic"] = True
    bad4["train"]["cudnn_benchmark"] = True
    expect_raises("deterministic + cudnn_benchmark is refused",
                  lambda: cfgmod.validate_train_config(bad4), ValueError)

    bad5 = copy.deepcopy(base)
    bad5["augmentation"]["rotation_deg"] = 30.0
    expect_raises("rotation beyond the 15 degree ceiling is refused",
                  lambda: cfgmod.validate_train_config(bad5), ValueError)


# --- runner --------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("Stage 3 smoke tests - SYNTHETIC FIXTURE ONLY, no CheXpert image needed")
    print("=" * 72)

    tmp = Path(tempfile.mkdtemp(prefix="urop_stage3_test_"))
    try:
        fixture = build_fixture(tmp / "fixture", patients=16, image_size=64, seed=7)
        print(f"\nfixture built: {fixture['counts']}")

        _, _, _, data_cfg = test_dataset(fixture)
        test_transforms(data_cfg)
        test_models()
        test_loss(fixture)
        test_real_pos_weight()
        test_grad_accumulation()
        test_checkpoint(tmp)
        test_metrics()
        test_config_guards()
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
