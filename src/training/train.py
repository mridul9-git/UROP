"""Stage 3 training entry point.

    python src/training/train.py                      # real run (needs images)
    python src/training/train.py --model densenet201
    python src/training/train.py --dry-run            # fixture, no CheXpert needed

Order of operations is deliberate. Everything that can fail cheaply fails before
anything expensive happens: config validation, split loading, patient-disjointness,
pos_weight cross-check and a synthetic forward pass all run before the first real
image is read.

The test split is scored ONCE, at the end, using the threshold selected on
validation, and only when --eval-test is passed. It is not touched during
training and never contributes to model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

from checkpoint import CheckpointManager  # noqa: E402
from config import (  # noqa: E402
    effective_batch, load_data_config, load_train_config, snapshot_config,
    validate_train_config,
)
from dataset import build_datasets, load_split_manifest  # noqa: E402
from engine import build_optimizer, build_scheduler, evaluate, fit  # noqa: E402
from losses import build_loss, resolve_pos_weight  # noqa: E402
from metrics import metrics_table  # noqa: E402
from models import build_from_config, check_output_shape, model_summary  # noqa: E402
from reproducibility import (  # noqa: E402
    FIXTURE_BANNER, build_run_manifest, make_run_dir, seed_worker, set_seed,
    sha1_of_file, write_json,
)
from transforms import describe_transform  # noqa: E402


def _loader(ds, batch_size: int, shuffle: bool, cfg: dict, generator=None):
    d = cfg["data"]
    workers = int(d.get("num_workers", 0))
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=bool(d.get("pin_memory", False)) and torch.cuda.is_available(),
        persistent_workers=bool(d.get("persistent_workers", False)) and workers > 0,
        prefetch_factor=int(d.get("prefetch_factor", 2)) if workers > 0 else None,
        drop_last=False,             # never silently discard a partial batch
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=generator,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to train.yaml")
    ap.add_argument("--model", default=None, help="override model.name")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build a synthetic fixture and run the whole pipeline on it")
    ap.add_argument("--fixture-dir", default=None,
                    help="where to build the fixture (default: temp dir)")
    ap.add_argument("--eval-test", action="store_true",
                    help="score the test split ONCE at the end, at the "
                         "validation-selected threshold")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    cfg = load_train_config(args.config)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    fixture_info = None
    if args.dry_run:
        import tempfile

        from make_fixture import build_fixture

        print("=" * 72)
        print(FIXTURE_BANNER)
        print("=" * 72)
        out = args.fixture_dir or tempfile.mkdtemp(prefix="urop_fixture_")
        fixture_info = build_fixture(out, patients=24, image_size=64, seed=42)

        # Point the run at the fixture. Nothing under artifacts/stage2 is read.
        cfg["experiment"]["fixture"] = True
        cfg["experiment"]["name"] = "dryrun"
        cfg["data"]["data_config"] = fixture_info["data_config"]
        cfg["data"]["splits_dir"] = fixture_info["splits_dir"]
        cfg["data"]["exclusions_csv"] = str(Path(out) / "no_exclusions.csv")
        cfg["data"]["num_workers"] = 0      # keep the dry run single-process
        cfg["data"]["persistent_workers"] = False
        cfg["model"]["pretrained"] = False  # no weight download in a smoke test
        cfg["train"]["epochs"] = min(int(cfg["train"]["epochs"]), 2)
        cfg["train"]["micro_batch"] = 4
        cfg["train"]["accum_steps"] = 2
        cfg["loss"]["pos_weight_source"] = "recompute"
        print(f"[FIXTURE] synthetic data at {out}")
        print(f"[FIXTURE] counts {fixture_info['counts']}")

    validate_train_config(cfg)
    data_cfg = load_data_config(cfg)

    fixture = bool(cfg["experiment"]["fixture"])
    tag = "[FIXTURE] " if fixture else ""
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed, deterministic=bool(cfg["train"]["deterministic"]),
             cudnn_benchmark=bool(cfg["train"].get("cudnn_benchmark", False)))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{tag}device={device}  seed={seed}  "
          f"effective_batch={effective_batch(cfg)}")

    # --- data ------------------------------------------------------------
    manifest = load_split_manifest(cfg["data"]["splits_dir"],
                                   cfg["data"]["split_manifest"])
    train_ds, val_ds, test_ds = build_datasets(cfg, data_cfg)
    stats = {s.split_name: s.stats() for s in (train_ds, val_ds, test_ds)}
    for name, st in stats.items():
        print(f"{tag}{name:5s} {st['images']:>7,} images  "
              f"{st['patients']:>6,} patients  prevalence {st['prevalence']}")

    # --- loss ------------------------------------------------------------
    pos_weight, pw_prov = resolve_pos_weight(cfg, manifest)
    print(f"{tag}pos_weight={pos_weight:.6f} "
          f"(source={pw_prov['source']}, "
          f"verified_against_train_csv={pw_prov['agrees_with_train_csv']})")

    # --- model -----------------------------------------------------------
    model = build_from_config(cfg)
    input_size = int(data_cfg["preprocessing"]["input_size"])
    shape = check_output_shape(model, input_size=input_size, batch=2, device="cpu")
    if shape != (2, 1):
        raise RuntimeError(
            f"model head produced {shape}, expected (2, 1). Binary Cardiomegaly "
            f"is a single logit."
        )
    model = model.to(device)
    print(f"{tag}model={cfg['model']['name']} "
          f"params={model_summary(model)['total_params_millions']}M "
          f"out_shape={shape}")

    # --- run directory + manifest ---------------------------------------
    run_dir = make_run_dir(cfg["experiment"]["output_root"], cfg["experiment"]["name"],
                           cfg["model"]["name"], seed, fixture=fixture)
    split_files = {
        name: {
            "path": str(st["csv"]),
            "sha1": sha1_of_file(st["csv"]),
            "images": st["images"], "patients": st["patients"],
            "prevalence": st["prevalence"],
        } for name, st in stats.items()
    }
    split_files["split_manifest_mode"] = manifest.get("mode")
    split_files["split_manifest_seed"] = manifest.get("seed")

    run_manifest = build_run_manifest(
        cfg, data_cfg, run_dir=run_dir, pos_weight=pos_weight,
        pos_weight_provenance=pw_prov, split_files=split_files,
        dataset_stats=stats, fixture=fixture,
    )
    run_manifest["model_summary"] = model_summary(model)
    run_manifest["transforms"] = {
        "train": describe_transform(train_ds.transform),
        "eval": describe_transform(val_ds.transform),
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    snapshot_config(cfg, run_dir / "config_snapshot.yaml")
    print(f"{tag}run dir: {run_dir}")

    # --- loaders / optimizer --------------------------------------------
    micro = int(cfg["train"]["micro_batch"])
    gen = torch.Generator().manual_seed(seed)
    train_loader = _loader(train_ds, micro, True, cfg, generator=gen)
    val_loader = _loader(val_ds, micro, False, cfg)

    criterion = build_loss(cfg, pos_weight, device=device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler(
        device="cuda" if device.startswith("cuda") else "cpu",
        enabled=bool(cfg["train"]["amp"]) and device.startswith("cuda"),
    )
    ckpt = CheckpointManager(
        run_dir, monitor=cfg["checkpoint"]["monitor"], mode=cfg["checkpoint"]["mode"],
        save_best=bool(cfg["checkpoint"].get("save_best", True)),
        save_last=bool(cfg["checkpoint"].get("save_last", True)),
    )

    provenance = {
        "git_sha": run_manifest["code"]["git_sha"],
        "seed": seed, "model": cfg["model"]["name"],
        "pos_weight": pos_weight, "fixture": fixture,
    }

    # --- train -----------------------------------------------------------
    result = fit(model, train_loader=train_loader, val_loader=val_loader,
                 criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                 scaler=scaler, device=device, cfg=cfg, ckpt_manager=ckpt,
                 provenance=provenance, run_dir=run_dir, fixture=fixture)

    payload = {"run_manifest": str(run_dir / "run_manifest.json"), **result}

    # --- test, once, at the validation threshold -------------------------
    if args.eval_test:
        best_epoch_val = next(
            (h["val"] for h in result["history"] if h["epoch"] == result["best_epoch"]),
            None,
        )
        thr = (best_epoch_val or {}).get("threshold")
        if thr is None:
            raise RuntimeError("no validation threshold available for test scoring")
        print(f"{tag}scoring TEST once at the validation-selected threshold {thr:.4f}")
        test_loader = _loader(test_ds, micro, False, cfg)
        test_metrics = evaluate(
            model, test_loader, device, threshold=thr, amp=bool(cfg["train"]["amp"]),
            split_name="test",
            calibration=bool(cfg["eval"].get("calibration", True)),
            calibration_bins=int(cfg["eval"].get("calibration_bins", 10)),
            fallback_test_caveat=(manifest.get("mode") == "fallback-carved-test"),
        )
        test_metrics.pop("_predictions", None)
        print(metrics_table(test_metrics))
        if "ground_truth_caveat" in test_metrics:
            print(f"{tag}CAVEAT: {test_metrics['ground_truth_caveat']}")
        payload["test"] = test_metrics

    write_json(run_dir / "metrics.json", payload)
    print(f"{tag}wrote {run_dir / 'metrics.json'}")

    if fixture:
        print("=" * 72)
        print(FIXTURE_BANNER)
        print("Metrics above are computed on synthetic noise and mean nothing.")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
