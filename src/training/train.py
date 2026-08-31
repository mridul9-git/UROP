"""Stage 3 training entry point.

    python src/training/train.py                      # real run (needs images)
    python src/training/train.py --model densenet201
    python src/training/train.py --dry-run            # fixture, no CheXpert needed
    python src/training/train.py --resume <run>/checkpoints/best.pt

Resume is explicit and never automatic. `--resume` continues an existing run
IN PLACE: it reuses that run's directory and its config_snapshot.yaml (so the
methodology cannot drift), restores model/optimizer/scheduler/scaler and the
best-metric bookkeeping, and starts at checkpoint_epoch + 1 against the SAME
total epoch budget. The original run_manifest.json and config_snapshot.yaml are
never rewritten; a resume writes its own record under <run>/resumes/.

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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "training"))
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))

from checkpoint import CheckpointManager, load_checkpoint  # noqa: E402
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


def _resume_run_dir(ckpt_path: Path) -> Path:
    """The run directory a checkpoint belongs to: <run>/checkpoints/<file>.

    Validated rather than assumed, because resuming into the wrong directory
    would append this run's epochs to another run's provenance.
    """
    if ckpt_path.suffix == ".tmp" or ckpt_path.name.endswith(".pt.tmp"):
        raise ValueError(
            f"Refusing to resume from {ckpt_path.name}: a .tmp file is a "
            f"partially written checkpoint, not a valid one."
        )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.parent.name != "checkpoints":
        raise ValueError(
            f"Expected the checkpoint to live in <run_dir>/checkpoints/, "
            f"got {ckpt_path.parent}"
        )
    run_dir = ckpt_path.parent.parent
    if not (run_dir / "run_manifest.json").exists():
        raise FileNotFoundError(
            f"{run_dir} has no run_manifest.json, so it is not a run directory "
            f"this resume can continue."
        )
    return run_dir


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
    ap.add_argument("--resume", default=None, metavar="CHECKPOINT",
                    help="continue an interrupted run from this checkpoint. "
                         "Never automatic: without this flag a run always "
                         "starts from epoch 1 in a fresh run directory.")
    args = ap.parse_args()

    resume_path = Path(args.resume).resolve() if args.resume else None
    resume_run_dir = _resume_run_dir(resume_path) if resume_path else None

    if resume_run_dir is not None:
        if args.dry_run:
            raise ValueError("--resume and --dry-run are mutually exclusive")
        # Hyperparameter overrides are refused on resume rather than merged:
        # changing epochs would change the cosine T_max, changing seed or model
        # would make the restored optimizer/scheduler state meaningless.
        blocked = [n for n, v in (("--model", args.model),
                                  ("--epochs", args.epochs),
                                  ("--seed", args.seed)) if v is not None]
        if blocked:
            raise ValueError(
                f"{', '.join(blocked)} cannot be combined with --resume: the "
                f"resumed run must use the original run's settings verbatim."
            )
        # Read the ORIGINAL run's snapshot, not the current config/train.yaml,
        # so a later edit to config/train.yaml cannot silently alter this run.
        snap = resume_run_dir / "config_snapshot.yaml"
        if args.config is None and snap.exists():
            args.config = str(snap)
            print(f"[RESUME] config from the original run: {snap}")

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
    # A resume continues the ORIGINAL directory. make_run_dir() is skipped so
    # the resumed epochs land beside the epochs they continue.
    run_dir = resume_run_dir or make_run_dir(
        cfg["experiment"]["output_root"], cfg["experiment"]["name"],
        cfg["model"]["name"], seed, fixture=fixture,
    )
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
    if resume_run_dir is None:
        write_json(run_dir / "run_manifest.json", run_manifest)
        snapshot_config(cfg, run_dir / "config_snapshot.yaml")
    else:
        # Requirement: the original run's provenance is immutable. The resume
        # gets its own dated pair alongside it, never an overwrite.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resume_dir = run_dir / "resumes"
        resume_dir.mkdir(parents=True, exist_ok=True)
        write_json(resume_dir / f"run_manifest__{stamp}.json", run_manifest)
        snapshot_config(cfg, resume_dir / f"config_snapshot__{stamp}.yaml")
        print(f"[RESUME] resume provenance -> {resume_dir} (stamp {stamp}); "
              f"original run_manifest.json / config_snapshot.yaml untouched")
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

    # --- resume ----------------------------------------------------------
    start_epoch, start_bad_epochs, resume_record, resume_stamp = 1, 0, None, None
    if resume_path is not None:
        ck_payload = load_checkpoint(
            resume_path, model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, restore_rng=True, map_location="cpu", strict=True,
        )

        # The checkpoint must belong to the run we are about to continue. A
        # mismatch here means the restored optimizer state describes a different
        # experiment, which would corrupt the result silently.
        ck_prov = ck_payload.get("provenance") or {}
        for key, live in (("seed", seed), ("model", cfg["model"]["name"])):
            if ck_prov.get(key) is not None and ck_prov[key] != live:
                raise ValueError(
                    f"Checkpoint provenance {key}={ck_prov[key]!r} does not "
                    f"match this run's {key}={live!r}."
                )
        ck_pw = ck_prov.get("pos_weight")
        if ck_pw is not None and abs(float(ck_pw) - float(pos_weight)) > 1e-6:
            raise ValueError(
                f"Checkpoint pos_weight={ck_pw} != resolved pos_weight="
                f"{pos_weight}. The loss would not be the loss the restored "
                f"optimizer state was produced under."
            )

        ckpt_epoch = int(ck_payload["epoch"])
        best_metric = ck_payload.get("best_metric")
        best_epoch = ck_payload.get("best_epoch")
        ckpt.load_state(best_metric, best_epoch)

        start_epoch = ckpt_epoch + 1
        # Patience already consumed: epochs completed since the best one. For a
        # checkpoint whose epoch IS the best epoch this is 0, as it must be.
        start_bad_epochs = (
            max(0, ckpt_epoch - int(best_epoch)) if best_epoch is not None else 0
        )

        # best.pt becomes writable again by CheckpointManager from here on. Keep
        # an immutable copy of the state we resumed from; nothing is deleted.
        resume_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = (
            resume_path.parent
            / f"pre_resume__{resume_path.stem}__epoch{ckpt_epoch}__{resume_stamp}.pt"
        )
        if not backup.exists():
            shutil.copy2(resume_path, backup)

        resume_record = {
            "resumed": True,
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "resumed_from_checkpoint": str(resume_path),
            "resumed_from_checkpoint_bytes": resume_path.stat().st_size,
            "preserved_copy": str(backup),
            "checkpoint_epoch": ckpt_epoch,
            "start_epoch": start_epoch,
            "total_epochs": int(cfg["train"]["epochs"]),
            "remaining_epochs": int(cfg["train"]["epochs"]) - start_epoch + 1,
            "restored_best_metric": best_metric,
            "restored_best_epoch": best_epoch,
            "restored_early_stopping_bad_epochs": start_bad_epochs,
            "restored": {
                "model": True,
                "optimizer": bool(ck_payload.get("optimizer_state")),
                "scheduler": bool(ck_payload.get("scheduler_state")),
                "scaler": bool(ck_payload.get("scaler_state")),
                "rng": ck_payload.get("rng_restored") or [],
            },
            "rng_state_present_in_checkpoint": bool(
                ck_payload.get("rng_state_present")
            ),
            "checkpoint_provenance": ck_prov,
            "config_source": cfg.get("_config_path"),
            "prior_epoch_history_recoverable": False,
        }
        if not resume_record["rng_state_present_in_checkpoint"]:
            resume_record["rng_caveat"] = (
                "This checkpoint predates RNG-state capture. Data order and "
                "augmentation for the resumed epochs come from set_seed(seed) "
                "rather than from the interrupted run's RNG stream, so the "
                "resumed epochs are NOT bit-identical to an uninterrupted run. "
                "Model/optimizer/scheduler/scaler state is restored exactly."
            )
        write_json(run_dir / "resumes" / f"resume__{resume_stamp}.json",
                   resume_record)

        es_patience = (cfg["train"].get("early_stopping") or {}).get("patience")
        restored_names = [k for k, v in resume_record["restored"].items() if v]
        print(f"[RESUME] from {resume_path}")
        print(f"[RESUME] restored epoch={ckpt_epoch} "
              f"best_{ckpt.monitor}={best_metric} best_epoch={best_epoch}")
        print(f"[RESUME] restored components={restored_names}")
        print(f"[RESUME] rng restored={resume_record['restored']['rng'] or 'NONE'} "
              f"(present in checkpoint: "
              f"{resume_record['rng_state_present_in_checkpoint']})")
        print(f"[RESUME] preserved copy -> {backup.name}")
        print(f"[RESUME] starting at epoch {start_epoch} of "
              f"{cfg['train']['epochs']} "
              f"({resume_record['remaining_epochs']} remaining), "
              f"early-stopping bad_epochs={start_bad_epochs}/{es_patience}")

    # --- train -----------------------------------------------------------
    result = fit(model, train_loader=train_loader, val_loader=val_loader,
                 criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                 scaler=scaler, device=device, cfg=cfg, ckpt_manager=ckpt,
                 provenance=provenance, run_dir=run_dir, fixture=fixture,
                 start_epoch=start_epoch, start_bad_epochs=start_bad_epochs)

    payload = {"run_manifest": str(run_dir / "run_manifest.json"), **result}
    if resume_record is not None:
        payload["resume"] = resume_record

    # --- test, once, at the validation threshold -------------------------
    if args.eval_test:
        best_epoch_val = next(
            (h["val"] for h in result["history"] if h["epoch"] == result["best_epoch"]),
            None,
        )
        thr = (best_epoch_val or {}).get("threshold")
        if thr is None:
            raise RuntimeError(
                f"no validation threshold available for test scoring: best "
                f"epoch is {result['best_epoch']}, which is not among the "
                f"epochs this process ran ({start_epoch}..). On a resumed run "
                f"whose best epoch predates the resume the threshold was not "
                f"carried in the checkpoint - re-evaluate validation rather "
                f"than substituting a different threshold."
            )
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

    # A resume never overwrites a completed run's metrics.json.
    metrics_path = run_dir / "metrics.json"
    if resume_record is not None and metrics_path.exists():
        metrics_path = (
            run_dir / "resumes"
            / f"metrics__from_epoch{start_epoch}__{resume_stamp}.json"
        )
    write_json(metrics_path, payload)
    print(f"{tag}wrote {metrics_path}")

    if fixture:
        print("=" * 72)
        print(FIXTURE_BANNER)
        print("Metrics above are computed on synthetic noise and mean nothing.")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
