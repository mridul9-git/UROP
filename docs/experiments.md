# Experiments — canonical ledger

**Last updated:** 31/08/2026 · **Stage:** 3 (baseline training)

This is the **experiment record**. One entry per training run that produced a checkpoint
and metrics. It is the only place experiment *results* are canonical; methodology
justification lives in `dataset_analysis.md`, scientific reasoning in `research.md`,
chronological engineering history in `notes.md`.

Certainty labels used throughout: **CONFIRMED** = measured and written to an artifact on
disk · **INFERRED** = follows from confirmed measurements · **VERIFY** = plausible but not
yet checked · **UNKNOWN** = not determined · **SUPERSEDED** = was true, no longer is.

## Rules for this ledger

1. **No experiment is recorded before it has run.** Placeholders below are explicitly
   marked and carry no numbers.
2. **Every number is transcribed from an artifact**, with the artifact path given. Nothing
   here is retyped from memory or estimated.
3. **The carved test split is never scored** except once, at the very end of the project,
   and only after the model set is frozen. Any entry whose *Test status* is not
   `NOT EVALUATED` must state the date and reason.
4. **A superseded result is struck through and kept**, never deleted.

---

## Entry format

```text
### E<NNN> — <short name>
Date · Objective · Model · Dataset/splits · Target & label policy · Configuration ·
Seed · Training outcome · Best checkpoint · Validation metrics ·
Official expert-validation metrics · Test status · Artifacts · Interpretation ·
Limitations · Status
```

---

## E001 — ResNet152 baseline

| Field | Value |
|---|---|
| **Experiment ID** | `E001` |
| **Date** | started 30/08/2026, completed 31/08/2026 |
| **Objective** | First of the four RQ1 baselines: establish ResNet152 performance for binary Cardiomegaly under the frozen protocol, and prove the Stage 3 training pipeline end-to-end on real data |
| **Model** | ResNet152, ImageNet-pretrained (`ResNet152_Weights.IMAGENET1K_V1`), single-logit head, dropout 0.0 — **58,145,857** trainable parameters |
| **Experiment name** | `baseline` |

### Dataset / splits

| | |
|---|---|
| Source metadata | `train_cheXbert.csv` (D211 — CheXbert relabelling, **not** the original rule-based labeller) |
| Split mode | `fallback-carved-test` — 70/15/15, patient-level, stratified on patient-level any-positive, seed 42 |
| Train | **40,002** images · 13,659 patients · prevalence 0.13799 |
| Validation | **29,380** images · 9,680 patients · prevalence 0.13652 |
| Test | **28,883** images · 9,682 patients · prevalence 0.13755 — **not evaluated** |
| Patient disjointness | **CONFIRMED** — `assert_no_patient_overlap` passed at run start |
| Views | Frontal only; AP/PA combined, retained as metadata |
| Exclusions applied | 0 rows excluded from each split via `excluded_images.csv` |

### Target and label policy

| | |
|---|---|
| Target | `Cardiomegaly`, binarised to a single logit |
| Uncertainty policy | `u_zeros` (`-1.0 → 0`) |
| `blank_is_negative` | `true` |
| Loss | `BCEWithLogitsLoss` with `pos_weight` = **6.246739**, computed from **train only**, verified against the train CSV at run start (**CONFIRMED**) |

### Configuration

| Parameter | Value |
|---|---|
| Input size | 320 × 320, direct resize, grayscale replicated to 3 channels, ImageNet normalisation |
| Micro-batch | 16 |
| Gradient accumulation | 2 |
| **Effective batch** | **32** |
| Optimiser | AdamW, lr `1e-4`, weight decay `1e-4`, betas `(0.9, 0.999)`, eps `1e-8` |
| Scheduler | Cosine annealing, `T_max = 14`, `min_lr = 1e-6`, linear warmup 1 epoch |
| AMP | enabled (`torch.amp`, CUDA) |
| Gradient clipping | disabled (`grad_clip_norm = 0.0`) |
| Determinism | `deterministic: true`, `cudnn_benchmark: false` |
| Epoch budget | **15** |
| Early stopping | on validation **PR-AUC**, `mode: max`, patience **5**, `min_delta 0.0` |
| Checkpoint monitor | `pr_auc` (max) |
| Augmentation | rotation ±10°, zoom 0.9–1.0, translation ±5%, brightness ±10%, contrast ±10%. **No** horizontal/vertical flip, shear, random erasing, MixUp, CutMix or CLAHE (D208, §11) |
| **Seed** | **42** |
| Hardware | NVIDIA RTX 4060 Laptop 8 GB, CUDA, torch 2.13.0+cu126 |
| Code version | git `6289e3b48391c9959de46faa6ebc0da098fc4b38` |

### Training outcome — **CONFIRMED**

The run was **interrupted after epoch 1** and **resumed from the epoch-1 checkpoint**; it
then ran epochs 2–8 and **early-stopped at epoch 8**. Process **exit code 0**.

| | |
|---|---|
| Epochs completed | 1 (original) + 7 (resumed, epochs 2–8) = **8 of the 15-epoch budget** |
| Stop reason | early stopping — validation PR-AUC did not improve for 5 consecutive epochs after epoch 3 |
| **Best epoch** | **3** |
| **Best validation PR-AUC** | **0.5417253378785041** |
| Resume provenance | restored model, optimizer, scheduler and AMP scaler state from the epoch-1 checkpoint; epoch counter and best-metric bookkeeping restored; started at epoch 2 against the unchanged 15-epoch budget |

Per-epoch validation (transcribed from `metrics.json`):

| Epoch | Train loss | PR-AUC | ROC-AUC | Threshold | F1 | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | *not recorded* | 0.5260268660904089 | — | — | — | yes |
| 2 | 0.8648 | 0.5386 | 0.8603 | 0.7649 | 0.5254 | yes |
| **3** | **0.7998** | **0.5417** | **0.8647** | **0.4421** | **0.5366** | **yes** |
| 4 | 0.7737 | 0.5301 | 0.8576 | 0.6086 | 0.5202 | no |
| 5 | 0.7485 | 0.5241 | 0.8546 | 0.3901 | 0.5150 | no |
| 6 | 0.7171 | 0.5341 | 0.8591 | 0.5111 | 0.5274 | no |
| 7 | 0.6805 | 0.5251 | 0.8566 | 0.6770 | 0.5264 | no |
| 8 | 0.6311 | 0.5407 | 0.8603 | 0.5206 | 0.5266 | no |

> **Epoch 1's full metric block is not recoverable.** The interruption preserved only the
> monitored value (`best_metric = 0.5260268660904089`) inside the checkpoint; the epoch-1
> `val` block — threshold, ROC-AUC, calibration — was never written to `metrics.json`
> before the process died. **UNKNOWN**, and it will stay unknown. See `notes.md`.

### Best checkpoint

| | |
|---|---|
| Path | `artifacts/stage3/runs/baseline__resnet152__seed42__20260830T185505Z/checkpoints/best.pt` |
| Epoch | **3** (**CONFIRMED**: checkpoint `epoch == 3`, `best_epoch == 3`, scheduler `last_epoch == 3`) |
| `best_metric` inside checkpoint | `0.5417253378785041` — identical to `metrics.json` |
| Also present | `last.pt` (epoch 8) and `pre_resume__best__epoch1__20260831T102547Z.pt` (the preserved epoch-1 state) |

### Validation metrics — carved CheXbert validation split, best epoch (3)

Selection surface. Threshold **selected here** by F1 policy.

| Metric | Value |
|---|---:|
| n | 29,380 |
| Prevalence | 0.136521 |
| **Threshold (selected)** | **0.44209548389690995** |
| ROC-AUC | 0.8647310646245118 |
| **PR-AUC** | **0.5417253378785041** (no-skill baseline 0.136521) |
| Sensitivity / recall | 0.604587 |
| Specificity | 0.897434 |
| PPV (precision) | 0.482395 |
| NPV | 0.934875 |
| F1 | 0.536623 |
| Accuracy | 0.857454 |
| Confusion | tn 22,767 · fp 2,602 · fn 1,586 · tp 2,425 |
| Brier | 0.096055 |
| ECE | 0.081674 |

### Official expert-validation metrics — secondary check, never tuned on

Threshold **inherited unchanged** from the carved validation split. No selection or tuning
occurred on this set (**CONFIRMED**: the evaluator records
`threshold_selection.policy == "inherited"` and refuses to report otherwise).

| | |
|---|---|
| Set | official CheXpert 200-patient validation set |
| Total images in CSV | 234 (200 patients) |
| **Frontal evaluated** | **202** |
| **Lateral excluded** | **32** (view policy — laterals cannot support a cardiothoracic-ratio judgement) |
| Patients | 200 |
| Positives / negatives | **66 / 136** |
| Prevalence | **0.32673** |
| Label source | radiologist-adjudicated (**not** CheXbert) |
| Uncertain / blank labels | 0 / 0 |

| Metric | Value |
|---|---:|
| ROC-AUC | **0.8577** |
| PR-AUC | **0.7547** (no-skill baseline 0.3267) |
| Sensitivity / recall | **0.3030** |
| Specificity | **0.9706** |
| PPV | **0.8333** |
| NPV | **0.7416** |
| F1 | **0.4444** |
| Accuracy | **0.7525** |
| Confusion | **tn 132 · fp 4 · fn 46 · tp 20** |
| Brier | **0.1766** |
| ECE | **0.1523** |

**Confidence intervals: NOT COMPUTED.** The Stage 3 metrics module contains no interval
estimator, and none was invented for this run. With 66 positives every rate above carries
wide uncertainty; differences of a few points are **not** interpretable without an
interval method that would have to be added deliberately.

### Test status

> **NOT EVALUATED.** The carved test split (28,883 images, 9,682 patients) has never been
> scored. **CONFIRMED** — `metrics.json` contains zero test-related entries at any depth,
> every history entry carries `split: "val"`, and no predictions file for the test split
> exists. The only test references in the run manifests are the split registry (path,
> SHA-1, row counts) computed from the CSV at start-up; no test image was read.

### Artifacts

```text
artifacts/stage3/runs/baseline__resnet152__seed42__20260830T185505Z/
├── run_manifest.json                     original run provenance (unmodified)
├── config_snapshot.yaml                  original config (unmodified)
├── metrics.json                          per-epoch history, best epoch, resume record
├── checkpoints/
│   ├── best.pt                           epoch 3  ← the model
│   ├── last.pt                           epoch 8
│   └── pre_resume__best__epoch1__20260831T102547Z.pt
└── resumes/
    ├── resume__20260831T102547Z.json     what was restored, and what was not
    ├── run_manifest__20260831T102547Z.json
    └── config_snapshot__20260831T102547Z.yaml

artifacts/stage3/eval/official_valid__resnet152__seed42__20260831T140009Z/
├── metrics.json                          official-validation metrics
├── eval_manifest.json                    checkpoint + threshold + dataset provenance
├── predictions.csv                       202 per-image predictions
└── official_valid_frontal.csv            the derived frontal-only evaluation frame
```

### Interpretation

Read carefully; two of these numbers are easy to misread.

1. **Ranking ability transfers.** ROC-AUC moves **0.8647 → 0.8577** between the carved
   CheXbert validation split and the expert-labelled official validation set. That is the
   most informative comparison available, because ROC-AUC is prevalence-independent. The
   model's ability to rank cardiomegaly above non-cardiomegaly survives a change of label
   source from NLP-derived to radiologist-adjudicated.

2. **PR-AUC 0.7547 is NOT an improvement over 0.5417.** The no-skill baseline for PR-AUC
   *is* the prevalence, and prevalence differs: **0.13652** on the carved split versus
   **0.32673** on the official frontal set. Expressed as lift over baseline the result
   moves the other way — **3.97× → 2.31×**. Any statement that the model "performs better
   on the expert set" is unsupported.

3. **The fixed threshold transfers conservatively.** The threshold 0.44209548389690995 was
   selected for F1 at 13.65% prevalence and applied unchanged at 32.67%. Sensitivity falls
   **0.6046 → 0.3030** while specificity rises **0.8974 → 0.9706**; the model misses 46 of
   66 expert-confirmed cases at this operating point. This is **consistent with a
   threshold-transfer effect** given that ROC-AUC is nearly unchanged. It does **not**
   demonstrate a discrimination failure, and must not be reported as one. Re-optimising
   the threshold on the official set would make that set a tuning surface and is
   forbidden (§13.2).

4. **Calibration degrades on expert labels** — ECE 0.0817 → 0.1523, Brier 0.0961 → 0.1766.
   The carved-split calibration bins already show systematic over-confidence (every bin's
   observed frequency below its mean prediction), which is the expected consequence of
   training with `pos_weight = 6.2467`. **INFERRED:** reporting calibrated probabilities
   would require an explicit recalibration step fitted on validation; the pipeline
   performs none.

5. **Training loss fell monotonically (0.865 → 0.631) while validation PR-AUC plateaued
   after epoch 3.** INFERRED: the model was overfitting from epoch 4 onward, so the
   early stop at epoch 8 with best epoch 3 is well placed rather than premature.

### Limitations

- Only **8 of 15** epochs ran. The budget was not exhausted; early stopping ended it.
  Whether a longer schedule or a different LR would improve on epoch 3 is **UNKNOWN**.
- **Single seed (42), single run.** No variance estimate exists. Any future comparison
  between architectures inherits this limitation.
- **No confidence intervals** anywhere in this entry.
- Epoch 1 ran under a different process than epochs 2–8 (see resume note); model,
  optimizer, scheduler and scaler state carried across exactly, but the **RNG stream did
  not** — see *Reproducibility* below.
- The test arm is a CheXbert-labelled fallback carve, not radiologist ground truth. Its
  eventual use will not produce an expert-adjudicated test result.
- Official-validation provenance carries a caveat — see `research.md`. Structure is
  confirmed; the chain of custody of this particular copy is not independently
  established.

### Reproducibility

- Seed 42; `deterministic: true`; `cudnn_benchmark: false`.
- Config snapshot and run manifest are preserved unmodified next to the checkpoints.
- **Exact re-execution of this run is NOT possible.** The epoch-1 checkpoint predates RNG
  state capture, so epochs 2–8 restarted their augmentation and shuffle streams from
  `set_seed(42)` rather than continuing epoch 1's stream. A fresh uninterrupted 15-epoch
  run at seed 42 would follow a different trajectory. Model, optimizer, scheduler and AMP
  scaler state were restored exactly; **only the RNG stream was not**. Checkpoints written
  from epoch 2 onward now carry RNG state, so this specific gap cannot recur.
- The DataLoader shuffle generator is re-seeded from the run seed at process start and is
  not carried in checkpoints. **Known residual gap**, not closed.

### Status

> **COMPLETE.** Model trained, best checkpoint selected on the carved validation split,
> secondary expert check performed on the official validation set. Test split untouched.
> This is baseline **1 of 4** for RQ1.

---

## Planned experiments — NOT YET RUN

No numbers appear below. These entries exist so that the ledger's shape is fixed before
results arrive, and so that an empty row is never mistaken for a null result.

### E002 — DenseNet201 baseline — **NOT RUN**

| Field | Value |
|---|---|
| Objective | Baseline 2 of 4 for RQ1 |
| Model | DenseNet201, ImageNet-pretrained |
| Protocol | **Must be identical to E001** — same splits, seed, preprocessing, augmentation, optimiser, scheduler, effective batch, epoch budget and early-stopping rule. Deviations invalidate the RQ1 comparison |
| Status | **PLACEHOLDER — no result** |

### E003 — EfficientNetV2-S baseline — **NOT RUN**

| Field | Value |
|---|---|
| Objective | Baseline 3 of 4 for RQ1 |
| Model | EfficientNetV2-S, ImageNet-pretrained |
| Protocol | Identical to E001 |
| Status | **PLACEHOLDER — no result** |

### E004 — MobileNetV3-Large baseline — **NOT RUN**

| Field | Value |
|---|---|
| Objective | Baseline 4 of 4 for RQ1 |
| Model | MobileNetV3-Large, ImageNet-pretrained |
| Protocol | Identical to E001 |
| Status | **PLACEHOLDER — no result** |

### E005+ — downstream stages — **NOT RUN, NOT SCOPED**

Hybrid feature fusion, hyperparameter optimisation, XGBoost-on-embeddings and Grad-CAM
are defined in the worklet and `literature_review_v2.md`. **None is scheduled and none
has a design fixed in code.** They must not be entered here until they do.

---

## Cross-experiment comparison — **EMPTY**

An architecture comparison table will appear here once **at least two** baselines have run
under the identical protocol. With one experiment complete there is nothing to compare,
and a single-row "ranking" would be misleading.

---

## Test-set policy

The carved test split has **never** been evaluated, in any experiment.

It will be scored **once**, after the model set is frozen, at the threshold selected on
validation, and the result will be reported with the standing caveat that it is a
CheXbert-labelled fallback carve rather than the official 5-radiologist consensus test
set. Any entry above that claims otherwise is a defect in this document.
