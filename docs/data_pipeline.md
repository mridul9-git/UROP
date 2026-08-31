# Data Pipeline — actual current state

**Stage:** 3 (baseline training). **Last updated:** 31/08/2026.

This document describes the **data** pipeline **as it exists in code today**, not as it is
planned. The training pipeline that consumes it is documented in `docs/flow.md` section 8.

Certainty labels: **CONFIRMED** = verified by running the code · **INFERRED** = follows
from confirmed behaviour · **VERIFY** = unchecked · **UNKNOWN** = not yet determined ·
**SUPERSEDED** = was true, no longer is.

> **Execution status — CONFIRMED:** every stage below has run end-to-end **on the real
> dataset**. Metadata and splits on 21/08/2026; image integrity, dimensions and duplicate
> analysis on 30/08/2026; training and evaluation on 30–31/08/2026.
>
> **SUPERSEDED (16/08/2026 -> 30/08/2026):** ~~every stage below has been executed
> end-to-end against a *synthetic* fixture. No stage has run against real CheXpert data,
> because none has been downloaded. Stages marked BLOCKED cannot run at all until specific
> files arrive.~~ The fixture runs still exist and are still how the smoke tests work; they
> are no longer the only evidence.

---

## 1. The pipeline

```text
Input                    data/{train.csv, valid.csv}                   ← on disk (30/08/2026)
  │                      + data/train/ data/valid/ image trees         ← on disk (30/08/2026)
  │                      train_cheXbert.csv (PRIMARY label source, D211)
  ▼
Metadata                 load_split_csv()
  │                      read CSV → assert required columns present
  ▼
Validation               attach_identifiers()
  │                      regex-parse patient / study / view from Path
  │                      RAISES on any unparseable row (never silent NaN)
  ▼
Label processing         apply_uncertainty_policy(policy="u_zeros")
  │                      1.0→1 · 0.0→0 · -1.0→0 · blank→0  ⇒ column `target`
  ▼
View selection           select_frontal()
  │                      keep Frontal/Lateral == "Frontal"
  │                      AP/PA retained as metadata; blanks → "Unknown"
  ▼
Patient-level split      patient_level_labels() → stratified_patient_split()
  │                      group = patient_id · stratify = any-positive · seed 42
  │                      assert_disjoint(train, val, test)   ← HARD GATE, raises
  ▼
Sampling / subset        subsample_patients(target_frontal_images=40000)
  │                      TRAIN ONLY · patient-level · seed 42 · nested supersets
  ▼
Output                   artifacts/stage2/splits/{train,val,test}.csv   ← FROZEN
                         artifacts/stage2/splits/split_manifest.json
  ▼
Preprocessing            IMPLEMENTED — src/training/transforms.py
  │                      resize 320×320 → gray→RGB×3 → /255 → ImageNet normalize
  ▼
Training                 src/training/  (see docs/flow.md section 8)
```

**Two side branches** run off the same frame but do not feed the split:

```text
select_frontal() ──┬──► validate_images.py  → excluded_images.csv, image_validation.md
                   └──► find_duplicates.py  → duplicates_exact.csv, duplicates.md,
                                              duplicates_near_crosspatient.csv
```

Both **have run on the real images (30/08/2026)**. Neither modifies the split; they produce
an exclusion manifest which the training loader consults. **0 rows were excluded** from any
split at training time.

---

## 2. Stage-by-stage detail

### 2.1 Input — **CONFIRMED present**

| Item | Path | Status |
|---|---|---|
| `train_cheXbert.csv` | repo root | **on disk** — the PRIMARY label source (D211) |
| `train.csv`, `valid.csv` | `E:/UROP/data/` | **on disk** 30/08/2026 |
| `train/`, `valid/` image trees | `E:/UROP/data/` | **on disk** 30/08/2026 |
| official test labels + images | `test_set.root` | **NOT ACQUIRED**; `test_set.enabled: false` |

> **SUPERSEDED (16/08/2026 -> 30/08/2026):** ~~`train.csv` / `valid.csv` not downloaded;
> image trees not downloaded — not yet approved.~~
>
> **Provenance caveat, still live:** the copy on disk matches the official release's
> structure exactly, but its chain of custody is **UNKNOWN**. See `docs/research.md`
> section 8. The official 500-patient **test** set remains unacquired, so the test arm is a
> fallback carve.

`load_split_csv()` raises `FileNotFoundError` with the config key to fix, so a missing
download can never be mistaken for an empty dataset. **CONFIRMED** by running it.

### 2.2 Metadata — `load_split_csv(cfg, which)`

Reads the CSV and checks that `Path`, `Frontal/Lateral` and `AP/PA` exist, raising
`ValueError` listing any missing column. Adds `source_csv` ("train"/"valid"), then calls
`attach_identifiers()`. **CONFIRMED.**

### 2.3 Validation — `attach_identifiers(df)`

Adds `patient_id`, `study_num`, `study_id`, `view_num`, `view_kind` by regex
(`patient(\d+)`, `study(\d+)`, `view(\d+)_(frontal|lateral)`) rather than by splitting on
`/` positionally, so a renamed root directory in a future release cannot silently shift
every field. Raises `ValueError` naming the first offending `Path` if any row fails.
**CONFIRMED on real data — 223,414 rows parsed, 0 unparseable.**

This is the only place patient identity is established, and **patient identity is what the
entire leakage guarantee rests on** — hence the hard failure rather than a warning.

### 2.4 Label processing — `apply_uncertainty_policy(df, column, policy, blank_is_negative)`

| policy | `-1.0` becomes | blank becomes |
|---|---|---|
| `u_zeros` *(primary)* | `0` | `0` |
| `u_ones` | `1` | `0` |
| `exclude` | row dropped from the frame | `0` |

Returns a **copy** with `target` (int) and `uncertainty_policy` columns. `exclude` drops
rows from the returned DataFrame only — **nothing is deleted on disk**. **CONFIRMED.**

`CardiomegalyDataset` re-applies this policy at training time and **raises** if the
re-derived target disagrees with the stored one, so a split built under a different policy
cannot be fed to a differently configured run.

### 2.5 View selection — `select_frontal(cfg, df)`

Keeps rows whose `Frontal/Lateral` is in `view.keep` (`["Frontal"]`). Fills blank `AP/PA`
with `"Unknown"` so projection-stratified reporting never silently drops rows.
**CONFIRMED.**

Also used by the official-validation evaluation, where it kept **202 frontal** and excluded
**32 lateral** images of 234.

### 2.6 Patient-level split — `patient_level_labels()` → `stratified_patient_split()`

`patient_level_labels(df)` collapses to one row per patient with `n_images`, `n_studies`,
`any_positive`, `n_positive_images`. `any_positive` is the stratification target.

`stratified_patient_split(patients, frac_b, seed)` cuts **20 fixed folds** with
`StratifiedGroupKFold` (groups = `patient_id`) and unions `round(frac_b × 20)` of them.

> **INFERRED → CONFIRMED by test.** The obvious implementation,
> `n_splits = round(1/frac_b)`, only hits the requested fraction when `1/frac_b` is an
> integer — asking for 15% yields 1/7 ≈ 14.3%. The fixture run exposed this; the 20-fold
> union fixes it. Recorded because it is the kind of bug that silently misreports a split.

`assert_disjoint(splits)` then raises `AssertionError` naming overlapping patient IDs if
any patient appears in more than one split. It runs **before any CSV is written**.
**CONFIRMED passing on real data — 33,021 patients.** Re-asserted at training time by
`assert_no_patient_overlap`, and **CONFIRMED passing there too** (30–31/08/2026).

### 2.7 Sampling / subset — `subsample_patients(patients, target_images, seed)`

Samples **patients**, in two strata (`any_positive` 1 and 0), taking a seeded permuted
prefix of each until the stratum's image budget is met. **TRAIN ONLY** — validation and
test are never subsampled.

**CONFIRMED by test** (fixture, seed 42): raising `target_images` yields a strict
**superset** at every tier (200 ⊂ 400 ⊂ 600), image counts land within ~2% of target, and
**image-level** prevalence is held within ~0.011 absolute. Patient-level prevalence differs
from image-level because positive patients contribute more images; image-level is the
quantity preserved, and it is the one the model and `pos_weight` see.

### 2.8 Preprocessing — **IMPLEMENTED** (`src/training/transforms.py`)

> **SUPERSEDED (16/08/2026 -> 30/08/2026):** ~~SPECIFIED, NOT IMPLEMENTED. No transform code
> exists yet. It belongs to Stage 3 and is deliberately not written here.~~

`config/data.yaml → preprocessing` holds the parameters (320×320, `direct` resize,
`replicate` to RGB, ImageNet mean/std) and `build_train_transform` /
`build_eval_transform` implement them. They are **separate functions**, not one function
with a flag, so training augmentation cannot reach validation or test by mistake.

Augmentation is **train-only**: rotation ±10°, zoom 0.9–1.0, translation ±5%, brightness
±10%, contrast ±10%. The config validator **refuses to start** if any augmentation
section 11 forbids is enabled (horizontal/vertical flip, shear, random erasing, MixUp,
CutMix, CLAHE).

### 2.9 Output — `make_splits.py::main`

| Artifact | Contents |
|---|---|
| `train.csv` / `val.csv` / `test.csv` | one row per image: `Path`, `patient_id`, `study_id`, `view_num`, `Sex`, `Age`, `Frontal/Lateral`, `AP/PA`, `Cardiomegaly`, `Support Devices`, `target`, `uncertainty_policy` |
| `split_manifest.json` | seed, policy, fractions, mode, per-split patient/study/image counts and prevalence, projection breakdown, `pos_weight_from_train`, and a **SHA-1 per split CSV** |

`pos_weight` is computed from **TRAIN ONLY** — deriving it from the full pool would leak
validation and test prevalence into training. **CONFIRMED**, measured value **6.246739**,
and cross-checked against the train CSV at the start of every training run.

**As produced (21/08/2026), and FROZEN since:**

| Split | Images | Patients | Prevalence |
|---|---:|---:|---:|
| train | 40,002 | 13,659 | 0.13799 |
| val | 29,380 | 9,680 | 0.13652 |
| test | 28,883 | 9,682 | 0.13755 |

`mode: fallback-carved-test` — 70/15/15, because the official test set was not acquired.
**The test split has never been evaluated.**

---

## 3. Configuration

Every data parameter lives in `config/data.yaml`; every training parameter in
`config/train.yaml`. No script hard-codes any of them.

Resolution order (**CONFIRMED**): explicit argument → `$UROP_DATA_CONFIG` /
`$UROP_TRAIN_CONFIG` → the repository config. The env var exists so the scripts can be
pointed at a fixture config without editing the real one — that is how the pipeline is
smoke-tested.

**Changed 30/08/2026**, when the images arrived in a different layout than anticipated:
`images.root` and `images.release_dir` were corrected, and `dataset.valid_csv` pointed at
the newly present file. **No code changed** — the centralised resolver absorbed the
difference, which is exactly what it exists for.

---

## 4. What is NOT in this pipeline

**Listed so the gaps are explicit rather than assumed.**

Now implemented (**SUPERSEDED** from the original list): transform/augmentation code,
`Dataset`/`DataLoader` classes, the model/training/evaluation code, and the join of the
exclusion manifest into the training loader — the loader **does** consult
`excluded_images.csv` (0 rows excluded).

Still absent:

- **any confidence-interval estimator** — `metrics.py` has none, so no interval is reported
  anywhere in the project;
- **any probability recalibration** — calibration is *measured* (Brier, ECE, bins) but
  never corrected, and the model is measurably over-confident;
- hyperparameter optimisation, hybrid feature fusion, XGBoost-on-embeddings and XAI code;
- carrying the DataLoader shuffle generator in checkpoints — it is re-seeded from the run
  seed at process start, so a resumed run does not reproduce the original batch order.

---

## 5. Reproducibility invariants

Any change to these invalidates existing splits and results, and must bump a new manifest:

| Invariant | Value | Where enforced |
|---|---|---|
| Split unit | `patient` | `make_splits.py` raises if `split.unit != "patient"` |
| Split seed | `42` | `config/data.yaml → split.seed` |
| Subset seed | `42` | `config/data.yaml → subset.seed` |
| Fold granularity | 20 | `make_splits.GRANULARITY` |
| Uncertainty policy | `u_zeros` | `config/data.yaml → target.policy`; re-asserted per row at train time |
| Views kept | `["Frontal"]` | `config/data.yaml → view.keep` |
| `pos_weight` | `6.246739`, train-only | `split_manifest.json`; cross-checked against the train CSV each run |
| Disjointness | asserted pre-write **and** at train time | `assert_disjoint`, `assert_no_patient_overlap` |
| Training seed | `42`, `deterministic: true` | `config/train.yaml`; `reproducibility.set_seed` |

**One known reproducibility gap.** The ResNet152 baseline was interrupted after epoch 1 and
resumed. The epoch-1 checkpoint predated RNG-state capture, so epochs 2–8 restarted their
augmentation and shuffle streams from the seed rather than continuing epoch 1's stream.
Model, optimizer, scheduler and AMP scaler state were restored exactly; **only the RNG
stream was not**. That run is therefore not reproducible as a single uninterrupted
trajectory. Checkpoints now carry RNG state, so the gap cannot recur — see
`docs/notes.md`.
