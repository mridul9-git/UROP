# Data Pipeline — actual current state

**Stage:** 2 (dataset verification). **Last updated:** 16/08/2026.

This document describes the pipeline **as it exists in code today**, not as it is planned.
No training pipeline exists yet and none is described here.

Certainty labels: **CONFIRMED** = verified by running the code · **INFERRED** = follows
from confirmed behaviour · **UNKNOWN** = not yet determined.

> **Execution status — CONFIRMED:** every stage below has been executed end-to-end against
> a *synthetic* CheXpert-shaped fixture (400 patients, 1,199 rows, real JPEGs). **No stage
> has run against real CheXpert data**, because none has been downloaded. Stages marked
> **BLOCKED** cannot run at all until specific files arrive.

---

## 1. The pipeline

```text
Input                    CheXpert-v1.0-small/{train.csv, valid.csv}   ← BLOCKED: not downloaded
  │                      + train/ valid/ image trees                  ← BLOCKED: not downloaded
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
Preprocessing            SPECIFIED, NOT IMPLEMENTED  (docs/dataset_analysis.md §10)
  │                      resize 320×320 → gray→RGB×3 → /255 → ImageNet normalize
  ▼
Output                   artifacts/stage2/splits/{train,val,test}.csv
                         artifacts/stage2/splits/split_manifest.json
```

**Two side branches** run off the same frame but do not feed the split:

```text
select_frontal() ──┬──► validate_images.py  → excluded_images.csv, image_validation.md
                   └──► find_duplicates.py  → duplicates_exact.csv, duplicates.md
```

Both are **BLOCKED** on the image download. Neither modifies the split; they produce an
exclusion manifest that the future training loader will consult.

---

## 2. Stage-by-stage detail

### 2.1 Input — **BLOCKED**

| Item | Path | Status |
|---|---|---|
| `train.csv`, `valid.csv` | `dataset.root` in `config/data.yaml` | **not downloaded** — approved for acquisition, §1.8 |
| `train/`, `valid/` image trees | same root | **not downloaded** — not yet approved |
| official test labels + images | `test_set.root` | **not acquired**; `test_set.enabled: false` |

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
**CONFIRMED.**

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

### 2.5 View selection — `select_frontal(cfg, df)`

Keeps rows whose `Frontal/Lateral` is in `view.keep` (`["Frontal"]`). Fills blank `AP/PA`
with `"Unknown"` so projection-stratified reporting never silently drops rows.
**CONFIRMED.**

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
**CONFIRMED passing** on the fixture.

### 2.7 Sampling / subset — `subsample_patients(patients, target_images, seed)`

Samples **patients**, in two strata (`any_positive` 1 and 0), taking a seeded permuted
prefix of each until the stratum's image budget is met. **TRAIN ONLY** — validation and
test are never subsampled.

**CONFIRMED by test** (fixture, seed 42): raising `target_images` yields a strict
**superset** at every tier (200 ⊂ 400 ⊂ 600), image counts land within ~2% of target, and
**image-level** prevalence is held within ~0.011 absolute. Patient-level prevalence differs
from image-level because positive patients contribute more images; image-level is the
quantity preserved, and it is the one the model and `pos_weight` see.

### 2.8 Preprocessing — **SPECIFIED, NOT IMPLEMENTED**

`config/data.yaml → preprocessing` holds the agreed parameters (320×320, `direct` resize,
`replicate` to RGB, ImageNet mean/std). **No transform code exists yet.** It belongs to
Stage 3 and is deliberately not written here.

### 2.9 Output — `make_splits.py::main`

| Artifact | Contents |
|---|---|
| `train.csv` / `val.csv` / `test.csv` | one row per image: `Path`, `patient_id`, `study_id`, `view_num`, `Sex`, `Age`, `Frontal/Lateral`, `AP/PA`, `Cardiomegaly`, `Support Devices`, `target`, `uncertainty_policy` |
| `split_manifest.json` | seed, policy, fractions, mode, per-split patient/study/image counts and prevalence, projection breakdown, `pos_weight_from_train`, and a **SHA-1 per split CSV** |

`pos_weight` is computed from **TRAIN ONLY** — deriving it from the full pool would leak
validation and test prevalence into training. **CONFIRMED.**

---

## 3. Configuration

Every parameter lives in `config/data.yaml`; no script hard-codes any of them.

Resolution order (**CONFIRMED**): explicit argument → `$UROP_DATA_CONFIG` → `E:/UROP/config/data.yaml`.
The env var exists so the scripts can be pointed at a fixture config without editing the
real one — that is how the pipeline was tested.

---

## 4. What is NOT in this pipeline yet

**UNKNOWN / not built** — listed so the gaps are explicit rather than assumed:

- transform / augmentation code (Stage 3)
- `Dataset` / `DataLoader` classes (Stage 3)
- any model, training loop, or evaluation code (Stage 4+)
- the exclusion-manifest join into the training loader — the manifest is produced by
  `validate_images.py`, but nothing consumes it yet, because nothing trains yet

---

## 5. Reproducibility invariants

Any change to these invalidates existing splits and must bump a new manifest:

| Invariant | Value | Where enforced |
|---|---|---|
| Split unit | `patient` | `make_splits.py` raises if `split.unit != "patient"` |
| Split seed | `42` | `config/data.yaml → split.seed` |
| Subset seed | `42` | `config/data.yaml → subset.seed` |
| Fold granularity | 20 | `make_splits.GRANULARITY` |
| Uncertainty policy | `u_zeros` | `config/data.yaml → target.policy` |
| Views kept | `["Frontal"]` | `config/data.yaml → view.keep` |
| Disjointness | asserted pre-write | `chexpert_metadata.assert_disjoint` |
