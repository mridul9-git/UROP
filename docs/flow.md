# Execution Flow — actual codebase

**Last updated:** 16/08/2026 · **Stage:** 2

Documents **only functions and classes that exist in the repository today**. No training,
model, optimisation or XAI code has been written, so none is described here.

Certainty: **CONFIRMED** = read from the source and executed · **INFERRED** = follows from
confirmed behaviour · **UNKNOWN** = not yet determined.

---

## 1. Entry points

There is **no single `main()` for the project**. Stage 2 is six independent scripts, each
with its own `main()` guarded by `if __name__ == "__main__":`.

| Entry point | Purpose | Needs CSV? | Needs images? | Runs today? |
|---|---|---|---|---|
| `src/data/plot_published_stats.py` | Step 18 figures from published values | no | no | ✅ **yes — has run** |
| `src/data/probe_vram.py` | Step 14 VRAM/throughput measurement | no | no | ✅ **yes — has run** (21/08/2026) |
| `src/data/analyze_metadata.py` | Steps 3, 4, 6, 12 tables + figures | **yes** | no | ⛔ blocked on CSV |
| `src/data/make_splits.py` | Steps 5, 13, 15 splits + manifest | **yes** | no | ⛔ blocked on CSV |
| `src/data/validate_images.py` | Steps 8, 9 integrity + dimensions | yes | **yes** | ⛔ blocked on images |
| `src/data/find_duplicates.py` | Step 7 duplicate analysis | yes | **yes** | ⛔ blocked on images |

`src/data/chexpert_metadata.py` is a **library, not an entry point** — it has no `main()`.

---

## 2. Shared library — `chexpert_metadata.py`

Every script begins by inserting its own directory on `sys.path` and importing from here,
so the definition of "a usable frontal Cardiomegaly example" exists in exactly one place.

### Configuration
| Function | Signature | Behaviour |
|---|---|---|
| `load_config` | `(path=None) -> dict` | resolves explicit path → `$UROP_DATA_CONFIG` → `DEFAULT_CONFIG` |
| `ensure_dirs` | `(cfg) -> None` | creates `paths.artifacts`, `paths.figures`, `paths.reports` |

### Loading and parsing
| Function | Signature | Behaviour |
|---|---|---|
| `load_split_csv` | `(cfg, which="train") -> DataFrame` | reads CSV, checks required columns, adds `source_csv`, calls `attach_identifiers` |
| `attach_identifiers` | `(df) -> DataFrame` | regex-parses `patient_id`, `study_num`, `study_id`, `view_num`, `view_kind`; **raises** on unparseable `Path` |

### Labels
| Function | Signature | Behaviour |
|---|---|---|
| `LabelCounts` | frozen dataclass | `positive`, `negative`, `uncertain`, `blank`; `.total` property |
| `raw_label_counts` | `(df, column) -> LabelCounts` | counts the four raw states |
| `apply_uncertainty_policy` | `(df, column, policy, blank_is_negative=True) -> DataFrame` | adds `target` + `uncertainty_policy`; `policy ∈ {u_zeros, u_ones, exclude}`; **raises** on unknown policy |

### Views and patients
| Function | Signature | Behaviour |
|---|---|---|
| `select_frontal` | `(cfg, df) -> DataFrame` | filters to `view.keep`; fills blank `AP/PA` with `"Unknown"` |
| `patient_level_labels` | `(df) -> DataFrame` | one row per patient: `n_images`, `n_studies`, `any_positive`, `n_positive_images` |
| `assert_disjoint` | `(dict[str, DataFrame]) -> None` | **raises `AssertionError`** naming overlapping patients |
| `imbalance_summary` | `(df) -> dict` | `positive`, `negative`, `total`, `prevalence`, `neg_per_pos`, `pos_weight` |

### Plotting
| Function | Signature | Behaviour |
|---|---|---|
| `compact_formatter` | `(vmax) -> callable` | picks `M` / `k` / plain units from the data range |
| `apply_plot_style` | `() -> None` | sets rcParams: recessive grid, hairline axes, system sans |

**Module constants:** `PATH_COL`, `SEX_COL`, `AGE_COL`, `VIEW_COL`, `PROJECTION_COL`,
`OBSERVATIONS` (the 14, in CSV order), `POSITIVE=1.0`, `NEGATIVE=0.0`, `UNCERTAIN=-1.0`,
`PALETTE`, `DEFAULT_CONFIG`.

---

## 3. Flow per entry point

### 3.1 `plot_published_stats.py` — **CONFIRMED, has run**

```text
main()
  ├─ load_config() → ensure_dirs() → apply_plot_style()
  ├─ fig_cardiomegaly_states(out)   → 01_cardiomegaly_label_states.png
  ├─ fig_uncertainty_rates(out)     → 02_uncertainty_rate_by_observation.png
  ├─ fig_policy_auc(out)            → 03_uncertainty_policy_auc.png
  ├─ fig_policy_balance(out)        → 04_uncertainty_policy_balance.png
  └─ fig_composition(out)           → 05_official_split_composition.png
```

Reads **no dataset file**. Its inputs are module-level constants transcribed from primary
sources — `TABLE1`, `N_STUDIES`, `TABLE3_CARDIOMEGALY`, `COMPOSITION` — each carrying its
citation in `SOURCE_T1` / `SOURCE_T3` / `SOURCE_DS`. Helpers: `_caption`, `_labels`.

### 3.2 `analyze_metadata.py` — ⛔ blocked on CSV (tested on fixture)

```text
main()
  ├─ load_config() → ensure_dirs() → apply_plot_style()
  ├─ load_split_csv(cfg,"train")            → attach_identifiers()
  ├─ load_split_csv(cfg,"valid")            [warns and continues if absent]
  ├─ corpus totals: patients / studies / images / frontal / lateral
  ├─ view × projection breakdown
  ├─ raw_label_counts()  image level AND study level (drop_duplicates("study_id"))
  ├─ all 14 observations, study level
  ├─ for policy in (u_zeros, u_ones, exclude):
  │      select_frontal() → apply_uncertainty_policy() → imbalance_summary()
  ├─ patient_level_labels()                 → repetition / leakage statistics
  ├─ _fig_label_states / _fig_view_breakdown / _fig_policy_balance / _fig_images_per_patient
  └─ writes reports/metadata_analysis.md, policy_class_balance.csv, patient_level_labels.csv
```

Helpers: `_md_table`, `_pct`, `_bar_labels`.

### 3.3 `make_splits.py` — ⛔ blocked on CSV (tested on fixture)

```text
main()
  ├─ load_config(); raise if split.unit != "patient"
  ├─ load_split_csv → select_frontal → apply_uncertainty_policy   ⇒ modelling frame
  ├─ patient_level_labels(frame)
  ├─ if test_set.enabled:
  │      stratified_patient_split(patients, val_frac, seed) → train_ids, val_ids
  │      _load_official_test(cfg, policy, target_col)       → test_frame
  │  else:
  │      two-step fallback carve → train / val / test        [prints a warning]
  ├─ if subset.enabled: subsample_patients(...) on TRAIN ONLY
  ├─ assert_disjoint({train, val, test})            ← HARD GATE
  ├─ summarise() per split
  └─ writes train.csv, val.csv, test.csv, split_manifest.json (incl. SHA-1 per file)
```

| Function | Signature | Notes |
|---|---|---|
| `stratified_patient_split` | `(patients, frac_b, seed) -> (ids_a, ids_b)` | 20 folds, union `round(frac_b×20)`; **raises** if `frac_b ∉ (0,1)` |
| `subsample_patients` | `(patients, target_images, seed) -> ids` | two strata, seeded permuted prefix — **CONFIRMED nested supersets** |
| `summarise` | `(name, df) -> dict` | counts, prevalence, projection breakdown |
| `_load_official_test` | `(cfg, policy, target_col) -> DataFrame` | **raises** with remediation text if labels absent |

Constants: `GRANULARITY = 20`, `KEEP_COLS`.

### 3.4 `validate_images.py` — ⛔ blocked on images (tested on fixture)

```text
main()
  ├─ argparse: --sample N | --dims-only | --workers
  ├─ select_frontal(load_split_csv(cfg,"train"))
  ├─ ThreadPoolExecutor → check_one(rel_path, root, full_decode) per image
  ├─ Step 8: status counts → excluded_images.csv (never deletes)
  ├─ Step 9: width/height/aspect percentiles, common dims, upsample counts per candidate size
  ├─ _fig_dimensions()                       → image_dimensions.png
  └─ writes reports/image_validation.md, image_validation_full.csv
```

`check_one` returns a record and **never raises**; statuses are `valid`, `missing`,
`unreadable`, `invalid_dims`, `constant`. Module sets
`ImageFile.LOAD_TRUNCATED_IMAGES = False` and promotes `DecompressionBombWarning` to an
error — **INFERRED significance:** Pillow's default silently pads a truncated JPEG with
grey, which would train happily and never be detected.

### 3.5 `find_duplicates.py` — ⛔ blocked on images (tested on fixture)

```text
main()
  ├─ argparse: --near | --max-hamming | --workers
  ├─ select_frontal(load_split_csv(cfg,"train"))
  ├─ L3/L4 legitimate repetition: groupby study_id / patient_id     [reported, never removed]
  ├─ L1/L2 exact: ThreadPool → sha1_of() → group by digest
  │      flags groups spanning >1 patient as the dangerous case
  ├─ L5 (only with --near): dhash() → bucket by first 16 bits
  │      → compare within buckets → keep CROSS-PATIENT pairs only
  └─ writes reports/duplicates.md, duplicates_exact.csv, duplicates_near_crosspatient.csv
```

| Function | Signature |
|---|---|
| `_resolve` | `(root, rel) -> Path` — handles `Path` values that already include the release dir |
| `sha1_of` | `(root, rel) -> (rel, hexdigest)`; returns `""` on failure |
| `dhash` | `(root, rel, size=8) -> (rel, bitstring)` — 64-bit difference hash |

### 3.6 `probe_vram.py` — **CONFIRMED, has run** (no data needed)

```text
main()                                    # PARENT
  ├─ import torch; exit cleanly if missing or CUDA unavailable
  ├─ report GPU, torch/CUDA versions, free-vs-total VRAM
  ├─ for name in MODELS:
  │      subprocess: python probe_vram.py --model NAME --json
  │      (own CUDA context per model — a hard OOM can wedge a context)
  │      parse the child's ---JSON--- payload
  └─ writes reports/vram_probe.json + prints the summary table

run_child(name, ...)                      # CHILD
  └─ probe_model() → sweep CANDIDATES ladder, recording throughput at every batch
```

| Function | Signature | Notes |
|---|---|---|
| `build` | `(name) -> nn.Module` | `weights=None` — irrelevant for a memory probe |
| `_is_oom` | `(exc) -> bool` | matches the OOM **message**, because torch ≤ 2.x raised `torch.cuda.OutOfMemoryError` while 2.13 surfaces `torch.AcceleratorError` |
| `try_batch` | `(name, batch, size, amp, steps=6) -> dict \| None` | real fwd+bwd+AdamW; returns `None` on OOM; skips steps 0–1 from timing; reports **both** `alloc_gib` and `reserved_gib`; `del` locals **inside** `finally` before `gc.collect()` |
| `probe_model` | `(name, size, amp, cap) -> dict` | sweeps `CANDIDATES`; returns max **efficient** batch, not max non-crashing batch |
| `run_child` | `(name, size, amp, cap) -> None` | emits JSON after a `---JSON---` marker |

Constants: `EFFECTIVE_BATCH = 32`, `CANDIDATES`, `EFFICIENCY_FLOOR = 0.90`,
`COLLAPSE_FLOOR = 0.70`.

**Why it is not a simple grow-until-OOM loop — CONFIRMED by measurement.** Windows CUDA
System Memory Fallback (driver default, ON) spills to host RAM instead of raising OOM, so
a naive probe reports an impossible batch that trains ~10× slower. Detection is by
throughput collapse plus `reserved_gib` exceeding physical VRAM. See
`dataset_analysis.md` §14.2.1.

**Three bugs in the first version, all found by running it:** cleanup ran outside the
`try` block; only `torch.cuda.OutOfMemoryError` was caught (torch 2.13 raises
`AcceleratorError`); and `gc.collect()` in `finally` could not free GPU tensors because
the frame's locals were still bound.

---

## 4. Data transformations, end to end

**CONFIRMED** — the actual column lifecycle:

```text
raw CSV row
  → +patient_id, study_num, study_id, view_num, view_kind      attach_identifiers
  → filtered to Frontal/Lateral == "Frontal"                    select_frontal
  → +target (int 0/1), +uncertainty_policy (str)                apply_uncertainty_policy
  → grouped to one row per patient (+any_positive)              patient_level_labels
  → patients partitioned into train / val / test                stratified_patient_split
  → train patients optionally reduced                           subsample_patients
  → written as KEEP_COLS subset, sorted by Path                 make_splits.main
```

---

## 5. Verification performed

**CONFIRMED** — executed 16/08/2026 against a synthetic fixture (400 patients, 1,199 rows,
real JPEGs, realistic label marginals; generator in the session scratchpad, not committed):

| Check | Result |
|---|---|
| All six modules byte-compile | pass |
| `analyze_metadata.py` full run | pass — report + 4 figures |
| `make_splits.py` full run | pass — **`assert_disjoint` passes**; train 124 / val 60 / test 60 patients |
| `validate_images.py` full run | pass — 1,001 checked, 0 excluded |
| `find_duplicates.py --near` | pass — L1–L5 report written |
| `plot_published_stats.py` | pass — 5 figures, on real published values |
| Subset nesting (200 ⊂ 400 ⊂ 600) | pass |
| `compact_formatter` at sub-1k scale | pass (regression: previously rendered "0k") |
| `probe_vram.py` on the real RTX 4060 (21/08/2026) | pass — 4 models measured, `vram_probe.json` written |

**Two real bugs were found and fixed by this testing**, both recorded in
`docs/data_pipeline.md` §2.6 and §2.7: the `StratifiedGroupKFold` fraction error, and the
axis formatter collapsing sub-1000 counts to "0k".

**UNKNOWN until real data arrives:** whether the real `Path` format, the `AP/PA` value set,
and the actual column list match §2.3 of `dataset_analysis.md`. The scripts raise loudly
rather than guessing if they do not.
