# UROP — Cardiomegaly Classification from Frontal Chest X-rays

Comparative deep learning study: **binary cardiomegaly classification from frontal chest
X-rays**, with hybrid feature fusion, hyperparameter optimisation and explainable AI.

The project makes **no architectural novelty claim**. It is a controlled comparative study
and partial replication/extension; its value is experimental rigour, reproducibility and
honest reporting of positive *or* negative results.

## Current stage

**Stage 2 — Dataset Acquisition & Analysis.** Research and analysis code complete;
**awaiting approval to download the dataset.** No dataset file is on disk yet.

## Documents

| File | What it is |
|---|---|
| `UROP_Heart_Disease_Classification_Worklet.md` | the project worklet — stages, scope, definition of done |
| `literature_review.md` | Stage 1 research archive (original, long) |
| `literature_review_v2.md` | **the working methodology document** — audited, scope-reduced |
| `docs/dataset_analysis.md` | **Stage 2 deliverable** — dataset facts, decisions and justifications |

Read `literature_review_v2.md` and `docs/dataset_analysis.md`; the other two are reference.

## Layout

```text
config/data.yaml          single source of truth for every data parameter
src/data/                 Stage 2 analysis scripts (no training code yet)
docs/                     stage deliverables
artifacts/stage2/         generated reports, figures and splits (git-ignored except splits)
requirements.txt
```

## Running the Stage 2 analysis

Requires the dataset. Set `dataset.root` in `config/data.yaml` first, then:

```bash
pip install -r requirements.txt

python src/data/analyze_metadata.py    # label distribution, views, imbalance, figures
python src/data/validate_images.py     # integrity + dimensions -> excluded_images.csv
python src/data/find_duplicates.py     # L1-L4 exact; add --near for cross-patient L5
python src/data/make_splits.py         # patient-level splits + split_manifest.json
python src/data/probe_vram.py          # measures real batch sizes on this GPU (no data needed)
```

Point the scripts at a different config with `UROP_DATA_CONFIG=/path/to/data.yaml`.

## Non-negotiables

- **Split by patient, never by image.** `assert_disjoint()` runs before any split is written.
- **Validation and test never receive training augmentation.**
- **The test set is touched once**, at the end. Never tune on it.
- **Accuracy is never reported alone** — PR-AUC with its prevalence baseline is the honest
  headline metric at ~12% positive.
- Preprocessing is **identical across all four baselines**; deviations invalidate RQ1.
- Performance is **agreement with a report-derived label**, not clinical diagnostic accuracy.
