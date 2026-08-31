# UROP — Cardiomegaly Classification from Frontal Chest X-rays

Comparative deep learning study: **binary cardiomegaly classification from frontal chest
X-rays**, with hybrid feature fusion, hyperparameter optimisation and explainable AI.

The project makes **no architectural novelty claim**. It is a controlled comparative study
and partial replication/extension; its value is experimental rigour, reproducibility and
honest reporting of positive *or* negative results.

## Current stage

**Stage 3 — Baseline training. In progress.** *(updated 31/08/2026)*

Stage 1 and Stage 2 are complete. The dataset is on disk, the splits are frozen, and the
training and evaluation pipeline has run end-to-end on real data.

| | |
|---|---|
| Baselines complete | **1 of 4** — ResNet152 |
| Best model | epoch 3 · validation PR-AUC **0.5417** · ROC-AUC **0.8647** |
| Official expert-validation check | 202 frontal images · ROC-AUC **0.8577** · PR-AUC **0.7547** |
| **Carved test split** | **NOT EVALUATED** — and will not be until the model set is frozen |
| Next | DenseNet201 under an identical protocol |

Full results and the exact configuration are in `docs/experiments.md`.

> **Two standing caveats.** The test arm is a CheXbert-labelled *fallback carve*, not the
> official 5-radiologist consensus test set, which has not been acquired — it must never be
> described as expert-adjudicated. And **no confidence intervals are computed anywhere**;
> the metrics module has no interval estimator.

## Documents

| File | What it is |
|---|---|
| `docs/experiments.md` | **the experiment ledger** — every run, its configuration and its results |
| `docs/research.md` | **scientific reasoning** — findings, hypotheses, limitations, open questions |
| `docs/dataset_analysis.md` | **Stage 2 deliverable** — dataset facts, decisions (D201–D211) and justifications |
| `docs/notes.md` | chronological engineering log — failures, fixes and operational knowledge |
| `docs/flow.md` | execution flow through the actual codebase |
| `docs/data_pipeline.md` | the data pipeline as it exists in code |
| `literature_review_v2.md` | Stage 1 working methodology document — audited, scope-reduced |
| `literature_review.md` | Stage 1 research archive (original, long) |
| `UROP_Heart_Disease_Classification_Worklet.md` | the project worklet — stages, scope, definition of done |

Start with `docs/experiments.md` and `docs/research.md` for where the project stands;
`docs/dataset_analysis.md` for why the methodology is what it is.

## Layout

```text
config/data.yaml          single source of truth for every data parameter
config/train.yaml         single source of truth for every training parameter
src/data/                 Stage 2 metadata, split and integrity scripts
src/training/             Stage 3 training and evaluation
docs/                     stage deliverables and living documentation
artifacts/stage2/         generated reports, figures and the frozen splits
artifacts/stage3/runs/    training runs — manifests, metrics, checkpoints
artifacts/stage3/eval/    evaluation runs — metrics, predictions, provenance
requirements.txt
```

## Running

```bash
pip install -r requirements.txt
```

### Stage 2 — data analysis and splits

Already executed; the splits under `artifacts/stage2/splits/` are **frozen**. Re-running
`make_splits.py` would invalidate every existing result.

```bash
python src/data/analyze_metadata.py    # label distribution, views, imbalance, figures
python src/data/validate_images.py     # integrity + dimensions -> excluded_images.csv
python src/data/find_duplicates.py     # L1-L4 exact; add --near for cross-patient L5
python src/data/make_splits.py         # patient-level splits + split_manifest.json
python src/data/probe_vram.py          # measures real batch sizes on this GPU (no data needed)
```

### Stage 3 — training and evaluation

```bash
# smoke test on a synthetic fixture — needs no dataset
python src/training/train.py --dry-run

# train a baseline
python src/training/train.py --model resnet152

# continue an interrupted run (explicit; never automatic)
python src/training/train.py --resume <run>/checkpoints/best.pt

# secondary expert check on the official validation set
# the threshold is REQUIRED and applied unchanged — this set is never tuned on
python src/training/eval_official_valid.py \
    --checkpoint <run>/checkpoints/best.pt \
    --threshold <threshold selected on the carved validation split>

# Stage 3 smoke suite
python src/training/test_training_smoke.py
```

Point the scripts at a different config with `UROP_DATA_CONFIG` / `UROP_TRAIN_CONFIG`.

## Non-negotiables

- **Split by patient, never by image.** Disjointness is asserted before any split is
  written, and again at training time.
- **Validation and test never receive training augmentation.**
- **The test set is touched once**, at the end. Never tune on it. It has not been touched.
- **The official validation set is a secondary expert check, never a selection surface.**
  The threshold is inherited, never re-optimised on it.
- **Accuracy is never reported alone** — PR-AUC with its prevalence baseline is the honest
  headline metric at ~13.7% positive.
- **PR-AUC is not comparable across sets with different prevalence**, because the no-skill
  baseline *is* the prevalence. Compare ROC-AUC, or compare lift over baseline.
- Preprocessing and every hyperparameter are **identical across all four baselines**;
  deviations invalidate RQ1.
- Performance is **agreement with a report-derived label**, not clinical diagnostic
  accuracy.
