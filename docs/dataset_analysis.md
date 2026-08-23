# Stage 2 — Dataset Acquisition & Analysis

**Project:** Comparative Deep Learning for Cardiomegaly Classification from Frontal Chest X-rays
**Stage:** 2 — dataset verification (no model training)
**Date:** 16/08/2026
**Status:** Research complete · analysis code written and smoke-tested · **blocked on download approval**

---

## 0. How to read this document

Every factual claim carries a certainty label:

| Label | Meaning |
|---|---|
| **CONFIRMED** | Verified against a primary source (dataset paper, official datasheet, official portal) or measured on this machine. Source cited inline. |
| **INFERRED** | Derived by arithmetic or logic from CONFIRMED facts. Reasoning shown. |
| **VERIFY** | Expected from secondary sources but **not yet checked against the actual files**. The script that will check it is named. |
| **UNKNOWN** | Not yet determined. What is needed is stated. |

Sections 3, 4, 10, 11, 12, 13 and 15 explicitly separate **WHAT THE DATASET SAYS**
from **WHAT WE RECOMMEND**.

**Nothing has been downloaded.** No dataset file exists on this machine. Everything
below §6 that requires the actual data is implemented as runnable code and marked
VERIFY; the numbers in the tables are the published expectations that the code will
either confirm or contradict.

### Where each required deliverable lives

| # | Required item | Section |
|---:|---|---|
| 1 | Dataset overview | **§0.1** |
| 2 | Access instructions | §1 (procedure: §1.8) |
| 3 | Dataset structure | §2 |
| 4 | Cardiomegaly label analysis | §3.1 |
| 5 | Uncertainty analysis | §3.2 |
| 6 | View analysis | §4 |
| 7 | Patient leakage analysis | §5 |
| 8 | Class distribution | §6 |
| 9 | Duplicate analysis | §7 |
| 10 | Corruption analysis | §8 |
| 11 | Image dimensions | §9 |
| 12 | Preprocessing recommendation | §10 |
| 13 | Augmentation recommendation | §11 |
| 14 | Class imbalance strategy | §12 |
| 15 | Split strategy | §13 |
| 16 | Compute feasibility | §14 |
| 17 | Dataset-size / subset strategy | §15 |
| 18 | Stage 2 decisions | **§20** (D201–D210) |
| 19 | Stage 2 unresolved questions | **§21** |
| — | Visualizations | **§18** |
| — | Final Stage 2 recommendation | **§22** |

---

## 0.1 Dataset overview

**CheXpert** is a large public chest-radiograph dataset released by the Stanford ML Group
in 2019, collected retrospectively from **Stanford Hospital between October 2002 and
July 2017** (**CONFIRMED** — Irvin et al. 2019).

| Property | Value | Certainty |
|---|---|---|
| Images | **224,316** | CONFIRMED |
| Patients | **65,240** | CONFIRMED |
| Studies | **187,641** (training portion) | CONFIRMED — Table 1 column sums |
| Observations labelled | **14**, including `Cardiomegaly` | CONFIRMED |
| Label states | positive / negative / uncertain / no-mention | CONFIRMED |
| Training label source | **NLP labeller over free-text radiology reports** | CONFIRMED |
| Valid / test label source | **radiologist annotation** — 3-way majority / 5-way consensus | CONFIRMED |
| Views | frontal (AP, PA) and lateral | CONFIRMED |
| Release we will use | 8-bit grayscale **JPEG**, ~390 × 320 | CONFIRMED |
| Institution | **single site**, Stanford Hospital | CONFIRMED |
| Licence | Stanford Research Use Agreement — non-commercial, no redistribution | CONFIRMED |

**Why it fits this project.** It carries an explicit `Cardiomegaly` observation at
sufficient scale, exposes **patient identifiers** (via `Path`) so leakage-controlled
splitting is possible, ships an **expert-annotated evaluation set**, and is obtainable
without credentialing — unlike MIMIC-CXR and VinDr-CXR, which sit behind PhysioNet
credentialing and were judged off the critical path in Stage 1.

**The limitation that must appear in every write-up.** Training labels are **derived from
radiology reports by an automated labeller, not from image review**:

> Model performance measures **agreement with a report-derived target**, not clinical
> diagnostic accuracy. The achievable ceiling is bounded by label noise we cannot directly
> measure.

It is also **single-institution, historical data**, so no claim of generalisation across
sites or eras is supportable from this dataset alone.

---

## 1. Step 1 — Dataset access

### 1.1 Exact release we will use

> **CheXpert v1.0, downsampled JPEG release (`CheXpert-v1.0-small`), plus the official
> expert-annotated test set.**

### 1.2 Access mechanism

**CONFIRMED** — Stanford AIMI has migrated dataset distribution to **Redivis**
(`stanford.redivis.com`), reachable from the AIMI shared-datasets index. Access requires:

1. A Redivis account (free; Google/institutional sign-in works — no Stanford affiliation needed).
2. Accepting the **Stanford University Dataset Research Use Agreement (RUA)** on the dataset page.
3. Download via the Redivis web UI or its CLI/Python client.

The original Stanford ML Group competition page (`stanfordmlgroup.github.io/competitions/chexpert/`)
is still live and still the canonical documentation entry point, but the competition
itself is closed and it now redirects downloads to the AIMI/Redivis portal.

### 1.3 Terms relevant to academic research

**CONFIRMED** — The Stanford RUA permits **non-commercial research use only** and
**prohibits redistribution** ("you may not distribute, publish or reproduce a copy of
this dataset").

Two consequences that shape this project:

- **A UROP research report is squarely within permitted use.** No credentialing exam,
  no CITI training, no PI sponsorship — unlike PhysioNet datasets (MIMIC-CXR, VinDr-CXR).
  This was the original reason CheXpert beat MIMIC on the critical path, and it holds.
- **The no-redistribution clause has a practical bite:** uploading the images to Google
  Drive, a Kaggle private dataset, or any third-party host to reach a cloud GPU is at
  best a grey area. **This is a real argument for training locally**, not merely a
  convenience — see §14.
- Published figures must never include identifiable patient images beyond what the RUA
  allows. Grad-CAM figures in the report use CheXpert images, which is standard practice
  in the literature, but check the RUA text at signup for any figure-publication clause.

### 1.4 Available releases and sizes

| Release | Format | Resolution | Download | Extracted | **CONFIRMED by** |
|---|---|---|---:|---:|---|
| CheXpert v1.0 (full) | JPEG | original DICOM dims | ~440 GB | ~440 GB | CheXpert datasheet, arXiv:2105.03020 |
| **CheXpert-v1.0-small** | JPEG | ~390 × 320 (aspect preserved) | **~11 GB** | **~11 GB** | Characteristics: same datasheet. **Current availability: UNVERIFIED** — no download source has ever been confirmed for this release (see D201) |
| CheXpert Plus (2024/25) | **DICOM** + reports | original | ~500 GB+ | — | Stanford AIMI / arXiv:2405.19538 |
| Official test set images | JPEG | original | < 1 GB (668 images) | — | CheXlocalize dataset, Stanford AIMI |
| Official test set labels | CSV | — | < 1 MB | — | `github.com/rajpurkarlab/cheXpert-test-set-labels` |

**Recommendation: `CheXpert-v1.0-small`.** Justification is not "it is smaller":

- Its native short side is **320 px**, which is exactly the input resolution we intend
  to train at (§9). The full release would be downsampled to 320 anyway — we would pay
  440 GB of storage and a large decode cost per epoch for pixels we then throw away.
- It is already **8-bit grayscale JPEG**. CheXpert Plus is DICOM, which would force us
  to make windowing / rescale-slope-intercept decisions that are a research project in
  themselves and are *not* part of our research questions.
- 11 GB fits comfortably alongside checkpoints on the E: drive (§14.5).

**The full release is only justified if we later decide to train above 320 px.** We do
not (§9).

### 1.5 Validation and test availability

| Set | Images | Patients | Labels | Available? |
|---|---:|---:|---|---|
| train | 223,414 | 64,540 | NLP-derived from reports (weak) | Yes, in the release |
| valid | 234 | 200 | **3 radiologists, majority vote** | Yes, in the release |
| test | 668 | 500 | **5 radiologists, consensus** | **Yes — but from two separate places** |

**CONFIRMED** — the test set was historically withheld for the leaderboard. It is now
public: **labels** at `github.com/rajpurkarlab/cheXpert-test-set-labels` (`groundtruth.csv`,
the 5-radiologist majority vote, plus per-radiologist annotations), and **images** as
part of the **CheXlocalize** dataset on Stanford AIMI.

This resolves open question #4 from the Stage 1 literature review. It matters a great
deal — see §13.

### 1.6 Additional files needed

- `train.csv`, `valid.csv` — ship inside the release archive.
- `groundtruth.csv` — separate GitHub clone (tiny).
- CheXlocalize images — separate AIMI/Redivis download.
- Nothing else. No DICOM headers, no separate metadata file.

### 1.7 Storage budget

| Item | Size |
|---|---:|
| `CheXpert-v1.0-small` archive | ~11 GB |
| extracted images | ~11 GB |
| peak during extraction (both present) | ~22 GB |
| official test images + labels | ~1 GB |
| model checkpoints (4 baselines × ~3 saves; ResNet152 fp32 ≈ 230 MB) | ~6 GB |
| cached embeddings for the XGBoost / fusion arms (float32, ~2k-dim × ~200k) | ~2 GB |
| logs, figures, Optuna study DB | < 1 GB |
| **Working total** | **~40 GB** |

**CONFIRMED (measured):** `E:` has **301.2 GB free**. Storage is a non-issue. Put
everything on `E:` — `C:` has only 90.6 GB free and is the OS drive.

### 1.8 Acquisition order — metadata first (APPROVED 16/08/2026)

> **Decision: obtain `train.csv` and `valid.csv` (~30 MB) before the ~11 GB of images.**

This is not caution for its own sake. The label CSV alone fully resolves **Steps 3, 4, 6,
12 and 13** — every label count, the real frontal/lateral and AP/PA breakdown, the patient
repetition statistics, the class imbalance, and the actual train/val split CSVs. Only
Steps 7, 8 and 9 (duplicates, integrity, dimensions) need pixels. It also confirms we have
the right release *before* committing the bandwidth: if the column schema or the row count
does not match §2.3 and §6.1, we have the wrong thing and should stop.

**Procedure** (**CONFIRMED** — Redivis exposes per-file download via the Files tab and
`File.download()` in its Python client):

1. Create a free **Redivis** account (`redivis.com`) — Google or institutional sign-in;
   no Stanford affiliation required.
2. Reach the CheXpert dataset from `aimi.stanford.edu/data` or the download link on
   `stanfordmlgroup.github.io/competitions/chexpert/`.
3. **Read and accept the Stanford Dataset Research Use Agreement.** Note the
   no-redistribution clause (§1.3) — it constrains §14.4.
4. Open the dataset's **Files** tab, filter to the `CheXpert-v1.0-small` folder, and
   download **only `train.csv` and `valid.csv`**.
5. Place them at `E:/UROP/data/raw/CheXpert-v1.0-small/` (the path already set as
   `dataset.root` in `config/data.yaml`).

**Do not** source these CSVs from Kaggle mirrors, HuggingFace uploads or third-party
GitHub repositories. Several exist, but they are redistributions of RUA-covered data of
unverifiable provenance — wrong on both compliance and correctness grounds.

**If the Files tab only offers the archive as a single object**, that is a hard blocker on
the CSV-only route: report back rather than downloading 11 GB unilaterally, since that
decision is the supervisor's to make, not a technical detail.

Once the two CSVs are in place, these two scripts run immediately and need no images:

```bash
python src/data/analyze_metadata.py    # Steps 3, 4, 6, 12 — all tables and figures
python src/data/make_splits.py         # Steps 5, 13, 15 — splits + manifest
```

`make_splits.py` runs in fallback mode until the official test set is acquired
(`test_set.enabled: false`), which is correct for now — the split it writes at this stage
is provisional and must be regenerated once the test images land (§13.3).

---

## 2. Step 2 — Dataset structure

**CONFIRMED** (CheXpert datasheet arXiv:2105.03020; path format cross-checked against
multiple independent reimplementations).

### 2.1 Directory layout

```text
CheXpert-v1.0-small/
├── train.csv
├── valid.csv
├── train/
│   ├── patient00001/
│   │   ├── study1/
│   │   │   ├── view1_frontal.jpg
│   │   │   └── view2_lateral.jpg
│   │   └── study2/
│   │       └── view1_frontal.jpg
│   └── patient00002/ ...
└── valid/
    └── patient64541/ ...
```

### 2.2 The hierarchy

```text
patient                      patient00001            ← the SPLIT UNIT (§5)
   ↓  1 : many
study                        study1, study2, …       ← one imaging visit; CARRIES THE LABEL
   ↓  1 : many
image                        view1_…, view2_…        ← one radiograph = one CSV row
   ↓  1 : 1
view                         Frontal / Lateral  +  AP / PA
   ↓
labels                       14 observations, shared by every image in the study
```

**The single most important structural fact: labels are assigned at the STUDY level,
not the image level.** They are extracted from the study's radiology report. Every image
in a study therefore inherits the same 14 labels. A lateral view of a
cardiomegaly-positive study is labelled positive even though a lateral film is not how
cardiomegaly is assessed. This is a known weakness (**INFERRED** from the labelling
procedure described in Irvin et al. §"Label Extraction"), and it is a further argument
for frontal-only selection (§4).

### 2.3 CSV columns

`train.csv` and `valid.csv` share one schema — **19 columns**, one row per image:

| # | Column | Type | Notes |
|---:|---|---|---|
| 1 | `Path` | str | `CheXpert-v1.0-small/train/patientNNNNN/studyN/viewN_frontal.jpg` |
| 2 | `Sex` | str | `Male` / `Female` / `Unknown` |
| 3 | `Age` | int | complete years at study time |
| 4 | `Frontal/Lateral` | str | `Frontal` / `Lateral` |
| 5 | `AP/PA` | str | projection; blank for laterals. **MEASURED value set:** `AP` / `PA` / `LL` / `RL` (§4.1) |
| 6–19 | the 14 observations | float | `1.0` / `0.0` / `-1.0` / blank |

The 14 observations, in CSV order: `No Finding`, `Enlarged Cardiomediastinum`,
**`Cardiomegaly`**, `Lung Opacity`, `Lung Lesion`, `Edema`, `Consolidation`, `Pneumonia`,
`Atelectasis`, `Pneumothorax`, `Pleural Effusion`, `Pleural Other`, `Fracture`,
`Support Devices`.

### 2.4 Identifiers

**There is no explicit patient-ID or study-ID column.** Both must be parsed out of `Path`.
This is a trap: a positional split on `/` breaks if a future release renames the root
directory. `src/data/chexpert_metadata.py::attach_identifiers` parses with regexes
(`patient(\d+)`, `study(\d+)`, `view(\d+)_(frontal|lateral)`) and **raises** on any row it
cannot parse, rather than silently producing `NaN` patient IDs that would quietly defeat
the leakage guard.

---

## 3. Step 3 — The Cardiomegaly label

### 3.1 WHAT THE DATASET SAYS

**CONFIRMED** — column name `Cardiomegaly`. Encoding (CheXpert datasheet §label values):

| State | CSV value | Meaning |
|---|---|---|
| Positive | `1.0` | report affirms cardiomegaly |
| Negative | `0.0` | report explicitly negates it |
| Uncertain | `-1.0` | report hedges ("borderline", "cannot exclude", "may represent") |
| No mention | *blank* → `NaN` in pandas | the observation is absent from the report |

**CONFIRMED** — study-level distribution over all 187,641 training studies
(Irvin et al. 2019, Table 1):

| State | Studies | % of studies |
|---|---:|---:|
| Positive | 23,002 | 12.26% |
| Uncertain | 6,597 | 3.52% |
| Negative *(incl. no-mention)* | 158,042 | 84.23% |
| **Total** | **187,641** | **100.00%** |

**INFERRED:** Table 1's three columns sum to exactly 187,641 — the full study count.
Therefore the paper's "Negative" column **merges explicit-negative and no-mention**.
This is easy to get wrong and it matters: it means treating blank as negative is not our
invention, it is the dataset paper's own accounting convention. Our config makes this
explicit (`target.blank_is_negative: true`) rather than leaving it to a silent `fillna`.

**Context — Cardiomegaly has unusually LOW label uncertainty.** Among the five CheXpert
competition tasks, its 3.52% uncertain rate is the lowest by a wide margin
(Consolidation 12.78%, Atelectasis 15.66%, Pneumonia 8.34%, Edema 6.17%). This single
fact drives the recommendation below — and it holds *even more strongly* in the file we
actually acquired (§3.3).

### 3.1a MEASURED distribution — `train_cheXbert.csv`, 21/08/2026

Everything above is the *published* description of the original release. What follows is
**measured from the file on disk** by `src/data/analyze_metadata.py`.

| State | Images | % images | Studies | % studies |
|---|---:|---:|---:|---:|
| Positive `1.0` | 30,566 | 13.68% | 25,840 | 13.77% |
| Negative `0.0` | 16,155 | 7.23% | 11,227 | 5.98% |
| Uncertain `-1.0` | 3,917 | 1.75% | 3,327 | 1.77% |
| Blank (no mention) | 172,776 | 77.33% | 147,247 | 78.47% |
| **Total** | **223,414** | **100.00%** | **187,641** | **100.00%** |

**Frontal images only (191,027) — the modelling frame:**

| State | Images | % of frontal |
|---|---:|---:|
| Positive | 26,283 | 13.76% |
| Negative | 11,456 | 6.00% |
| Uncertain | 3,372 | 1.77% |
| Blank | 149,916 | 78.48% |

This **resolves U3–U4**: the image-level and frontal-only distributions were previously
unpublished and are now measured. Negative + blank = 84.57% (images), within 0.3 pp of the
paper's 84.23% study-level figure — the two labellers agree closely on the negative mass.

### 3.3 Label provenance — `train_cheXbert.csv`, NOT the original rule-based labels

> **This is a deliberate substitution and must be stated in the report. Decision D211.**

The file acquired from Stanford AIMI is **`train_cheXbert.csv`** — the CheXpert training
set relabelled with **CheXbert** (Smit et al. 2020), a BERT-based report labeller. It is
**not** the original rule-based labeller output of Irvin et al. 2019 that `train.csv`
contains, and which §3.1's published tables describe.

**What is identical (CONFIRMED by measurement):**

| Property | Value |
|---|---|
| Images / patients / studies | 223,414 / 64,540 / 187,641 — exact match to the published release |
| Frontal / lateral | 191,027 / 32,387 — exact match |
| Columns | all 19, including `Path`, `Sex`, `Age`, `Frontal/Lateral`, `AP/PA` |
| Label encoding | same four states: `1.0` / `0.0` / `-1.0` / blank |
| `Path` values | identical, and in identical order to `train_visualCheXbert.csv` |

**What differs:** the labels themselves. Cardiomegaly uncertainty is **1.77% of studies
here vs 3.52% reported for the original labeller** — CheXbert resolves roughly half of the
original hedged cases. Positive rate is 13.77% vs 12.26%.

**Consequence for D203, stated honestly:** Irvin et al. Table 3's uncertainty-policy AUCs
were measured with the *original* labeller, so they are **indicative for this file, not
exactly applicable**. D203's reasoning nonetheless *strengthens*: with uncertainty at 1.77%
rather than 3.52%, the policy choice moves only **3,372 frontal images (1.77%)**, so the
case for the simple U-Zeros option over a task-restructuring U-MultiClass is stronger than
when the decision was made. **D203 is unchanged.**

**One schema difference worth recording:** in this file `No Finding` is the *last* column
(19) rather than the first label (6). All project code addresses observations by name, so
nothing breaks — but any positional indexing would silently mislabel.

**`train_visualCheXbert.csv` is an ablation only, never the primary** (config
`dataset.ablation_train_csv`). VisualCheXbert predicts image-derived rather than
report-derived labels: **58.33% frontal prevalence with zero uncertain and zero blank
states**. Substituting it would void D203 (no uncertainty to police) and invert D209
(`pos_weight` 0.71 — positives would be *down*weighted). It is a different task, not a
different file.

### 3.2 WHAT WE RECOMMEND

**Comparison of the four candidate policies.**

Irvin et al. Table 3 reports validation AUC per policy for Cardiomegaly
(**CONFIRMED**, 200-study validation set, 30-checkpoint ensemble):

| Policy | Cardiomegaly AUC | 95% CI | Binary task? | Verdict |
|---|---:|---|---|---|
| U-Ignore (= exclusion) | 0.828 | (0.769, 0.888) | yes | worst reported |
| U-Zeros | 0.840 | (0.783, 0.897) | yes | **recommended primary** |
| U-Ones | 0.832 | (0.773, 0.890) | yes | recommended ablation |
| U-SelfTrained | 0.831 | (0.770, 0.891) | yes | out of scope (needs a two-pass regime) |
| U-MultiClass | **0.854** | (0.800, 0.909) | **no — 3-class** | rejected, see below |

> **RECOMMENDATION: `U-Zeros` as the primary policy, with `U-Ones` and `exclusion`
> reported as sensitivity ablations. U-MultiClass is rejected.**

Four reasons, in order of weight:

1. **The reported U-MultiClass advantage is not statistically distinguishable.** Its CI
   (0.800–0.909) overlaps U-Zeros' (0.783–0.897) almost completely. The 0.014 AUC gap
   was measured on **200 studies** — roughly 25–30 cardiomegaly positives. Restructuring
   the entire pipeline on that evidence would be over-fitting to a single small-sample
   number. *This is exactly the kind of "general knowledge" decision the Stage 2 brief
   warned against; the actual numbers, including their confidence intervals, argue
   against it.*
2. **U-MultiClass breaks the binary task definition.** It requires a 3-class head. That
   propagates into every downstream component: the XGBoost head (§Stage 8) would need a
   different objective, PR-AUC and the threshold policy would need redefining, `pos_weight`
   would not apply, and the frozen-feature fusion comparison would no longer be
   apples-to-apples. Literature review v2 §3 fixes the task as **binary**. The cost of
   U-MultiClass is paid across six later stages for a gain inside the noise.
3. **The decision affects only 3.52% of studies.** Cardiomegaly is the *least* uncertain
   of the competition tasks. Whatever we choose moves a small minority of labels — which
   is precisely why the CIs overlap. Choosing the simple option costs almost nothing.
4. **U-Zeros is semantically defensible for this specific finding.** An uncertain
   cardiomegaly mention is typically "borderline cardiomegaly" or "heart size difficult
   to assess". The report did *not* affirm the finding. Mapping to 0 means "not
   affirmed", which is exactly what our target measures — agreement with a report-derived
   label, not clinical ground truth.

**Why exclusion is an ablation and not the primary:** it scored worst in the paper
(0.828), and dropping 6,597 studies removes precisely the *hard, borderline* cases —
which optimistically biases every metric. That is a real methodological cost, and it is
worth reporting as an ablation exactly so we can show it.

**Reporting rule:** all three policies are computed by `analyze_metadata.py` and the
primary is set in `config/data.yaml`. Report all three. **Do not silently adopt whichever
gives the best number** — declare U-Zeros as primary before seeing test results.

---

## 4. Step 4 — Frontal image selection

### 4.1 WHAT THE DATASET SAYS

All **MEASURED** from `train_cheXbert.csv`, 21/08/2026 — every item below was previously
VERIFY and is now settled.

- `Frontal/Lateral` ∈ {`Frontal`, `Lateral`} — **CONFIRMED**.
- **Frontal 191,027 (85.50%) / Lateral 32,387 (14.50%)** — **U1 CONFIRMED**, matching the
  previously-secondary figure exactly.
- **U2 CONFIRMED — and `LL`/`RL` do occur.** Full measured value set:

| `Frontal/Lateral` | `AP/PA` | Images | % of all |
|---|---|---:|---:|
| Frontal | AP | 161,590 | 72.33% |
| Frontal | PA | 29,420 | 13.17% |
| Frontal | **LL** | **16** | 0.01% |
| Frontal | **RL** | **1** | 0.00% |
| Lateral | *(blank)* | 32,387 | 14.50% |

- Within frontals: **AP 84.59% / PA 15.40%** — confirming the expected inpatient-heavy
  portable-AP skew that D204's confound analysis assumes.
- `AP/PA` is blank for **exactly** the 32,387 laterals and never for a frontal, so the
  blank is structural, not missing data. `select_frontal()` maps it to `Unknown`.
- **17 frontal images carry a lateral projection code** (`LL`/`RL`) — an internal
  contradiction in the source metadata. See §8.4; retained and flagged, not silently
  dropped.

### 4.2 WHAT WE RECOMMEND

> **RECOMMENDATION: `Frontal/Lateral == "Frontal"` only. Combine AP and PA into one
> training set, but retain `AP/PA` as a metadata column and report test metrics
> stratified by projection.**

**Excluding laterals** is not merely a scope choice. Cardiomegaly is assessed on the
frontal projection via the cardiothoracic ratio; the cardiac silhouette on a lateral film
does not support that measurement. Because CheXpert labels are **study-level** (§2.2),
every lateral in a positive study carries a positive label it cannot visually justify.
Including laterals would inject a block of systematically unlearnable positives.

**On combining AP and PA — this is the genuinely contested decision.**

The medical concern is real and specific: in an **AP** projection the heart lies further
from the detector and is **magnified**, so an AP film overstates cardiac size. The
conventional CTR > 0.5 threshold is defined for **erect PA** films. AP and PA are
therefore not interchangeable inputs for this task.

Worse, the confound is not just geometric. AP films are portable/supine studies of
sicker, less mobile inpatients. Projection correlates with acuity, which correlates with
cardiac disease. A model can learn *"this is a portable film, therefore this patient is
sick, therefore predict cardiomegaly"* — the same shortcut class as the Support Devices
problem already identified in literature review v2.

| Option | For | Against | Verdict |
|---|---|---|---|
| PA only | Clinically cleanest; CTR conventions apply | ~15% of frontals (**VERIFY**); loses most positives (PA is outpatient, lower prevalence); **the official valid/test sets contain both projections, so a PA-only model could not be evaluated on the expert-annotated benchmark without discarding most of it** | Rejected |
| AP + PA combined | Full data volume; matches the entire CheXpert literature; evaluable on the official test set | AP/PA is a confound | **Recommended** |
| Separate models | Isolates the confound cleanly | Doubles every experiment; halves data per model; not affordable in scope | Rejected (future work) |

The decisive argument for combining is **not** convenience — it is that our primary
evaluation set is the official 500-patient expert-annotated test set, which is mixed AP/PA.
A PA-only model would be unevaluable on the strongest ground truth available to us.

**The confound is handled by measurement, not by exclusion.** `AP/PA` is carried through
into every split CSV (`make_splits.py::KEEP_COLS`), and the Stage 4 evaluation must report:

- test ROC-AUC / PR-AUC on the **AP subset**
- test ROC-AUC / PR-AUC on the **PA subset**
- positive-prediction rate for each

If those diverge sharply, that is a **reportable finding about a projection shortcut**,
not a failure. It runs alongside the Support Devices probe using the same machinery.

---

## 5. Step 5 — Patient-level leakage

### 5.1 The risk is present and large

**CONFIRMED** — 223,414 training images across 187,641 studies from **64,540 patients**.

**INFERRED** — that is **≈ 2.91 studies per patient** and **≈ 3.46 images per patient**.

**MEASURED, 21/08/2026** (frontal images only, the actual modelling frame — **U5 resolved**):

| Statistic | Value |
|---|---:|
| Patients with ≥1 frontal image | 64,534 |
| Mean frontal images per patient | 2.96 |
| **Median** frontal images per patient | **1** |
| Max frontal images for one patient | **91** |
| Patients with >1 frontal image | **31,744 (49.19%)** |
| Patients with >1 study | 30,975 (48.00%) |
| Patients positive in any study | 12,759 (19.77%) |

The distribution is **heavily right-skewed**: the median patient contributes a single
image, but the mean is 2.96 and one patient contributes 91. Roughly **half of all patients
contribute more than one frontal image**, and those patients carry disproportionate weight
in any leakage — a random image-level split would scatter one person's 91 radiographs
across all three sets. Plotted in `figures/dataset/images_per_patient.png`.

### 5.2 Why image-level random splitting leaks

A random image-level split puts *the same person's chest* in train and test. Three
distinct failure modes compound:

1. **Anatomical identity.** Rib cage geometry, body habitus, thoracic width and spine
   shape are stable, highly distinctive, and directly visible. A CNN can memorise an
   individual and recall their label rather than learning what an enlarged heart looks
   like.
2. **Label correlation across time.** Cardiomegaly is a chronic structural finding. A
   patient positive in March is almost certainly positive in July. Memorising the patient
   *is* memorising the label — the shortcut is close to perfectly rewarded.
3. **Same-study near-copies.** Two frontal views from one visit are minutes apart and
   near-identical. An image-level split can put one in train and its twin in test.

The result is inflated, unrecoverable metrics: **the architecture ranking — the entire
point of RQ1 — degenerates into noise**, because all four models saturate on memorisation
rather than separating on their ability to learn the finding. This is the project's
single largest threat, and it is silent: nothing in the training logs looks wrong.

### 5.3 The guarantee

Enforced in code by `chexpert_metadata.py::assert_disjoint`, which raises on any overlap
and is called by `make_splits.py` before a single CSV is written:

```text
TRAIN ∩ VALIDATION = ∅
TRAIN ∩ TEST       = ∅
VALIDATION ∩ TEST  = ∅
```

Verified passing on the synthetic fixture. **Every training script must call this on the
split CSVs at load time** — an assertion that runs once at split-creation and never again
is an assertion you will eventually break by editing a CSV by hand.

### 5.4 Splitting strategy

Split unit is the **patient**, stratified on a patient-level label:
a patient is positive if **any** retained frontal study is Cardiomegaly-positive under the
primary policy. This is what lets us stratify *and* group at the same time
(`sklearn.model_selection.StratifiedGroupKFold`), so prevalence stays balanced across
splits without ever splitting a person.

The exact fractions and the use of the official test set are argued in §13.

---

## 6. Step 6 — Data distribution

### 6.1 Published expectations

**CONFIRMED** unless marked. These are what the code should reproduce; a mismatch means
we have the wrong release and must stop.

| Category | Count | Source |
|---|---:|---|
| Total patients (whole dataset) | 65,240 | Irvin et al. 2019 |
| Total images (whole dataset) | 224,316 | Irvin et al. 2019 |
| — train images | 223,414 | CheXpert datasheet |
| — valid images | 234 | CheXpert datasheet |
| — test images | 668 | CheXpert datasheet |
| Train patients | 64,540 | datasheet / independent reimplementations |
| Train studies | 187,641 | Irvin Table 1 (column sums) |
| Valid patients / studies | 200 / 200 | Irvin et al. 2019 |
| Test patients / studies | 500 / 500 | Irvin et al. 2019 |
| Frontal images (train) | 191,027 (85.5%) | ✅ **MEASURED — exact match** |
| Lateral images (train) | 32,387 (14.5%) | ✅ **MEASURED — exact match** |

**INFERRED consistency check:** 64,540 + 200 + 500 = 65,240 ✓ and
223,414 + 234 + 668 = 224,316 ✓. The published figures are internally consistent.

**MEASURED confirmation, 21/08/2026.** `train_cheXbert.csv` reproduces **every** published
training-set figure exactly: 223,414 images · 64,540 patients · 187,641 studies · 191,027
frontal · 32,387 lateral. This is the §6.1 stop-check passing — we have the right release.

### 6.2 Cardiomegaly, study level

| Category | Published (Irvin Table 1, rule-based labeller) | **MEASURED (`train_cheXbert.csv`)** |
|---|---:|---:|
| Total studies | 187,641 (100.00%) | 187,641 (100.00%) |
| Positive | 23,002 (12.26%) | **25,840 (13.77%)** |
| Uncertain | 6,597 (3.52%) | **3,327 (1.77%)** |
| Negative + no-mention | 158,042 (84.23%) | **158,474 (84.46%)** |

The two labellers agree on the **negative mass to within 0.23 pp**. CheXbert's difference
is concentrated exactly where expected: it resolves ~half the original hedged cases,
moving them predominantly to positive (§3.3).

### 6.3 Usable frontal examples — **MEASURED**

| Policy | Usable frontal images | Positive | Negative | Prevalence | Pos : Neg | `pos_weight` |
|---|---:|---:|---:|---:|---:|---:|
| **U-Zeros** *(primary)* | **191,027** | **26,283** | **164,744** | **13.76%** | **1 : 6.27** | **6.268** |
| U-Ones | 191,027 | 29,655 | 161,372 | 15.52% | 1 : 5.44 | 5.442 |
| Exclusion | 187,655 | 26,283 | 161,372 | 14.01% | 1 : 6.14 | 6.140 |

The pre-measurement projection was ~12.3% prevalence at ~1:7.2; **measured 13.76% at
1:6.27**. The projection assumed prevalence is independent of how many frontal images a
study contributes — that assumption is now tested and holds well (study-level 13.77% vs
image-level frontal 13.76%, a 0.01 pp difference). **U3–U4 resolved.**

### 6.4 Class imbalance verdict — **CONFIRMED**

**1 positive to 6.27 negatives is moderate, not severe** — slightly *better* than the ~1:7.2
projected. §12's conclusion stands unchanged and is now measured rather than estimated:
severe-imbalance machinery (focal loss, heavy oversampling, SMOTE) is not warranted and
would add tunable hyperparameters that confound the architecture comparison.

### 6.5 Plots produced

`analyze_metadata.py` writes to `artifacts/stage2/figures/`:

| Figure | Shows |
|---|---|
| `cardiomegaly_label_states.png` | the four raw label states, image level |
| `view_breakdown.png` | images by `Frontal/Lateral` × `AP/PA` |
| `uncertainty_policy_balance.png` | positive/negative counts under all three policies |
| `images_per_patient.png` | patient repetition — the visual leakage argument |

---

## 7. Step 7 — Duplicates

### 7.1 Five distinct things, only two of them defects

| Level | Kind | Detection | Is it a defect? | Action |
|---|---|---|---|---|
| L1 | Exact duplicate file | SHA-1 over bytes | **Yes** | flag; keep one copy after review |
| L2 | Same image, several paths | same SHA-1, different `Path` | **Yes** | flag |
| L3 | Same study, several frontal views | group by `study_id` | **No** — separate radiographs | **keep both** |
| L4 | Same patient, repeated studies | group by `patient_id` | **No** — genuine follow-up | **keep all** |
| L5 | Near-duplicate images | dHash + Hamming distance | **Only across patients** | flag cross-patient only |

### 7.2 DUPLICATE IMAGE vs LEGITIMATE MULTIPLE STUDIES

This distinction decides whether deleting data is right or destructive:

- A **duplicate image** is the same pixels stored twice. It adds no information, silently
  reweights the loss toward whatever it depicts, and — if it spans two patient IDs —
  **defeats patient-level splitting entirely**.
- **Legitimate multiple studies** are a real person imaged on different days. They are the
  clinical reality the dataset is meant to represent. Deleting them would discard genuine
  signal *and would not improve leakage safety by one bit*, because **patient-level
  splitting already contains them completely**.

> **Patient-level splitting — not deduplication — is the primary protection against
> leakage.** Deduplication addresses a different, narrower problem: identical content
> appearing under two identities.

### 7.3 Is exact detection computationally practical? Yes — implemented

SHA-1 over ~191,000 JPEGs totalling ~11 GB is I/O-bound, not CPU-bound. With 12 threads on
a 14-core/20-thread CPU this is a few minutes. **Implemented** in
`src/data/find_duplicates.py`, verified on the fixture.

### 7.4 Near-duplicates — implemented, with a scoping insight

Naive pairwise comparison of 191k images is ~1.8 × 10¹⁰ pairs — infeasible. Two
observations make it tractable:

1. **Within-patient near-duplicates do not matter.** They are expected (serial films of
   one person) and are already neutralised by patient-level splitting. Only
   **cross-patient** near-duplicates threaten the experiment.
2. Bucketing 64-bit dHashes by their first 16 bits and comparing only within buckets turns
   the sweep into a group-by that finishes in minutes.

`find_duplicates.py --near` does exactly this and reports **only cross-patient pairs**.
It is off by default so the fast exact check can run first; it is a documented option,
not an unexplained omission.

### 7.5 Expected findings

**VERIFY** — an independent manual review of CheXpert
(`github.com/ayhyap/CheXpert-review`, an unofficial amateur audit — treat as indicative,
not authoritative) reported **688 images (0.3%)** warranting exclusion. We expect exact
duplicates to be rare. **Nothing is deleted**; findings go to
`reports/duplicates_exact.csv` for a human decision that gets recorded.

---

## 8. Step 8 — Corrupted and unreadable images

### 8.1 The validation step

`src/data/validate_images.py` opens **every retained frontal image** and records:

| Check | Method | Status on failure |
|---|---|---|
| file exists | path resolution | `missing` |
| image opens | `PIL.Image.open` header parse | `unreadable` |
| dimensions valid | `im.size` both > 0 | `invalid_dims` |
| pixel data readable | forced full decode to a numpy array | `unreadable` |
| not blank | `min(pixels) != max(pixels)` | `constant` |

Two settings make this strict rather than cosmetic:

- `ImageFile.LOAD_TRUNCATED_IMAGES = False` — Pillow's default is to **silently pad a
  truncated JPEG with grey**. Left at the default, a corrupt file trains happily as a
  half-grey image and is never detected.
- `DecompressionBombWarning` is promoted to an error, so an absurdly-sized file is caught
  rather than exhausting RAM.

### 8.2 Reporting

`reports/image_validation.md`:

| Status | Count | Percentage |
|---|---:|---:|
| valid | — | — |
| missing | — | — |
| unreadable | — | — |
| invalid_dims | — | — |
| constant | — | — |
| **total checked** | — | — |

### 8.3 Metadata-level data quality — **MEASURED, 21/08/2026**

Checks that need only the CSV, run before any image exists. Overall the metadata is
**clean**; four items are worth recording.

| Check | Result | Assessment |
|---|---|---|
| Duplicate `Path` rows | **0** of 223,414 | clean |
| Missing `Path` / `Sex` / `Age` / `Frontal/Lateral` | **0** | clean |
| Missing `AP/PA` | 32,387 — **exactly** the laterals, never a frontal | structural, not missing data |
| `Path` filename vs `Frontal/Lateral` column | **0 disagreements** | clean — view is recoverable from either |
| Lateral rows with a non-blank `AP/PA` | **0** | clean |
| **Frontal rows with `LL`/`RL` projection** | **17** | ⚠️ **internal contradiction** |
| **`Age == 0`** | **3** | ⚠️ implausible; almost certainly "unknown" encoded as 0 |
| **`Age > 89`** | **7,579** (max **110**) | ⚠️ not HIPAA-capped at 90 |
| **`Sex == "Unknown"`** | **1** | ⚠️ trivial |
| Patients with no frontal image | **6** (lateral-only) | excluded by design |
| Studies with >1 frontal image | 3,361 of 187,625 (max 3) | legitimate L3 repetition (§7.2) |

**The 17 `Frontal` + `LL`/`RL` rows.** The `Frontal/Lateral` column says frontal, the
filename says `..._frontal.jpg`, but the projection code is a *lateral* one. The source
metadata contradicts itself. **Disposition: retain and flag.** They are 0.009% of frontals
and `select_frontal()` already carries them through as their own `AP/PA` category, so they
are visible in every stratified report rather than silently merged into AP or PA. Dropping
them would be defensible too, but retaining is the lower-risk default — no data is
discarded on the strength of a 17-row anomaly, and they can be excluded later by a one-line
filter if the projection-stratified analysis warrants it.

**Age.** 3 zeros and 7,579 records above 89 (up to 110). Age is not a model input, so
neither affects training. Both matter only if we report demographic breakdowns, where the
zeros should be treated as missing and the >89 tail as unreliable.

### 8.4 Nothing is deleted

Failures are written to **`reports/excluded_images.csv`** with `Path`, `status` and a
`reason` string. The training pipeline reads that manifest and skips those rows. The files
stay on disk, the exclusion list is version-controlled, and the count goes in the report's
limitations section.

**What this check cannot catch:** the manual review cited in §7.5 found 244 "striped"
images (shifted rows), 354 with incomplete chest coverage, 6 non-chest scans, and 155
needing rotation correction. These decode perfectly and will pass every automated check
here. Detecting them requires visual review, which is **out of Stage 2 scope** — but it
must be named as a known residual data-quality limitation in the final report rather than
left unmentioned.

---

## 9. Step 9 — Image dimensions and model input size

### 9.1 WHAT THE DATASET SAYS

**CONFIRMED** (CheXpert datasheet) — the small release is downsampled to approximately
**390 × 320**, preserving each image's original aspect ratio, at 256 grayscale levels,
JPEG. **INFERRED:** since aspect ratio is preserved and one dimension is quoted as 320,
the **short side is fixed at 320 px** and the long side varies.

**VERIFY** — min / max / median width and height, the exact common-dimension table, and
the aspect-ratio distribution. `validate_images.py` computes all of these and plots
`figures/image_dimensions.png`.

### 9.2 WHAT WE RECOMMEND

> **RECOMMENDATION: 320 × 320.**

| Candidate | Argument for | Argument against | Verdict |
|---|---|---|---|
| 224 × 224 | ImageNet-native; cheapest; ~2× throughput of 320 | **Discards ~51% of available pixels.** Cardiomegaly is judged by the cardiac border against the thoracic wall — a boundary-localisation task that degrades as edges blur | Rejected as primary; kept as a **speed ablation** |
| **320 × 320** | **Exactly the native short side** — no upsampling, no waste; **what Irvin et al. used**, aiding comparability; fits 8 GB VRAM with AMP | ~2× the compute of 224 | **Recommended** |
| 384 × 384 | Higher nominal resolution | **Would upsample every image from 320.** Interpolation adds zero information while costing ~44% more compute than 320 | Rejected |

The reasoning is not "224 is the default" or "bigger is better". It is that **320 is where
the actual information in this release ends.** Below it we throw away pixels we have;
above it we spend compute on pixels we do not have.

Cross-checks:

- **Literature.** The CheXpert baseline trained at 320 × 320 and its cardiomegaly results
  are the numbers we will sanity-check against.
- **GPU memory.** 320 × 320 with AMP fits all four architectures on 8 GB (§14) —
  confirmed by `probe_vram.py`, not assumed.
- **Information loss.** Zero downsampling relative to the release.
- **Architectures.** All four accept arbitrary input sizes via adaptive pooling; none
  requires 224.

**If `probe_vram.py` shows ResNet152 cannot reach a workable micro-batch at 320**, the
fallback is **gradient accumulation at 320**, not a drop to 224 — changing resolution
changes the task for every model and would have to be applied to all four to stay fair.

---

## 10. Step 10 — Preprocessing

One pipeline, **byte-identical across all four baselines**. Any per-model deviation
invalidates RQ1.

| Step | Decision | Reasoning |
|---|---|---|
| **Load** | PIL, JPEG → single-channel `L` | The release is already 8-bit grayscale. No DICOM windowing, no rescale slope/intercept — a real advantage of this release over CheXpert Plus |
| **Resize** | direct bilinear to **320 × 320** | See below |
| **Channels** | **replicate the grayscale channel 3×** | See below |
| **Scale** | `/255` → `[0,1]` | standard |
| **Normalize** | ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` | All four backbones are ImageNet-pretrained; their first-layer statistics expect this |
| **Crop** | **none** | See below |
| **CLAHE / histogram equalisation** | **not in core** | See below |

### 10.1 Resize: why direct, not centre-crop

**Cropping is rejected outright.** Cardiomegaly is a *ratio* — cardiac width against
thoracic width. Cropping risks removing a lateral chest wall, which deletes the
denominator of the very measurement the task depends on. This is the same reason
literature review v2 rejects lung-field ROI extraction.

That leaves direct resize to 320 × 320 versus aspect-preserving resize + zero-padding.
**Direct resize is recommended**, on a specific technical ground: a purely horizontal
rescale multiplies cardiac width and thoracic width by *the same* factor, so the
cardiothoracic **ratio is invariant** under it. The clinically meaningful quantity
survives. Padding, by contrast, introduces black borders whose size correlates with the
original aspect ratio — a spurious feature a CNN can read. Direct resize also matches the
dominant CheXpert convention.

`config/data.yaml` exposes `preprocessing.resize_mode: direct | aspect_pad`, so the
alternative is a one-line ablation rather than a rewrite. **Fix this once and never vary
it between models.**

### 10.2 Grayscale → RGB: the cleanest 3-channel method

Two options exist for feeding 1-channel data to ImageNet-pretrained backbones:

1. **Replicate the channel 3× (recommended).** The pretrained first-conv weights are used
   exactly as trained. Identical, trivial, and provably consistent across all four
   architectures.
2. Sum the first conv layer's weights across the input dimension to make a 1-channel stem.
   Saves a little memory and compute, but **changes the pretrained filter statistics**, and
   would have to be re-derived separately for four different stem designs (ResNet's 7×7,
   DenseNet's 7×7, EfficientNetV2's fused-MBConv stem, MobileNetV3's hard-swish stem).
   Four bespoke surgeries is four chances to introduce an architecture-specific difference
   into what is supposed to be a controlled comparison.

**Replication wins on experimental-control grounds, not just simplicity.**

### 10.3 CLAHE: deliberately excluded

Literature review v2 §7 is explicit that CLAHE must not silently become a preprocessing
trick. It is excluded from core preprocessing. It may be run later as a **single
documented ablation on the best model only**, reported whether it helps or not. It is not
added because it is popular in CXR papers.

---

## 11. Step 11 — Data augmentation

**Training split only.** Validation and test receive **resize → to-RGB → normalize** and
nothing else. No TTA in the core protocol. Enforced by constructing two separate transform
objects — never one transform with an `is_train` flag that can be passed wrongly.

| Augmentation | Recommended? | Reason | Risk |
|---|---|---|---|
| **Rotation ±10°** | **Yes** | Patient positioning and detector alignment genuinely vary this much between films | Beyond ~15° the cardiac silhouette's orientation relative to the thoracic cage stops resembling any real radiograph |
| **Horizontal flip** | **No** | See §11.1 | Produces an anatomically impossible body plan |
| **Zoom / scale, isotropic 0.9–1.0** | **Yes** | Source-to-detector distance genuinely varies; isotropic scaling **preserves the cardiothoracic ratio** | Must be isotropic. **Anisotropic aspect-ratio jitter is forbidden** — it changes CTR directly, i.e. it changes the label |
| **Translation ±5%** | **Yes** | Centring varies between films | Must **pad**, never crop out — losing a chest wall removes the CTR denominator (§10.1) |
| **Brightness ±10%** | **Yes** | Exposure genuinely varies between machines and studies | Extreme values wash out the cardiac border |
| **Contrast ±10%** | **Yes** | Same rationale | Beyond ~20% the mediastinal border can be obliterated |
| Vertical flip | **No** | No radiograph is ever upside-down | Anatomically impossible |
| Shear / elastic / perspective | **No** | Deforms the cardiac silhouette non-physically | **Directly alters the measured quantity** |
| Cutout / random erasing | **No** | May erase the cardiac border itself | Can delete the finding while keeping the positive label |
| MixUp / CutMix | **No** | Blends two chests into an anatomically meaningless image | Also blends labels — incompatible with our clean binary framing |

### 11.1 Horizontal flip — the required justification

Literature review v2 §7 states that flipping is "not established by this review as
universally harmful or universally safe" and that it is a methodological choice
**requiring justification**. Here is ours.

> **Recommendation: do NOT horizontally flip in the core protocol.**

1. **Cardiac position is left-sided and that is diagnostic information.** Situs inversus /
   dextrocardia occurs in roughly 1 in 10,000 people. Flipping half the training images
   fabricates a population in which ~50% have dextrocardia — a distribution that exists
   neither in CheXpert nor in deployment.
2. **The task is defined by the cardiac silhouette's position and extent relative to the
   thorax.** That relationship is precisely what flipping destroys. This is unlike, say,
   pneumothorax detection, where a lesion is genuinely lateralisable and flipping is
   more defensible — which is why the literature is split, and why the answer is
   pathology-specific rather than universal.
3. **CheXpert images carry laterality markers** (L/R annotations, and the manual review in
   §7.5 found 155 images with non-standard orientation). Mirroring produces reversed
   markers — a visible artifact present only in training data.

**Ablation:** because the review declined to settle this, run flip-on as a single
documented ablation on the best model and report the result either way. That converts a
contested assumption into a small piece of evidence.

---

## 12. Step 12 — Class imbalance

### 12.1 The actual distribution

**~12.3% positive, ~7.2 negatives per positive** (§6.3) — moderate. For calibration:
severe medical-imaging imbalance means 1:100 or worse. At 1:7 the minority class still has
**~23,000 positive frontal images**, which is a large absolute number.

### 12.2 Strategy comparison

| Strategy | Mechanism | Cost to a controlled comparison | Verdict |
|---|---|---|---|
| **Class weighting (`pos_weight`)** | reweights the loss term | **None** — data pipeline stays byte-identical across models; **zero new hyperparameters** (`pos_weight = N_neg / N_pos`, computed from train only) | **Recommended** |
| Weighted sampling | rebalances each epoch's composition | Changes what "an epoch" means; adds sampling stochasticity that interacts with augmentation and complicates seed control | Rejected |
| Focal loss | down-weights easy examples | Adds **two** tunable hyperparameters (α, γ). Tuning them per architecture confounds RQ1; not tuning them is arbitrary. Designed for ~1:1000 regimes, not 1:7 | Rejected |
| Oversampling / SMOTE | duplicates or synthesises minority examples | Duplication risks memorisation and inflates epoch time; SMOTE on natural images produces non-anatomical blends | Rejected |

> **RECOMMENDATION: one primary strategy — `pos_weight` in
> `torch.nn.BCEWithLogitsLoss`, computed from the TRAINING split only.**

`make_splits.py` already writes it to `split_manifest.json` (`pos_weight_from_train`), so
it is fixed, recorded and reproducible rather than recomputed ad hoc in each script.

Two supporting notes:

- **Computing it from train only is not pedantry.** Deriving `pos_weight` from the full
  dataset leaks validation and test prevalence into training.
- **A validation-selected decision threshold does most of the work for threshold-based
  metrics, and is mandatory regardless of the loss.** Never evaluate F1 or recall at a
  hard-coded 0.5 under imbalance. Select the threshold on the validation split, freeze it,
  then apply it once to test. `pos_weight` also distorts calibration — another reason the
  threshold must be chosen empirically rather than assumed.
- Plain unweighted BCE is worth **one ablation run**; at 1:7 the gap is often small, and
  a null result there is informative.

### 12.3 How imbalance affects each metric

At 12.3% prevalence:

| Metric | Effect | Consequence |
|---|---|---|
| **Accuracy** | A model that predicts "negative" for every image scores **87.7%** | Near-useless. Must never be reported alone — literature review v2 §8 and the worklet §19 both require this |
| **Precision** | Falls as prevalence falls, for a fixed classifier quality | Not comparable across datasets with different prevalence |
| **Recall / Sensitivity** | Conditioned on positives only — **prevalence-invariant** | Comparable across datasets. Report it |
| **F1** | Harmonic mean of precision and recall → inherits precision's prevalence dependence | Report, but never compare to a paper with different prevalence |
| **ROC-AUC** | **Prevalence-invariant**, which is its strength and its trap. FPR's denominator is the large negative class, so hundreds of false positives barely move the curve | Can look reassuring while the model is unusable at any operating point |
| **PR-AUC** | Prevalence-**dependent**; the no-skill baseline is **0.123**, not 0.5. Directly sensitive to false positives among the minority class | **The honest headline metric here.** Always report the 0.123 baseline beside it |

**Reporting rule:** ROC-AUC **and** PR-AUC together, with the PR-AUC baseline stated, plus
recall, specificity, precision and F1 at the pre-declared validation-selected threshold,
plus the confusion matrix. This is already the requirement in literature review v2 §8;
§12.3 explains *why* each element is there.

---

## 13. Step 13 — Final split methodology

### 13.1 The official splits are not symmetric in quality

| Set | Size | Label quality | Usable as… |
|---|---|---|---|
| official train | 223,414 images / 64,540 patients | **weak** — NLP-derived from reports | training pool |
| official valid | 234 images / 200 patients | **strong** — 3-radiologist majority | too small for model selection |
| official test | 668 images / 500 patients | **strongest** — 5-radiologist consensus | **the test set** |

The official validation set is **200 studies**, of which roughly 25–30 are cardiomegaly
positive. Early-stopping and Optuna decisions made on ~30 positives would be dominated by
sampling noise, and every model would be selected on a different accident. It is excellent
*evidence* and a poor *tuning signal*.

### 13.2 WHAT WE RECOMMEND

> **RECOMMENDATION: use the official test set as the test set. Carve validation from the
> official training patients. Do not use the 70/15/15 default.**

| Split | Source | Patients | Purpose |
|---|---|---:|---|
| **train** | official train patients, 85% | ~54,900 | training only |
| **val** | official train patients, 15% | ~9,700 | early stopping, checkpoint selection, Optuna, **threshold selection** |
| **test** | **official CheXpert test set** | 500 | **touched once**, at the very end |
| *(extra)* | official valid set | 200 | a second expert-labelled evaluation, reported, **never tuned on** |

**Why not 70/15/15?** The brief invited it but told us not to adopt it if the official
split or the literature provides something stronger. It does, on two counts:

1. **A 5-radiologist consensus test set is dramatically stronger ground truth than any
   test split we could carve from NLP-derived training labels.** Carving our own test set
   would mean measuring agreement with a noisy report labeller and calling it performance.
   Literature review v2 §3 flags exactly this limitation; using the official test set is
   the one available mitigation.
2. **It is the standard CheXpert benchmark**, so our numbers are comparable to published
   work — one of the few comparisons that will be methodologically valid.

Holding out 15% of ~64,500 patients gives ~9,700 validation patients (~28,000 frontal
images, ~3,400 positives) — far more than enough for stable model selection, while costing
little training data.

### 13.3 Contingency

If the CheXlocalize test images cannot be obtained, `test_set.enabled: false` switches
`make_splits.py` to a **70 / 15 / 15 patient-level** carve of the training pool
(config `split.fallback`), keeping the official 200-patient valid set as an extra
expert-labelled check. **If this path is taken it must be stated prominently in the
report**, because it materially weakens the ground truth. The console prints a warning and
`split_manifest.json` records `"mode": "fallback-carved-test"` so it cannot happen silently.

### 13.4 The exact methodology

| Parameter | Value |
|---|---|
| Split unit | **patient** (`config: split.unit`, validated at runtime) |
| Random seed | **42** |
| Fractions | 85 / 15 of official train patients; official test used as-is |
| Grouping | `patient_id` parsed from `Path` |
| Stratification | **yes** — patient-level "any positive study", via `StratifiedGroupKFold` |
| Official valid/test used? | **yes** — test as primary held-out; valid as secondary evidence |
| Uncertainty handling | applied **before** splitting, so stratification reflects the actual target |
| Disjointness | asserted in code before any CSV is written |

**On stratifying without breaking grouping** — the brief asked us to investigate this.
It is achievable and is implemented. `StratifiedGroupKFold` takes the patient-level
"any-positive" flag as the stratification target and `patient_id` as the group, satisfying
both constraints simultaneously. Note the consequence: because a positive *patient* may
contribute a mix of positive and negative *images*, image-level prevalence will vary by a
few tenths of a percent between splits. That is expected and correct — it is the price of
never splitting a person, and it is recorded per split in the manifest.

One implementation detail worth recording, because it was a real bug found during testing:
setting `n_splits = round(1/frac)` in `StratifiedGroupKFold` does **not** produce the
requested fraction unless `1/frac` is an integer — asking for 15% that way yields 1/7 ≈
14.3%. `make_splits.py` instead cuts a fixed 20 folds and unions `round(frac × 20)` of
them, so 0.15 means exactly 3/20.

### 13.4a The split as actually produced — **MEASURED, 21/08/2026**

> **Mode: `fallback-carved-test` (70/15/15), NOT the recommended official-test split.**
> The official 500-patient test set has not been acquired, so `test_set.enabled: false` and
> `make_splits.py` carved a test set from the training patients, printing a warning and
> recording `"mode": "fallback-carved-test"` in the manifest. **This split is provisional
> and must be regenerated once the official test set is obtained** (§13.3).

| Split | Patients | Studies | Images | Positive | Prevalence | AP / PA / LL / RL |
|---|---:|---:|---:|---:|---:|---|
| **train** *(T1 subset applied)* | 13,659 | 39,285 | **40,002** | 5,520 | **13.80%** | 33,912 / 6,086 / 4 / 0 |
| **val** | 9,680 | 28,862 | 29,380 | 4,011 | 13.65% | 24,871 / 4,507 / 2 / 0 |
| **test** | 9,682 | 28,377 | 28,883 | 3,973 | 13.76% | 24,585 / 4,295 / 2 / 1 |

- **Patient-disjointness verified** — `assert_disjoint` passed before any CSV was written.
- **Prevalence is well balanced across splits**: 13.80% / 13.65% / 13.76%, a spread of
  0.15 pp against a pool value of 13.76%. Patient-level stratification worked as designed.
- **`pos_weight` (train only) = 6.246739** — computed from the *subsetted* train split, not
  the pool, and recorded in the manifest.
- Train patients before subsetting: 45,172 (132,764 frontal images). The T1 subset reduced
  this to 13,659 patients / 40,002 images; **val and test were untouched**, as designed.

### 13.5 Reproducibility artifacts

`make_splits.py` writes to `artifacts/stage2/splits/`:

- `train.csv`, `val.csv`, `test.csv` — one row per image, carrying `Path`, `patient_id`,
  `study_id`, `Sex`, `Age`, `AP/PA`, raw `Cardiomegaly`, `Support Devices`, and the binary
  `target`.
- `split_manifest.json` — seed, policy, fractions, mode, per-split counts and prevalence,
  projection breakdown, `pos_weight_from_train`, and a **SHA-1 of each split CSV**.

**Commit all four files to version control.** The SHA-1s are what let a result be traced
to the exact data it was trained on.

---

## 14. Step 14 — Compute feasibility

### 14.1 Available hardware — CONFIRMED (measured on this machine, 16/08/2026)

| Component | Spec |
|---|---|
| **GPU** | **NVIDIA GeForce RTX 4060 Laptop GPU — 8,188 MiB (~8 GB) VRAM**, 120 W cap |
| Driver / CUDA | 596.08 / CUDA 13.2 capable |
| CPU | Intel i7-13650HX — **14 cores / 20 threads** |
| RAM | **31.7 GiB** |
| Storage | `E:` 301.2 GB free · `C:` 90.6 GB free · `F:` 102.2 GB free |
| Python | **3.13.0** |
| PyTorch | **2.13.0+cu126**, torchvision 0.28.0+cu126 — installed 21/08/2026, CUDA available, compute capability **8.9** (Ada) |
| Driver quirk | **CUDA System Memory Fallback is ON** (Windows driver default) — see §14.2.1 |

**This resolves the "we do not have an A100" instruction concretely: 8 GB VRAM is the
binding constraint on this project.** It is enough, but only with mixed precision and
gradient accumulation.

### 14.2 Architecture requirements — **MEASURED**

Parameter counts are **CONFIRMED** (torchvision reference implementations). VRAM and
throughput are **CONFIRMED BY MEASUREMENT** on this exact card — no longer estimates.

**Measurement provenance:** `src/data/probe_vram.py`, 21/08/2026, torch 2.13.0+cu126,
CUDA 12.6, RTX 4060 Laptop (8.00 GiB, 6.94 GiB free at probe start), 320 × 320, AMP on,
real forward+backward+AdamW steps on synthetic batches, each model in its own subprocess.
Raw data: `artifacts/stage2/reports/vram_probe.json`.

#### 14.2.1 The Windows system-memory-fallback trap — **CONFIRMED, and it invalidates naive probing**

The first probe run reported ResNet152 surviving **batch 64**, which is impossible on 8 GB.
It was not fitting — it was **spilling into host RAM**.

Recent NVIDIA Windows drivers enable **CUDA System Memory Fallback by default**. When VRAM
is exhausted, allocations silently migrate to system RAM across PCIe instead of raising
`OutOfMemoryError`. Training continues and produces correct results, ~10× slower.

The measured signature is unambiguous:

| Model | Batch | Reserved | Throughput | vs peak |
|---|---:|---:|---:|---:|
| ResNet152 | 32 | 6.44 GiB | 81.0 img/s | 96% |
| ResNet152 | **48** | **9.36 GiB** | **7.7 img/s** | **9%** |
| DenseNet201 | 32 | 7.23 GiB | 86.7 img/s | 100% |
| DenseNet201 | **48** | **10.56 GiB** | **7.3 img/s** | **8%** |
| MobileNetV3-L | 128 | 6.96 GiB | 498.7 img/s | 94% |
| MobileNetV3-L | **192** | **10.37 GiB** | **80.1 img/s** | **15%** |

**Reserved memory exceeding the card's 8.00 GiB total is the proof** — 10.56 GiB cannot
exist in 8 GiB of VRAM. Throughput collapsing to 8–15% at exactly those batches is the
consequence.

> **Any "grow the batch until it OOMs" probe is worthless on this machine.** It never OOMs;
> it silently degrades. The probe therefore reports the largest **efficient** batch —
> throughput within 90% of that model's peak — not the largest batch that avoids a crash.

#### 14.2.2 Measured results

| Model | Params | Max **efficient** batch | Reserved @ that batch | Throughput | Batch 16 | Batch 32 |
|---|---:|---:|---:|---:|---:|---:|
| ResNet152 | 60.2 M | **32** | 6.44 GiB | 81.0 img/s | 84.0 img/s @ 3.55 GiB | 81.0 img/s @ 6.44 GiB |
| DenseNet201 | 20.0 M | **32** | **7.23 GiB** | 86.7 img/s | 61.0 img/s @ 3.80 GiB | 86.7 img/s @ 7.23 GiB |
| EfficientNetV2-S | 21.5 M | **32** | 5.46 GiB | 144.0 img/s | 152.1 img/s @ 3.05 GiB | 144.0 img/s @ 5.46 GiB |
| MobileNetV3-Large | 5.5 M | **128** | 6.96 GiB | 498.7 img/s | 486.1 img/s @ 1.00 GiB | 530.4 img/s @ 1.82 GiB |

**The DenseNet prediction held.** §14.2's original note — "few parameters but dense
concatenation makes it activation-hungry; memory does not track parameter count" — is
confirmed: DenseNet201 has **one third** of ResNet152's parameters yet consumes **more**
memory (7.23 vs 6.44 GiB at batch 32), the highest of all four.

#### 14.2.3 The recommended configuration — **micro-batch 16, accumulation 2, all four models**

> **REVISED from the pre-measurement plan.** The earlier text said micro-batches "differ by
> up to 8×" and prescribed per-model accumulation. Measurement shows the *efficient*
> batches are 32/32/32/128 — a 4× spread — and that a **single uniform configuration works
> for every model**.

| Model | Micro-batch | Accum steps | Effective batch | Reserved | Headroom vs 6.94 GiB free |
|---|---:|---:|---:|---:|---:|
| ResNet152 | 16 | 2 | 32 | 3.55 GiB | 3.39 GiB |
| DenseNet201 | 16 | 2 | 32 | 3.80 GiB | 3.14 GiB |
| EfficientNetV2-S | 16 | 2 | 32 | 3.05 GiB | 3.89 GiB |
| MobileNetV3-Large | 16 | 2 | 32 | 1.00 GiB | 5.94 GiB |

Three reasons this beats running each model at its own maximum:

1. **DenseNet201 at micro-batch 32 reserves 7.23 GiB against 6.94 GiB free** — it is
   *already marginally spilling*, and under real training (DataLoader workers, pinned
   memory, fragmentation over thousands of steps) it would spill hard. Micro-batch 16
   halves that to 3.80 GiB.
2. **Throughput barely changes.** Between batch 16 and 32 the heavy models move by a few
   percent, and ResNet152 and EfficientNetV2-S are actually *faster* at 16. The safety
   margin is nearly free.
3. **Identical configuration across all four models** is exactly what D020 demands — batch
   size cannot confound RQ1 if it is literally the same number everywhere.

### 14.3 Training cost — **MEASURED**

Recomputed from measured throughput at the recommended micro-batch 16, 12 epochs:

| Model | Measured img/s | T1 pool (~40,000/epoch) | Full pool (~162,000/epoch) |
|---|---:|---:|---:|
| ResNet152 | 84.0 | 7.9 min/epoch → **1.6 h** | 32 min/epoch → **6.4 h** |
| DenseNet201 | 61.0 | 10.9 min/epoch → **2.2 h** | 44 min/epoch → **8.9 h** |
| EfficientNetV2-S | 152.1 | 4.4 min/epoch → **0.9 h** | 18 min/epoch → **3.5 h** |
| MobileNetV3-Large | 486.1 | 1.4 min/epoch → **0.3 h** | 5.6 min/epoch → **1.1 h** |
| **Four baselines** | — | **~5.0 h** | **~20 h** |

The pre-measurement estimate for the four baselines at T1 was ~5.1 h; **measured ~5.0 h**.
The full-pool estimate was ~17 h at 10 epochs; **measured ~20 h at 12 epochs** — consistent.
The §15.2 budget therefore stands without revision.

**One caveat these numbers do not capture.** They are pure GPU compute on synthetic
tensors. Real epochs also decode ~40,000 JPEGs, which is CPU work. With 20 logical cores
this should keep the GPU fed for the heavy models, but **MobileNetV3 at 486 img/s may
become data-loader-bound rather than GPU-bound**. That will show as low GPU utilisation
during Stage 4 and is a tuning problem (`num_workers`, `persistent_workers`), not a
capacity problem. It cannot be measured until real images exist.

> **Verdict — now measured, not estimated: the project is feasible on this hardware.**
> T1 four-baseline cost is **~5 GPU-hours**, comfortably inside the ~25–30 h budget for the
> 4-week horizon. Full data scale (~20 h for baselines alone, ~50–70 h with Optuna and
> ablations) remains outside it, so §15's subset strategy is still a requirement.

### 14.4 Training environments

**PRIMARY — the local RTX 4060.**

- No session limits, no re-uploading 11 GB, no queue.
- **The data never leaves the machine**, which cleanly satisfies the RUA's
  no-redistribution clause (§1.3). This is a compliance argument, not a convenience one.
- Ada Lovelace: full AMP (bf16/fp16) and TF32 support.
- Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
  (the cu126 wheels run fine on a newer driver; Python 3.13 needs torch ≥ 2.6).

**BACKUP — Kaggle Notebooks.**

- Free P100 (16 GB) or T4 ×2, ~30 GPU-hours/week, 12 h max session, persistent datasets.
- **Double the VRAM of the local card**, which removes gradient accumulation for the heavy
  models and roughly halves ResNet152's wall-clock.
- **Check the RUA before uploading** (§1.3). If the terms disallow it, the backup becomes
  *a subset small enough to be defensible*, or Colab under the same constraint — flag this
  to the supervisor rather than deciding it unilaterally.
- Preferred over Colab free tier, which disconnects aggressively and offers no persistent
  storage.

### 14.5 Windows-specific practicalities

- PyTorch `DataLoader` workers use **spawn** on Windows, so each worker re-imports the
  module. Start at `num_workers=8`, set `persistent_workers=True` and
  `pin_memory=True`, and **guard the entry point with `if __name__ == "__main__":`** —
  without it, worker spawning recurses and the run fails in a confusing way.
- Decoding ~162k JPEGs per epoch is CPU work; with 20 threads this should keep the 4060
  fed, but if GPU utilisation sits low, the data loader is the bottleneck, not the GPU.

### 14.6 Planning horizon

**4 weeks** is the confirmed planning horizon (16/08/2026); the actual UROP deadline is
not yet fixed. §15 is sized for four weeks and structured to scale up by changing one
config value if more time appears (§15.5).

---

## 15. Step 15 — Data subset strategy

### 15.1 The principle

> Subsetting must never be "take the first N rows" or an unseeded `df.sample()`. It must
> preserve **class distribution**, **patient-level separation**, and **reproducibility** —
> and it must never touch validation or test.

### 15.2 Recommended tiers — sized for a 4-week horizon

**Planning horizon: 4 weeks** (confirmed 16/08/2026). The actual UROP deadline is not
fixed, so the tiers are built to scale up without rework — see §15.6.

| Tier | Scope | Frontal train images | Est. GPU cost | Reported? |
|---|---|---:|---:|---|
| **T0 — pilot** | pipeline debugging only | ~5,000 | < 1 h | **No** — results discarded by policy |
| **T1 — core comparison** | 4 baselines, feature analysis, hybrid, XGBoost, Optuna, ablations | **~40,000** | ~25–30 h | **Yes** — the main results |
| **T2 — scaling check** | retrain **only** the winner and runner-up on the full pool | ~162,000 | ~11 h | **Stretch** — only if ahead of schedule |

**T1 cost breakdown at ~40,000 images** (12 epochs, throughput estimates from §14.2):

| Work | Est. GPU hours |
|---|---:|
| ResNet152 | ~1.9 |
| DenseNet201 | ~1.5 |
| EfficientNetV2-S | ~1.2 |
| MobileNetV3-Large | ~0.5 |
| Optuna, 20 trials with pruning on the winner | ~8–12 |
| Frozen-feature hybrid + XGBoost (embeddings cached once) | ~1 |
| Ablations: U-Ones, no-flip→flip, plain BCE, CLAHE, 224 px | ~10 |
| Grad-CAM + sanity check | ~1 |
| **Total** | **~25–30** |

That fits four weeks with real slack for debugging — which is the point, since debugging
is what actually consumes a student project's schedule.

**Validation and test are NEVER subsetted.** Every tier is evaluated on the same full
validation split and the same official test set, so results remain directly comparable
across tiers. This is what makes T2 a meaningful check on T1 rather than a different
experiment.

### 15.3 Why ~40,000, and why this is not arbitrary

- It preserves **~4,900 positive examples**. On a binary task the binding constraint is
  the *minority* count, not the total, and ~4,900 positives is comfortably above the
  point at which ImageNet-pretrained CNNs become data-starved on a single well-represented
  finding.
- It is ~25% of the pool for ~25% of the per-epoch cost, and CXR performance on one
  well-represented pathology is well into diminishing returns by this scale.
- It buys the thing this project actually needs: **budget for Optuna and the ablations**
  (§3.2, §10.3, §11.1, §12.2). Each ablation is a full training run, and the ablations are
  what make the conclusions defensible. A larger subset that crowds them out would be a
  worse project, not a better one.

**The honest caveat:** at 40k the absolute AUCs will likely sit slightly below published
full-data CheXpert cardiomegaly numbers. That is expected and must be stated in the report
rather than glossed over. It does **not** undermine the research questions, because every
comparison — RQ1 through RQ6 — is *internal*, run on identical data under identical
conditions. We are comparing architectures against each other, not against a leaderboard.

**T2 is what converts the subset from an assumption into evidence.** If the ranking and
approximate AUC hold between 40k and the full pool, the subset was adequate and we can say
so with data. If they do not, that is itself a finding worth reporting.

### 15.4 The sampling method

`make_splits.py::subsample_patients` samples **patients**, not images, in two strata
(patient any-positive = 1 and = 0), taking a randomly permuted prefix of each until the
per-stratum image budget is met. Seeded (`subset.seed: 42`). This guarantees:

- **class distribution preserved** — strata are sampled in proportion to their *image*
  share, so it is **image-level** prevalence that is held, not patient-level. These differ
  because positive patients tend to contribute more images; image-level is the right one to
  preserve, since it is what the model and `pos_weight` actually see.
- **patient separation preserved** — patients are the sampling unit, so no patient is
  partially included
- **reproducible** — same seed, same patients, and the resulting CSV's SHA-1 is recorded
  in `split_manifest.json`

**Measured on the synthetic fixture** (400 patients; the real pool is ~135× larger):

| Target images | Actual images | Image-level prevalence | Nested in the next tier? |
|---:|---:|---:|---|
| — (full pool) | 1,001 | 0.1299 | — |
| 200 | 208 | 0.1346 | ✅ superset holds |
| 400 | 403 | 0.1191 | ✅ superset holds |
| 600 | 604 | 0.1242 | ✅ superset holds |

Image counts land within ~2% of target and prevalence within ~0.011 absolute. The residual
wobble is small-sample noise at fixture scale and will shrink substantially on the real
pool — but it is **approximate stratification, not exact**, and the achieved prevalence per
split is recorded in `split_manifest.json` rather than assumed.

Set `subset.enabled: false` for the T2 run. Nothing else changes.

### 15.4a T1 as actually produced — **MEASURED, 21/08/2026**

| Property | Target | **Measured** |
|---|---:|---:|
| Frontal training images | 40,000 | **40,002** (+0.005%) |
| Training patients | — | **13,659** of 45,172 available (30.2%) |
| Positive examples | ~4,900 projected | **5,520** |
| Prevalence | pool 13.76% | **13.80%** (+0.04 pp) |
| `pos_weight` | — | **6.246739** |
| Studies | — | 39,285 |

The sampler hit its image target to within 2 images and held prevalence to within
0.04 pp of the pool. **The projected ~4,900 positives was conservative — 5,520 measured**,
because the projection used the pre-measurement 12.3% prevalence estimate rather than the
true 13.76%.

Validation (29,380 images) and test (28,883 images) were **not** subsampled, so all three
tiers remain directly comparable.

### 15.5 Scaling up if more time appears

The deadline is not confirmed, so the plan is built to grow without rework.
**`subset.target_frontal_images` is the single value to change:**

| If the horizon becomes… | Set it to | Est. T1 cost |
|---|---:|---:|
| 4 weeks (current plan) | `40000` | ~25–30 h |
| 6–8 weeks | `60000` | ~35–40 h |
| 3 months | `enabled: false` (full pool) | ~60–70 h |

Two properties make this safe rather than a rerun of everything:

- **The split logic is unchanged** — only how many patients enter TRAIN. Validation and
  test are untouched at every tier, so results across tiers stay directly comparable.
- **The seed is fixed at 42 for the subset independently of the split seed.** Raising the
  target grows the sampled set from the same permutation, so a larger subset is a
  superset of a smaller one. Models trained at 40k and 60k differ only by *added* data,
  which is what makes the T1→T2 comparison a clean scaling check rather than a confound.

Escalation order if time runs short instead: cut the ablation set before cutting data
(the ablations are ~10 of the ~28 hours), then reduce Optuna trials, and only then reduce
epochs. **Never** cut the test set, the split discipline, or the metric set.

### 15.6 T1 image acquisition requirement — **exact, 21/08/2026**

Derived from the split CSVs now on disk.

| Need | Images | Est. size @ ~49 KB/image |
|---|---:|---:|
| T1 **train** only | **40,002** | ~1.9 GB |
| T1 train + **val** (needed to train *and* select) | 69,382 | ~3.3 GB |
| T1 train + val + **test** (full T1 experiment) | **98,265** | **~4.7 GB** |
| All frontal images (enables T2 without re-downloading) | 191,027 | ~9.2 GB |
| Entire `CheXpert-v1.0-small` release | 223,414 | **~11 GB** |

**Recommendation: take the whole ~11 GB small release.** The full T1 experiment already
needs 98,265 images — **51% of all frontal images** — because validation and test are
deliberately *not* subsampled (§15.2). At that point selective retrieval of ~98k individual
files is slower and more error-prone than one archive, saves only ~6 GB, and would force a
second download for T2. Storage is not the constraint: ~11 GB against 301 GB free on `E:`.

**Two concrete blockers to resolve before downloading:**

1. ~~**Path prefix mismatch**~~ — ✅ **FIXED 21/08/2026.** The split CSVs carry
   `CheXpert-v1.0/train/patient…` (the **full-release** prefix, because that is what
   `train_cheXbert.csv` ships) while the downsampled release unpacks to
   `CheXpert-v1.0-small/`. The two per-script resolvers were replaced by a single explicit
   one in `chexpert_metadata.py` — `strip_release_prefix()` + `resolve_image_path()` +
   `verify_image_root()` — driven by the new `images:` config block. Verified on **all
   98,265 split paths (0 malformed)** and covered by `src/data/test_path_resolution.py`.
   See §15.6.1 for the exact directory layout the resolver now expects.
2. **Release availability — UNRESOLVED.** D201 assumes the ~11 GB `CheXpert-v1.0-small`
   release. If Stanford AIMI now offers only the **471 GB** full release, D201 must be
   revisited: 471 GB **exceeds the 301 GB free on `E:`** and would be discarded down to
   320 px anyway. **Confirm the small release is still listed before committing.**

#### 15.6.1 Directory layout the resolver expects

```text
E:/UROP/data/raw/                      <- images.root   (must exist)
└── CheXpert-v1.0-small/               <- images.release_dir
    ├── train/                         <- images.expect_subdir (must exist)
    │   └── patientNNNNN/
    │       └── studyN/
    │           ├── viewN_frontal.jpg
    │           └── viewN_lateral.jpg
    └── valid/                         (optional; only if valid.csv is obtained)
```

Resolution is: **strip the leading `CheXpert-v1.0` or `CheXpert-v1.0-small` segment from
the metadata `Path`, then re-root under `images.root / images.release_dir`.** A `Path`
that carries no release prefix is passed through unchanged, so the resolver is correct
whichever prefix a future CSV ships.

`verify_image_root()` runs once at the start of both image-dependent scripts and raises
`FileNotFoundError` naming the offending directory if `images.root`, the release directory,
or `train/` is missing — including a listing of what *is* present, since a nested
`CheXpert-v1.0-small/CheXpert-v1.0-small/` is a common archive quirk. Without this guard a
missing download produces ~191,000 individual `missing` records that read like a corrupt
dataset rather than an un-downloaded one.

**If the archive unpacks under a different name**, do not rename it — set
`images.release_dir` to the actual name. Nothing else changes.

### 15.7 The trade-off, stated plainly

**Data size buys** a slightly better final model and tighter confidence intervals.
**Experimental speed buys** more experiments, more ablations, more seeds, and the ability
to recover from mistakes.

For a controlled *comparative* study whose research questions are all of the form "does A
differ from B under identical conditions", the second is worth more. We are not trying to
win a leaderboard; we are trying to make four fair comparisons and report them honestly.
A subset that lets us run each comparison properly — with the ablations that make the
conclusions trustworthy — serves the research questions better than a full-data run that
consumes the entire budget on four numbers.

---

## 16. Step 16 — The data pipeline

**Design only. No CNN training code is written at this stage.**

```text
Raw Dataset  (CheXpert-v1.0-small + official test set)
      │
      ▼
Metadata Validation ............ load_split_csv → schema check → attach_identifiers
      │                          (raises on unparseable Path — never silent NaN)
      ▼
Label Processing ............... apply_uncertainty_policy(U-Zeros)  ← §3
      │                          blank → 0, uncertain → 0, target ∈ {0,1}
      ▼
Frontal View Selection ......... Frontal/Lateral == "Frontal"      ← §4
      │                          AP/PA retained as metadata
      ▼
Patient-Level Split ............ StratifiedGroupKFold, seed 42     ← §5, §13
      │                          assert_disjoint(train, val, test)
      │                          [optional] patient-level subset    ← §15
      ▼
Image Integrity Check .......... validate_images.py                ← §8
      │                          excluded_images.csv → skipped, never deleted
      ▼
Preprocessing .................. resize 320×320 → gray→RGB×3       ← §10
      │                          → /255 → ImageNet normalize
      │                          (IDENTICAL for train / val / test)
      ▼
Training Augmentation .......... rotation ±10°, isotropic zoom     ← §11
      │                          0.9–1.0, translate ±5%,
      │                          brightness/contrast ±10%
      │                          ***TRAIN SPLIT ONLY — no flip***
      ▼
DataLoader ..................... num_workers=8, persistent_workers,
                                 pin_memory, effective batch 32
                                 via gradient accumulation          ← §14.2
```

**Note the ordering.** Label processing and view selection happen **before** splitting, so
stratification reflects the real target and the split fractions describe the data we will
actually train on. Integrity checking happens **after** splitting so that exclusions are
attributable to a split and cannot change split membership.

---

## 17. Deliverables produced in Stage 2

| File | Purpose | Status |
|---|---|---|
| `docs/dataset_analysis.md` | this document | done |
| `config/data.yaml` | single source of truth for every Stage 2/3 parameter | done |
| `requirements.txt` | pinned dependency set | done |
| `src/data/chexpert_metadata.py` | shared loading, label, view, split-guard, plot style | done, tested |
| `src/data/analyze_metadata.py` | Steps 3, 4, 6, 12 — tables + 4 figures | done, tested |
| `src/data/validate_images.py` | Steps 8, 9 — integrity + dimensions | done, tested |
| `src/data/find_duplicates.py` | Step 7 — L1–L5 duplicate analysis | done, tested |
| `src/data/make_splits.py` | Steps 5, 13, 15 — patient-level splits + manifest | done, tested |
| `src/data/probe_vram.py` | Step 14 — empirical VRAM/throughput measurement | **done, executed 21/08/2026** → `reports/vram_probe.json` |
| `src/data/plot_published_stats.py` | Step 18 — figures from CONFIRMED published values | **done, executed** |
| `docs/data_pipeline.md` | Step 19 — the actual current data pipeline | done |
| `docs/flow.md` | Step 19 — actual execution flow, existing code only | done |
| `README.md` | project entry point | done |

**All data scripts were executed end-to-end against a synthetic CheXpert-shaped fixture**
(400 patients, 1,199 rows, real JPEGs, realistic label marginals). They run, produce their
reports and figures, and the patient-disjointness assertion passes. Three real bugs were
found and fixed this way: the `StratifiedGroupKFold` fraction error (§13.4), an axis
formatter that collapsed sub-1000 counts to "0k", and a split-composition figure that
rendered two of three categories as invisible (§18).

They have **not** been run against real CheXpert data, because none has been downloaded.
`plot_published_stats.py` is the exception — it needs no dataset and **has run on real
published values**.

---

## 18. Visualizations

Figures live in **`figures/dataset/`**. `config/data.yaml → paths.figures` points there, so
the analysis scripts write to the same place once data arrives.

### 18.1 Produced now — from CONFIRMED published values

These use numbers transcribed from primary sources, each carrying its citation in the
figure caption and in `plot_published_stats.py`. **No value is assumed, estimated or
fabricated.**

| Figure | Shows | Basis |
|---|---|---|
| `01_cardiomegaly_label_states.png` | positive / negative / uncertain, study level | **CONFIRMED** — Irvin Table 1 |
| `02_uncertainty_rate_by_observation.png` | uncertain-label rate across all 14 observations, Cardiomegaly highlighted | **CONFIRMED** — Irvin Table 1 |
| `03_uncertainty_policy_auc.png` | Cardiomegaly AUC per uncertainty policy **with 95% CIs** | **CONFIRMED** — Irvin Table 3 |
| `04_uncertainty_policy_balance.png` | class balance under U-Zeros / U-Ones / exclusion | **INFERRED** — arithmetic on Table 1, labelled as such on the figure |
| `05_official_split_composition.png` | patients / studies / images per official split | **CONFIRMED** — Irvin + datasheet |

Figure 03 is the visual form of the D203 argument: **U-MultiClass's higher point estimate
lies inside every other policy's confidence interval.** Figure 02 is its companion —
Cardiomegaly has one of the lowest uncertainty rates of the 14 observations, so the policy
choice moves very little data.

Two charting decisions worth recording, both made because the first attempt failed on
inspection:

- **Figure 05 is a log-axis dot plot, not a bar chart.** The counts span 200 → 223,414;
  on a linear bar chart the valid and test bars render as literally nothing. Bars on a log
  axis are invalid (bar length must be zero-anchored), so the form changed to dots, where
  value is encoded by position.
- **Figure 04 has no value axis.** Every bar is directly labelled, so an axis would only
  repeat the numbers.

### 18.1a Produced 21/08/2026 — from the measured metadata

`analyze_metadata.py` generated these from `train_cheXbert.csv`. They **supersede** the
corresponding deferred entries below.

| Figure | Shows | Basis |
|---|---|---|
| `cardiomegaly_label_states.png` | the four raw label states, **image level** | MEASURED |
| `view_breakdown.png` | images by `Frontal/Lateral` × `AP/PA`, incl. `LL`/`RL` | MEASURED |
| `uncertainty_policy_balance.png` | frontal pos/neg under all three policies | MEASURED |
| `images_per_patient.png` | patient repetition — the leakage argument, median 1 / max 91 | MEASURED |

Figures `01`–`05` from `plot_published_stats.py` remain valid and are **not** superseded:
they document the *published* release against which the measured file was validated, and
figure `03` (policy AUC with CIs) has no measured counterpart.

### 18.2 DEFERRED until data arrives

Deliberately **not** produced, because the required data does not exist locally and
fabricating it would defeat the purpose:

| Figure | Blocked on | Produced by | Why deferred |
|---|---|---|---|
| ~~View distribution~~ | — | — | ✅ **PRODUCED 21/08/2026** (§18.1a) |
| ~~Label distribution by view~~ | — | — | ✅ **PRODUCED** — the view × AP/PA breakdown is in §4.1 and `view_breakdown.png` |
| ~~Image-level label counts~~ | — | — | ✅ **PRODUCED** (§3.1a) |
| ~~Images per patient~~ | — | — | ✅ **PRODUCED** (§5.1) |
| Image dimension / aspect-ratio distribution | **image files** | `validate_images.py` | Requires opening every JPEG |
| Representative example images | **image files** | — | Requires the images; also check RUA figure terms first |
| Split composition after splitting | `train.csv` | `make_splits.py` | Depends on the actual split |

**The ~11 GB image download was still not triggered to satisfy any visualization.** Four of
the seven deferred figures needed only the metadata CSV and were produced on 21/08/2026.
The **three remaining** all genuinely require pixels: image dimensions, representative
example images, and the post-split composition against real files.

---

## 19. Stage 2 verdict against the entry criteria

| # | Objective | Status |
|---:|---|---|
| 1 | Exact dataset/release | ✅ `CheXpert-v1.0-small` + official test set (§1) |
| 2 | Cardiomegaly label representation | ✅ column `Cardiomegaly`, 1.0/0.0/−1.0/blank (§3) |
| 3 | Uncertain label representation | ✅ `-1.0`; 6,597 studies (3.52%) (§3) |
| 4 | Number of patients | ✅ 65,240 total / 64,540 train (§6) |
| 5 | Studies / images | ✅ 187,641 studies / 224,316 images (§6) |
| 6 | Frontal images | ✅ **191,027 MEASURED** (85.50%); lateral 32,387 (§4.1) |
| 7 | Usable pos/neg | ✅ **26,283 / 164,744 MEASURED**, 13.76% prevalence, 1:6.27 (§6.3) |
| 8 | Class imbalance | ✅ ~12.3% positive, ~1:7 — moderate (§6.4, §12) |
| 9 | Patient-level leakage | ✅ risk quantified; guard implemented and tested (§5) |
| 10 | Duplicate risk | ✅ 5 levels distinguished; L1–L5 implemented (§7) |
| 11 | Corrupted-image risk | ✅ validator implemented; strict-truncation configured (§8) |
| 12 | Dimensions / formats | ✅ ~390×320 8-bit gray JPEG; exact stats **VERIFY** (§9) |
| 13 | Train/val/test split | ✅ official test + 85/15 patient-level, seed 42 (§13) |
| 14 | Preprocessing | ✅ specified and justified (§10) |
| 15 | Augmentation | ✅ specified, with the flip question resolved (§11) |
| 16 | Storage | ✅ ~40 GB working; 301 GB free on `E:` (§1.7) |
| 17 | GPU/CPU requirements | ✅ measured: RTX 4060 8 GB / i7-13650HX / 31.7 GiB. **Per-model VRAM and throughput now MEASURED** (§14.2.2), not estimated |
| 18 | Computationally feasible? | ✅ **Yes — confirmed by measurement.** Four baselines at T1 = **~5.0 GPU-hours**, inside the ~25–30 h budget (§14.3). **Not** at full data scale (~20 h for baselines alone) |

### What is needed to close Stage 2

| # | Action | Owner | Unblocks |
|---:|---|---|---|
| 1 | Redivis account + accept the Stanford RUA | you | everything |
| ~~2~~ | ~~Download the metadata CSV~~ | — | ✅ **DONE 21/08/2026** — `train_cheXbert.csv` acquired (D211); note `valid.csv` was **not** available and remains outstanding (U28) |
| ~~3~~ | ~~Run `analyze_metadata.py` and `make_splits.py`~~ | — | ✅ **DONE 21/08/2026** — U1–U7, U27 resolved |
| 4 | Approve the ~11 GB image download | you | Steps 7, 8, 9 |
| 5 | Acquire the official test set (labels + CheXlocalize images) | you | the §13.2 split; can wait until Stage 3 |
| ~~6~~ | ~~`pip install torch` + run `probe_vram.py`~~ | — | ✅ **DONE 21/08/2026** — §14.2/§14.3 now measured |

Steps 3 and 6 need no approval and no images — 6 can run right now if you want the real
batch-size numbers before committing to anything else.

**Still unknown, but not blocking:** whether any university GPU resource exists besides
this laptop. If one does, §14.4 changes and T1 could be sized larger.

---

## 20. Stage 2 decision log

All dated **16/08/2026**. Confidence is about the *decision*, not the underlying fact:
**FINAL** = settled, changing it would require new evidence · **PROVISIONAL** = settled in
principle but one input is still unverified · **UNKNOWN** = not decidable yet.

### D201 — Dataset release

| | |
|---|---|
| **Decision** | **CheXpert v1.0, `CheXpert-v1.0-small`** (~11 GB), plus the official 500-patient test set (labels from `rajpurkarlab/cheXpert-test-set-labels`, images from CheXlocalize). Full 440 GB release and CheXpert Plus (DICOM) rejected. |
| **Reason** | Its native short side is **320 px** — exactly our input size (D206), so the full release would be downsampled to 320 anyway at 40× the storage. Already 8-bit grayscale JPEG, avoiding the DICOM windowing decisions CheXpert Plus would force, which are a research project in themselves and not one of our research questions. |
| **Evidence** | CheXpert datasheet arXiv:2105.03020 (release sizes and resolutions); Irvin et al. 2019; Stanford AIMI / Redivis portal. |
| **Confidence** | **Corrected 21/08/2026.** The *preference* for the small release is **FINAL** — its 320 px native short side and 8-bit JPEG encoding are confirmed by the datasheet and are the reasons to want it. Its **availability is UNKNOWN**: no download source for the small-release images has ever been verified. The earlier 'FINAL' wrongly extended a characteristics judgement to an obtainability claim, contradicting §15.6. **PROVISIONAL** for the test set — availability confirmed from the GitHub repo and AIMI, not yet acquired. |
| **Date** | 16/08/2026 |

### D202 — Target label

| | |
|---|---|
| **Decision** | Binary target from the **`Cardiomegaly`** column. Raw encoding `1.0` / `0.0` / `-1.0` / blank → binary `target ∈ {0,1}`. Blank (no mention) → `0`. `Support Devices` retained as metadata for the shortcut probe. |
| **Reason** | Cardiomegaly is the defensible, directly-observable radiographic target chosen in Stage 1 (D001). Treating blank as negative is **the dataset paper's own convention**, not our shortcut: Table 1's three columns sum to exactly 187,641 studies, so its "Negative" column already merges explicit-negative with no-mention. |
| **Evidence** | CheXpert datasheet (label values); Irvin et al. Table 1 (column sums — INFERRED by arithmetic). |
| **Confidence** | **FINAL** |
| **Date** | 16/08/2026 |

### D203 — Uncertainty strategy

| | |
|---|---|
| **Decision** | **U-Zeros** as primary (`-1.0 → 0`). **U-Ones** and **exclusion** reported as sensitivity ablations. **U-MultiClass rejected.** |
| **Reason** | Four grounds, in order of weight: (1) U-MultiClass's 0.854 vs U-Zeros' 0.840 has **fully overlapping 95% CIs** (0.800–0.909 vs 0.783–0.897) on a **200-study** validation set — ~25–30 positives; (2) it requires a 3-class head, breaking the binary task across the XGBoost arm, PR-AUC, threshold policy, `pos_weight` and the fusion comparison; (3) Cardiomegaly has the **lowest uncertainty rate of the five competition tasks (3.52%)**, so the choice moves very little data — which is why the CIs overlap; (4) "uncertain" here typically means *borderline* — the report did not affirm the finding, which is exactly what our target measures. |
| **Evidence** | Irvin et al. Table 3 (AUC + CIs), Table 1 (3.52% uncertain). Figures `03_uncertainty_policy_auc.png`, `02_uncertainty_rate_by_observation.png`. |
| **Confidence** | **FINAL** — and deliberately *against* the paper's headline number, on stated grounds. |
| **Date** | 16/08/2026 |

### D204 — View selection

| | |
|---|---|
| **Decision** | `Frontal/Lateral == "Frontal"` only. **AP and PA combined** into one training set, with `AP/PA` retained as a metadata column and **test metrics reported stratified by projection**. |
| **Reason** | Laterals cannot support a cardiothoracic-ratio judgement, and because CheXpert labels are **study-level**, every lateral in a positive study carries a positive label it cannot visually justify. On combining AP/PA: the decisive argument is that **the official expert-annotated test set is mixed-projection**, so a PA-only model could not be evaluated on our strongest ground truth. The AP confound (magnification + inpatient acuity) is real and is handled by **measurement**, not exclusion. |
| **Evidence** | §2.2 (study-level labelling, Irvin et al.); §4.2 option analysis; official test set composition. |
| **Confidence** | **FINAL** (upgraded 21/08/2026). The last open detail is measured: the `AP/PA` value set is **`AP` 161,590 · `PA` 29,420 · `LL` 16 · `RL` 1**, with blanks confined to laterals. `LL`/`RL` do occur but total **17 images (0.009% of frontals)** — retained as their own category so they surface in stratified reports (§8.3). Frontal AP/PA split measured at **84.59% / 15.40%**, confirming the inpatient-AP skew the confound analysis assumes. |
| **Date** | 16/08/2026 · AP/PA set measured 21/08/2026 |

### D205 — Patient split

| | |
|---|---|
| **Decision** | Split unit = **patient**, never image or study. **Test = the official 500-patient set.** Train/val = **85 / 15** of the official training patients, **stratified** on patient-level any-positive, **seed 42**, via `StratifiedGroupKFold` (20 folds, union 3). Official 200-patient valid set kept as a **secondary expert-labelled check, never tuned on**. Fallback 70/15/15 carve if the test set proves unobtainable. |
| **Reason** | 64,540 patients contribute 223,414 images (~3.46 each), so image-level splitting puts the same chest in train and test; cardiomegaly is chronic, so memorising the patient *is* memorising the label, and the architecture ranking — the entire point of RQ1 — would degenerate into noise. **70/15/15 was not adopted** because a 5-radiologist consensus test set is far stronger ground truth than anything carvable from NLP-derived labels, and it is the standard benchmark. The official valid set (200 studies, ~30 positives) is too small to tune on. |
| **Evidence** | Irvin et al. (test annotation protocol, counts); `rajpurkarlab/cheXpert-test-set-labels`; `assert_disjoint` **CONFIRMED passing** on the fixture. |
| **Confidence** | **Methodology FINAL; the produced split remains PROVISIONAL** (updated 21/08/2026). The patient-level machinery is now **executed and verified on real data**: `assert_disjoint` passed, and stratification held prevalence to 13.80% / 13.65% / 13.76% across splits (§13.4a). What is still provisional is the **test arm only** — the official 500-patient set is not acquired, so the run fell back to a 70/15/15 carve. The train/val split will survive regeneration; the test split will not. |
| **Date** | 16/08/2026 · executed and verified 21/08/2026 |

### D206 — Image size

| | |
|---|---|
| **Decision** | **320 × 320**. |
| **Reason** | 320 is **exactly the native short side** of the small release — 224 would discard ~51% of the pixels we have, and 384 would upsample, adding zero information for ~44% more compute than 320. It is also what Irvin et al. used, aiding comparability. If VRAM proves tight, the fallback is **gradient accumulation at 320**, not a drop to 224: changing resolution changes the task for every model. |
| **Evidence** | CheXpert datasheet (~390 × 320, aspect preserved); Irvin et al. training setup. |
| **Confidence** | **FINAL in substance** (upgraded from PROVISIONAL, 21/08/2026). The VRAM question is now **MEASURED, not estimated**: 320 × 320 fits all four models at micro-batch 16 using 1.00–3.80 GiB of 6.94 GiB free, with >3 GiB headroom on every model (§14.2.2). The one remaining unverified detail is the exact dimension *distribution*, still **VERIFY** until `validate_images.py` runs — but the datasheet already CONFIRMS the 320 px short side, so that check can only refine, not overturn, this decision. |
| **Date** | 16/08/2026 · VRAM half resolved 21/08/2026 |

### D207 — Normalization

| | |
|---|---|
| **Decision** | Direct bilinear resize to 320 × 320 (**no cropping**) → replicate the grayscale channel **3×** → `/255` → normalize with **ImageNet** mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`. **CLAHE excluded from core.** Identical for train, val and test. |
| **Reason** | Cropping is rejected outright because cardiomegaly is a *ratio* and cropping risks removing a lateral chest wall — the denominator of the measurement. Direct resize is preferred over aspect-preserving padding because a uniform horizontal rescale multiplies cardiac and thoracic width by the same factor, leaving **CTR invariant**, whereas padding introduces borders whose size correlates with original aspect ratio — a spurious feature. Channel replication preserves the pretrained stem weights exactly, and avoids performing four bespoke first-conv surgeries on four different stem designs, each a chance to inject an architecture-specific difference into a controlled comparison. |
| **Evidence** | §10.1–10.3; literature review v2 §7 (CLAHE must not become a silent trick). |
| **Confidence** | **FINAL** on channel handling and normalization. **PROVISIONAL** on `direct` vs `aspect_pad` — both defensible; exposed as a one-line config ablation. |
| **Date** | 16/08/2026 |

### D208 — Augmentation

| | |
|---|---|
| **Decision** | **Train split only.** Rotation ±10°, **isotropic** zoom 0.9–1.0, translation ±5% (pad, never crop out), brightness ±10%, contrast ±10%. **No horizontal flip.** No vertical flip, shear, elastic, cutout, MixUp or CutMix. No TTA in the core protocol. |
| **Reason** | Each accepted transform corresponds to genuine acquisition variation. Two constraints are specific to this task: **anisotropic aspect jitter is forbidden** because it changes the cardiothoracic ratio — i.e. it changes the label; and translation must pad rather than crop, for the same reason as D207. On flipping — literature review v2 §7 explicitly left this open and demanded justification: flipping fabricates a population with ~50% dextrocardia (true rate ~1:10,000), destroys the left-sided cardiac position the task depends on, and mirrors the L/R laterality markers present in CheXpert images. |
| **Evidence** | §11 table; §11.1; literature review v2 §7. |
| **Confidence** | **FINAL** — with flip-on retained as a single reported ablation, since the review declined to settle it. |
| **Date** | 16/08/2026 |

### D209 — Class imbalance strategy

| | |
|---|---|
| **Decision** | **One** strategy: `pos_weight` in `torch.nn.BCEWithLogitsLoss`, computed from the **training split only**. Weighted sampling, focal loss and oversampling rejected. Decision threshold selected on **validation**, frozen, then applied once to test. |
| **Reason** | At ~12.3% positive (~1:7) the imbalance is **moderate**, not severe — severe-imbalance machinery is unwarranted. `pos_weight` adds **zero tunable hyperparameters** and keeps the data pipeline byte-identical across all four architectures, which matters because anything that varies per model confounds RQ1. Focal loss would add α and γ: tuning them per architecture confounds the comparison, not tuning them is arbitrary. Computing `pos_weight` from the full pool would leak val/test prevalence into training. |
| **Evidence** | §6.3 projected prevalence; §12.2 comparison; `split_manifest.json → pos_weight_from_train`. |
| **Confidence** | **FINAL** (upgraded 21/08/2026). The strategy was already settled; the value is now **measured**: frontal prevalence **13.76%** at **1 : 6.27**, giving **`pos_weight` = 6.246739** computed from the T1 train split alone and recorded in `split_manifest.json`. The measured imbalance is slightly *milder* than the 1:7.2 projection, so the argument against focal loss and resampling only strengthens. |
| **Date** | 16/08/2026 · value measured 21/08/2026 |

### D210 — Dataset subset vs full data

| | |
|---|---|
| **Decision** | Tiered. **T0** ~5,000 images (pipeline debugging, results never reported) · **T1 ~40,000 frontal training images** — the core comparison, all four baselines, fusion, XGBoost, Optuna and ablations · **T2** full pool, retraining only the winner and runner-up as a **stretch** scaling check. Validation and test are **never** subsampled. |
| **Reason** | Sized to the confirmed **4-week planning horizon**: T1 costs ~25–30 GPU-hours on the available RTX 4060, versus ~50–70 at full scale, which would not fit. T1 preserves ~4,900 positives — the binding constraint on a binary task is the *minority* count, and that is well above data-starvation. Critically, it buys the **~10 hours of ablations** (D203, D207, D208, D209) that make the conclusions defensible; a larger subset that crowded them out would be a worse project. Every research question is an *internal* comparison under identical conditions, so absolute AUC matters less than comparability. |
| **Evidence** | §14.3 cost model; §15.2 breakdown; subset nesting and prevalence preservation **CONFIRMED by test** (§15.4). |
| **Confidence** | **PROVISIONAL** — but for one reason only now (updated 21/08/2026). The throughput half is **MEASURED**: four baselines at T1 cost **~5.0 GPU-hours**, against a pre-measurement estimate of ~5.1 h (§14.3). The cost model is confirmed and T1 = 40k is comfortably affordable — in fact conservative, since the measured baseline cost consumes only ~20% of the ~25–30 h budget. What remains open is **non-technical**: the real UROP deadline is still unconfirmed (U17), and T1 sizing is pegged to a 4-week *planning horizon*, not a known deadline. `subset.target_frontal_images` remains the single value to change. **The sampling strategy has deliberately not been altered on the strength of this measurement.** |
| **Date** | 16/08/2026 · throughput half resolved 21/08/2026 |

### D211 — Primary label source *(new, 21/08/2026)*

| | |
|---|---|
| **Decision** | **`train_cheXbert.csv`** (CheXbert-relabelled training set, Smit et al. 2020) is the primary label source, in place of the original rule-based `train.csv` of Irvin et al. 2019. `train_visualCheXbert.csv` is retained as an **optional ablation only** (`dataset.ablation_train_csv`), never the primary. |
| **Reason** | It is what Stanford AIMI exposes as a standalone metadata download, and it is structurally a drop-in: **measured** to reproduce every published training-set figure exactly (223,414 images / 64,540 patients / 187,641 studies / 191,027 frontal), with all 19 columns and the same four-state encoding. It preserves the task definition, the uncertainty structure D203 rests on, and comparability with the CheXpert literature. CheXbert is also a stronger labeller than the original rule-based system. VisualCheXbert was rejected as primary because at **58.33% frontal prevalence with zero uncertain and zero blank states** it is a *different task*: it would void D203 and invert D209 (`pos_weight` 0.71). |
| **Evidence** | Direct measurement of both files, 21/08/2026 (§3.3, §6.2). Column/row/patient/study counts verified against Irvin et al. and the CheXpert datasheet. |
| **Confidence** | **FINAL** — with the provenance caveat stated in §3.3 and required in the report: Irvin Table 3's uncertainty-policy AUCs are *indicative* for this file, not exactly applicable, because they were measured with the original labeller. D203's reasoning strengthens rather than weakens under CheXbert (uncertainty 1.77% vs 3.52%). |
| **Date** | 21/08/2026 |

### Supplementary detailed log

The granular decisions behind the ten above, retained for traceability:

| ID | Decision | Reason | Section |
|---|---|---|---|
| D009 | `CheXpert-v1.0-small` over full/Plus releases | native 320 px short side = our input size; 8-bit JPEG avoids DICOM windowing decisions | §1.4 |
| D010 | U-Zeros primary; U-Ones + exclusion as ablations; **U-MultiClass rejected** | U-MultiClass's 0.014 AUC edge has fully overlapping CIs on 200 studies and breaks the binary task across six downstream stages | §3.2 |
| D011 | Frontal only; AP + PA **combined**, projection retained as metadata | official expert test set is mixed-projection, so PA-only would be unevaluable on our strongest ground truth; confound handled by stratified reporting | §4.2 |
| D012 | **Official 500-patient test set** as the test set | 5-radiologist consensus vs NLP-derived labels; also the standard benchmark | §13.2 |
| D013 | 85/15 patient-level train/val, seed 42, stratified | official valid set (200 studies) is too small to tune on; 70/15/15 unnecessary once the official test set is used | §13.2 |
| D014 | Input size **320 × 320** | native resolution of the release; matches Irvin et al.; 224 discards ~51% of pixels, 384 upsamples | §9.2 |
| D015 | Direct resize, **no cropping** | cropping can remove a chest wall = the CTR denominator; horizontal rescale preserves CTR | §10.1 |
| D016 | Grayscale → RGB by **channel replication** | preserves pretrained stem weights identically across four different stem designs | §10.2 |
| D017 | **No horizontal flip** in core protocol | fabricates ~50% dextrocardia; destroys the left-sided cardiac position the task depends on; ablation retained | §11.1 |
| D018 | `pos_weight` as the **single** imbalance strategy | zero new hyperparameters, identical data pipeline across models; 1:7 does not warrant focal loss | §12.2 |
| D019 | Local RTX 4060 primary, Kaggle backup | 8 GB is sufficient with AMP + gradient accumulation; **local training satisfies the RUA no-redistribution clause** | §14.4 |
| D020 | **Fixed effective batch size 32 via gradient accumulation** | per-model max batch varies up to 8×; unequal batch size would confound RQ1 | §14.2 |
| D021 | Tiered subset: **T1 ~40k images** core (4-week horizon), T2 full-pool as a stretch scaling check | preserves ~4,900 positives; buys the ~10 h ablation budget the research questions require; supersets cleanly if the horizon extends | §15 |
| D022 | Acquire **metadata CSVs before images** | `train.csv` (~30 MB) resolves Steps 3, 4, 6, 12 and 13 outright and confirms we have the right release before committing 11 GB | §1.8 |

---

## 21. Stage 2 unresolved questions

Nothing here blocks Stage 3 *planning*; items marked **BLOCKING** block Stage 3
*execution*.

### 21.1 ~~Resolved by the ~30 MB metadata CSV~~ — **ALL RESOLVED 21/08/2026**

| # | Question | Measured answer |
|---:|---|---|
| U1 | Exact frontal / lateral image counts | ✅ **191,027 / 32,387** — exact match to the secondary figure |
| U2 | The exact `AP/PA` value set — do `LL` / `RL` occur? | ✅ **Yes.** `AP` 161,590 · `PA` 29,420 · `LL` 16 · `RL` 1 · blank 32,387 (laterals only) |
| U3 | **Image-level** Cardiomegaly counts | ✅ pos 30,566 / neg 16,155 / unc 3,917 / blank 172,776 (§3.1a) |
| U4 | Usable frontal counts per policy | ✅ U-Zeros **26,283 pos / 164,744 neg, 13.76%** (§6.3) |
| U5 | Images-per-patient distribution | ✅ mean 2.96, **median 1**, max **91**; 49.19% of patients have >1 (§5.1) |
| U6 | Real `pos_weight` (D209) | ✅ **6.246739** from the T1 train split |
| U7 | Does the real CSV schema match §2.3? | ✅ All 19 columns present. **One difference:** `No Finding` is last, not first — harmless, all code addresses by name (§3.3) |
| U27 | *(new)* Which labeller does the acquired file use? | ✅ **CheXbert, not the original rule-based labeller** — D211, §3.3 |

### 21.2 Resolved by the ~11 GB image download — **BLOCKING for Stage 3 execution**

| # | Question | Certainty now | Resolved by |
|---:|---|---|---|
| U8 | Exact dimension / aspect-ratio distribution | INFERRED (short side 320) | `validate_images.py` |
| U9 | Count of corrupted / missing / unreadable files | UNKNOWN | `validate_images.py` |
| U10 | Exact-duplicate count, and any **cross-patient** duplicates | UNKNOWN | `find_duplicates.py` |
| U11 | Cross-patient near-duplicates | UNKNOWN | `find_duplicates.py --near` |

### 21.3 Resolved by acquiring the official test set — **BLOCKING for D205**

| # | Question | Certainty now |
|---:|---|---|
| U12 | Can the CheXlocalize test **images** actually be downloaded? | CONFIRMED available; **not yet acquired** |
| U13 | Does `groundtruth.csv` carry `Path` values that join to those images? | UNKNOWN — the repo documents labels, not the join key |
| U14 | Test-set frontal/lateral and AP/PA composition | UNKNOWN |
| U28 | *(new)* Is the official **`valid.csv`** obtainable separately? | **UNKNOWN** — neither acquired CSV contains any validation rows; all 223,414 are `train/`. Needed for the §13.2 secondary expert check, not for training |

**The fallback is currently in force.** `test_set.enabled: false`, so the split produced on
21/08/2026 is `mode: fallback-carved-test` (70/15/15). This must be stated prominently in
the report, since it materially weakens the ground truth — the test set is currently
CheXbert-labelled, not radiologist-adjudicated.

### 21.4 ~~Resolved by installing `torch`~~ — **RESOLVED 21/08/2026**

| # | Question | Status |
|---:|---|---|
| U15 | Real max micro-batch per architecture at 320 × 320 on 8 GB | ✅ **MEASURED** — max efficient batch 32/32/32/128; recommended uniform micro-batch **16 × 2 accumulation** (§14.2.2–14.2.3) |
| U16 | Real throughput, hence the true T1 cost | ✅ **MEASURED** — 84.0 / 61.0 / 152.1 / 486.1 img/s; four baselines at T1 = **~5.0 GPU-hours** (§14.3) |
| U25 | *(new, found by the probe)* Does Windows system-memory fallback distort batch sizing? | ✅ **CONFIRMED PRESENT** — reserved memory reaching 10.56 GiB on an 8.00 GiB card, throughput collapsing to 8–15%. Handled: the probe reports max *efficient* batch, and the recommended config sits >3 GiB clear of the spill threshold (§14.2.1) |
| U26 | *(new, deferred)* Will the JPEG data loader bottleneck MobileNetV3 at ~486 img/s? | **UNKNOWN** — measured throughput is pure GPU compute on synthetic tensors. Cannot be tested until real images exist; a `num_workers` tuning problem, not a capacity problem (§14.3) |

`probe_vram.py` needed no dataset and has run. Raw data:
`artifacts/stage2/reports/vram_probe.json`.

### 21.5 Needs a human answer — non-blocking

| # | Question | Why it matters |
|---:|---|---|
| U17 | **The actual UROP deadline** | 4 weeks is a *planning horizon*, not a confirmed deadline. Sets T1 size (D210) |
| U18 | Any university/lab GPU besides this laptop? | Would change D210 sizing and §14.4 |
| U19 | Does the RUA permit uploading data to Kaggle/Colab? | Determines whether the backup environment (§14.4) is usable at all |
| U20 | Does the RUA restrict publishing example images in the report? | Affects Grad-CAM figures and the deferred example-image visualization |

### 21.6 Known and accepted — will NOT be resolved in this project

| # | Limitation | Disposition |
|---:|---|---|
| U21 | Training labels are report-derived, not image-verified | **Accepted.** Stated in abstract, methods and limitations |
| U22 | ~688 images (0.3%) reported as striped / incomplete / non-CXR decode fine and pass every automated check | **Accepted.** Detecting them needs visual review, out of scope. Named as a residual data-quality limitation |
| U23 | Single-institution, historical (2002–2017) data | **Accepted.** No cross-site generalisation claims |
| U24 | AP/PA is confounded with patient acuity | **Measured, not removed** — stratified test reporting (D204) |

---

## 22. FINAL STAGE 2 RECOMMENDATION

| # | Question | Answer |
|---:|---|---|
| 1 | **Can we use CheXpert?** | **Yes.** It has an explicit `Cardiomegaly` label at scale, recoverable patient identifiers, an expert-annotated evaluation set, and access without credentialing. The binding condition is honest framing: performance is **agreement with a report-derived label**, not diagnostic accuracy. |
| 2 | **Which exact release?** | **`CheXpert-v1.0-small`** (~11 GB, ~390 × 320, 8-bit grayscale JPEG), plus the **official 500-patient test set** (labels from `rajpurkarlab/cheXpert-test-set-labels`, images via CheXlocalize). Not the 440 GB release; not CheXpert Plus. |
| 3 | **Which images?** | **Frontal only** (`Frontal/Lateral == "Frontal"`), **AP and PA combined**, with `AP/PA` retained as metadata and test metrics reported stratified by projection. Laterals excluded — they cannot support a CTR judgement yet inherit study-level positive labels. |
| 4 | **Which label?** | The **`Cardiomegaly`** column, binarised to `target ∈ {0,1}`. Blank (no mention) → `0`, matching the dataset paper's own accounting. `Support Devices` retained as metadata for the shortcut probe. |
| 5 | **How are uncertain labels handled?** | **U-Zeros** (`-1.0 → 0`) as primary. **U-Ones** and **exclusion** as reported ablations. **U-MultiClass rejected** — its apparent advantage has fully overlapping CIs on 200 studies and it would break the binary task across six downstream stages. |
| 6 | **How many usable examples? — MEASURED** | **191,027 usable frontal images** from **64,534 patients**. Under U-Zeros: **26,283 positive / 164,744 negative = 13.76% prevalence, 1 : 6.27**. Study level: 25,840 pos / 3,327 uncertain / 158,474 neg of 187,641. All measured from `train_cheXbert.csv`, 21/08/2026 (§3.1a, §6.3). |
| 7 | **What split? — EXECUTED** | **Patient-level, always**, seed 42, stratified on patient-level any-positive. `assert_disjoint` **passed on real data**. **Currently in fallback mode (70/15/15)** because the official test set is not acquired: train 13,659 pts / 40,002 imgs (13.80%), val 9,680 pts / 29,380 imgs (13.65%), test 9,682 pts / 28,883 imgs (13.76%). **The intended split is unchanged** — official 500-patient test set + 85/15 train/val — and the test arm **must be regenerated** once that set is obtained (§13.4a). |
| 8 | **What preprocessing?** | Direct resize to **320 × 320** (no cropping) → replicate grayscale **3×** → `/255` → **ImageNet** normalization. Identical across all four baselines and across train/val/test. CLAHE excluded from core. |
| 9 | **What augmentation?** | **Train split only.** Rotation ±10°, isotropic zoom 0.9–1.0, translation ±5% with padding, brightness ±10%, contrast ±10%. **No horizontal flip.** No anisotropic aspect jitter — it changes the cardiothoracic ratio, i.e. the label. No shear, elastic, cutout, MixUp or CutMix. No TTA. |
| 10 | **How is imbalance handled? — MEASURED** | **One** strategy: `pos_weight` in `BCEWithLogitsLoss`, computed from **train only** = **6.246739**. Measured imbalance is **1 : 6.27** (13.76% positive) — moderate, and milder than the 1:7.2 projection, so focal loss and resampling remain unjustified. Threshold selected on validation, frozen, applied once to test. **Report PR-AUC with its baseline — now 0.138, not the projected 0.123 — alongside ROC-AUC; never accuracy alone.** |
| 11 | **Full dataset or subset? — PRODUCED** | **Subset — T1 delivered at 40,002 frontal training images**, 13,659 patients, **5,520 positives**, prevalence 13.80% (pool 13.76%). Patient-level stratified, seed 42. Full pool (T2) only as a stretch scaling check. **Validation (29,380) and test (28,883) were not subsampled.** Raising `subset.target_frontal_images` is the single change if more time appears; subsets are nested supersets, **CONFIRMED by test**. |
| 12 | **What hardware?** | **Primary: the local RTX 4060 Laptop (8 GB), i7-13650HX, 31.7 GiB RAM, 301 GB free on `E:`.** All four models **measured** at 320 × 320 with AMP: run every one at **micro-batch 16 with 2 gradient-accumulation steps** (effective batch 32, identical across models per D020), using 1.00–3.80 GiB and leaving >3 GiB headroom. Measured throughput 84.0 / 61.0 / 152.1 / 486.1 img/s. **Critical machine-specific caveat:** Windows CUDA System Memory Fallback is ON, so exceeding VRAM does not error — it silently spills to host RAM at ~10× slowdown (§14.2.1). The recommended config sits well clear of that threshold. Local training also keeps data on-machine, satisfying the RUA. **Backup: Kaggle Notebooks** (16 GB P100, ~30 h/week) — subject to U19. |
| 13 | **What is still unknown?** | 28 items in §21. **Resolved 21/08/2026:** all metadata facts (U1–U7, U27) and all compute facts (U15–U16, U25). **Still blocking Stage 3 execution:** the four image-dependent data-quality checks (U8–U11 — integrity, dimensions, exact and near duplicates) and the official test set (U12–U14, U28). **Non-blocking:** the data-loader bottleneck (U26) and four human answers (U17–U20), of which the **actual deadline** (U17) is the only thing still keeping D210 provisional. |
| 14 | **Ready for Stage 3?** | **Not yet — one acquisition away.** The metadata half of Stage 2 is **complete and measured**: labels, views, prevalence, `pos_weight`, patient-level splits and the T1 subset all exist on disk with verified patient-disjointness. Seven of eleven decisions are now FINAL. **Two things remain before Stage 3 opens:** (a) the **T1 images** (§15.6) — without them nothing can train and U8–U11 cannot be checked; (b) the **official test set**, without which the current test split is a CheXbert-labelled fallback rather than radiologist-adjudicated ground truth. Stage 3 must not begin before (a). |

### Stage 2 status

> **CONDITIONALLY COMPLETE — pending data acquisition.**
>
> The metadata half of Stage 2 is **complete and measured**. Of the **eleven** formal
> decisions (updated 21/08/2026 after the VRAM probe and the metadata run):
>
> - **Seven FINAL** — D202, D203, D208, **D204** (AP/PA set measured), **D209**
>   (`pos_weight` = 6.246739 measured), **D211** (label source), and D206 in substance.
> - **Two FINAL with one unverified detail** — D201 (test set not acquired), D207
>   (`direct` vs `aspect_pad`, an exposed ablation).
> - **Two PROVISIONAL** — **D205**: the *methodology* is executed and verified on real
>   data, but the produced split is in `fallback-carved-test` mode and its **test arm must
>   be regenerated** once the official set is obtained. **D210**: throughput measured and
>   T1 delivered, pending only the real deadline (U17).
>
> **Stage 2 cannot close yet.** The four image-dependent checks (U8–U11) are untouched
> because no image has been downloaded, and the test set is currently a CheXbert-labelled
> fallback rather than radiologist ground truth. Declaring closure now would be exactly the
> kind of unverified assertion this document is structured to prevent.

---

## 23. Sources

Primary:

- Irvin, J. et al. (2019). *CheXpert: A Large Chest Radiograph Dataset with Uncertainty
  Labels and Expert Comparison.* AAAI 33(01), 590–597. arXiv:1901.07031 — dataset scale,
  Table 1 label counts, Table 3 uncertainty-policy AUCs, validation/test annotation protocol.
- Garbin, C. et al. (2021). *Structured dataset documentation: a datasheet for CheXpert.*
  arXiv:2105.03020 — label encoding values, image release formats and sizes, per-split image counts.
- Stanford ML Group — CheXpert competition page: `stanfordmlgroup.github.io/competitions/chexpert/`
- Stanford AIMI shared datasets / Redivis: `aimi.stanford.edu/data`, `stanford.redivis.com`
- `github.com/rajpurkarlab/cheXpert-test-set-labels` — official test-set ground truth.
- Chambon, P. et al. (2024). *CheXpert Plus.* arXiv:2405.19538 — DICOM release scope.

Secondary (used only where marked VERIFY, and flagged as such):

- `github.com/ayhyap/CheXpert-review` — unofficial manual data-quality audit (§7.5, §8.3).
- `github.com/fau-masters-collected-works-cgarbin/chexpert_explorer` — independent
  reimplementation corroborating patient/study/image counts.

Measured on this machine, 16/08/2026: `nvidia-smi`, `Win32_Processor`,
`Win32_ComputerSystem`, `Get-PSDrive`, `python --version`.
