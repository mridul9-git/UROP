# Research — reasoning, evidence and interpretation

**Last updated:** 31/08/2026 · **Stage:** 3 (baseline training)

This document holds the **scientific reasoning**: why the study is built the way it is,
what the evidence currently supports, and what it does not. It does not restate dataset
measurements (`dataset_analysis.md`), experiment results (`experiments.md`), pipeline
mechanics (`flow.md`, `data_pipeline.md`) or chronological history (`notes.md`).

Certainty labels: **CONFIRMED** · **INFERRED** · **VERIFY** · **UNKNOWN** · **SUPERSEDED**.

---

## 1. Research objective

The study is a **controlled comparative deep-learning study of binary cardiomegaly
classification from frontal chest radiographs**, with downstream hybrid feature fusion,
hyperparameter optimisation and explainable AI.

It makes **no architectural novelty claim**. It is a controlled comparison and partial
replication/extension. Its value is experimental rigour, reproducibility, and honest
reporting of positive *or* negative results.

### Research questions — as established in Stage 1

These are carried forward unchanged from `literature_review_v2.md` §2. They are not
re-scoped here.

| # | Question | Status |
|---:|---|---|
| **RQ1** | Which selected CNN architecture performs best for cardiomegaly classification under identical conditions? | **IN PROGRESS** — 1 of 4 baselines complete |
| **RQ2** | Do the strongest CNNs learn meaningfully different feature representations? | NOT STARTED — requires ≥2 baselines |
| **RQ3** | Does combining the top-two representations improve performance? | NOT STARTED |
| **RQ4** | Does hyperparameter optimisation improve the strongest model? | NOT STARTED |
| **RQ5** | Does XGBoost provide competitive performance against a neural classifier on the same frozen embeddings? | NOT STARTED |
| **RQ6** | What do Grad-CAM explanations reveal about correct and incorrect predictions? | NOT STARTED |

**RQ1 cannot be answered yet.** One architecture has been trained. A single result is not
a comparison, and no ranking claim is available.

---

## 2. The binding framing condition

> Performance in this study is **agreement with a report-derived label**, not clinical
> diagnostic accuracy.

This is not a disclaimer appended to the end; it constrains every claim the project can
make. Training labels come from an NLP labeller applied to free-text radiology reports.
A model that perfectly reproduces those labels has reproduced *the labeller*, including
its errors. Any statement of the form "the model detects cardiomegaly with X accuracy" is
outside what this evidence supports.

The one place this weakens is the official validation set, whose labels are
radiologist-adjudicated rather than report-derived — which is precisely why it is worth
scoring against, and precisely why it must not become a tuning surface.

---

## 3. Methodology rationale

### 3.1 Why a controlled comparison rather than a novel architecture

A four-week horizon with a single 8 GB laptop GPU cannot support both architecture search
and rigorous evaluation. Choosing rigour means: one preprocessing pipeline, one split, one
seed policy, one loss, one metric protocol, applied identically to every architecture. Any
deviation between baselines makes the comparison uninterpretable, so the protocol is
frozen in configuration and asserted in code rather than left to operator discipline.

### 3.2 Label-source reasoning

The primary training metadata is `train_cheXbert.csv` — the **CheXbert** relabelling, not
the original rule-based CheXpert labeller. This matters for three reasons:

1. **The literature's uncertainty-policy results are indicative, not directly
   applicable.** Published per-policy AUCs were measured against the original labeller.
   Under a different labeller the uncertain-label population differs, so those numbers
   guide the choice without determining it.
2. **The project must never describe this as "the original CheXpert labels".** That is a
   standing prohibition, not a stylistic preference.
3. **A visually-derived alternative was rejected.** A VisualCheXbert-labelled file was
   available. It predicts image-derived rather than report-derived labels, carries a
   dramatically different frontal prevalence, and contains **no uncertain or blank
   states at all**. Adopting it would have silently voided the uncertainty policy (there
   would be no uncertainty to police) and inverted the class-imbalance weighting. It is
   retained as an **optional ablation only**, never as the primary target.

### 3.3 Uncertainty-policy reasoning

Uncertain labels (`-1.0`) are mapped to negative (**U-Zeros**) as the primary policy, with
U-Ones and exclusion available as reported ablations.

The reasoning is that a multi-class treatment of uncertainty would break the binary task
across every downstream stage — hybrid fusion, HPO, the XGBoost comparison and Grad-CAM
all assume a single logit — while its apparent advantage in the literature rests on a
small evaluation set with overlapping confidence intervals. Trading a coherent
six-stage pipeline for an effect that may not survive its own error bars is a bad trade.

Blank (no mention) is treated as negative, matching the dataset paper's own accounting.

### 3.4 Patient-level split rationale

Splitting is **by patient, always** — never by image or study.

The dataset's patients contribute multiple images each, and **cardiomegaly is chronic**.
A patient appearing in both train and test means memorising the patient *is* memorising
the label. Under image-level splitting, RQ1 — the entire point of the study — would
degenerate into a measurement of memorisation capacity rather than of architecture
quality. The guarantee is therefore enforced as a hard assertion that runs before any
split CSV is written **and** again at training time, because CSVs on disk can be
regenerated or mixed between runs.

Stratification is on patient-level any-positive, seed 42.

### 3.5 View-selection rationale

Frontal images only; AP and PA combined, with projection retained as metadata.

Laterals are excluded because they cannot support a cardiothoracic-ratio judgement, yet
they inherit study-level positive labels — a lateral in a positive study carries a
positive label it cannot visually justify. AP and PA are combined rather than restricted
to PA because the expert-annotated evaluation sets are mixed-projection; a PA-only model
could not be evaluated against the strongest available ground truth. The AP confound
(magnification, and inpatient acuity) is real, and is handled by **measurement** —
projection-stratified reporting — rather than by exclusion.

### 3.6 Metric rationale

At ~13.7% positive prevalence, **accuracy alone is misleading** and is never reported
alone. An all-negative predictor achieves ~86% accuracy and zero recall.

- **PR-AUC is the headline metric**, always reported with its no-skill baseline, which
  equals the prevalence of the scored set.
- **ROC-AUC is reported alongside** because it is prevalence-independent and therefore the
  only metric directly comparable across sets with different prevalence.
- **Threshold is selected on validation, frozen, and applied unchanged elsewhere.**
- **Calibration (Brier, ECE, binned reliability) is recorded** because a weighted loss
  distorts probability estimates and a distorted probability is not a usable one.

### 3.7 Class-imbalance reasoning

Exactly **one** strategy is used: `pos_weight` in `BCEWithLogitsLoss`, computed from the
**training split only** — deriving it from the full pool would leak validation and test
prevalence into training. The measured value is **6.246739**.

The measured imbalance (~1:6.27) is moderate. Focal loss and resampling are therefore
unjustified: they would add a hyperparameter and a confound to a comparison whose whole
purpose is to isolate architecture.

### 3.8 The role of the official expert-validation set

The official 200-patient validation set is a **secondary expert-labelled check, never a
model-selection surface**.

It is small — 202 usable frontal images, 66 positives — which is too small to tune on
without overfitting to it, and it is one of only two radiologist-adjudicated surfaces the
project can reach. Spending it as a tuning set would destroy its value as an independent
check. This is enforced mechanically: the evaluation entry point requires the threshold as
a command-line argument, has no threshold-selection code path at all, and refuses to
report if the applied policy is anything other than `inherited`.

---

## 4. Evidence discipline

Every substantive claim carries a certainty label, and labels are **never upgraded without
new evidence**.

| Label | Means |
|---|---|
| **CONFIRMED** | measured, executed, or read directly from an artifact on disk |
| **INFERRED** | follows logically from confirmed facts, but was not itself measured |
| **VERIFY** | plausible and load-bearing, but unchecked — flagged for action |
| **UNKNOWN** | not determined |
| **SUPERSEDED** | was accurate, has been overtaken; retained, not deleted |

Three standing rules:

1. **A projection is never presented as a measurement.** Where a projected figure has been
   replaced by a measured one, both are kept and the projection is marked SUPERSEDED.
2. **A contradiction between documents is preserved and marked**, not silently reconciled.
   The historical record of what was believed, and when, is part of the evidence.
3. **A failed check is recorded as prominently as a passing one.** Silent failure is the
   specific failure mode this project is structured to prevent.

---

## 5. Current findings

### 5.1 The pipeline works end-to-end on real data — **CONFIRMED**

Metadata, splits, image resolution, preprocessing, training, checkpointing, resume and
evaluation have all executed against the real dataset. This was the primary risk in
Stage 3 and it is retired.

### 5.2 ResNet152 discriminates cardiomegaly well above chance — **CONFIRMED**

Best epoch 3: validation ROC-AUC **0.8647**, PR-AUC **0.5417** against a no-skill baseline
of **0.1365** — a lift of **3.97×**.

### 5.3 Discrimination survives a change of label source — **CONFIRMED, and the key result so far**

Moving from CheXbert-labelled carved validation to radiologist-adjudicated official
validation, **ROC-AUC moves 0.8647 → 0.8577**.

This is the most scientifically interesting number the project has produced. It is
evidence — on 202 images — that what the model learned is not purely an artefact of the
NLP labeller. Because ROC-AUC is prevalence-independent, it is the only one of the
reported metrics that supports a direct cross-set comparison.

**Strength of this evidence is limited.** 202 images, 66 positives, no confidence
interval. It is consistent with genuine transfer; it does not establish it tightly.

### 5.4 PR-AUC across the two sets is NOT comparable — **CONFIRMED**

Official-validation PR-AUC is **0.7547**, carved-validation PR-AUC is **0.5417**. This is
**not** an improvement. The PR-AUC no-skill baseline equals prevalence, and prevalence
differs: **0.13652** versus **0.32673**.

Expressed as lift over the baseline, performance moves in the *opposite* direction:

| Set | PR-AUC | Baseline | Lift |
|---|---:|---:|---:|
| Carved validation (CheXbert) | 0.5417 | 0.1365 | **3.97×** |
| Official validation (expert) | 0.7547 | 0.3267 | **2.31×** |

Any report stating that the model performs better on expert labels because 0.7547 > 0.5417
is wrong, and this document exists partly to prevent that sentence being written.

### 5.5 The fixed threshold transfers conservatively — **CONFIRMED as an observation, INFERRED as a mechanism**

| | Carved validation | Official validation |
|---|---:|---:|
| Prevalence | 0.13652 | 0.32673 |
| Sensitivity | 0.6046 | **0.3030** |
| Specificity | 0.8974 | **0.9706** |

The threshold **0.44209548389690995** was selected to maximise F1 at 13.65% prevalence and
applied unchanged at 32.67%. At that operating point the model misses 46 of 66
expert-confirmed cases.

**INFERRED, not confirmed:** that this is a threshold-transfer effect rather than a
discrimination failure. The supporting evidence is that ROC-AUC — which is independent of
any threshold — barely moves. A genuine collapse in discrimination would have depressed
ROC-AUC too, and it did not. **This must not be reported as evidence of discrimination
failure.** Equally it must not be dismissed: at this operating point the model is
clinically unusable for case-finding, and that is a real result.

The correct response is *not* to re-tune on the official set. Operating-point selection is
a separate question from discrimination, and answering it on the expert set would consume
the only independent check the project has.

### 5.6 The model is over-confident, and worse so on expert labels — **CONFIRMED**

ECE **0.0817 → 0.1523**, Brier **0.0961 → 0.1766**. The carved-validation reliability bins
show observed frequency below mean predicted probability in **every** bin.

**INFERRED:** this is the expected consequence of training with `pos_weight = 6.2467`,
which deliberately inflates positive scores. Producing usable probabilities would need an
explicit recalibration step fitted on validation. The pipeline performs none, and no
calibrated-probability claim should be made until it does.

### 5.7 Overfitting begins early — **INFERRED**

Training loss fell monotonically 0.865 → 0.631 across epochs 2–8 while validation PR-AUC
plateaued after epoch 3. The early stop at epoch 8 selecting epoch 3 is therefore well
placed rather than premature. **UNKNOWN** whether a lower learning rate, stronger
regularisation or a longer warmup would move the plateau.

---

## 6. Current hypotheses

Stated so they can be tested or refuted, not assumed.

| # | Hypothesis | Basis | How it would be tested |
|---:|---|---|---|
| H1 | Architecture choice produces a measurable difference in cardiomegaly PR-AUC under this protocol | RQ1 premise | Run E002–E004 under the identical protocol and compare |
| H2 | The four architectures' differences will be small relative to run-to-run variance | Single-seed design gives no variance estimate; published architecture gaps on this task are modest | Requires multi-seed runs, which are **not currently budgeted** |
| H3 | Discrimination learned from CheXbert labels transfers to radiologist labels | ROC-AUC 0.8647 → 0.8577 on 202 images | Repeat for each baseline; the pattern holding across architectures would strengthen it considerably |
| H4 | The operating point, not the representation, is what fails on the expert set | ROC-AUC stable while sensitivity halves | Sweep the decision threshold on the official set **for reporting only**, never feeding it back into selection |
| H5 | Explicit recalibration would materially improve Brier/ECE without changing ROC-AUC | Reliability curves are monotone but systematically over-confident | Fit a calibrator on carved validation, evaluate on official validation |

H2 is the uncomfortable one. If it holds, RQ1's honest answer may be "the architectures
are not distinguishable under this protocol at this scale" — a **negative result the
project has committed to reporting**.

---

## 7. Limitations

### Dataset and labels

- Training labels are **report-derived, not image-verified**. Accepted, and stated in
  every reporting surface.
- Single-institution, historical data. **No cross-site generalisation claims.**
- AP/PA is confounded with patient acuity. Measured and reported stratified; not removed.
- A small population of visually degraded images decodes cleanly and passes every
  automated check. Detecting them needs visual review, which is out of scope. Named as a
  residual data-quality limitation.

### Evaluation

- **The test arm is a CheXbert-labelled fallback carve**, not the official 5-radiologist
  consensus test set. It is therefore not radiologist ground truth, and must never be
  described as expert-adjudicated.
- The carved test split **has not been evaluated** and will be scored once only, after the
  model set is frozen.
- **No confidence intervals exist anywhere in the project.** The metrics module has no
  interval estimator. Adding one is a deliberate, unbudgeted decision, not an oversight to
  paper over.
- The official validation set is **202 usable images with 66 positives** — enough for a
  sanity check, not enough for a precise estimate.

### Experimental design

- **One seed. One run per architecture.** No variance estimate. This limits how strongly
  any RQ1 ranking can be stated, and the limitation must appear beside any such ranking.
- Only one of four baselines is complete.
- The completed run consumed 8 of its 15 permitted epochs; it was not trained to budget
  exhaustion.

### Provenance of the official validation copy

Handled separately in §8 because it is easy to overstate.

---

## 8. Official-validation provenance — exact status

The project holds a `valid.csv` and a `valid/` image tree that match the official CheXpert
validation set on every structural property that can be checked locally.

**CONFIRMED:**

- 234 rows, 234 image files, an **exact 1:1 match** between CSV paths and files on disk.
- 200 patients, 200 studies, patient identifiers forming a contiguous block **disjoint
  from every training patient**.
- Column schema **identical** to the training metadata.
- All 14 observation columns contain **only `0.0` / `1.0`** — no uncertain values, no
  blanks. This is the structural signature of an adjudicated set rather than labeller
  output.
- 202 frontal / 32 lateral, matching the expected composition.

**INFERRED, not confirmed:** that these labels are the published radiologist majority-vote
annotations. The file contains no annotator columns and no provenance metadata; the
inference rests on the structural match plus the published description of the release.

**UNKNOWN:** the chain of custody of this particular copy. The project has previously
investigated third-party redistributions of this dataset and rejected them on licensing
grounds, and that concern is not retired by a structural match.

**VERIFY before any publication:** checksum this copy against a first-party release, and
confirm the applicable data-use agreement covers it.

> The correct statement is: *"a copy matching the official validation set's structure was
> evaluated."* The statement *"the official expert-adjudicated validation set was
> evaluated"* carries more certainty than the evidence supports.

---

## 9. Open research questions

| # | Question | Blocking? |
|---:|---|---|
| R1 | Which of the four architectures performs best under the frozen protocol? | RQ1 — needs E002–E004 |
| R2 | Is the between-architecture difference larger than run-to-run variance? | Needs multi-seed runs; currently unbudgeted |
| R3 | Does the ROC-AUC transfer to expert labels hold across architectures? | Needs E002–E004 plus their expert checks |
| R4 | What operating point would make the model clinically useful for case-finding, and what is the precision cost? | Reporting-only threshold sweep |
| R5 | Would explicit recalibration fix the over-confidence without harming ranking? | Needs a calibrator; none implemented |
| R6 | How much do the conclusions depend on the uncertainty policy? | U-Ones / exclusion ablations, specified but not run |
| R7 | Can the official 5-radiologist consensus **test** set be obtained? | Would upgrade the test arm from fallback carve to expert ground truth |
| R8 | Do the strongest models learn different representations, and does fusing them help? | RQ2/RQ3 — needs ≥2 baselines |

---

## 10. Logic for the next experiments

The next step is **E002 (DenseNet201)** under a protocol **identical** to E001.

The reasoning is ordering, not preference: RQ2–RQ6 all depend on having at least two
comparable baselines, and every deviation introduced before the baseline set is complete
compounds into the comparison. Specifically, the following must **not** change between
E001 and E004:

split files and their hashes · seed · uncertainty policy · view policy · `pos_weight` ·
preprocessing · augmentation · optimiser and its hyperparameters · scheduler · effective
batch · epoch budget · early-stopping rule · monitored metric · threshold policy.

Two things should be decided **before** E002 rather than after, because deciding them
later would require re-running completed work:

1. **Whether confidence intervals will be reported.** If yes, the estimator belongs in the
   metrics module now, so every baseline is scored identically. Retrofitting after four
   runs means recomputing all of them.
2. **Whether multi-seed runs are affordable.** H2 is untestable without them, and RQ1's
   answer is weak without H2 resolved. This is a compute-budget decision, and it is a
   research decision as much as an engineering one.

Neither is resolved. Both are **UNKNOWN** and should be settled explicitly rather than by
default.
