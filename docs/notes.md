# Notes — chronological engineering and research log

**Last updated:** 31/08/2026

Operational history in the order it happened, plus the hard-earned knowledge that came out
of it. Entries are **historical records**: each says what was true *at that time*. Where a
past belief has since been overturned it is marked **SUPERSEDED** and kept, never deleted —
the record of what was believed, and when, is part of the evidence.

For **current** state see `MASTER_CONTEXT.md`; for decisions and dataset evidence see
`dataset_analysis.md`; for results see `experiments.md`; for reasoning see `research.md`.

> **Reading rule.** Nothing in the dated log below is a statement about the present.
> Section 3, *Current state*, is the only part of this file that describes now.

---

## 1. Chronological log

### 16/08/2026 — Stage 1 closes, Stage 2 opens

Literature review completed and audited. Scope narrowed from generic "heart disease
classification" to **binary cardiomegaly classification from frontal chest X-rays** — a
more defensible and reproducible public-data task. Four baseline architectures fixed:
ResNet152, DenseNet201, EfficientNetV2, MobileNetV3.

The Stage 2 methodology document was written **before any data existed**, deliberately: the
point was to fix the split rule, label policy and metric protocol while there was no
temptation to tune them to observed numbers.

Hardware measured: RTX 4060 Laptop 8 GB, i7-13650HX, 31.7 GiB RAM.

### 16/08/2026 — pipeline verified against a synthetic fixture

All Stage 2 modules were exercised end-to-end against a synthetic dataset-shaped fixture
(400 patients, 1,199 rows, real JPEGs) because no real data was available.

**Two real bugs were found this way**, both of the silent-wrong-answer kind:

1. **Split fractions were wrong.** The obvious `StratifiedGroupKFold(n_splits=round(1/frac))`
   only lands on the requested fraction when `1/frac` is an integer — asking for 15%
   silently produced 1/7 ≈ 14.3%. Fixed by cutting 20 fixed folds and unioning
   `round(frac × 20)` of them.
2. **The axis formatter collapsed sub-1000 counts to "0k".** Cosmetic, but it would have
   put a wrong number in a report figure.

**Knowledge:** a fixture that only exercises the happy path finds nothing. Both bugs
surfaced because the fixture had *awkward* proportions, not clean ones.

### 16–21/08/2026 — the dataset acquisition problem

This period was dominated by a single problem: **obtaining a legitimate copy of the data.**

**Failure — assuming the small release was simply downloadable.** The methodology was
written assuming the current official portal exposed a downsampled release containing
`train.csv` and `valid.csv`. It did not. What the portal exposed was the full multi-hundred-
gigabyte archive and a newer DICOM-plus-reports release. The ~11 GB downsampled release —
the one the entire compute plan depends on, because its native short side is exactly the
320 px training resolution — had **no confirmed first-party download source**.

**SUPERSEDED belief:** *"the small release is available from the official portal."* It was
never verified, and treating it as settled cost time.

**Knowledge:** an availability assumption is a dependency. It deserves the same
verification as a measurement, and the same certainty label.

**Failure — treating the release decision as final too early.** The release decision had
been recorded as settled while one of its inputs was unverified. It was downgraded and the
certainty vocabulary tightened so that "settled in principle, one input unverified" became
a distinct, visible status rather than being rounded up to "final".

**Failure — a third-party mirror looked attractive.** A redistribution of the dataset was
investigated and **rejected**. It was structurally consistent with the genuine release and
contained the files needed, including the `valid.csv` that was otherwise missing. It was
rejected on licensing grounds: the copy asserted a permissive licence that the dataset's
actual research-use agreement does not grant, and using it would have meant representing a
redistribution as first-party data.

**Knowledge, and the standing rule that came from it:** convenience is not provenance. A
structural match to the real dataset says nothing about whether the copy may lawfully be
used. This rule outlived the incident and still binds — see §3 and `research.md` §8.

**Failure — the full archive as a workaround.** Downloading the several-hundred-gigabyte
release was considered and rejected: it does not fit the storage budget, and every pixel
above 320 px would be discarded at load time anyway. Paying that cost to obtain images the
pipeline immediately downsamples is not a trade worth making.

Fallback datasets were surveyed against the criterion *what survives a dataset switch* —
the split rule, label policy and metric protocol are dataset-independent; the specific
labeller semantics are not.

### 21/08/2026 — metadata acquired, first real-data run

Training metadata obtained. Two label files were present, and the difference between them
turned out to be decision-grade:

- **`train_cheXbert.csv`** — chosen as primary. It is the **CheXbert** relabelling, **not**
  the original rule-based labeller. Same images, same 4-state encoding, different and
  improved labels. Consequence recorded permanently: published uncertainty-policy results
  are *indicative* for this file, not directly applicable.
- **`train_visualCheXbert.csv`** — rejected as primary, retained as an optional ablation.

**Failure — the visually-derived file looked like an obvious upgrade.** It is
image-derived rather than report-derived, which sounds strictly better for an imaging
task. It is not, for this project: it has a completely different frontal prevalence and
**no uncertain or blank states at all**. Adopting it would have silently voided the
uncertainty policy — there would be no uncertainty left to police — and inverted the
class-imbalance weighting. Two of the project's settled decisions would have become
meaningless without any error being raised.

**Knowledge:** "better labels" is not a property of a file. It is a property of a file
*relative to the decisions already built on top of it*.

First real-data runs: metadata analysis and split generation both executed. **223,414
paths parsed, zero unparseable.** Patient disjointness held on the real patient
population. The split landed at **40,002 / 29,380 / 28,883** images.

**`valid.csv` was not in the acquired metadata.** Every one of the 223,414 rows was a
training row. The warn-and-continue branch that had been written speculatively became the
live code path. The official validation set — the secondary expert check the methodology
depends on — was recorded as an open, non-blocking gap.

Because the official test set was also not obtained, the split was generated in
**fallback-carved-test** mode: a 70/15/15 patient-level carve. This materially weakens the
ground truth and is flagged everywhere it appears — the test arm is labeller-derived, not
radiologist-adjudicated.

### 21/08/2026 — the image path-prefix defect

The split CSVs carry a **full-release** path prefix, while the downsampled release unpacks
under a **different top-level directory name**. Joining the two naively resolves to
nothing — silently, for every image in the dataset.

This is the defect class the project is most exposed to: it produces no error, just an
empty result that looks like a legitimately empty dataset.

**Fix.** The two ad-hoc per-script resolvers were replaced with **one** explicit resolver
in the shared metadata library, driven by a new `images:` configuration block. It strips
the leading release segment, normalises separators, and re-roots the path. A companion
`verify_image_root()` raises a listing of what *is* present when the layout is wrong,
rather than failing per-file thousands of times.

Regression-tested against a throwaway release tree: loud failure on every missing-layout
case, correct stripping across four prefix variants including Windows separators, correct
resolution for frontal AP, frontal PA and lateral, and an explicit assertion that the old
naive join does **not** resolve. Verified against real data: **all 98,265 split-CSV paths
resolve, 0 malformed.**

**Knowledge:** two implementations of one concept will diverge. The bug was possible only
because "where is this image" had two answers.

### 21/08/2026 — the VRAM probe, and a machine-specific trap

Per-model VRAM and throughput were measured rather than estimated.

**Failure — the probe initially accepted a fallback as success.** This machine's CUDA
driver has **system-memory fallback enabled**, so exceeding VRAM does **not** raise an
out-of-memory error. It silently spills to host RAM and keeps running at roughly a tenth
of the speed. A naive grow-until-it-crashes probe therefore reports an impossible batch
size that "works". Detection is by *throughput collapse* plus reserved memory exceeding
the physical card, and the probe now reports the maximum **efficient** batch rather than
the maximum non-crashing one.

**Three further bugs in the probe, all found by running it:** cleanup ran outside the
`try` block; only one OOM exception type was caught while the installed framework raises a
different one; and garbage collection in `finally` could not free GPU tensors because the
frame's locals were still bound.

**Knowledge:** "it did not crash" is not evidence that it fit. On this machine it is
evidence of almost nothing.

### 24/08/2026 — silent hash failures

The duplicate sweep returned an empty string on any hashing exception, and the caller
dropped those rows with a bare truthiness check. A damaged or unreadable file therefore
**vanished from the analysis** while the report still printed "Files hashed: N" as though
complete — and damaged files are exactly what the sweep exists to find.

Failures are now returned as data, collected, written to a dedicated failures CSV, and
raised. The script exits non-zero and writes **no** duplicates report at all. Covered by a
29-check regression test.

**Knowledge:** an error swallowed by a falsy return is indistinguishable from a clean
result. This is the same failure shape as the path-prefix defect, and the third instance
of it in the project.

### 30/08/2026 — images obtained; the directory layout was not what the config expected

The image archive was extracted successfully. Both a `train/` tree and a **`valid/` tree**
were present — and with the `valid/` tree came `valid.csv`, the official validation
metadata that had been missing since 21/08.

The extracted layout did not match the configured expectation: the configuration pointed
at a nested raw-download directory containing a named release folder, while the actual
extraction placed `train/`, `valid/`, `train.csv` and `valid.csv` directly under the
project data directory. Two configuration values were corrected — the image root and the
release directory name — and the validation-metadata path was pointed at the newly present
file. **No code changed**; the centralised resolver absorbed the difference, which is what
it was built for.

Pre-training integrity work ran on the real images: validation, dimension profiling,
exact-duplicate detection, and the cross-patient near-duplicate sweep. No rows were
excluded from any split as a result.

### 30/08/2026 — ResNet152 baseline started, then interrupted

Training began. **Epoch 1 completed** — validation PR-AUC **0.5260268660904089** — and the
epoch-1 checkpoint was written. The process was then interrupted overnight, mid-write.

What was left on disk:

- `best.pt` — complete and valid, epoch 1.
- `last.pt` — complete, epoch 1.
- `last.pt.tmp` — a **partially written** file, roughly half the size of a complete
  checkpoint, from the write that was in progress when the process died.

The temporary file was correctly identified as unusable and left alone. The atomic-write
design worked exactly as intended: an interruption mid-write left a truncated `.tmp`
rather than a corrupted `best.pt`.

### 31/08/2026 — resume support

The training entry point had no resume capability. It was added as the smallest safe
change that could continue that specific run:

- an explicit `--resume <checkpoint>` argument — **never** automatic, because a silent
  auto-resume is a silent methodology change;
- restoration of model, optimizer, scheduler, AMP scaler, epoch counter and best-metric
  bookkeeping;
- the epoch loop starting at `checkpoint_epoch + 1` against the **unchanged** total
  budget, so the budget is a total rather than something that quietly restarts;
- reconstruction of the early-stopping patience counter, so a resume cannot reset
  patience;
- the resumed run reusing the **original** run directory and reading that run's own
  configuration snapshot, so a later configuration edit cannot drift the methodology;
- refusal of `--epochs`, `--seed` and `--model` alongside `--resume`, since changing the
  epoch count would change the cosine schedule length;
- the original run manifest and configuration snapshot left **byte-identical**, with
  resume provenance written alongside them.

Verified before use: 36 checks restoring the real checkpoint into freshly built objects —
all 932 model tensors bit-exact, all optimizer moments and step counters exact, scheduler
and AMP scaler state restored — plus 24 end-to-end checks on a synthetic fixture, and the
existing suite still passing.

#### The missing RNG state — a permanent, unfixable gap in this run

The epoch-1 checkpoint contained **no RNG state**. Nothing had ever captured it.

Consequence, stated plainly and permanently: epochs 2–15 restarted their augmentation and
shuffle streams from the run seed rather than continuing epoch 1's stream. Model,
optimizer, scheduler and scaler state carried across **exactly**; only the random stream
did not. **This specific run is therefore not reproducible as a single uninterrupted
trajectory**, and no amount of later work can recover it.

RNG capture was added going forward, so checkpoints from epoch 2 onward carry it and this
gap cannot recur. One residual gap remains open and is **not** closed: the data-loader
shuffle generator is re-seeded at process start and is still not carried in checkpoints.

**Knowledge:** "restores training state" is an incomplete specification. It is worth
enumerating *which* state, because the pieces that are easy to forget are the ones nobody
notices are missing until a run is interrupted.

#### The `last.pt.tmp` collision — an instruction that could not be honoured

A constraint was set for the resumed run: do not touch the partially written temporary
file.

**It was not honoured, and it could not have been.** The checkpoint writer builds its
temporary path by appending `.tmp` to the destination — so the routine per-epoch write of
`last.pt` uses **exactly** that filename. The stale partial was overwritten by epoch 2's
save and renamed away, and the same mechanism replaced the epoch-1 `last.pt`.

This should have been predicted before launch rather than discovered after. The collision
was visible from the writer's implementation.

**No state was lost** — the partial was an unloadable fragment already ruled out, and the
epoch-1 weights it shared with `last.pt` were preserved in a backup copy made before the
resume. But the instruction was violated.

**Knowledge:** a "do not touch this file" constraint is unenforceable when a routine code
path derives that exact filename. Making it enforceable requires a unique temporary
suffix, which has **not** been implemented.

### 31/08/2026 — the resumed run completes

Resumed from the epoch-1 checkpoint at epoch 2. A backup of the epoch-1 state was made
before training, and the original run provenance was left untouched.

Epochs 2 and 3 improved; epochs 4–8 did not. **Early stopping fired at epoch 8** after
five consecutive non-improving epochs. **Exit code 0.**

**Best epoch 3, validation PR-AUC 0.5417253378785041**, ROC-AUC 0.8647.

Training loss fell monotonically 0.865 → 0.631 while validation PR-AUC plateaued after
epoch 3 — the signature of overfitting from epoch 4 onward, and a sign the stop was well
placed.

### 31/08/2026 — official expert-validation check

The newly available validation set was audited before use. Structure matched the official
set on every locally checkable property: 234 rows, 234 image files in exact 1:1
correspondence, 200 patients in a contiguous block disjoint from every training patient,
schema identical to the training metadata, and **only `0.0`/`1.0` across all 14
observation columns** — no uncertain values, no blanks, which is the structural signature
of an adjudicated set.

The evaluation reused the existing pipeline rather than introducing a parallel evaluator:
the same identifier parsing, view filter, uncertainty policy, path resolver, dataset class,
evaluation transform and metric functions the frozen splits use. The only new code is a
thin driver.

**Frontal-only: 202 evaluated, 32 laterals excluded.** The carved-validation threshold was
applied **unchanged**; the driver has no threshold-selection path at all and refuses to
report if the applied policy is not `inherited`.

Result: **ROC-AUC 0.8577, PR-AUC 0.7547** at a prevalence of 0.32673.

**The interpretation trap, recorded because it is easy to fall into.** PR-AUC rose from
0.5417 to 0.7547. That is **not** an improvement — the PR-AUC no-skill baseline *is* the
prevalence, and prevalence rose from 0.13652 to 0.32673. Lift over baseline actually fell,
3.97× → 2.31×. The genuinely informative comparison is ROC-AUC, which is
prevalence-independent and barely moved.

Sensitivity fell 0.6046 → 0.3030 while specificity rose 0.8974 → 0.9706: a threshold tuned
for one prevalence behaving conservatively at another. Consistent with threshold transfer,
**not** evidence of discrimination failure, given ROC-AUC held.

**Confidence intervals were not computed.** The metrics module has no interval estimator,
and inventing one for a single report would have produced a number with no defensible
method behind it. Recorded as a limitation instead.

---

## 2. Recurring lessons

Four incidents, one shape.

| Incident | What it looked like | What it was |
|---|---|---|
| Path-prefix mismatch | zero images found | every path silently unresolvable |
| Hash failures | "Files hashed: N", report complete | damaged files silently dropped |
| VRAM fallback | large batch "fits" | spilling to host RAM at ~10× slowdown |
| Missing RNG state | resume "restores training state" | one component silently absent |

**The pattern: the system reported success while doing the wrong thing.** None of these
raised an error. Each was found by measuring something the code did not itself check.

Consequences now built into the project:

1. **Failures are returned as data and raised**, never converted to a falsy value.
2. **One concept, one implementation.** Two resolvers meant two answers.
3. **"It did not crash" is not evidence.** Especially not on this machine.
4. **Enumerate what is restored, not just that state is restored.**
5. **A fixture with awkward proportions finds bugs a clean one hides.**

---

## 3. Current state — as of 31/08/2026

This is the only section describing the present.

| | |
|---|---|
| **Stage** | 3 — baseline training, in progress |
| **Data on disk** | training metadata, validation metadata, and both image trees |
| **Splits** | frozen: 40,002 / 29,380 / 28,883, patient-disjoint, seed 42, fallback-carved-test mode |
| **Experiments complete** | **1 of 4** — ResNet152 (`E001`) |
| **Best model** | epoch 3, validation PR-AUC 0.5417 |
| **Expert check** | performed on 202 frontal official-validation images |
| **Carved test split** | **NOT EVALUATED** |
| **Test arm quality** | labeller-derived fallback carve, **not** radiologist-adjudicated |
| **Confidence intervals** | none, anywhere — no estimator implemented |
| **Next step** | DenseNet201 (`E002`) under an identical protocol |

### Standing operational cautions

- **The official test set has not been obtained.** Until it is, the test arm is a fallback
  carve and must never be called expert-adjudicated.
- **Provenance of the validation copy is structurally confirmed but not independently
  established.** See `research.md` §8. Checksum against a first-party release before
  publishing anything that rests on it.
- **System-memory fallback is enabled on this machine.** Never size a batch by raising it
  until something crashes.
- **The `.tmp` filename collision is still live.** Any future instruction to preserve a
  specific `*.pt.tmp` file cannot be honoured while a routine save targets the same name.
- **No confidence intervals.** Decide before the next baseline whether they will be
  reported, because retrofitting means recomputing every completed run.
