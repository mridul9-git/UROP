# Literature Review v2 — Audited Working Version

**Project:** Comparative Deep Learning for Cardiomegaly Classification from Chest X-ray Images with Hybrid Feature Fusion, Hyperparameter Optimization, and Explainable AI  
**Version:** 2.0 — audited working version  
**Date:** 16/08/2026  
**Status:** Stage 1 approved for transition to Stage 2

> This document is the audited working version of the original `literature_review.md`. The original remains the research archive; this version contains the corrections and scope reductions agreed after audit.

## 1. Executive Summary

The project is narrowed from generic “heart disease classification” to **binary cardiomegaly classification from frontal chest X-rays**. This is a more defensible and reproducible public-data task.

The current primary dataset candidate is **CheXpert**, subject to final Stage 2 verification. Its major strengths are scale, a Cardiomegaly observation, patient identifiers, and expert-annotated evaluation data. Its major limitation is that training labels are report-derived rather than image-confirmed clinical ground truth.

The baseline study will compare **ResNet152, DenseNet201, EfficientNetV2, and MobileNetV3** under one controlled protocol. The strongest two models will feed one frozen-feature concatenation hybrid. Optuna/TPE will optimize the strongest model, and CNN embeddings will be compared using a conventional neural head and XGBoost. Grad-CAM will be the core XAI method.

The project makes **no architectural novelty claim**. It is a controlled comparative study / partial replication-extension whose value is experimental rigor, reproducibility, and honest reporting of positive or negative results.

## 2. Core Research Questions

1. Which selected CNN architecture performs best for cardiomegaly classification under identical conditions?
2. Do the strongest CNNs learn meaningfully different feature representations?
3. Does combining the top-two representations improve performance?
4. Does hyperparameter optimization improve the strongest model?
5. Does XGBoost provide competitive performance against a neural classifier on the same frozen embeddings?
6. What do Grad-CAM explanations reveal about correct and incorrect predictions?

## 3. Dataset Recommendation — CheXpert

### Working decision
**CheXpert, frontal views, binary Cardiomegaly target.**

### Why
- Large public CXR dataset.
- Contains Cardiomegaly.
- Patient identifiers support leakage-controlled splitting.
- Expert-annotated validation/test subsets provide stronger evaluation context.
- Extensive literature makes benchmarking practical.
- A downsampled release is more realistic for student GPU/Colab resources.

### Critical limitation
CheXpert training labels are automatically extracted from radiology reports. Therefore:

> **Performance should be described as agreement with a report-derived target, not proof of clinical diagnostic accuracy.**

Also document the single-institution/historical-data limitation and avoid generalization claims.

## 4. Dataset Alternatives

### VinDr-CXR
Useful secondary asset because it contains radiologist annotations and cardiomegaly localization information. Access requires credentialing/training, so it is **not on the critical path**.

Do not claim a fixed “1–3 week” access time. Use: **credentialing may introduce a critical-path delay.**

If available later, VinDr-CXR may support optional Grad-CAM localization evaluation.

### MIMIC-CXR-JPG
Strong large-scale option, but access requirements and download volume make it less attractive under the current time constraint. If the team already has access, reassess the primary dataset choice before implementation.

## 5. Label Uncertainty

CheXpert contains uncertain labels. Do **not** hard-code a universal policy before Stage 2.

Possible strategies include U-Zero, U-One, U-MultiClass, or exclusion. The original CheXpert benchmark used pathology-specific uncertainty strategies.

**Stage 2 must inspect the actual release and justify the selected Cardiomegaly policy.**

## 6. CNN Architecture Research

| Model | Architectural role |
|---|---|
| ResNet152 | Residual learning |
| DenseNet201 | Dense feature reuse |
| EfficientNetV2 | Efficient modern scaling |
| MobileNetV3 | Lightweight architecture |

The purpose is architectural diversity under one controlled protocol, **not** a claim that these are universally the four best CXR models.

Literature comparing many CXR architectures indicates that architecture family can matter substantially and that ImageNet performance does not automatically predict medical-imaging performance.

## 7. Transfer Learning and Preprocessing

Transfer-learning evidence is nuanced: some medical-imaging studies found limited benefit in certain settings, while CXR-specific work has found measurable benefits. Therefore, transfer learning is a reasonable starting point, not an assumption that must produce an improvement.

Use one common preprocessing pipeline across models:
- resize
- normalization
- consistent grayscale/RGB handling
- medically defensible augmentation
- training-only augmentation

CLAHE may be considered later as an ablation; it should not silently become a model-specific preprocessing trick.

Horizontal flipping is **not established by this review as universally harmful or universally safe** for cardiomegaly. Treat it as a methodological choice requiring justification.

## 8. Evaluation Metrics

Accuracy must not be the sole metric under class imbalance.

Core reporting:
- ROC-AUC
- PR-AUC
- Recall/Sensitivity
- Specificity
- Precision
- F1
- Confusion matrix

Threshold selection must be defined before final test evaluation and must never tune on the test set.

## 9. Feature Analysis

After the four baselines, select the strongest two using predefined evaluation criteria. Extract embeddings from a consistent feature stage and visualize them with UMAP, t-SNE, or PCA.

Embedding plots are descriptive: they do not prove clinical superiority. They are used to inspect class separability, overlap, and potential differences between learned representations.

## 10. Hybrid Feature Fusion

The literature contains both positive and null fusion results. Therefore, fusion is an empirical question.

Core design:

```text
X-ray
  ├── CNN A → frozen Feature A ─┐
  └── CNN B → frozen Feature B ─┤
                                ↓
                          Concatenate
                                ↓
                           Small MLP
                                ↓
                           Prediction
```

The two backbones are frozen at their selected baseline checkpoints for the initial hybrid experiment. This keeps the question focused on whether their learned representations complement each other.

Do not expand into multiple hybrid architectures or attention/weighted fusion unless the core project is already stable.

## 11. Fusion Controls

Core:
- Best CNN A
- Best CNN B
- A+B concatenation

Optional:
- self-concatenation `[A ; A]` as a capacity control

The self-concatenation idea should **not** be described as universally novel. Correct wording:

> “We did not identify a cardiomegaly/CXR study using this control in the surveyed literature.”

That is a scoped literature observation, not a “first ever” claim.

## 12. XGBoost vs Neural Head

Existing CNN→XGBoost/SVM/RF literature provides methodological precedent, but many comparisons use unequal optimization/training conditions.

Our question is:

> **Under a controlled setup using the same frozen CNN embeddings, does XGBoost provide competitive performance against the neural classification head?**

Use:
- same cached embeddings
- same split
- comparable tuning budget
- same metrics
- same threshold policy

Do not assume XGBoost will win. Direct cardiomegaly evidence includes null/near-equivalent results, making a null result meaningful.

## 13. Hyperparameter Optimization

Use **Optuna with TPE**. BOHB is removed from core scope.

Initial search:
- learning rate
- dropout
- weight decay
- batch size

Optimize the strongest selected model. If compute permits, the hybrid may receive an equivalent budget for a fair comparison.

Never use the test set for hyperparameter selection.

## 14. Explainable AI

### Core
**Grad-CAM**

Generate examples covering:
- true positive
- true negative
- false positive
- false negative

Select examples using a predefined rule rather than by visually choosing attractive heatmaps.

### Reliability
Core: Grad-CAM + **one model-weight randomization sanity check**.

Optional if ahead of schedule:
- cross-seed repeatability
- cross-architecture comparison
- support-device analysis
- VinDr-CXR localization evaluation

The full SSIM/IoU/AUPRC XAI battery from the original review is **not core scope**.

## 15. XAI Claims We Will Avoid

Do NOT claim:
- Grad-CAM proves clinical validity.
- Grad-CAM validates the model.
- Grad-CAM proves the model is trustworthy.
- Attractive heatmaps prove correct reasoning.

Correct framing:

> **Grad-CAM is used as a model debugging and failure-analysis tool to inspect spatial attribution patterns.**

## 16. Research Gaps

### Gap 1 — Controlled architecture comparison
Published CXR studies often differ in dataset, preprocessing, splits, and training protocol. Our contribution is a controlled comparison of four architectural families.

**Type:** controlled comparative study, not architectural novelty.

### Gap 2 — Feature fusion vs capacity
Fusion is often motivated by “complementary features,” but increased feature capacity is not always isolated.

**Type:** potential methodological extension; self-concatenation is optional.

### Gap 3 — Neural vs classical classifier fairness
CNN→XGBoost literature often compares unequal training/tuning regimes.

**Type:** controlled replication/extension.

### Gap 4 — XAI reliability
Medical-imaging studies frequently present saliency maps without rigorous falsification.

**Type:** application of established XAI validation methodology; not a new XAI method.

## 17. Honest Overall Research Claim

> **This project makes no methodological or architectural novelty claim. It is a controlled comparative study and partial replication/extension that evaluates whether architecture-family effects reported in CXR research hold for cardiomegaly under a fixed protocol, investigates whether combining learned representations provides a measurable benefit, compares neural and classical classifier heads under controlled conditions, and uses Grad-CAM as a cautious failure-analysis tool. Its value is experimental rigor, reproducibility, and tolerance for negative results rather than novelty theatre.**

## 18. Proposed Experimental Pipeline

```text
CheXpert
  ↓
Frontal CXR + Cardiomegaly
  ↓
Patient-level split
  ↓
Fixed preprocessing
  ↓
ResNet152 / DenseNet201 / EfficientNetV2 / MobileNetV3
  ↓
Baseline comparison
  ↓
Top 2 models
  ↓
Feature analysis
  ↓
Frozen feature-concatenation hybrid
  ↓
Best-model selection
  ↓
Neural head vs XGBoost
  ↓
Optuna
  ↓
Final model
  ↓
Grad-CAM
  ↓
Final analysis
```

## 19. Experimental Fairness

Must remain identical across baseline comparisons:
- dataset split
- patient-level separation
- preprocessing
- augmentation
- evaluation set
- metric definitions
- threshold policy
- checkpoint-selection policy
- seed policy

Architecture-specific differences are allowed only where technically necessary.

## 20. Reproducibility

Every experiment records:

```text
Experiment ID
Date
Dataset version
Dataset split
Model
Pretrained weights
Image size
Batch size
Optimizer
Learning rate
Weight decay
Dropout
Epochs
Early stopping
Random seed
Hardware
Training time
Checkpoint
Metrics
Observations
```

Recommended assets:
- committed split CSV
- pinned requirements
- configuration files
- run logs
- git commit SHA
- raw per-seed results

## 21. Failure Modes

We must avoid:
- patient/image leakage
- test-set tuning
- inconsistent preprocessing
- accuracy-only reporting
- cherry-picked XAI examples
- calling cardiomegaly “general heart-disease diagnosis”
- clinical-readiness claims
- unsupported “first ever” novelty claims
- silently changing experimental conditions between models

## 22. Scope After Audit

### Core
- CheXpert cardiomegaly task
- dataset analysis
- leakage-controlled split
- fixed preprocessing
- four CNN baselines
- feature analysis
- one concatenation hybrid
- Optuna
- neural vs XGBoost head comparison
- Grad-CAM
- one XAI sanity check
- final comparative analysis
- report/paper/presentation

### Optional
- self-concatenation capacity control
- cross-seed Grad-CAM repeatability
- cross-architecture Grad-CAM comparison
- support-device analysis
- VinDr-CXR localization
- additional fusion methods

### Removed from core
- VGG19
- InceptionV3
- MobileNetV2
- multiple hybrids
- attention/weighted fusion
- BOHB
- SHAP
- LIME
- large XAI falsification battery
- CKA/linear probes/McNemar unless later justified

## 23. Stage 1 Decision Log

| ID | Decision | Reason |
|---|---|---|
| D001 | Cardiomegaly rather than generic heart disease | Defensible public CXR target |
| D002 | CheXpert as primary candidate | Label + scale + patient IDs + practical access |
| D003 | Four CNN baselines | Architectural diversity without scope explosion |
| D004 | Top-2 frozen feature concatenation | Tests representation complementarity cheaply |
| D005 | Optuna/TPE only | Sufficient for time-boxed project |
| D006 | Neural vs XGBoost on same frozen embeddings | Fair classifier-head comparison |
| D007 | Grad-CAM as core XAI | Strong precedent and spatial interpretation |
| D008 | Controlled study/replication framing | Avoids unsupported novelty claims |

## 24. Stage 2 Entry Criteria

Before implementation, verify:

1. Exact CheXpert release/access.
2. Cardiomegaly label distribution.
3. Uncertainty-label handling.
4. Patient identifiers.
5. Available official splits.
6. Test-label availability.
7. Image format/resolution.
8. Storage requirements.
9. Available GPU/compute.
10. Exact binary classification definition.

Only then should the preprocessing/training pipeline be implemented.

## 25. Final Stage 1 Verdict

# APPROVED WITH AUDIT CORRECTIONS

The original `literature_review.md` remains the **research archive**.

This v2 is the **working methodology document**.

Stage 1 is complete enough to move to:

> **Stage 2 — Dataset Acquisition & Analysis**

The purpose of Stage 2 is not yet to train models. It is to prove that the selected dataset, labels, split strategy, preprocessing requirements, and compute environment are suitable for the experiments defined above.