# Heart Disease Classification Research Worklet (UROP)

## Project Title

**Comparative Deep Learning for Heart Disease Classification from Chest X-ray Images with Hybrid Feature Fusion, Hyperparameter Optimization, and Explainable AI**

---

## 1. Project Objective

Develop a focused and reproducible deep learning research pipeline for heart disease classification from chest X-ray images.

The project will:

1. Analyze and preprocess a suitable chest X-ray dataset.
2. Benchmark multiple CNN architectures under controlled experimental conditions.
3. Analyze learned feature representations from the strongest models.
4. Design and evaluate one hybrid CNN feature-fusion model.
5. Optimize the strongest model using automated hyperparameter tuning.
6. Compare conventional neural classification with XGBoost on extracted CNN features.
7. Apply Grad-CAM for visual interpretation of model predictions.
8. Perform a consolidated experimental analysis and document the findings as a UROP research report/paper.

### Research Philosophy

The original worklet proposed a very broad experimental space. This version intentionally reduces the number of experiments while preserving a complete research narrative.

The priority is:

> **Fewer experiments, stronger experimental control, reproducibility, and genuine understanding of the resulting system.**

---

# 2. Research Questions

The project will focus on the following research questions.

### RQ1 — CNN Architecture

Which selected CNN architecture provides the strongest performance for chest X-ray heart disease classification under identical experimental conditions?

### RQ2 — Feature Representation

Do the strongest CNN architectures learn meaningfully different and separable feature representations?

### RQ3 — Hybrid Feature Fusion

Can combining feature representations from two complementary CNN architectures improve classification performance over the individual models?

### RQ4 — Hyperparameter Optimization

Can automated hyperparameter optimization improve the performance of the strongest baseline or hybrid model?

### RQ5 — Classifier Strategy

Does XGBoost applied to CNN-extracted feature vectors provide competitive performance compared with the conventional neural-network classifier?

### RQ6 — Explainability

Does Grad-CAM indicate that the final model relies on meaningful spatial regions of chest X-ray images when making predictions?

---

# 3. Final Project Pipeline

```text
                    CHEST X-RAY DATASET
                            |
                            v
                DATASET ANALYSIS & CLEANING
                            |
                            v
                  PREPROCESSING PIPELINE
                            |
                            v
                    CNN BENCHMARKING
                            |
          +---------+-------+-------+---------+
          |         |       |       |         |
          v         v       v       v         |
       ResNet152 DenseNet201 EfficientNetV2 MobileNetV3
          |         |       |       |
          +---------+-------+-------+
                            |
                            v
                    MODEL COMPARISON
                            |
                            v
                     TOP 2 MODELS
                            |
                            v
                 FEATURE REPRESENTATION
                            |
                            v
                   HYBRID FEATURE FUSION
                            |
                            v
                    BEST MODEL SELECTION
                            |
                +-----------+-----------+
                |                       |
                v                       v
          SOFTMAX CLASSIFIER       XGBOOST
                |                       |
                +-----------+-----------+
                            |
                            v
                HYPERPARAMETER OPTIMIZATION
                            |
                            v
                     FINAL MODEL
                            |
                            v
                       GRAD-CAM
                            |
                            v
                 FINAL RESULT ANALYSIS
                            |
                            v
                UROP REPORT / PAPER / DECK
```

---

# 4. Stage 1 — Literature Survey

## Objective

Understand existing research in chest X-ray based cardiac/heart disease classification and establish the motivation for the proposed experiments.

## Tasks

Review approximately **10–12 recent research papers** covering:

- Chest X-ray based disease classification
- CNN-based medical image classification
- Transfer learning for chest X-rays
- DenseNet / ResNet / EfficientNet based approaches
- Hybrid or feature-fusion approaches
- Explainable AI in medical imaging
- Grad-CAM and related visual explanation methods

## Information to Extract

For every paper, record:

| Field | Description |
|---|---|
| Paper | Title / citation |
| Year | Publication year |
| Dataset | Dataset used |
| Images | Approximate dataset size |
| Task | Classification task |
| Architecture | Model(s) used |
| Preprocessing | Image processing methods |
| Metrics | Evaluation metrics |
| Best Result | Reported performance |
| XAI | Explainability method |
| Limitation | Main limitation |
| Relevance | Why it matters to this project |

## Deliverable

`literature_review.md`

Containing:

- Literature summary
- Research gap
- Dataset comparison
- Model comparison
- Evaluation metric comparison
- Motivation for the selected methodology

---

# 5. Stage 2 — Dataset Acquisition and Analysis

## Objective

Acquire and prepare a suitable chest X-ray dataset for binary or multiclass heart disease classification, depending on the selected dataset.

## Dataset Requirements

The selected dataset should have:

- Clearly defined labels
- Sufficient image count
- Consistent metadata
- Public/research-appropriate access
- A classification task compatible with the project objective

## Dataset Exploration

Determine:

- Total number of images
- Number of classes
- Images per class
- Class imbalance
- Image formats
- Image dimensions/resolutions
- Availability of patient identifiers
- Duplicate images
- Corrupted/unreadable files
- Missing or inconsistent labels

## Critical Data Integrity Requirement

If patient identifiers are available:

> **Train, validation, and test sets must be separated at the patient level rather than randomly splitting individual images.**

This is necessary to reduce the risk of patient-level data leakage.

## Data Cleaning

- Remove or isolate corrupted images.
- Detect duplicate images where practical.
- Verify labels.
- Identify missing metadata.
- Record all cleaning decisions.

## Deliverable

`dataset_analysis.md`

Containing:

- Dataset statistics
- Class distribution
- Data quality findings
- Cleaning decisions
- Dataset limitations

---

# 6. Stage 3 — Preprocessing Pipeline

## Objective

Create one reproducible preprocessing pipeline that is used consistently across all baseline models.

## Preprocessing

Potential steps:

1. Image loading
2. Color/grayscale handling according to model requirements
3. Image resizing
4. Pixel normalization
5. Dataset splitting
6. Data augmentation for training data only

## Augmentation

Potential augmentations:

- Small rotations
- Horizontal flipping where medically appropriate
- Moderate zoom
- Brightness adjustment

Augmentation must not create unrealistic medical images.

## Data Leakage Rule

Validation and test images must **not** receive training-only augmentation.

## Deliverables

- Reproducible preprocessing code
- `data_pipeline.md`
- Dataset distribution visualizations

---

# 7. Stage 4 — CNN Benchmarking

## Objective

Compare selected CNN architectures under controlled and consistent experimental conditions.

## Selected Models

### Model 1 — ResNet152

Represents deep residual learning.

### Model 2 — DenseNet201

Represents dense feature reuse and strong feature propagation.

### Model 3 — EfficientNetV2

Represents modern compound scaling and efficient CNN design.

### Model 4 — MobileNetV3

Represents lightweight and computationally efficient CNN architectures.

## Why These Four?

The selected models intentionally represent different architectural philosophies:

```text
ResNet152       → Residual learning
DenseNet201     → Dense connectivity
EfficientNetV2  → Efficient scaling
MobileNetV3     → Lightweight architecture
```

This provides meaningful architectural diversity without unnecessarily expanding the experiment count.

## Training Strategy

Use a common training framework for all models.

Potential components:

- Transfer learning
- Frozen backbone phase
- Fine-tuning phase
- Common optimizer
- Early stopping
- Model checkpointing
- Reproducible random seeds
- Consistent train/validation/test splits

## Evaluation Metrics

At minimum:

- Accuracy
- Precision
- Recall / Sensitivity
- F1 Score
- ROC-AUC
- Confusion Matrix

Where appropriate, additionally report:

- Specificity
- Per-class metrics
- ROC curves

## Primary Benchmark Table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| ResNet152 | | | | | |
| DenseNet201 | | | | | |
| EfficientNetV2 | | | | | |
| MobileNetV3 | | | | | |

## Deliverables

- Common model-training framework
- Training logs
- Evaluation results
- Confusion matrices
- Benchmark comparison table
- `models.md`
- `training_pipeline.md`

---

# 8. Stage 5 — Feature Analysis

## Objective

Investigate the representations learned by the strongest CNN architectures.

## Selection

Select the **top two baseline models** based on the primary evaluation criteria.

Selection must be based on recorded experimental results rather than subjective preference.

## Tasks

1. Extract feature vectors from the selected models.
2. Reduce feature dimensionality using an appropriate visualization method.
3. Visualize class separation.
4. Compare feature representations between models.

Possible visualization methods:

- t-SNE
- UMAP

## Questions

- Are the classes visually separable?
- Does one model produce more compact clusters?
- Are there overlapping regions?
- Do the two models appear to capture complementary information?

## Deliverable

`feature_analysis.md`

Including:

- Feature extraction methodology
- Embedding visualizations
- Observations
- Limitations

---

# 9. Stage 6 — Hybrid CNN Feature Fusion

## Objective

Determine whether combining complementary feature representations improves classification.

## Model Selection

Use the two strongest baseline CNNs identified in Stage 4.

## Initial Fusion Strategy

Use **feature concatenation**.

```text
                    INPUT X-RAY
                         |
             +-----------+-----------+
             |                       |
             v                       v
        CNN MODEL A             CNN MODEL B
             |                       |
             v                       v
        FEATURE A                FEATURE B
             |                       |
             +-----------+-----------+
                         |
                         v
                    CONCATENATE
                         |
                         v
                    DENSE LAYER
                         |
                         v
                      OUTPUT
```

## Evaluation

Compare:

- Model A
- Model B
- Hybrid Model

using the same test set and metrics.

## Important Research Outcome

If the hybrid model does not improve performance, this is **not automatically a failure**.

The result can be reported as:

> Feature fusion did not provide a meaningful improvement over the strongest individual backbone under the tested configuration.

## Scope Restriction

The initial implementation will **not** include:

- Multiple hybrid combinations
- Attention fusion
- Weighted fusion

These may be considered future work if time and compute permit.

## Deliverables

- Hybrid architecture diagram
- Hybrid implementation
- Performance comparison
- `hybrid_models.md`

---

# 10. Stage 7 — Hyperparameter Optimization

## Objective

Determine whether automated hyperparameter optimization can improve the strongest model.

## Optimization Framework

Use:

**Optuna with TPE-based optimization**

## Model to Optimize

Optimize only the strongest baseline or hybrid model rather than every architecture.

## Initial Search Space

Potential parameters:

- Learning rate
- Dropout rate
- Weight decay
- Batch size

Additional parameters may be added only if justified.

## Experimental Process

```text
Best Baseline / Hybrid
          |
          v
     Optuna Search
          |
          v
 Candidate Configurations
          |
          v
   Best Configuration
          |
          v
 Final Retraining
          |
          v
 Performance Comparison
```

## Evaluation

Compare:

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Original | | | | | |
| Optimized | | | | | |

## Scope Restriction

BOHB is removed from the core implementation.

It may be mentioned as a possible future optimization strategy.

## Deliverables

- Optimization configuration
- Trial results
- Best hyperparameter configuration
- Performance comparison
- `optimization.md`

---

# 11. Stage 8 — Classifier Architecture Study

## Objective

Compare conventional neural classification with a classical machine-learning classifier applied to learned CNN features.

## Selected Models

Use:

- Best CNN
- Best Hybrid

## Classification Strategies

### Approach A — Neural Classifier

```text
CNN
 ↓
Feature Representation
 ↓
Dense Layer
 ↓
Softmax / Sigmoid
 ↓
Prediction
```

### Approach B — XGBoost

```text
CNN
 ↓
Feature Extraction
 ↓
Feature Vector
 ↓
XGBoost
 ↓
Prediction
```

## Maximum Core Experiments

```text
Best CNN + Neural Classifier
Best CNN + XGBoost

Best Hybrid + Neural Classifier
Best Hybrid + XGBoost
```

## Evaluation

Use the same test set and evaluation metrics.

## Deliverable

`classifier_comparison.md`

---

# 12. Stage 9 — Explainable AI

## Objective

Investigate which image regions influence the final model's predictions.

## Primary XAI Method

### Grad-CAM

Grad-CAM will be the primary explainability technique because it provides spatial heatmaps for CNN predictions.

## Analysis Categories

Generate explanations for examples representing:

1. True Positive
2. True Negative
3. False Positive
4. False Negative

## Pipeline

```text
Chest X-ray
     |
     v
Final CNN
     |
     v
Prediction
     |
     v
Grad-CAM
     |
     v
Activation Heatmap
     |
     v
Overlay on X-ray
```

## Interpretation

Analyze whether the model appears to focus on meaningful anatomical/image regions.

### Important Limitation

Grad-CAM visualizations should **not** be treated as proof of clinical validity.

They indicate regions associated with the model's prediction, not definitive medical reasoning.

## Scope Restriction

SHAP and LIME are removed from the core implementation.

They may be listed as future work or added only if the core pipeline is already complete.

## Deliverables

- Grad-CAM implementation
- Heatmap examples
- Explanation analysis
- `xai.md`

---

# 13. Stage 10 — Final Comparative Analysis

## Objective

Consolidate all experiments into a single research narrative.

## Master Results Table

| Experiment | Model | Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|---:|
| Baseline | ResNet152 | Neural | | | | | |
| Baseline | DenseNet201 | Neural | | | | | |
| Baseline | EfficientNetV2 | Neural | | | | | |
| Baseline | MobileNetV3 | Neural | | | | | |
| Hybrid | Best Hybrid | Neural | | | | | |
| Classifier | Best CNN | XGBoost | | | | | |
| Classifier | Best Hybrid | XGBoost | | | | | |
| Optimized | Best Model | Neural | | | | | |

## Analysis

The final report should answer:

- Which CNN performed best?
- Why might that architecture have performed better?
- Did feature fusion help?
- Did optimization improve performance?
- Did XGBoost outperform the neural classifier?
- Which model was most computationally efficient?
- What did Grad-CAM reveal?
- What are the limitations of the dataset and methodology?
- What experiments should be performed in future work?

---

# 14. Experimental Reproducibility

Every meaningful experiment should record:

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
Number of epochs
Early stopping configuration
Random seed
Hardware
Training time
Evaluation metrics
Checkpoint location
Observations
```

## Rule

> **A result without enough information to reproduce it is not considered a completed experiment.**

---

# 15. Proposed Repository Structure

The exact structure will be determined after inspecting the existing repository.

A possible target structure is:

```text
project/
│
├── README.md
├── requirements.txt
├── config/
│   └── ...
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── features/
│   ├── optimization/
│   ├── xai/
│   └── utils/
│
├── experiments/
│   ├── configs/
│   ├── logs/
│   └── results/
│
├── notebooks/
│   └── ...
│
├── figures/
│   ├── dataset/
│   ├── embeddings/
│   ├── confusion_matrices/
│   └── gradcam/
│
└── docs/
    ├── flow.md
    ├── architecture.md
    ├── data_pipeline.md
    ├── training_pipeline.md
    ├── models.md
    ├── feature_analysis.md
    ├── hybrid_models.md
    ├── optimization.md
    ├── classifier_comparison.md
    ├── xai.md
    ├── experiments.md
    ├── decisions.md
    ├── debugging.md
    └── glossary.md
```

The actual repository structure should **not** be forced to match this before reconnaissance.

---

# 16. Living Documentation System

Documentation must be maintained alongside the implementation.

## `flow.md`

Must describe the actual execution path through the code.

It should include:

- Entry point
- Main execution flow
- Important functions
- Important classes
- Function call relationships
- Data transformations
- Model creation
- Training flow
- Evaluation flow
- XAI flow

Example:

```text
Entry Point
    ↓
main()
    ↓
load_config()
    ↓
load_dataset()
    ↓
preprocess_data()
    ↓
create_model()
    ↓
train_model()
    ↓
evaluate_model()
    ↓
save_results()
```

The example above is illustrative only. `flow.md` must always reflect the **actual codebase**.

## Documentation Rule

Whenever a meaningful code change is made:

```text
CODE CHANGE
    ↓
UPDATE RELEVANT DOCS
    ↓
UPDATE flow.md
    ↓
RECORD EXPERIMENT / DECISION
    ↓
EXPLAIN CHANGE
```

## Certainty Labels

Documentation should distinguish between:

- **CONFIRMED** — directly verified from the code/data.
- **INFERRED** — logically inferred but not directly confirmed.
- **UNKNOWN** — not yet verified.

Undocumented architecture must not be invented.

---

# 17. Experimental Scope

## Core Scope — Must Complete

> **Reading note (added 31/08/2026).** The ticked boxes below mark items that are **in
> scope and required**, not items that are finished. This is the original scope
> definition and is retained unchanged as a historical document. For actual completion
> status see `docs/experiments.md`; as of 31/08/2026 **one** of the four baseline
> benchmarks (ResNet152) has been run, and every item from *Model comparison* onward is
> **not started**.

- [x] Literature review
- [x] Dataset analysis
- [x] Data preprocessing pipeline
- [x] ResNet152 benchmark
- [x] DenseNet201 benchmark
- [x] EfficientNetV2 benchmark
- [x] MobileNetV3 benchmark
- [x] Model comparison
- [x] Feature analysis
- [x] One hybrid feature-fusion model
- [x] Optuna optimization
- [x] Softmax/neural classifier vs XGBoost comparison
- [x] Grad-CAM
- [x] Final comparative analysis
- [x] UROP report
- [x] Research paper draft
- [x] Presentation

> The checkboxes indicate planned core deliverables, not completed work.

## Optional Scope

Only attempt these after the core pipeline is stable:

- [ ] Additional hybrid architecture
- [ ] Weighted feature fusion
- [ ] Attention-based fusion
- [ ] BOHB
- [ ] SHAP
- [ ] LIME
- [ ] Additional CNN architectures
- [ ] Extensive ablation studies

---

# 18. Explicitly Removed from Core Scope

The following items from the original worklet are intentionally removed from the minimum viable research project:

| Original Item | Decision |
|---|---|
| VGG19 | Removed |
| InceptionV3 | Removed |
| MobileNetV2 | Removed |
| Three hybrid architectures | Reduced to one |
| Multiple fusion strategies | Reduced to concatenation |
| Attention-based fusion | Optional |
| Weighted fusion | Optional |
| BOHB | Removed from core |
| SHAP | Optional |
| LIME | Optional |
| XGBoost across many architectures | Reduced to best CNN + best hybrid |

This is a **scope-control decision**, not a claim that these techniques are unimportant.

---

# 19. Research Integrity Requirements

Because this is a medical-imaging research project:

### No Data Leakage

Never allow information from validation/test data to influence training or hyperparameter selection.

### Fixed Test Set

The final test set should remain untouched until final evaluation.

### Reproducibility

Record seeds, configurations, dataset splits, and model versions.

### Honest Reporting

Do not hide unsuccessful experiments.

A model that performs worse is still a useful result when the experiment was properly controlled.

### No Unsupported Clinical Claims

The project demonstrates model behavior on a dataset.

It does **not** establish clinical diagnostic capability.

### Class Imbalance

Accuracy must not be used as the sole indicator of performance when classes are imbalanced.

---

# 20. Definition of Done

The project is considered successfully completed when:

- [ ] Dataset is documented and validated.
- [ ] Preprocessing pipeline is reproducible.
- [ ] Four baseline CNNs have been evaluated under comparable conditions.
- [ ] Results are logged.
- [ ] Two strongest models have been analyzed at the feature level.
- [ ] One hybrid model has been implemented and evaluated.
- [ ] Best model has undergone Optuna optimization.
- [ ] Best CNN and hybrid have been compared using neural and XGBoost classifiers.
- [ ] Grad-CAM explanations have been generated.
- [ ] All major experiments are documented.
- [ ] Final results are reproducible.
- [ ] Research conclusions answer the defined research questions.
- [ ] UROP report is complete.
- [ ] Research paper draft is complete.
- [ ] Presentation is complete.

---

# 21. Final Expected Outcome

The final project should demonstrate the following research pipeline:

```text
Chest X-ray
     ↓
Dataset Analysis
     ↓
Preprocessing
     ↓
CNN Benchmarking
     ↓
Feature Representation Analysis
     ↓
Hybrid Feature Fusion
     ↓
Hyperparameter Optimization
     ↓
Softmax vs XGBoost
     ↓
Explainable AI with Grad-CAM
     ↓
Comparative Research Analysis
```

The goal is **not** to maximize the number of architectures or experiments.

The goal is to produce a **reproducible, technically sound, experimentally justified, and well-understood research system** within the available time and computational resources.

---

# 22. Working Principle

> **Build less. Measure properly. Document everything. Understand the system.**

Implementation may be produced rapidly, but every major component must be understood by the project owner through the living documentation and execution-flow analysis.

The repository should remain understandable enough that the project owner can explain:

1. Where execution starts.
2. How an image enters the system.
3. How preprocessing works.
4. How each model is created.
5. How training is performed.
6. How evaluation is performed.
7. How features are extracted.
8. How the hybrid model works.
9. How Optuna changes the model configuration.
10. How XGBoost consumes CNN features.
11. How Grad-CAM generates explanations.
12. How final conclusions are derived from the experimental results.
