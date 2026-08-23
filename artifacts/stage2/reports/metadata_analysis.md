# Stage 2 — CheXpert metadata analysis (generated)

Release: `CheXpert-v1.0-small`  ·  target: `Cardiomegaly`


## Corpus totals — official training CSV

| Category | Count | Percentage |
|---|---:|---:|
| Total patients | 64,540 | 100.00% |
| Total studies | 187,641 | — |
| Total images | 223,414 | 100.00% |
| Frontal images | 191,027 | 85.50% |
| Lateral images | 32,387 | 14.50% |

## View / projection breakdown (training CSV)

| Frontal/Lateral | AP/PA | Images | % of all images |
|---|---|---:|---:|
| Frontal | AP | 161,590 | 72.33% |
| Lateral | (blank) | 32,387 | 14.50% |
| Frontal | PA | 29,420 | 13.17% |
| Frontal | LL | 16 | 0.01% |
| Frontal | RL | 1 | 0.00% |

## `Cardiomegaly` raw label states

Counted at IMAGE level (train.csv rows) and at STUDY level.

| State | Images | % images | Studies | % studies |
|---|---:|---:|---:|---:|
| Positive (1.0) | 30,566 | 13.68% | 25,840 | 13.77% |
| Negative (0.0) | 16,155 | 7.23% | 11,227 | 5.98% |
| Uncertain (-1.0) | 3,917 | 1.75% | 3,327 | 1.77% |
| Blank (no mention) | 172,776 | 77.33% | 147,247 | 78.47% |

### All 14 observations (study level, for context)

| Observation | Positive | Uncertain | Negative + blank |
|---|---:|---:|---:|
| No Finding | 15,695 | 0 | 171,946 |
| Enlarged Cardiomediastinum | 6,342 | 12,545 | 168,754 |
| Cardiomegaly | 25,840 | 3,327 | 158,474 |
| Lung Opacity | 90,933 | 231 | 96,477 |
| Lung Lesion | 6,946 | 1,170 | 179,525 |
| Edema | 49,737 | 10,906 | 126,998 |
| Consolidation | 11,946 | 23,367 | 152,328 |
| Pneumonia | 3,653 | 16,280 | 167,708 |
| Atelectasis | 29,783 | 29,966 | 127,892 |
| Pneumothorax | 15,903 | 2,202 | 169,536 |
| Pleural Effusion | 78,474 | 6,234 | 102,933 |
| Pleural Other | 2,734 | 1,830 | 183,077 |
| Fracture | 7,016 | 379 | 180,246 |
| Support Devices | 104,016 | 53 | 83,572 |

## Usable frontal examples under each uncertainty policy

Frontal images only. `exclude` drops uncertain rows from the modelling frame; no file is removed from disk.

| Policy | Usable images | Positive | Negative | Prevalence | Pos : Neg | pos_weight |
|---|---:|---:|---:|---:|---:|---:|
| u_zeros | 191,027 | 26,283 | 164,744 | 13.76% | 1 : 6.27 | 6.268 |
| u_ones | 191,027 | 29,655 | 161,372 | 15.52% | 1 : 5.44 | 5.442 |
| exclude | 187,655 | 26,283 | 161,372 | 14.01% | 1 : 6.14 | 6.140 |

## Patient repetition — the leakage argument

| Statistic | Value |
|---|---:|
| Patients with frontal images | 64,534 |
| Mean images per patient | 2.96 |
| Median images per patient | 1 |
| Max images per patient | 91 |
| Patients with >1 image | 31,744 (49.19%) |
| Mean studies per patient | 2.91 |
| Patients with >1 study | 30,975 (48.00%) |
| Patients positive (any study) | 12,759 (19.77%) |

> 31,744 of 64,534 patients contribute more than one frontal image. Under a random image-level split those patients' images land on both sides of the train/test boundary.


## Figures

- `figures/cardiomegaly_label_states.png`
- `figures/view_breakdown.png`
- `figures/uncertainty_policy_balance.png`
- `figures/images_per_patient.png`
