# Stage 2 — image integrity and dimension report (generated)

Checked `191,027` frontal images from `CheXpert-v1.0-small`.


## Step 8 — integrity

| Status | Count | Percentage |
|---|---:|---:|
| valid | 191,027 | 100.0000% |
| missing | 0 | 0.0000% |
| unreadable | 0 | 0.0000% |
| invalid_dims | 0 | 0.0000% |
| constant | 0 | 0.0000% |
| **total checked** | **191,027** | **100.0000%** |

Exclusion manifest: `reports/excluded_images.csv` (0 rows). No file was deleted or modified.


## Step 9 — dimensions

| Statistic | Width | Height | Aspect ratio (W/H) |
|---|---:|---:|---:|
| min | 320 | 320 | 0.742 |
| p05 | 320 | 320 | 1.000 |
| median | 390 | 320 | 1.219 |
| mean | 382 | 320 | 1.192 |
| p95 | 390 | 320 | 1.219 |
| max | 930 | 431 | 2.906 |

### Most common exact dimensions

| Width × Height | Images | Percentage |
|---|---:|---:|
| 390 × 320 | 134,602 | 70.46% |
| 389 × 320 | 21,871 | 11.45% |
| 320 × 320 | 13,163 | 6.89% |
| 369 × 320 | 3,432 | 1.80% |
| 371 × 320 | 1,590 | 0.83% |
| 384 × 320 | 777 | 0.41% |
| 370 × 320 | 749 | 0.39% |
| 320 × 321 | 483 | 0.25% |
| 368 × 320 | 364 | 0.19% |
| 321 × 320 | 341 | 0.18% |
| 373 × 320 | 303 | 0.16% |
| 375 × 320 | 265 | 0.14% |
| 367 × 320 | 236 | 0.12% |
| 362 × 320 | 232 | 0.12% |
| 382 × 320 | 221 | 0.12% |

### Implication for model input size

- **224×224** — 0 images (0.0%) have a short side below 224 and would be UPSAMPLED (no information gained).
- **320×320** — 0 images (0.0%) have a short side below 320 and would be UPSAMPLED (no information gained).
- **384×384** — 191,027 images (100.0%) have a short side below 384 and would be UPSAMPLED (no information gained).
| Colour mode | Images |
|---|---:|
| L | 191,027 |

Figure: `figures/image_dimensions.png`

