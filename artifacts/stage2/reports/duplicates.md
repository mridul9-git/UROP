# Stage 2 — duplicate analysis (generated)

Scope: `191,027` frontal training images.


## L3/L4 — legitimate repetition (NOT removed)

| Level | Statistic | Value |
|---|---|---:|
| L3 | Studies with >1 frontal image | 3,361 of 187,625 |
| L3 | Max frontal images in one study | 3 |
| L4 | Patients with >1 study | 30,975 of 64,534 |
| L4 | Max studies for one patient | 91 |

L3 rows are separate radiographs acquired in the same visit and L4 rows are genuine follow-up imaging. Both are kept. Patient-level splitting — not deletion — is what stops them leaking.


## L1/L2 — exact duplicates (SHA-1 over file bytes)

| Finding | Value |
|---|---:|
| Files hashed | 191,027 |
| Distinct content hashes | 191,006 |
| Duplicate groups | 21 |
| Redundant files (group size − 1) | 21 |
| Duplicate rows spanning >1 patient | 8 |

Detail: `reports/duplicates_exact.csv`. Nothing deleted.

> **Cross-patient exact duplicates found.** These are the dangerous kind: identical pixels under two patient IDs defeat patient-level splitting. Resolve before training — keep one copy and record the decision.


## L5 — near-duplicates across different patients

dHash 64-bit, Hamming ≤ 4, cross-patient pairs only.

- Cross-patient near-duplicate pairs: **3,900,837**
- Distinct patients involved: **52,525**

Detail: `reports/duplicates_near_crosspatient.csv`.

