# Literature Review — Stage 1

**Project:** Comparative Deep Learning for Heart Disease Classification from Chest X-ray Images with Hybrid Feature Fusion, Hyperparameter Optimization, and Explainable AI

**Stage:** 1 (research and documentation only — no implementation)
**Status:** Draft for team review
**Last updated:** 2026-08-16

---

## 0. Executive Summary

> *What does the literature actually tell us about how to build this project?*

### 0.1 The single most important finding

**"Heart disease" is not a label that exists in any accessible public chest X-ray dataset.** This is the finding that should shape every downstream decision.

The literature splits cleanly into two families:

1. **Studies that predict genuine cardiac conditions from CXR** (reduced LVEF, aortic stenosis, mitral regurgitation, atrial fibrillation, elevated pulmonary artery wedge pressure, coronary artery disease). These are the studies whose *title* matches our project. Every single one of them derives ground truth from a paired modality the CXR cannot see — echocardiography, right heart catheterisation, or coronary angiography — held in **private, institutional, non-public** datasets (Ueda et al. 2023; Hsiang et al. 2022; Kusunose et al. 2020; D'Ancona et al. 2023). We cannot replicate this class of study. There is no public CXR dataset with paired echo labels.

2. **Studies that predict a radiographic *sign* of cardiac disease from CXR** — overwhelmingly **cardiomegaly**. These use public datasets (ChestX-ray14, CheXpert, MIMIC-CXR, VinDr-CXR, PadChest) and are directly replicable (Sogancioglu et al. 2020; CELM 2025).

**Our project must live in family 2.** The honest framing is: *binary classification of cardiomegaly — the principal radiographic marker of cardiac enlargement and a recognised early sign of underlying heart disease — from frontal chest radiographs.* Anything stronger is overclaiming, and reviewers who know this literature will spot it immediately.

### 0.2 Headline recommendations

| Decision | Recommendation | Evidence strength |
| --- | --- | --- |
| **Dataset** | **CheXpert** (downsampled release), frontal views only, binary `Cardiomegaly` target, patient-level splits | Moderate (our inference from access + label + patient-ID constraints) |
| **Backup dataset** | **VinDr-CXR** — radiologist-annotated with cardiomegaly bounding boxes; better labels, worse access/logistics | Moderate |
| **Explicitly avoid** | NIH ChestX-ray14 as *primary* — label PPVs measured 10–30 percentage points below documentation | Strong (Oakden-Rayner 2020) |
| **Baselines** | ResNet152, DenseNet201, EfficientNetV2-S, MobileNetV3-Large — justified as **four distinct architectural families**, not as "the four best models" | Strong (Ke et al. 2021: family matters more than size; ImageNet rank does not predict CXR rank) |
| **Transfer learning** | ImageNet init + full fine-tune at low LR. Expect **faster convergence**, not necessarily better final AUC | Strong (Raghu et al. 2019; Ke et al. 2021) |
| **Preprocessing** | Resize 224², replicate grayscale→3ch, ImageNet normalisation. **CLAHE as an ablation, not a default.** | Moderate |
| **Augmentation** | Small rotation (±10°), small translate/scale, mild brightness/contrast. **No horizontal flip.** | Our inference, anatomically grounded |
| **Primary metric** | **ROC-AUC** primary; **PR-AUC + sensitivity at fixed specificity** co-primary; accuracy reported but never headlined | Strong |
| **Feature viz** | **UMAP** primary, PCA variance as a sanity companion. Treat as *descriptive only* | Moderate |
| **Hybrid model** | **Concatenate penultimate-layer embeddings of the top-2 CNNs → small MLP head.** Do not build attention fusion without evidence | Moderate — fusion gains in the literature are real but small and inconsistently reported |
| **XGBoost** | Run it, but the fair comparison is *same frozen embeddings → softmax head vs XGBoost head*. Expect a small or null difference | Moderate |
| **Optuna** | TPE + MedianPruner, ~30–50 trials, **on the single best baseline and the hybrid only**. Report the tuning budget | Strong (Akiba et al. 2019) |
| **BOHB** | Documented and deliberately excluded — it is a multi-fidelity method whose advantage appears at large trial budgets we cannot afford | Strong (Falkner et al. 2018) |
| **Grad-CAM** | Core XAI method, but presented as a **debugging and failure-inspection tool**, never as evidence of clinical trustworthiness | Strong (Arun et al. 2021; Ghassemi et al. 2021) |

### 0.3 Key research gap we can honestly claim

Not a novel architecture. Not a novel algorithm. The defensible gap is:

> **The cardiomegaly-from-CXR literature is a collection of mutually incomparable single-paper results.** Studies use different datasets, different splits, different preprocessing, different augmentation, and different metrics, then report headline accuracies between roughly 90% and 99%. Almost none isolate *which* factor produced the gain. What is missing is a **controlled comparison holding everything constant except the variable under test** — architecture family, then feature fusion, then classifier head — with feature-space evidence for *why* fusion helps or doesn't, and with Grad-CAM subjected to at least one falsification test rather than presented as a set of cherry-picked pretty heatmaps.

This is a **rigorous replication-and-controlled-extension** contribution. It is genuinely useful and genuinely publishable at student/workshop level. It is **not** methodological novelty, and the write-up must say so.

### 0.4 The five risks most likely to sink this project

1. **Patient-level leakage.** CheXpert averages multiple studies per patient. A random image-level split inflates every number we report. This is the #1 killer.
2. **Weak labels.** CheXpert train labels come from an NLP labeler over reports, not from image review. Our ceiling is bounded by label noise we cannot measure.
3. **Silent unfairness between baselines.** Different input resolutions, different normalisation, or different early-stopping patience per model turns "architecture comparison" into noise.
4. **Grad-CAM theatre.** Presenting plausible-looking heatmaps as validation. Multiple studies show saliency maps fail basic sanity and repeatability checks.
5. **Overclaiming.** Writing "heart disease detection" when we built a cardiomegaly classifier, or "clinically useful" when we have a single-centre benchmark with no external validation.

---

## 1. Scope, Method, and How to Read This Document

### 1.1 Search scope

Sources were drawn from peer-reviewed journals (Lancet Digital Health, Radiology: Artificial Intelligence, European Radiology, IEEE Access, Scientific Reports, Academic Radiology, PLOS Medicine, Canadian Journal of Cardiology), high-quality ML conferences (AAAI, KDD, ICML, NeurIPS, CVPR, ICCV, CHIL), official dataset documentation (PhysioNet, Stanford AIMI, NIH), and preprints where the preprint is the canonical record.

Priority was given to 2020–2026 work, with foundational papers included where the project depends directly on them (ResNet, DenseNet, EfficientNetV2, MobileNetV3, Grad-CAM, XGBoost, Optuna, t-SNE/UMAP).

### 1.2 Two relevance classes

Throughout this document:

- **[DR] DIRECTLY RELEVANT** — the study attempts to infer a cardiac or cardiovascular condition/finding from a chest radiograph. These inform our *task framing, dataset choice, target label, and expected performance ceiling*.
- **[MR] METHODOLOGICALLY RELEVANT** — the study is about something else (pneumonia, COVID, TB, general thoracic multi-label, or pure ML method) but supplies methodology we will reuse: preprocessing, transfer learning, fusion, XGBoost heads, HPO, or XAI.

Pneumonia/COVID/TB studies appear **only** under [MR] and are never treated as heart-disease evidence.

### 1.3 Evidence labels

Major conclusions carry one of:

- **[Strong]** — multiple high-quality studies or a systematic review support it.
- **[Moderate]** — several relevant studies support it, but the evidence is heterogeneous.
- **[Limited]** — a small number of studies directly support it.
- **[Our inference]** — our interpretation, extrapolated from the literature rather than directly established by it.

### 1.4 Honest statement on paper count

The worklet asks for 15–20 highly relevant papers. We found **19 directly relevant CXR-cardiovascular studies/reviews** and a further **~25 methodologically relevant works**. However:

> **For the *exact* task we can execute — binary cardiomegaly classification on a public CXR dataset with a controlled multi-architecture comparison — there are fewer than 10 genuinely comparable studies, and no two of them share a dataset, split, and metric set.** This scarcity of *comparable* work is itself the finding that motivates our research gap (§16).

---

## 2. Paper Selection Tables

`NR` = not reported / not retrieved in this pass. Where a value comes from a review rather than the primary paper, it is marked `†`.

### 2.1 Table A — DIRECTLY RELEVANT: cardiovascular/cardiac inference from chest radiographs

| ID | Year | Paper | Task | Dataset | Images | Model | Training Strategy | Metrics | Best Result | XAI | Main Finding | Limitation | Relevance |
| -- | ---: | ----- | ---- | ------- | -----: | ----- | ----------------- | ------- | ----------- | --- | ------------ | ---------- | --------- |
| D1 | 2023 | Ueda et al., *Lancet Digit Health* | Multi-label: LVEF<40%, TR velocity, IVC dilation + 6 valvular diseases | Private, 4 institutions (Japan) | 22,551 CXR / 16,946 patients | CNN ensemble (arch NR) | Transfer learning; multi-label heads; echo-derived labels | AUC | Mean AUC 0.89/0.90/0.92 internal; **0.87 external**; LVEF<40% external AUC 0.92 | Saliency maps (concentrated on cardiac shadow) | First multi-institutional model classifying cardiac function + VHD from CXR; CXR carries more cardiac signal than CTR alone | Retrospective; Japanese cohorts only; labels are echo-derived proxies; data not public | **Gold standard for the task our title implies — and proof we cannot replicate it publicly** |
| D2 | 2022 | Hsiang et al., *Can J Cardiol* | Left ventricular systolic dysfunction (LVEF<35%) | Private (Taiwan military hospital system) | 90,547 CXR w/ paired TTE | CNN (arch NR) | Supervised, echo-derived binary label | AUC, HR | AUC 0.88 internal, **0.87 external** | NR | Large-scale evidence that LVSD is inferable from CXR; false positives carried elevated future cardiomyopathy risk (HR 3.91) | Single health system; retrospective; not public | Establishes realistic AUC ceiling (~0.87–0.88) for *genuine* cardiac function tasks |
| D3 | 2020 | Matsumoto et al., *Eur Heart J* (abstract)† | Heart failure detection | Private, relabelled by 2 cardiologists | 952 | CNN + transfer learning | Augmentation + transfer learning | Accuracy | ~82% accuracy | NR | Earliest DL-for-HF-from-CXR demonstration | **Conference abstract, not a full paper**; tiny dataset; accuracy-only reporting | Historical anchor; also a cautionary example of accuracy-only reporting |
| D4 | 2022 | Matsumoto et al., *Eur Radiol* | Atrial fibrillation detection | Private (single institution) | 13,868 CXR / 7,047 patients | CNN (arch NR) | PA standing views only, within 30 days of TTE; all eligible CXRs per patient used | AUC, Sn, Sp | Test AUC 0.80, Sn 0.70, Sp 0.74 | Saliency (focus on left then right atrium) | Arrhythmia is partly inferable from atrial silhouette | Single-centre; **multiple images per patient — leakage control not clearly described** | Directly illustrates the multi-image-per-patient split hazard we must avoid |
| D5 | 2022 | Ueda et al., *Eur Heart J Digit Health* | Aortic stenosis | Private | 10,433 CXR / 5,638 patients | InceptionV3 + ResNet50 + DenseNet121, **soft-voting ensemble** | Transfer learning; echo-labelled | AUC, Sn, Sp, Acc | AUC 0.83, Sn 0.83, Sp 0.69, Acc 0.71 | Visualisation (attention on aortic valve *and* LV regions) | **Multi-architecture ensembling beat single models** on a cardiac CXR task | Single-centre; modest specificity | **Closest published precedent for our hybrid/fusion hypothesis** |
| D6 | 2022 | Ueda et al., *Radiol Artif Intell* | Mitral regurgitation | Private | NR | CNN (as D5 family) | Transfer learning; echo-labelled | AUC, Sn, Sp, Acc | AUC 0.80, Sn 0.71, Sp 0.74, Acc 0.73 | Visualisation (left atrium, hilum) | Valve-specific lesions produce learnable silhouette changes | Single-centre; not public | Reference performance for a hard cardiac CXR task |
| D7 | 2020 | Kusunose et al., *Sci Rep* | Elevated pulmonary artery pressure (mPAP>20 mmHg) | Private, RHC-referenced | 900 patients | 5 CNNs compared | Supervised | AUC vs physician readers | AUC **0.71** (vs 0.60 measurement, 0.63 eyeball) | NR | Beat human readers — but absolute performance is *poor* | Small; single-centre; low AUC | Honest evidence that some cardiac targets are near the CXR information limit |
| D8 | 2023 | D'Ancona et al., *Int J Cardiol* (AI4CAD) | Significant coronary artery disease | Private, angiography-referenced | 7,728 patients | DL (arch NR) | PA/AP only; binary stenosis threshold | AUC, OR | AUC 0.73 (0.77 + angina status) | NR | CAD is weakly inferable from CXR | Single institution; CAD not directly visible on CXR | Example of a **proxy target with weak radiographic basis** — the failure mode our §5 analysis warns about |
| D9 | 2021 | Kamel et al., *Radiol Cardiothorac Imaging* | Coronary artery calcium score | Private, cardiac-CT-referenced | 1,689 CXR | CNN | Frontal + lateral compared | AUC | 0.73 frontal / 0.70 lateral (zero vs non-zero CAC); 0.74 at CAC>100 | NR | Modest CAC inference; independent of traditional risk factors | Small; modest performance | Reference for "modest is the honest expectation" |
| D10 | 2020 | Sogancioglu et al., *IEEE Access* | **Cardiomegaly**: segmentation vs classification | **ChestX-ray14** (public) + 778 seg annotations | 65k for classification; 367 held-out radiologist-labelled test | U-Net-style seg vs image-level CNN classifier; systematic hyperparameter search for both | Both arms hyperparameter-searched; CTR thresholded at 0.5 for reference | Sn, Sp, PPV, NPV, AUC | **Seg-method AUC 0.978**, equal to second radiologist; class-method clearly lower specificity at all sensitivities (AUC NR here) | Segmentation output is inherently interpretable | **Segmentation needed ~100× fewer annotated images and beat image-level classification** | Only 367 test images; single reference reader | **The single most important comparator paper for us** — sets the honest performance context and warns that image-level classification is the weaker paradigm |
| D11 | 2019 | Lu et al., *JAMA Netw Open* | 12-year all-cause and cardiovascular mortality | PLCO (dev/test) + NLST (external) — public-on-request trial data | >50,000 | CNN risk score | Trial-cohort development, external validation | HR | HR 18.3 (PLCO) / 15.2 (NLST) very-high vs very-low risk; significant for non-cancer CV death | NR | A single CXR carries long-horizon cardiovascular prognostic signal | Scanned film; prognostic not diagnostic | Motivates "CXR contains latent cardiac signal" framing |
| D12 | 2024 | Weiss et al., *Ann Intern Med* (CXR CVD-Risk) | 10-year ASCVD risk estimation | PLCO (dev, scanned film) + Mass General Brigham (external, digital) | NR | Deep learning risk model | Trained on film, validated on digital | Risk discrimination | Externally validated in ~8,869 patients with missing ASCVD inputs and ~2,132 with computable risk | NR | Opportunistic CV risk stratification from routine CXR | Film→digital domain shift; population-specific | Supports the "opportunistic screening" clinical motivation section |
| D13 | 2022 | Ieki et al., *Commun Med* | "CXR age" → cardiovascular prognosis | Private (Japan) | >100,000 | DNN regression | Age regression, then applied to HF/ICU cohorts | MAE, outcome association | Beat expert physicians at age estimation; higher CXR-age → worse HF readmission/mortality | NR | Latent CXR features encode cardiovascular ageing | Not a diagnostic classifier; single-country | Background motivation only |
| D14 | 2022 | Kim et al., *Eur Radiol* | Cardiovascular border (CB) analysis for valvular heart disease | Private + external validation | 816 normal / 798 VHD | DL segmentation of cardiovascular borders | Dev + external validation | ICC vs manual, correlation with TTE | ICC > 0.98 vs manual CB drawing; CB parameters correlated with LVEF, LV volumes, LA dimensions | Inherently interpretable (geometric) | **Geometric/anatomical features are highly reproducible and clinically correlated** | Not an end-to-end classifier | Second strong argument that anatomy-aware approaches beat black-box classification |
| D15 | 2025 | CELM (Kavitha et al. / *Diagnostics*-family) | **Cardiomegaly** binary classification | **PadChest + NIH ChestX-ray14 + VinDr-CXR + CheXpert** (all public), PA only | NR (multi-source merge) | VGG16, ResNet50, InceptionV3, DenseNet121, **DenseNet201**, AlexNet, ViTs + **stacking ensemble meta-classifier (CELM)** | Transfer learning + contrast enhancement + noise filtering; stacked meta-classifier over CNN features | Acc, Prec, Rec, F1, Sp, AUC | CELM: Acc 0.92, Prec 0.99, Rec 0.89, F1 0.94, Sp 0.92, **AUC 0.90** | NR | **Fusing complementary CNN features via a meta-classifier outperformed every single CNN** | Multi-dataset merge risks site-confounding (cf. Zech 2018); label provenance differs per source | **Closest published analogue to our entire pipeline** — and our most important comparator for the fusion claim |
| D16 | 2022 | Multimodal Cardiomegaly Classification with Image-Derived Digital Biomarkers (Springer/MICCAI-workshop) | **Cardiomegaly** | Public CXR (NR which) | NR | **XGBoost on 2 hand-derived features (CTR, CPAR)** vs ResNet-50 vs multimodal fusion | Compared interpretable-feature model against black-box | Acc, F1, AUC | XGBoost(CTR+CPAR): Acc 81.4%, F1 0.859, **AUC 0.810**; ResNet-50: Acc 81.9%, F1 0.874, AUC 0.767; Multimodal: 81.9%, 0.873, 0.768 | Features are inherently interpretable | **A 2-feature XGBoost matched — and on AUC beat — a full ResNet-50**; multimodal fusion added ~nothing | Single dataset; modest absolute performance | **Critical negative result for our §12/§13 hypotheses. Must be cited honestly.** |
| D17 | 2020 | Hirata et al., *Circulation* (abstract)† | Elevated pulmonary arterial wedge pressure (>18 mmHg) | Private, RHC-referenced | 1,013 patients | 5 CNNs | Supervised | Sn/Sp vs BNP and TTE | AI 0.77 vs BNP 0.77 vs TTE 0.70 (n.s.); adding AI to conventional model raised AUC 0.80→0.86 | NR | AI-CXR is complementary to, not a replacement for, existing tests | Abstract-level record | Supports "complementary tool" framing |
| D18 | 2023 | Farina et al., *J Imaging* 9(11):236 | **Narrative review**: AI-based prediction of CVD from CXR | — | — | — | — | — | — | — | Comprehensive map of the field: HF, PH, CAD, VHD, aortic disease, AF, outcomes | Narrative, not systematic; no meta-analysis | **The best single entry point to this literature; source for D3, D7, D8, D9, D14, D17** |
| D19 | 2025 | *Explainable AI in Radiological Cardiovascular Imaging — A Systematic Review*, *Diagnostics* 15(11):1399 | **Systematic review** of XAI (Grad-CAM, SHAP, LIME, saliency) in cardiovascular imaging incl. CXR | Jan 2015–Mar 2025, PubMed/Scopus/WoS | — | — | — | — | — | — | Grad-CAM dominates cardiovascular imaging XAI; rigorous validation of explanations remains rare | Heterogeneous inclusion | Directly supports our XAI method choice **and** our XAI-reliability caveats |
| D20 | 2024 | *Measurement of Cardiothoracic Ratio on Chest X-rays Using AI — Systematic Review and Meta-Analysis*, *JCM* 13:4659 (PROSPERO CRD42023437459) | **Systematic review + meta-analysis** of AI CTR measurement vs human | — | — | — | — | — | — | — | AI CTR measurement is approaching clinical viability | CTR ≠ heart disease; CTR correlates only weakly with LVEF | Supports cardiomegaly as the tractable public target, while capping how much we may claim from it |

### 2.2 Table B — METHODOLOGICALLY RELEVANT: empirical studies supplying pipeline methodology

| ID | Year | Paper | Contribution we use | Dataset | Key result | Why it matters to us |
| -- | ---: | ----- | ------------------- | ------- | ---------- | -------------------- |
| M1 | 2019 | Irvin et al., *CheXpert* (AAAI) | Dataset + uncertainty-label handling strategies | 224,316 CXR / 65,240 patients, Stanford | Different uncertainty policies (U-Ones, U-Zeros, U-MultiClass, U-Ignore) suit different pathologies; **U-MultiClass reported effective for Cardiomegaly** | Our target label carries explicit uncertainty; this is the reference for how to handle it |
| M2 | 2019 | Johnson et al., *MIMIC-CXR-JPG* | Dataset + **official reference splits** | 377,110 CXR / 227,827 studies / 65,379 patients, BIDMC | 14 CheXpert/NegBio NLP labels; official train/val/test ≈ 97.2/0.8/1.9% | Reference for split discipline and for label-tool disagreement (~3.6% of studies) |
| M3 | 2017 | Wang et al., *ChestX-ray8/14* (CVPR) | Dataset | 112,120 frontal / 30,805 patients, NIH CC | 14 text-mined labels, weakly supervised | Widely used; also the dataset whose labels were later shown to be unreliable |
| M4 | 2022 | Nguyen et al., *VinDr-CXR* (*Sci Data* 9:429) | **Radiologist-annotated** dataset with bounding boxes | 18,000 PA CXR (15,000 train / 3,000 test) from 2 Vietnamese hospitals | 22 local (box) + 6 global labels; train = 3 independent radiologists each, test = 5-radiologist consensus; **Cardiomegaly is local label #3**; train 10,606 normal / 4,394 abnormal, test 2,052 / 948 | The only realistic public source of **cardiomegaly bounding boxes** — enables quantitative Grad-CAM localisation checking |
| M5 | 2020 | Oakden-Rayner, *Acad Radiol* 27:106–112 | Label-quality audit | ChestX-ray14, MURA | **ChestX-ray14 label PPVs were 10–30 percentage points below documented values**; hidden stratification and label-disambiguation failures found (e.g. 86% of inspected "emphysema" cases were subcutaneous, not pulmonary) | The reason ChestX-ray14 is not our primary dataset |
| M6 | 2018 | Zech et al., *PLOS Med* 15:e1002683 | Confounding and external generalisation | NIH (112,120) + Mount Sinai (42,396) + Indiana (3,807) | External performance worse than internal in 3/5 natural comparisons; **CNNs identified the source hospital with 99.95%/99.98% accuracy** and calibrated to site prevalence | The reason merging datasets (as D15 does) is risky, and the reason we must not claim generalisation |
| M7 | 2019 | Raghu et al., *Transfusion* (NeurIPS) | Transfer learning mechanics | CXR + retinal fundus | ImageNet transfer gave **little final-performance benefit but substantially faster convergence**; pretrained and random-init models were surprisingly similar at higher layers | Sets correct expectations: we use ImageNet for convergence speed and stability, not as a performance guarantee |
| M8 | 2021 | Ke et al., *CheXtransfer* (ACM CHIL) | Architecture selection for CXR | CheXpert, 16 architectures | (1) **No relationship between ImageNet accuracy and CheXpert AUC**; (2) without pretraining, **model *family* matters more than size within a family**; (3) ImageNet pretraining gives a statistically significant boost, **larger for smaller architectures**; (4) truncating final blocks made models 3.25× more parameter-efficient with no significant AUC drop | **The core justification for our four-family baseline design** and for including a lightweight model (MobileNetV3) |
| M9 | 2017 | Rajpurkar et al., *CheXNet* | DenseNet-121 baseline for CXR | ChestX-ray14 | Widely-used 121-layer dense baseline | Establishes DenseNet as the de facto CXR backbone |
| M10 | 2023 | *Fusion-Extracted Features by Deep Networks…*, PMC10218019 | **Feature-level fusion** [MR: COVID/bacterial pneumonia] | CXR | Features from **5 transfer-learned CNNs concatenated → SVM (RBF)**; Acc 0.994, κ 0.991 | Direct methodological template for our feature-concatenation hybrid |
| M11 | 2021 | *Deep features + PSO-optimised XGBoost for COVID-19 CXR*, PMID 34177133 | **CNN features → XGBoost** [MR] | 2 public CXR databases | VGG19 + InceptionV3 + ResNet50 features → XGBoost (PSO-tuned); Acc 98.71%, F1 99.25% | Direct template for our CNN→XGBoost arm; also an example of implausibly high numbers on small merged datasets |
| M12 | 2022 | *Hybrid CNN + XGBoost…*, *Electronics* 11:3798 | CNN flatten-layer → XGBoost [MR] | COVID CXR | Replacing FC layers with XGBoost improved accuracy and cut training time | Shows the swap is cheap; also shows the literature rarely controls the comparison |
| M13 | 2024 | *MultiFusionNet* (arXiv 2401.00728) | **Multi-layer** feature fusion [MR] | CXR 3-class / 2-class | 97.21% / 99.60% accuracy with fused different-sized feature maps (FDSFM) | Evidence that fusing *across layers* (not just across models) can help — noted as future work, out of scope |
| M14 | 2026 | *Optimized EfficientNetB0 with CLAHE preprocessing*, *Sci Rep* 16:10811 | CLAHE + augmentation recipe [MR] | NIH ChestX-ray14, multi-label | CLAHE clipLimit=2.0, tileGridSize=(8,8); resize 224²; horizontal flip p=0.5; macro-AUC 0.906, beat DenseNet121 and MobileNetV2 | Concrete CLAHE parameters; **also an example of using horizontal flip on CXR, which we argue against** |
| M15 | 2021 | DeGrave et al., *Nat Mach Intell* | Shortcut learning in CXR | COVID CXR | Models selected shortcuts over signal | Central to §15 XAI-reliability and §19 failure modes |
| M16 | 2021 | Arun et al., *Radiol Artif Intell* 3(6):e200267 | **Saliency-map trustworthiness** | SIIM-ACR Pneumothorax + RSNA Pneumonia (both CXR) | 8 saliency methods tested on 4 criteria (localisation utility, weight-randomisation sensitivity, repeatability, reproducibility). **All failed at least one.** Only XRAI passed localisation utility, and it failed randomisation. **No method was more repeatable/reproducible than a plain localisation network.** | **The single most important paper for §15.** Grad-CAM cannot be presented as validation |
| M17 | 2018 | Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS) | Model/data randomisation tests | — | Several popular saliency methods are insensitive to model weights and to labels | Gives us a **concrete, cheap falsification test** to run |
| M18 | 2021 | Ghassemi, Oakden-Rayner & Beam, *Lancet Digit Health* 3:e745–e750 | Position paper on XAI in medicine | — | Current explainability methods are unlikely to deliver trust, transparency, or bias mitigation for **patient-level** decision support | Required framing for how we discuss Grad-CAM results |

### 2.3 Table C — FOUNDATIONAL method papers

| ID | Year | Paper | What we take from it |
| -- | ---: | ----- | -------------------- |
| F1 | 2016 | He et al., *Deep Residual Learning* (CVPR) | ResNet152; identity/residual shortcuts enabling very deep optimisation |
| F2 | 2017 | Huang et al., *Densely Connected Convolutional Networks* (CVPR) | DenseNet201; dense connectivity, feature reuse, parameter efficiency |
| F3 | 2021 | Tan & Le, *EfficientNetV2* (ICML) | EfficientNetV2-S; Fused-MBConv, training-aware NAS, progressive learning |
| F4 | 2019 | Howard et al., *Searching for MobileNetV3* (ICCV) | MobileNetV3-Large; NAS + NetAdapt, inverted residuals, SE blocks, h-swish |
| F5 | 2016 | Chen & Guestrin, *XGBoost* (KDD) | Regularised gradient-boosted trees; our alternative classifier head |
| F6 | 2019 | Akiba et al., *Optuna* (KDD) | Define-by-run HPO; TPE sampler; pruners. Pruning let TPE explore ~1,279 trials in the time non-pruned TPE managed ~36 |
| F7 | 2018 | Falkner, Klein & Hutter, *BOHB* (ICML) | Bayesian optimisation + Hyperband via KDE-guided successive halving. Explicitly excluded — see §13.4 |
| F8 | 2017 | Selvaraju et al., *Grad-CAM* (ICCV) | Gradient-weighted class activation mapping — our core XAI method |
| F9 | 2018 | Chattopadhay et al., *Grad-CAM++* (WACV) | Pixel-wise gradient weighting; better multi-instance/partial-object localisation |
| F10 | 2008 | van der Maaten & Hinton, *t-SNE* (JMLR) | Nonlinear neighbour-preserving 2-D embedding |
| F11 | 2018 | McInnes, Healy & Melville, *UMAP* (arXiv 1802.03426) | Faster, better global-structure-preserving manifold embedding |
| F12 | 2016 | Wattenberg, Viégas & Johnson, *How to Use t-SNE Effectively* (Distill) | t-SNE cluster sizes and inter-cluster distances are **not** meaningful; perplexity dominates appearance |
| F13 | 2019 | Kobak & Berens, *Nat Commun* 10:5416 | Practical t-SNE protocol; initialisation and exaggeration matter more than people assume |
| F14 | 2017 | Lundberg & Lee, *SHAP* (NeurIPS) | Shapley-value attributions — listed as optional future work |
| F15 | 2016 | Ribeiro et al., *LIME* (KDD) | Local surrogate explanations — listed as optional future work |

---

## 3. Directly Relevant Cardiovascular / CXR Research — Detailed Analysis

This section answers, for each key [DR] study, the twelve questions the worklet specifies. The purpose is not to summarise abstracts — it is to establish **what a defensible cardiac-CXR study looks like**, and **which of those properties we can and cannot reproduce**.

### 3.1 The three-tier structure of the field

Reading the [DR] literature, the studies sort into three tiers by how directly the label is visible on the radiograph:

| Tier | Target | Radiographic basis | Typical AUC | Public data exists? |
| --- | --- | --- | ---: | --- |
| **1 — Directly observable** | Cardiomegaly, aortic enlargement, cardiovascular border geometry | The cardiac silhouette *is* the finding | 0.90–0.98 | **Yes** (ChestX-ray14, CheXpert, MIMIC-CXR, VinDr-CXR, PadChest) |
| **2 — Indirectly inferable** | Reduced LVEF, aortic stenosis, mitral regurgitation, atrial fibrillation, elevated PAWP | Silhouette/chamber/hilar changes correlate with the condition, but the condition itself is invisible | 0.80–0.92 | **No** |
| **3 — Proxy prediction** | Coronary artery disease, coronary calcium score, elevated mean PAP, 10-year ASCVD risk, mortality | Almost no direct radiographic sign; the model is learning correlates of body habitus, age, comorbidity, and device presence | 0.70–0.77 | Partially (PLCO/NLST, on request) |

**[Strong] Observation:** performance degrades monotonically with distance from the radiographic sign. Tier 3 AUCs cluster near 0.73 — Kusunose 0.71, D'Ancona 0.73, Kamel 0.73/0.70/0.74. This is not a coincidence and it is not a modelling failure; it is the information limit of the modality.

**Implication for us:** any result we obtain on cardiomegaly (Tier 1) sits in the 0.90+ regime, and **must not be compared to Tier 2/3 numbers**. Doing so would make our project look better than it is.

### 3.2 Detailed study analysis

#### D1 — Ueda et al. 2023, *Lancet Digital Health* [DR, Tier 2]

- **Condition predicted:** LVEF<40%, tricuspid regurgitant velocity >2.8 m/s, IVC dilation >21 mm, plus six valvular diseases (none–mild vs moderate–severe cutoffs).
- **Ground truth:** transthoracic echocardiography reports.
- **Dataset:** private, **four institutions** in Japan; 22,551 CXRs from 16,946 patients.
- **Centres:** **multi-centre**, with a genuinely separate external test institution.
- **Architecture:** CNN ensemble; specific backbones not stated in the retrieved text (`NR`).
- **Transfer learning:** yes.
- **Preprocessing:** `NR`.
- **Metrics:** AUC for nine primary classifiers.
- **Results:** internal mean AUCs 0.89 / 0.90 / 0.92; **external mean AUC 0.87**; external valvular AUCs 0.83–0.92; LVEF classification external AUC 0.92.
- **External validation:** **yes** — the defining strength of this paper.
- **Patient-level splitting:** not explicitly described in retrieved text; the multi-institution design provides stronger separation than a patient split would.
- **Limitations:** retrospective; single-country; echo-derived labels inherit echo's own inter-observer variability.
- **Explainability:** saliency maps, which the authors report concentrated on the cardiac shadow across all primary classifiers.
- **Label observability:** **proxy.** LVEF cannot be measured from a radiograph. The authors are explicit that even CTR — the classical cardiac-enlargement marker — correlates only weakly with LVEF.

> **Take-away for us:** this is the paper our project title gestures at. It required four hospitals, ~17,000 patients, and paired echocardiography. **We cannot do this.** Stating so in our final report is a strength, not a weakness.

#### D10 — Sogancioglu et al. 2020, *IEEE Access* [DR, Tier 1] — **our closest comparator**

- **Condition:** cardiomegaly.
- **Ground truth:** two independent expert radiologists hand-labelled a 367-image held-out set; reference CTR values thresholded at 0.5.
- **Dataset:** **public ChestX-ray14** — 65k images with image-level labels for the classification arm; **778** images with heart/lung segmentation annotations for the segmentation arm.
- **Centres:** single (NIH Clinical Center).
- **Architectures:** a segmentation network (anatomical) vs an image-level CNN classifier. **Both arms received systematic hyperparameter searches over architectural, learning, and regularisation parameters** — a fairness discipline we should copy.
- **Metrics:** sensitivity, specificity, PPV, NPV, AUC.
- **Results:** the **segmentation-based method reached AUC 0.978**, statistically indistinguishable from the second human reader. The classification method "performed reasonably well, but with clearly much lower specificity at all sensitivity settings."
- **External validation:** no.
- **Patient-level splitting:** ChestX-ray14's official split is patient-disjoint; the held-out 367 were separately curated.
- **Limitations:** small test set; single reference reader for the primary comparison; ChestX-ray14 label noise (see M5) affects the classification arm's *training* labels.
- **Explainability:** the segmentation output is intrinsically interpretable — a measured CTR, not a heatmap.

> **Take-away for us [Strong]:** on the exact task we plan to run, the published evidence says **image-level classification is the weaker paradigm** and needs roughly 100× more annotated data than an anatomy-aware approach to do worse. We are choosing the classification paradigm anyway — because comparing CNN architectural families is the point of the project — but the write-up must acknowledge this explicitly, and the discussion should note segmentation/CTR as the stronger clinical approach.

#### D15 — CELM 2025 [DR, Tier 1] — **our closest pipeline analogue**

- **Condition:** cardiomegaly, binary.
- **Datasets:** PadChest + NIH ChestX-ray14 + VinDr-CXR + CheXpert, PA views only, merged.
- **Architectures compared:** VGG16, ResNet50, InceptionV3, DenseNet121, **DenseNet201**, AlexNet, plus Vision Transformers.
- **Hybrid:** a **stacking ensemble (CELM)** integrating complementary CNN features via a meta-classifier.
- **Results:** Acc 0.92, Prec 0.99, Rec 0.89, F1 0.94, Sp 0.92, AUC 0.90 — the best of all models tested.
- **Preprocessing:** image enhancement for low-contrast regions + noise filtering, reported to improve accuracy over raw data.
- **Reproducibility:** the authors emphasise releasing architecture, data splits, metrics, code, and trained models.
- **Limitation we must flag:** merging four datasets from four countries invites exactly the **site-confounding** failure Zech et al. (M6) demonstrated. A model can reach high apparent accuracy by learning which hospital produced the image, since cardiomegaly prevalence differs across those four sources. The paper does not appear to address this.

> **Take-away for us [Moderate]:** feature-level fusion via a meta-classifier *did* beat every constituent CNN on cardiomegaly. That is genuine support for our hybrid hypothesis. But the effect is confounded with a multi-site merge, and **our single-dataset design will be a cleaner test of the fusion claim than theirs was.** That is a legitimate, modest contribution.

#### D16 — Multimodal Cardiomegaly with Image-Derived Biomarkers 2022 [DR, Tier 1] — **the negative result we must not hide**

- Compared: XGBoost on **two** hand-derived geometric features (cardiothoracic ratio, cardiopulmonary area ratio) vs a full ResNet-50 vs a multimodal fusion of both.
- Results: XGBoost(2 features) Acc 81.4%, F1 0.859, **AUC 0.810**. ResNet-50 Acc 81.9%, F1 0.874, **AUC 0.767**. Multimodal Acc 81.9%, F1 0.873, AUC 0.768.
- The authors describe the difference as **negligible** — and on AUC specifically, the two-feature interpretable model *beat* the deep model.

> **Take-away for us [Limited but important]:** on cardiomegaly, a deep CNN may add little over two measurable geometric quantities, and multimodal fusion added nothing here. This is a direct challenge to both our hybrid-fusion and our CNN-features-plus-XGBoost hypotheses. **We should pre-register the possibility of a null result and treat it as a valid finding rather than a failure.**

### 3.3 Cross-cutting observations from the [DR] literature

| Observation | Evidence | Consequence for our design |
| --- | --- | --- |
| Almost every Tier 2 study is **single-centre and retrospective**; Ueda 2023 is the notable exception | D1–D9, D14, D17; Farina et al. review discussion | We cannot claim generalisation; we should not even imply it |
| Reported saliency consistently lands on **anatomically plausible** regions (cardiac shadow, LA, hilum) | D1, D4, D5, D6 | Plausible-looking Grad-CAM is the *expected* outcome, therefore it is not evidence of anything (see §15) |
| **Ensembling multiple CNN architectures improved cardiac CXR performance** in at least two studies | D5 (soft-voting), D15 (stacked meta-classifier) | Genuine support for our hybrid arm |
| The strongest cardiomegaly results come from **anatomy-aware**, not black-box, methods | D10 (AUC 0.978), D14 (ICC>0.98) | Frame our black-box comparison honestly against this |
| **Multiple images per patient is the norm** in cardiac CXR studies and leakage control is often under-described | D4 explicitly uses all eligible CXRs per patient | Our patient-level split discipline is a real (if unglamorous) methodological improvement |
| Accuracy-only reporting persists in the weaker papers | D3 | Never headline accuracy |

---

## 4. Dataset Investigation

### 4.1 Candidate comparison

| Dataset | Size | Labels | Cardiac Label? | Patient IDs | Public Access | Recommended Splits | Image Format | Major Advantages | Major Problems |
| ------- | ---: | ------ | -------------- | ----------- | ------------- | ------------------ | ------------ | ---------------- | -------------- |
| **CheXpert** (Stanford) | 224,316 CXR / 65,240 patients; ~191,027 frontal | 14 observations, 4-state (positive / negative / **uncertain** / unmentioned) | **Yes — `Cardiomegaly` and `Enlarged Cardiomediastinum`** | **Yes** (`patientID` encoded in the path) | Registration + research-use agreement with Stanford ML Group / Stanford AIMI. **No CITI training required.** Downsampled release available | No single mandated split for train; official **validation set of 200 studies** manually labelled by 3 radiologists (majority vote), and a **500-study test set** annotated by consensus of 5 (historically withheld for the leaderboard — **verify current public availability**) | JPEG | Largest easily-obtainable CXR set with a cardiac label; patient IDs enable true patient-level splits; radiologist-adjudicated eval set; downsampled version is manageable on student hardware | Training labels are **NLP-extracted from reports**, not image-reviewed; uncertainty label needs an explicit policy; single institution (Stanford, 2002–2017) so no generalisation claim is possible; official test set availability must be confirmed |
| **MIMIC-CXR-JPG** (MIT/BIDMC) | 377,110 CXR / 227,827 studies / 65,379 patients | 14 CheXpert-vocabulary labels from **two** NLP tools (CheXpert labeler + NegBio) | **Yes — `Cardiomegaly`** | **Yes** (`subject_id`, `study_id`) | **PhysioNet credentialed access**: account, identity verification, **CITI "Data or Specimens Only Research" training**, and a signed DUA | **Official reference split provided** (~97.2% train / 0.8% val / 1.9% test) | JPEG (DICOM in MIMIC-CXR) | Best-engineered public CXR resource; official splits remove split-design ambiguity; ~3.5 studies/patient makes it a good longitudinal resource; linkable to MIMIC-IV | Credentialing + CITI training takes days-to-weeks — **a real schedule risk for a UROP**; very large download; NLP labels; the two labelers disagree on ~3.6% of studies; ED population at a single Boston hospital |
| **NIH ChestX-ray14** | 112,120 frontal / 30,805 patients | 14 text-mined labels | **Yes — `Cardiomegaly`** | **Yes** (`Patient ID` column) | **Fully open**, no registration | Official patient-level `train_val_list.txt` / `test_list.txt` provided | PNG | Easiest access by a wide margin; smallest download; official patient-disjoint split; the dataset used by our closest comparator (D10) | **Label quality is the worst of the candidates** — radiologist review found PPVs 10–30 points below documentation, plus hidden stratification and label-disambiguation failures (M5). Using it as the primary source would put a known-bad ceiling on everything downstream |
| **VinDr-CXR** (VinBigData) | 18,000 PA CXR (15,000 train / 3,000 test) from >100,000 raw; train: 10,606 normal / 4,394 abnormal; test: 2,052 / 948 | 22 local labels (**bounding boxes**) + 6 global labels | **Yes — `Cardiomegaly` is local label #3, with bounding boxes** | **No usable patient ID** — images carry anonymised per-study hashes; patients were not linked | **PhysioNet credentialed access**: credentialing + **CITI training** + DUA | Official 15,000 / 3,000 train/test | **DICOM** | **Best label quality of any candidate**: every training scan independently read by 3 radiologists (≥8 yrs experience), every test scan by 5-radiologist consensus. **Bounding boxes make quantitative Grad-CAM localisation evaluation possible.** Small enough to handle. Two hospitals, not one | Credentialing + CITI delay; DICOM requires conversion (windowing decisions add a preprocessing variable); **no patient IDs means we cannot verify or enforce patient-level separation** — though random sampling from 100k+ scans makes overlap unlikely, we cannot prove it; adults and PA only; Vietnamese population |
| **PadChest** | 160,861 images / >67,000 patients, Hospital San Juan (Spain) | 174 findings, 19 differential diagnoses; ~27% manually labelled by physicians, rest NLP | **Yes** (cardiomegaly among findings) | Yes | Open with a use agreement | No mandated split | — | Large partially-hand-labelled Spanish cohort; useful as an external test set | Labels mostly in Spanish taxonomy; heterogeneous label provenance; less community tooling |
| **PLCO / NLST** | >50,000 (PLCO) | Trial outcomes incl. cardiovascular mortality | Outcome, not finding | Yes | NCI CDAS application (weeks) | Trial-defined | Scanned film | The only public-ish resource with true cardiovascular *outcomes* | Application process; **scanned film** creates a domain shift vs digital; prognosis, not diagnosis — wrong task for us |

### 4.2 Critical issue: label availability

**A chest X-ray dataset is not automatically a heart-disease dataset.** [Strong]

Of the 14 CheXpert/MIMIC observations, exactly two are cardiac: `Cardiomegaly` and `Enlarged Cardiomediastinum`. Neither is "heart disease". Both are radiographic *findings*.

The clinical literature is unambiguous that the gap between them is large:

- The cardiothoracic ratio — the quantitative basis of cardiomegaly — has only a **weak association with left ventricular ejection fraction** (Philbin et al. 1998, cited in Farina et al. 2023).
- Chest radiography has only **moderate accuracy for diagnosing congestive heart failure** in acutely dyspnoeic ED patients (Mueller-Lenke et al. 2006).

**Conclusion [Strong]:** we should target `Cardiomegaly` as a binary label, and describe it as *cardiomegaly*, with a sentence in the introduction explaining that cardiomegaly is a recognised radiographic marker of underlying cardiac pathology **but is neither necessary nor sufficient for a heart-disease diagnosis**.

**Optional extension (only if time permits):** a 3-class or multi-label variant adding `Enlarged Cardiomediastinum`. Not recommended for the core comparison — it complicates every downstream metric and fusion analysis for little gain.

### 4.3 Critical issue: data leakage

This is where student CXR projects most reliably go wrong.

| Leakage vector | Present in | Mitigation |
| --- | --- | --- |
| **Same patient in train and test** | CheXpert (~3.4 studies/patient avg), MIMIC (~3.5), ChestX-ray14 (~3.6 images/patient) | **Split by patient ID, never by image.** Use `GroupShuffleSplit` / `StratifiedGroupKFold` with the patient ID as the group |
| **Frontal + lateral of the same study in different splits** | CheXpert, MIMIC | Use **frontal only**; if both views are kept, split by `study_id` at minimum, patient ID preferably |
| **Multiple studies of the same patient across time** | All | Patient-level split handles this |
| **Duplicate/near-duplicate studies** | Possible in all | Optional: perceptual-hash duplicate scan before splitting |
| **Test-set tuning** | Any project | Tune only on validation; touch the test set once |
| **Site confounding across merged datasets** | Any multi-dataset merge (as in D15) | **Do not merge datasets for training.** Zech et al. showed CNNs identify the source hospital with ~99.95% accuracy and calibrate to site prevalence |
| **Patient re-identification via image biometrics** | Demonstrated on CXR (verification AUC ~0.994) | Not a performance leak, but a reminder that "de-identified" images still carry patient identity — relevant if we ever publish derived embeddings |

**Recommended split policy [Our inference, grounded in M2/M6]:**

```
Split by patient_id (never image_id)
  train 70% / val 15% / test 15%
Stratify on the Cardiomegaly label at patient level
Fix the split ONCE with a recorded seed, save it to disk as a CSV
Every model in the study reads that exact CSV
```

Saving the split to disk is not bureaucratic overhead — it is the single mechanism that makes the four-model comparison meaningful.

### 4.4 Critical issue: label quality

| Dataset | Label provenance | Quality verdict |
| --- | --- | --- |
| ChestX-ray14 | Automated text mining of reports | **Noisy/weak.** Radiologist audit found PPVs 10–30 points below documentation (M5) |
| CheXpert (train) | Rule-based CheXpert labeler over reports; 4-state incl. explicit uncertainty | **Weak but honest** — uncertainty is encoded rather than hidden |
| CheXpert (val/test) | Manual radiologist annotation; val = 3-radiologist majority, test = 5-radiologist consensus | **Strong** |
| MIMIC-CXR-JPG | CheXpert labeler + NegBio over reports; the two disagree on ~3.6% of studies | **Weak**, with a measurable disagreement rate |
| VinDr-CXR (train) | **3 independent radiologists per scan**, ≥8 years' experience | **Strong** |
| VinDr-CXR (test) | **5-radiologist consensus**, two-stage review with disagreement resolution | **Strongest of any candidate** |
| PadChest | ~27% physician-labelled, remainder NLP | Mixed |

An important second-order point: report-derived labels describe **what the radiologist wrote**, not what is in the image. A finding present but unmentioned becomes a negative. This systematically depresses measured sensitivity and is not fixable downstream.

**Handling CheXpert uncertainty for Cardiomegaly:** Irvin et al. found no single uncertainty policy is best for all pathologies, and reported **U-MultiClass** as effective for Cardiomegaly specifically. Practical recommendation: **use U-Zeros (treat uncertain as negative) as the default for a clean binary task, and run U-Ones as a single sensitivity ablation.** Report both. Do not silently pick whichever gives the better number.

### 4.5 Accessibility and timeline reality check

| Dataset | Registration | Institutional credential | CITI training | Realistic time to data-in-hand |
| --- | --- | --- | --- | --- |
| NIH ChestX-ray14 | None | No | No | **Hours** |
| CheXpert | Yes (Stanford agreement) | No | No | **1–3 days** |
| PadChest | Yes | No | No | 1–3 days |
| VinDr-CXR | PhysioNet credentialing | Yes (reference/supervisor) | **Yes** | **1–3 weeks** |
| MIMIC-CXR-JPG | PhysioNet credentialing | Yes | **Yes** | **1–3 weeks** + very large download |
| PLCO/NLST | NCI CDAS application | Yes | — | **Weeks** |

For a time-boxed UROP, the credentialing gap is the deciding practical factor.

### 4.6 Dataset Recommendation

#### 1. Recommended dataset

**CheXpert (downsampled release), frontal views only, binary `Cardiomegaly` target.**

#### 2. Why it fits

- It is the only candidate that combines **(a) a cardiac label, (b) usable patient IDs for leakage-free splitting, (c) a radiologist-adjudicated evaluation set, and (d) access without CITI training.**
- The downsampled release is tractable on student-grade GPU/Colab hardware, which matters when we must train four architectures plus a hybrid plus an Optuna study.
- Cardiomegaly is one of the five CheXpert "competition" tasks, so there is abundant published context to sanity-check our numbers against.
- Single-institution sourcing is a *limitation for generalisation* but an *advantage for a controlled comparison* — it removes the site-confounding variable that muddies D15.

#### 3. Expected target label

`Cardiomegaly` → binary {0, 1}, frontal (PA/AP) views only, uncertain labels mapped to 0 in the primary analysis with a U-Ones ablation reported.

#### 4. Main limitation

**Training labels are NLP-extracted from radiology reports, not verified against the images.** Our measured performance is therefore an estimate of *agreement with a text-mined proxy of a radiologist's written report*, not of diagnostic accuracy. This must appear in the abstract-level framing, not buried in limitations.

Secondary limitation: single centre, single country, 2002–2017 acquisition era — **no generalisation claim is available to us**.

#### 5. Backup dataset

**VinDr-CXR.**

#### 6. Why the backup is less desirable

- **Access latency.** PhysioNet credentialing plus CITI training is a 1–3 week critical-path risk we cannot absorb if it fails.
- **No patient IDs.** We cannot *prove* patient-level separation, only argue it is probable. For a project whose stated contribution is methodological rigour, an unverifiable leakage assumption is a weak spot.
- **DICOM.** Requires windowing/rescale decisions during conversion, adding a preprocessing degree of freedom that must itself be held constant — extra work and extra risk.
- **Smaller.** 18,000 images with ~5,300 abnormal total; the cardiomegaly-positive subset will be modest.

**However — VinDr-CXR has one capability CheXpert does not: cardiomegaly bounding boxes.** If credentialing completes in time, VinDr-CXR should be acquired as a **secondary asset for two specific purposes**: (a) quantitative Grad-CAM localisation evaluation against ground-truth boxes (§15), and (b) an honest external-validation check of the CheXpert-trained models. Both are high-value additions; neither is on the critical path.

> **Note we chose against size.** MIMIC-CXR-JPG is the largest and best-documented option and would be the right choice for a longer project. We are recommending against it purely on credentialing latency and download volume. If the team already holds PhysioNet credentials, **MIMIC-CXR-JPG becomes the better primary choice** because of its official reference splits — flag this to the supervisor early.

---

## 5. CNN Architecture Research

### 5.1 The correct framing

The objective is **not** to prove these four are the best models for CXR. Ke et al. (M8) showed decisively that **ImageNet accuracy does not predict CheXpert accuracy**, so "best on ImageNet" is not an argument. The objective is to select four models that **span distinct architectural design philosophies**, so that "architecture family" is a meaningful independent variable.

Ke et al.'s second finding makes this precise: for models without pretraining, **the choice of model family influenced performance more than size within a family**. That is the empirical justification for a four-*family* design rather than, say, ResNet-50/101/152/200.

### 5.2 Architecture comparison

| Architecture | Core Idea | Strengths | Weaknesses | Medical Imaging Relevance | Expected Compute Cost |
| ------------ | --------- | --------- | ---------- | ------------------------- | --------------------- |
| **ResNet152** (He et al. 2016) | Identity/residual shortcut connections let gradients bypass blocks, making very deep (152-layer) optimisation stable | Extremely well-understood; stable training; deep hierarchical features; universal framework support | ~60M params, heaviest of the four; parameter-inefficient relative to modern designs; long-range feature reuse only via addition | The default deep backbone in CXR work; used in cardiac CXR ensembles (D5) and aortic-dissection detection (Lee et al. 2022) | **Highest.** Slowest per epoch, largest memory footprint |
| **DenseNet201** (Huang et al. 2017) | Every layer receives the concatenated feature maps of all preceding layers within a block — maximal feature reuse | Strong gradient flow; **parameter-efficient for its depth**; concatenation preserves early low-level features that later layers can still access | Memory-hungry at train time due to concatenation; slower than parameter count suggests | **The de facto CXR backbone.** CheXNet (M9) used DenseNet-121; DenseNet-121/201 appear throughout the cardiomegaly literature (D15) and cardiac CXR ensembles (D5). Feature reuse suits the fine texture/edge cues of radiographs | **High.** Moderate params, high activation memory |
| **EfficientNetV2-S** (Tan & Le 2021) | Training-aware NAS + compound scaling; **Fused-MBConv** in early stages (a plain 3×3 conv replaces the depthwise+expand pair) to fix the throughput bottleneck of V1; progressive learning during training | Best accuracy-per-FLOP of the four; markedly faster training than EfficientNetV1; modern design | Sensitive to input resolution and to LR schedule; progressive resizing complicates *fair* comparison and should be disabled for our study | Represents the "modern efficient architecture" family; EfficientNet variants are increasingly used in CXR work including cardiomegaly (M14, EfficientNetB7 cardiomegaly work) | **Medium.** Efficient per FLOP |
| **MobileNetV3-Large** (Howard et al. 2019) | Platform-aware NAS + NetAdapt layer-wise tuning; inverted residual bottlenecks, squeeze-and-excitation blocks, h-swish activation | Smallest and fastest; deployable; useful **lower bound** on the capacity needed | Lowest capacity; typically the weakest single performer; SE blocks add a mild architectural confound vs the other three | Directly tests Ke et al.'s finding that **ImageNet pretraining gives a larger boost to smaller architectures** and that CXR models may be unnecessarily large. Also the realistic choice for any deployed screening tool | **Lowest.** Fastest by a wide margin |

### 5.3 Why these four make a defensible comparison

1. **They span four genuinely different information-routing strategies**: additive residual (ResNet), concatenative reuse (DenseNet), NAS-derived compound-scaled efficiency (EfficientNetV2), and NAS-derived mobile-constrained efficiency (MobileNetV3).
2. **They span a ~10× parameter range**, which lets us test the capacity question empirically rather than assuming it.
3. **Two are hand-designed (2016/2017) and two are NAS-derived (2019/2021)** — a secondary axis worth commenting on.
4. **Ke et al. (M8) predicts specific outcomes we can check**: no correlation with ImageNet rank; family > size; larger pretraining benefit for MobileNetV3; and possible over-parameterisation of the big models. If our results reproduce these, that is a meaningful confirmation on a cardiac-specific task. If they don't, that is also interesting.

### 5.4 What we should explicitly *not* claim

- Not "these are the best four architectures."
- Not "DenseNet is better than ResNet for medical imaging" — Ke et al. found family matters, but which family wins is task-dependent.
- Not any generalisation from four models to "CNNs in general."
- Not that our ranking would hold on another dataset. Zech et al. (M6) makes that claim untenable from single-site data.

---

## 6. Transfer Learning and Fine-Tuning

### 6.1 What the literature actually found

This is an area where the common practitioner belief and the evidence diverge, so it is worth stating carefully.

| Claim | Evidence | Verdict |
| --- | --- | --- |
| ImageNet pretraining improves final CXR performance | Raghu et al. (M7): "transfer offers little benefit to performance"; Ke et al. (M8): "statistically significant boost across architectures, **higher boost for smaller architectures**" | **Partially true.** The boost is real but modest, and it is largest for small models. [Moderate] |
| ImageNet pretraining speeds up convergence | Raghu et al.: transfer "significantly helps convergence speed" | **True.** [Strong] — and this is the *main* practical reason to use it |
| Better ImageNet models transfer better to CXR | Ke et al.: **no relationship** between ImageNet and CheXpert performance | **False for CXR.** [Strong] |
| Pretrained features are deeply reused in medical tasks | Raghu et al.: pretrained models are "surprisingly similar to random initialization at higher layers"; random-init models don't learn Gabor filters on medical images | **Largely false.** Much of the benefit is better weight conditioning, not sophisticated feature reuse. [Moderate] |
| Large ImageNet architectures are appropriately sized for CXR | Ke et al.: truncating final blocks made models **3.25× more parameter-efficient with no significant AUC drop** | **False.** CXR models are typically over-parameterised. [Moderate] |

### 6.2 What the literature commonly does

Across the [DR] and [MR] CXR studies surveyed:

- **ImageNet-pretrained initialisation** — near-universal.
- **Full fine-tuning** of all layers is the most common strategy in recent work; frozen-backbone feature extraction is more common in the older or resource-constrained papers and in the CNN→classical-ML hybrids (M10, M11).
- **Adam / AdamW** optimisers dominate; SGD+momentum appears in the more carefully-tuned studies.
- **Learning rates** of 1e-4 to 1e-5 for full fine-tuning; 1e-3 for a freshly-initialised head.
- **Class imbalance** handled by class weighting, weighted sampling, focal loss, or SMOTE on features (the latter only in the CNN→classical-ML pipelines).
- **Early stopping on validation loss or validation AUC** with patience 5–15.
- **Best-checkpoint selection on validation metric**, not last epoch.
- **Batch sizes** 16–64, usually GPU-memory-bound rather than chosen.

### 6.3 What we recommend

| Element | Recommendation | Rationale |
| --- | --- | --- |
| **Initialisation** | ImageNet-pretrained for all four models | Convergence speed and stability; keeps the four comparable [Strong] |
| **Fine-tuning schedule** | **Two-phase.** Phase 1: freeze backbone, train the new head for 3 epochs at LR 1e-3. Phase 2: unfreeze everything, train at LR 1e-4 with cosine decay | Phase 1 prevents large random-head gradients from destroying pretrained features on the first batches; Phase 2 gets the accuracy. Simple, deterministic, identical across models |
| **Gradual unfreezing** | **Do not use.** | It introduces per-architecture scheduling decisions (which blocks, when) that are impossible to make equivalently across four different block structures. That would break experimental fairness for a small, unproven gain [Our inference] |
| **Optimiser** | **AdamW**, weight_decay = 1e-4, identical for all four | Decoupled weight decay is better behaved than Adam's L2; one optimiser removes a confound |
| **Base LR** | 1e-4 (fine-tune phase), identical for all four in the baseline comparison | See caveat below |
| **LR schedule** | Cosine annealing, no warm restarts | Deterministic, one hyperparameter |
| **Batch size** | Largest power of two that fits **the most memory-hungry model** (likely DenseNet201), then use that same value for all four | Batch size affects effective regularisation; varying it across models silently confounds the comparison |
| **Loss** | `BCEWithLogitsLoss` with `pos_weight = N_neg / N_pos` | Standard, principled imbalance handling; avoids resampling artefacts |
| **Class imbalance** | `pos_weight` only. **No SMOTE, no oversampling, in the CNN arm.** | Synthetic image oversampling on radiographs risks unrealistic anatomy; duplication risks memorisation |
| **Early stopping** | Monitor **validation ROC-AUC**, patience 10, `min_delta` 0.001, max 40 epochs, identical for all four | Identical stopping policy is essential to fairness (§18) |
| **Checkpointing** | Save best-validation-AUC weights; evaluate the test set **once**, from that checkpoint | Prevents test-set tuning |
| **Seeds** | 3 seeds per model (e.g. 42/1337/2024); report **mean ± SD**, never the best run | Single-run reporting is the most common inflation mechanism in this literature |

### 6.4 The learning-rate fairness caveat

There is a genuine tension here, and it should be discussed openly in the report rather than resolved silently.

Using one identical LR for all four models is *procedurally* fair but potentially *substantively* unfair: MobileNetV3 and ResNet152 may simply have different optimal LR ranges. Two defensible resolutions:

- **Option A (recommended for the main comparison):** identical LR for all. Cleanest attribution — any difference is attributable to architecture-under-a-fixed-protocol. State this framing explicitly.
- **Option B (recommended as a supplementary result if compute allows):** run a small identical LR sweep — the *same* three-point grid {3e-5, 1e-4, 3e-4} for every model — and select per-model best on validation. This gives each architecture a fair shot while keeping the *search procedure* identical.

Sogancioglu et al. (D10) took the Option-B spirit: both of their competing methods received systematic hyperparameter searches. That is the higher standard. **If compute permits, do Option B and report Option A as an ablation.**

---

## 7. Preprocessing and Augmentation

### 7.1 Preprocessing options and medical appropriateness

| Step | What the literature commonly does | Medical appropriateness | Our recommendation |
| --- | --- | --- | --- |
| **Resize** | 224×224 is the overwhelming default; some work uses 320×320 | Cardiomegaly is a **large, global, low-frequency** shape feature (the cardiac silhouette relative to the thorax). Unlike nodules, it does not require high resolution | **224×224.** Sufficient for this target, and it lets us afford four models × three seeds |
| **Aspect ratio** | Often ignored (direct squash to square) | Squashing distorts the cardiothoracic ratio — **the exact quantity that defines our target** | **Resize the short side then centre-crop to square**, or pad to square then resize. Do *not* squash. This is a task-specific decision that generic pipelines get wrong |
| **Grayscale → RGB** | Channel replication to 3 channels | Loses nothing; required by ImageNet-pretrained stems | **Replicate the single channel 3×** |
| **Normalisation** | ImageNet mean/std is standard when using pretrained weights; some studies use dataset-specific statistics | ImageNet statistics are technically mismatched for radiographs, but empirically fine and keep pretrained stems in their expected input regime | **ImageNet mean/std.** One ablation with dataset statistics if time permits |
| **Histogram equalisation (global)** | Occasionally used | Global HE can crush the very intensity relationships that distinguish soft tissue from lung field, and amplifies noise | **Do not use** |
| **CLAHE** | Widely used with clipLimit≈2.0, tileGridSize=(8,8) (M14); improves contrast in low-visibility regions without global noise amplification | Defensible for radiographs. But CLAHE is a **per-tile local** operation and can alter the apparent heart–lung boundary; it is not obviously helpful for a global shape task | **Treat as an ablation, not a default.** Train the best baseline with and without CLAHE and report the delta. Claiming CLAHE helps without measuring it is exactly the kind of unsupported claim this project should avoid |
| **Lung-field cropping / ROI extraction** | Used in some CheXpert pipelines via template matching | Cropping to the lung fields could remove thoracic boundary information needed to judge *relative* heart size | **Do not crop to ROI.** The thorax width is part of the signal |
| **View filtering** | Frontal-only is standard for cardiomegaly | Lateral views encode cardiac size differently; mixing views without a view indicator adds noise | **Frontal (PA + AP) only.** Optionally record PA vs AP as a metadata column — AP portable films magnify the cardiac silhouette, a known confounder worth noting in the discussion |

### 7.2 Augmentation: which are medically appropriate

This is where medical imaging diverges sharply from natural-image practice, and where uncritical use of a default `torchvision` pipeline causes real harm.

| Augmentation | Common in CXR papers? | Anatomically plausible? | Risk to *our* target | Verdict |
| --- | --- | ---: | --- | --- |
| **Small rotation (±5–10°)** | Yes | Yes — patient rotation on the detector is a real acquisition variation | Minimal; slightly perturbs CTR | **Use, ±10°** |
| **Small translation (±5%)** | Yes | Yes — centring varies | Minimal | **Use, ±5%** |
| **Small scale/zoom (0.9–1.1×)** | Yes | Yes — source-to-detector distance varies | **Caution:** scaling the whole image preserves *ratios*, so CTR is unaffected. Scaling that crops the thorax edge would not be | **Use, 0.9–1.1×, whole-image only** |
| **Mild brightness/contrast jitter (±10%)** | Yes | Yes — exposure varies genuinely between machines and patients | Low | **Use, mild** |
| **Horizontal flip** | **Yes — very common** (e.g. M14 uses p=0.5) | **No.** The human thorax is not left–right symmetric. The heart sits left of midline; a flipped radiograph depicts **dextrocardia or situs inversus**, a rare congenital condition | **High.** For a task defined by the position and extent of the cardiac silhouette, flipping teaches the model that a right-sided heart is normal. It also destroys any hope of interpreting Grad-CAM laterality | **DO NOT USE.** This is our most important preprocessing recommendation and it contradicts common practice |
| **Vertical flip** | Occasionally (in careless pipelines) | **No.** Upside-down chest radiographs do not exist | High | **DO NOT USE** |
| **Large rotation (>15°)** | Occasionally | No — clinically implausible acquisition | High | **DO NOT USE** |
| **Shear / elastic deformation** | Sometimes | **No** for our task — elastic warping directly deforms the cardiac silhouette and therefore the label-defining geometry | **Very high** — can flip the ground truth | **DO NOT USE** |
| **MixUp / CutMix** | Increasingly common in CXR papers | Produces anatomically impossible images | Blends label semantics; also makes Grad-CAM uninterpretable | **DO NOT USE** in the core study |
| **Random erasing / coarse dropout** | Sometimes | Can occlude the heart border entirely | High for our target | **DO NOT USE** |
| **Gaussian noise** | Sometimes | Plausible (quantum mottle) | Low | Optional, low priority |

### 7.3 The uncomfortable point about horizontal flip

Horizontal flip appears in a large fraction of published CXR pipelines, including recent, well-performing ones. It probably does little harm on lung-parenchyma tasks where left–right position is not diagnostic. **For cardiomegaly it is different**, because the target is defined by a laterally-asymmetric structure. We should:

1. Not use it.
2. Say why in the paper.
3. If we have the budget, **run it as a single ablation** — training the best baseline with flip p=0.5 and reporting the delta. If it makes no difference, that is a useful negative finding. If it helps, that is evidence the model is not using laterality — which would itself be worth reporting.

### 7.4 Proposed preprocessing strategy

```
DETERMINISTIC (train + val + test, identical):
  1. Load frontal view only (PA or AP)
  2. Convert to single-channel float32, scale to [0, 1]
  3. Pad to square (reflect or constant), preserving aspect ratio   # protects CTR
  4. Resize to 224 x 224 (bilinear)
  5. Replicate channel -> 3 channels
  6. Normalise with ImageNet mean/std

STOCHASTIC (train split only):
  7. RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))
  8. ColorJitter(brightness=0.10, contrast=0.10)
  NO horizontal flip. NO vertical flip. NO shear. NO elastic. NO MixUp/CutMix. NO erasing.

ABLATIONS (best baseline only, one at a time):
  A1. + CLAHE(clipLimit=2.0, tileGridSize=(8,8)) inserted at step 2.5
  A2. + RandomHorizontalFlip(p=0.5) at step 7.5
  A3. Dataset-specific normalisation statistics instead of ImageNet at step 6
```

**Every model in the comparison uses the identical deterministic and stochastic pipeline.** Ablations are run only after the main comparison is complete and are reported separately.

---

## 8. Evaluation Metrics

### 8.1 The metrics and what each actually measures

Let TP/FP/TN/FN be the confusion-matrix cells for the positive class (cardiomegaly present).

| Metric | Definition | What it tells us | Failure mode |
| --- | --- | --- | --- |
| **Accuracy** | (TP+TN)/N | Overall correctness at one threshold | **Dominated by the majority class.** With 20% positives, predicting "no cardiomegaly" always gives 80% accuracy while detecting nothing |
| **Precision (PPV)** | TP/(TP+FP) | Of flagged cases, how many are real | Depends on prevalence; not comparable across datasets with different base rates |
| **Recall / Sensitivity** | TP/(TP+FN) | Of real cases, how many we catch | Trivially maximised by predicting everything positive |
| **Specificity** | TN/(TN+FP) | Of healthy cases, how many we correctly clear | Trivially maximised by predicting everything negative |
| **F1** | Harmonic mean of precision and recall | Balances the two; threshold-dependent | Ignores TN entirely; not prevalence-invariant |
| **ROC-AUC** | Area under TPR-vs-FPR | **Threshold-free ranking quality**; probability a random positive is ranked above a random negative | Can look optimistic under heavy imbalance because FPR has a large TN denominator |
| **PR-AUC (average precision)** | Area under precision-vs-recall | **Threshold-free, and sensitive to the positive class under imbalance** | Baseline equals prevalence, so PR-AUC values are not comparable across datasets |
| **Confusion matrix** | The raw 2×2 | The only artefact that shows *how* the model fails | Threshold-dependent; must state the threshold |

### 8.2 Why accuracy alone is misleading here

[Strong] Three independent reasons apply to our specific task:

1. **Class imbalance.** Cardiomegaly prevalence in public CXR datasets is well below 50%. In VinDr-CXR, the *overall* abnormal rate is ~29% train / ~32% test, and cardiomegaly is one finding among 22 — so the positive rate for our specific label is a minority class by a wide margin. Accuracy is therefore anchored to the negative class.
2. **Asymmetric clinical cost.** A missed cardiomegaly (FN) means an undetected potential marker of cardiac pathology. A false positive means an unnecessary echocardiogram. These costs are not equal, and accuracy weights them equally.
3. **It hides the operating point.** Two models with identical accuracy can have completely different sensitivity/specificity trade-offs. Sogancioglu et al. (D10) found precisely this — their classification method "performed reasonably well, but with clearly much lower specificity at all sensitivity settings." An accuracy-only comparison would have concealed that.

The literature demonstrates this concretely. A systematic review of AI for aortic-arch calcification on CXR found one model achieving 95.9% recall while another achieved near-perfect specificity (0.99) alongside **sensitivity of only 0.22** — a model that would be nearly useless as a screening tool despite excellent specificity and probably respectable accuracy.

### 8.3 Recommended metric strategy

Grounded in the dataset (imbalanced, single-label, binary) and the literature (D1, D2, D5, D10 all report AUC as primary):

| Tier | Metric | Role |
| --- | --- | --- |
| **Primary** | **ROC-AUC** (mean ± SD over 3 seeds, with 95% bootstrap CI on the test set) | Threshold-free, directly comparable to the published cardiac-CXR literature, and the metric on which model selection and Optuna optimise |
| **Co-primary** | **PR-AUC** | The imbalance-sensitive complement. If ROC-AUC and PR-AUC rank models differently, that disagreement is itself a finding |
| **Co-primary** | **Sensitivity at fixed specificity = 0.90** | Forces a clinically-shaped operating point and prevents the "high AUC, useless threshold" failure |
| **Secondary** | F1, precision, recall, specificity, accuracy — all at a **validation-selected threshold**, stated explicitly | Comparability with the weaker literature; completeness |
| **Diagnostic** | Confusion matrix + ROC and PR curves per model, overlaid | Shows *how* models differ, not just *that* they do |
| **Statistical** | **DeLong test** for pairwise ROC-AUC differences between models | Prevents claiming a 0.004 AUC difference as a result |

**Threshold policy:** select the operating threshold on the **validation** split (e.g. maximising Youden's J, or at fixed specificity 0.90), then apply that frozen threshold to the test set. Never select a threshold on test data.

**Reporting rule:** report **mean ± SD across 3 seeds** for every metric. A single-run number is not a result.

---

## 9. Feature Representation Analysis

### 9.1 The three methods

| Method | Mechanism | Preserves | Deterministic? | Cost |
| --- | --- | --- | --- | --- |
| **PCA** (linear) | Orthogonal projection onto directions of maximum variance | **Global** linear structure; distances along retained components; gives an interpretable *explained-variance* number | Yes (up to sign) | Trivial |
| **t-SNE** (van der Maaten & Hinton 2008) | Matches pairwise neighbour probabilities in high-D and 2-D by minimising KL divergence | **Local** neighbourhoods | No — stochastic, initialisation- and perplexity-dependent | Moderate; poor scaling beyond ~10k points without approximation |
| **UMAP** (McInnes et al. 2018) | Fuzzy simplicial-set / topological approximation of the manifold, optimised by cross-entropy | Local structure, with **better global structure retention than t-SNE**; faster | No — stochastic, but more stable across runs | Low; scales well |

### 9.2 What these methods show — and what they emphatically do not prove

**What they show:** whether the classes are *linearly or nonlinearly separable in the learned embedding space*, whether there are visible sub-clusters (a hint of hidden stratification), and whether two models' embeddings organise the data differently.

**What they do NOT prove** [Strong]:

1. **Cluster sizes in t-SNE are meaningless.** t-SNE expands sparse regions and contracts dense ones; a big blob is not a "more variable" class (Wattenberg et al., F12).
2. **Distances *between* clusters in t-SNE are meaningless.** Two clusters appearing far apart may be adjacent in the original space (F12).
3. **Apparent clusters can be artefacts of the hyperparameters.** Perplexity (t-SNE) and `n_neighbors`/`min_dist` (UMAP) dominate the visual outcome. Kobak & Berens (F13) showed initialisation and early-exaggeration choices materially change the picture.
4. **Visual separation ≠ classifier performance.** A projection that looks messy can still be trivially separable by a linear probe in the original 1000+-dimensional space, and vice versa. **A 2-D picture is a lossy summary of a high-dimensional object.**
5. **They cannot establish complementarity.** Two embeddings looking "different" in 2-D is not evidence that concatenating them helps.

### 9.3 How to use them to compare two CNNs' representations

The honest use is **descriptive, hypothesis-generating, and always paired with a quantitative measure**. Specifically:

| Question | Wrong tool | Right tool |
| --- | --- | --- |
| Are the classes separable in model A's features? | "The UMAP looks separated" | **Linear probe**: fit logistic regression on the frozen embeddings, report AUC |
| Do models A and B encode different information? | "The UMAPs look different" | **CKA (Centered Kernel Alignment)** or **SVCCA** between the two embedding matrices — a single number in [0,1] |
| Is fusion likely to help? | Visual inspection | **Prediction disagreement rate** on the test set + a **linear probe on the concatenated embeddings vs each alone** |
| Is there hidden stratification? | — | UMAP **coloured by metadata** (view PA/AP, sex, age band, support-device label). This is the one genuinely strong use of the visualisation |

Raghu et al. (M7) is the model here: they did not eyeball projections, they used **CCA** to compare representations quantitatively and drew a real conclusion (pretrained ≈ random at higher layers).

### 9.4 Recommendation

**Primary visualisation method: UMAP.** Reasons:

1. **Better global structure preservation** than t-SNE, which matters when the comparative question is "do these two models organise the data similarly?"
2. **More stable across runs** with a fixed seed, which matters for a study whose contribution is reproducibility.
3. **Faster**, which matters when we render this for four baselines plus a hybrid, coloured by several metadata fields.
4. **PCA reported alongside** as a sanity companion — specifically the cumulative explained variance of the first 2, 10, and 50 components. If PCA-2 already separates the classes, the nonlinear embedding is decoration.

**Mandatory pairing:** every UMAP figure in our report must be accompanied by (a) a linear-probe AUC on the same embeddings, and (b) a caption stating that inter-cluster distances and cluster sizes are not interpretable. **t-SNE is skipped** — it adds a second set of hyperparameters and a second set of caveats for no additional insight.

**Concrete protocol:**

```
For each trained model M in {ResNet152, DenseNet201, EffNetV2-S, MobileNetV3-L, Hybrid}:
  1. Extract penultimate (post-global-pool, pre-classifier) embeddings for the TEST split
  2. StandardScaler fit on TRAIN embeddings, applied to test        # fit on train only
  3. PCA -> report cumulative explained variance @ 2/10/50 comps
  4. UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42) -> 2D
  5. Plot 4x: coloured by (a) true label, (b) predicted label,
              (c) correct/incorrect, (d) view PA vs AP
  6. Linear probe: LogisticRegression on embeddings -> test AUC
Between the top-2 models:
  7. CKA(embeddings_A, embeddings_B) -> single scalar
  8. Prediction disagreement rate + McNemar test
```

Step 5(d) is the one most likely to find something genuinely alarming — if PA and AP images separate cleanly in the embedding, the model may be partly encoding acquisition type rather than cardiac size, which is a Zech-style confound (AP portable films magnify the heart).

---

## 10. Hybrid CNN / Feature Fusion Research

### 10.1 Feature-level vs prediction-level fusion

| | **Feature-level (early/intermediate) fusion** | **Prediction-level (late) fusion** |
| --- | --- | --- |
| **What is combined** | Internal representations (e.g. two 1×D penultimate embedding vectors) | Output probabilities or logits |
| **Mechanism** | Concatenate / weighted-sum / attention-weight → new classifier head trained on the joint space | Average, weighted average, soft/hard voting, or a stacked meta-learner over the probability vectors |
| **What it can learn** | **Cross-model interactions** — the joint head can learn "feature 341 from DenseNet matters only when feature 88 from ResNet is high" | Only a reweighting of already-collapsed decisions. Interactions are lost before fusion |
| **Dimensionality** | High (e.g. 2048 + 1920 = 3968) — overfitting risk with few samples | Very low (one number per model) |
| **Cost** | Requires a new head trained on the joint space; backbones can be frozen | Nearly free; no retraining |
| **Interpretability** | Harder — Grad-CAM must be computed per-branch | Easier — per-model contributions are explicit |
| **CXR precedent** | M10 (5 CNNs → concat → SVM), M13 (multi-layer fusion), D15 (stacked meta-classifier) | **D5 (Ueda et al., soft-voting ensemble of InceptionV3 + ResNet50 + DenseNet121 for aortic stenosis — the best-performing model in that study)** |

### 10.2 Fusion strategies, ordered by complexity

1. **Concatenation** — `[f_A ; f_B]` → dense head. Simplest, no extra hyperparameters beyond the head. This is what we plan.
2. **Weighted feature fusion** — `α·f_A + (1−α)·f_B`, requires equal dimensionality (projection layers) and adds α as a hyperparameter.
3. **Attention-based fusion** — learn per-feature or per-branch attention weights. Adds parameters, adds a design space, and requires more data to fit.
4. **Late fusion / ensembling** — average or vote over probabilities.
5. **Stacked meta-learner** — train a second-level model (logistic regression, XGBoost, MLP) on out-of-fold predictions. This is CELM's approach (D15).

### 10.3 Does combining architectures actually help? The honest answer

**Cases where it did help:**

- **D5 (Ueda et al. 2022, aortic stenosis):** soft-voting ensemble of InceptionV3 + ResNet50 + DenseNet121 showed the **highest overall performance** of all models tested (AUC 0.83). [DR, Tier 2]
- **D15 (CELM 2025, cardiomegaly):** stacking-based ensemble integrating complementary CNN features via a meta-classifier **outperformed every individual CNN and the ViTs** (AUC 0.90, F1 0.94). [DR, Tier 1 — our exact task]
- **M10 (Fusion-extracted features, COVID/bacterial pneumonia CXR):** features from five transfer-learned CNNs concatenated into an SVM reached Acc 0.994, κ 0.991. [MR]
- **M13 (MultiFusionNet):** fusing feature maps from *different layers* of one network improved accuracy to 97.21%/99.60%. [MR]

**Cases where it did not help:**

- **D16 (Multimodal Cardiomegaly, 2022):** multimodal fusion of ResNet-50 image features with hand-derived CTR/CPAR biomarkers gave **Acc 81.9%, F1 0.873, AUC 0.768** — statistically indistinguishable from ResNet-50 alone (81.9%, 0.874, 0.767), and *worse on AUC* than a two-feature XGBoost (AUC 0.810). Fusion added nothing. [DR, Tier 1 — our exact task]

**Critical caveat on all the positive results** [Moderate]: the fusion successes above are reported without controlling for the extra capacity and extra effective training the fused model receives. A concatenated two-CNN model has more parameters and has seen more gradient updates in total than either constituent. **Almost none of these papers include the control that would separate "fusion helps" from "more capacity helps."**

### 10.4 Recommendation

**Use feature-level concatenation of the penultimate embeddings of the top-2 baseline CNNs, feeding a small MLP head.**

```
CNN_A (frozen or fine-tuned) --> GAP --> f_A  (D_A dims)
                                              \
                                               concat --> Dropout(p) --> Dense(256, ReLU) --> Dense(1)
                                              /
CNN_B (frozen or fine-tuned) --> GAP --> f_B  (D_B dims)
```

**Rationale:**

- It is the simplest fusion that can learn cross-model interactions, and both directly-relevant precedents (D5, D15) support the general claim on cardiac CXR tasks.
- It requires **no new hyperparameters** beyond the head width and dropout — both of which Optuna will handle.
- Attention-based fusion is **not justified by the evidence available**. No paper in this survey demonstrated that attention fusion beats concatenation on cardiomegaly. Adding it would be complexity without evidence, and the worklet explicitly asks us not to do that.

**The control experiment that makes this claim defensible** [Our inference — and the part most papers skip]:

| Arm | Purpose |
| --- | --- |
| A1. Best single CNN, standard head | Baseline |
| A2. **Best single CNN with a head of equal parameter count to the fusion head** | Isolates head capacity |
| A3. **Same CNN concatenated with itself** (`[f_A ; f_A]`) → fusion head | Isolates the effect of doubled input dimensionality with **zero** new information |
| A4. **Top-2 different CNNs concatenated** → fusion head | The actual hypothesis |
| A5. Late fusion (probability averaging) of the top-2 | Cheap comparator; if A5 ≈ A4, feature fusion bought us nothing |

If A4 > A3 and A4 > A5 by a margin exceeding seed variance (DeLong-tested), we have real evidence for feature complementarity. If not, we report a null result. **Arm A3 is the cheapest and most informative control in the entire study and almost nobody runs it.** It should be a highlighted part of our methodology.

---

## 11. CNN Features + XGBoost

### 11.1 The two pipelines

```
Pipeline 1 (neural head):      CNN --> GAP --> Dropout --> Dense --> Softmax/Sigmoid --> P(cardiomegaly)
Pipeline 2 (tree head):        CNN --> GAP --> [frozen embeddings] --> XGBoost --> P(cardiomegaly)
```

### 11.2 Why XGBoost might work on CNN embeddings

| Reason | Explanation |
| --- | --- |
| **Non-linear axis-aligned partitioning** | Gradient-boosted trees carve the feature space with axis-aligned splits and can capture threshold effects and feature interactions that a single linear layer cannot |
| **Built-in regularisation** | XGBoost's L1/L2 penalties on leaf weights, `min_child_weight`, `subsample`, and `colsample_bytree` provide strong regularisation on small datasets where a wide dense head overfits |
| **Robustness to irrelevant features** | CNN embeddings contain many dead or near-constant dimensions; trees simply never split on them |
| **No gradient-flow constraints** | The head is trained by a completely different optimisation procedure, so it does not inherit the CNN's LR schedule or convergence pathologies |
| **Fast to fit and re-fit** | Once embeddings are cached, a full XGBoost fit takes seconds, making the comparison cheap |

### 11.3 Disadvantages and risks

| Issue | Detail | Mitigation |
| --- | --- | --- |
| **Dimensionality** | ResNet152 → 2048-d, DenseNet201 → 1920-d, concatenated hybrid → ~3968-d. Tree ensembles degrade in very high dimensions where most features are weakly informative | Optional PCA to 128–256 components (M10-style pipelines commonly do this); **or** rely on `colsample_bytree` and accept the dimensionality. **Report which** |
| **Overfitting** | With ~few thousand training samples and ~4000 features, XGBoost can memorise. Deep trees + many estimators is the failure mode | `max_depth` ≤ 6, `subsample` 0.7–0.9, `colsample_bytree` 0.5–0.8, early stopping on the validation set |
| **Feature scaling** | Trees are **scale-invariant** — standardisation is *not* required for XGBoost itself. But it *is* required if PCA precedes it (PCA is variance-driven) | If using PCA: `StandardScaler` fit on **train embeddings only**. If not using PCA: no scaling needed. Do not scale "just in case" and then forget you did |
| **No end-to-end learning** | The backbone cannot adapt to the tree head; the embeddings are frozen at whatever the softmax-trained backbone produced. **This structurally advantages the softmax head**, which was trained jointly with those very features | Acknowledge explicitly. Optionally also extract embeddings from a *self-supervised or ImageNet-only* backbone as a neutral third condition |
| **Loses spatial information** | Global average pooling collapses spatial maps before the tree head, so **Grad-CAM cannot be computed through XGBoost** | Grad-CAM is only run on the CNN arms. For the XGBoost arm, use SHAP over embedding dimensions if any explanation is wanted — noted as optional |
| **Computational** | Cheap once embeddings are cached; the expensive part is inference over the dataset to build the embedding matrix | Cache embeddings to `.npy` once and reuse |

### 11.4 Is this common in medical imaging?

[Moderate] Yes — CNN-feature-extraction-plus-classical-classifier is a well-populated sub-genre of the CXR literature:

- **M11**: VGG19 + InceptionV3 + ResNet50 deep features → PSO-optimised XGBoost, reporting Acc 98.71%, F1 99.25% on COVID CXR.
- **M12**: LeNet-5-style CNN's flatten layer → XGBoost, reporting improved accuracy and reduced training time versus the fully-connected head.
- **M10**: five CNNs' fused features → SVM (RBF), Acc 0.994.
- **Recent tree-based CXR work** (Bioengineering 2026): ResNet-18 fine-tuned features + PCA + SMOTE → decision tree / random forest / XGBoost, with the notable finding that **XGBoost showed stronger tolerance to feature-distribution shift** than the other tree learners, and that **feature-representation quality mattered more than classifier complexity** for cross-source generalisation.

**But the critical caveat** [Strong]: essentially all of these report headline accuracies of 98–99% on small, merged, class-balanced COVID/pneumonia datasets. Those numbers are not credible as estimates of clinical performance and are not a reason to expect similar gains on a properly-split cardiomegaly task. The one directly-relevant cardiomegaly data point we have — **D16** — found that XGBoost on two interpretable features **matched a full ResNet-50**, with the neural model *ahead* on accuracy/F1 and *behind* on AUC. That is the realistic expectation.

### 11.5 Recommendation: how to run this fairly

The experiment is only meaningful if **the only thing that changes is the classifier head.**

```
1. Take the SINGLE best-performing baseline CNN (by validation ROC-AUC), fully trained.
2. FREEZE it. Extract penultimate (post-GAP) embeddings for train / val / test
   using the IDENTICAL deterministic preprocessing, no augmentation, eval mode.
   Cache to disk.
3. HEAD A (neural):  train Dropout -> Dense(256) -> Dense(1) on the SAME frozen
                     train embeddings. Early stop on val AUC.
4. HEAD B (XGBoost): fit on the SAME frozen train embeddings, early stop on the
                     SAME val embeddings.
5. Give BOTH heads an IDENTICAL Optuna budget (same n_trials, same sampler, same
   pruner, same objective = val ROC-AUC).
6. Evaluate BOTH on the SAME test embeddings, at thresholds selected on the SAME
   val split.
7. Report mean +/- SD over 3 seeds, plus a DeLong test on the AUC difference.
```

**Three fairness points that are usually violated in the literature:**

- **Do not compare XGBoost-on-frozen-features against the *end-to-end fine-tuned* CNN.** That compares a head swap against a whole different training regime. The correct softmax comparator is a head trained on the **same frozen embeddings** (step 3).
- **Give both heads the same tuning budget.** Comparing a heavily-tuned XGBoost against a default dense layer is the most common way this comparison is rigged, including in M11 where XGBoost got PSO optimisation and the neural comparator did not.
- **State the structural bias.** The frozen embeddings were produced by a backbone trained jointly with a softmax head. That is a real, unavoidable advantage for Head A, and we should say so rather than pretend the comparison is perfectly neutral.

**Expected outcome [Our inference]:** a small difference in either direction, likely within seed variance. **This is a legitimate and publishable finding** — the literature is full of papers claiming large XGBoost gains under uncontrolled comparisons, and a clean null would be a genuine contribution.

---

## 12. Hyperparameter Optimization

### 12.1 Optuna

Optuna (Akiba et al. 2019, F6) is built on three design criteria: a **define-by-run API** (the search space is constructed dynamically in Python during the trial rather than declared upfront), efficient **searching and pruning**, and a lightweight, scalable architecture.

**TPE (Tree-structured Parzen Estimator)** is Optuna's default sampler and is a form of sequential model-based optimisation. Rather than modelling `p(objective | params)` directly as classical Bayesian optimisation with Gaussian Processes does, TPE models `p(params | objective)` by splitting observed trials into "good" and "bad" groups and fitting a density to each, then sampling candidates that maximise the ratio of good-density to bad-density. Practically this means it handles conditional and categorical parameters gracefully and scales better than GP-based BO in moderate dimensions.

**Pruning** is where the real budget saving lives. Optuna monitors intermediate objective values via a `report` API and terminates unpromising trials via `should_prune`. The impact in the original paper is dramatic: on a simplified AlexNet/SVHN benchmark within a fixed 4-hour budget, **TPE without pruning completed ~35.8 trials, while TPE with pruning explored ~1,278.6 trials (of which ~1,271.5 were pruned)**. The authors also found their **ASHA implementation significantly outperformed median pruning**, and concluded that "sampling algorithm alone is not sufficient for cost-effective optimization."

**Trial management:** Optuna persists trials in a study (in-memory or RDB-backed), supports resuming interrupted studies, and parallelises across workers without code changes — all directly useful when a Colab session dies mid-search.

### 12.2 Commonly optimised parameters and recommended ranges

| Parameter | Typical range in the literature | Our recommendation | Why |
| --- | --- | --- | --- |
| **Learning rate** | 1e-6 – 1e-2, log-uniform | `suggest_float("lr", 1e-5, 1e-3, log=True)` | Highest-impact single hyperparameter for fine-tuning; log scale is essential |
| **Weight decay** | 1e-6 – 1e-2, log-uniform | `suggest_float("wd", 1e-6, 1e-2, log=True)` | Main regulariser alongside dropout |
| **Dropout** | 0.0 – 0.6 | `suggest_float("dropout", 0.0, 0.5)` | Cheap regularisation on the head |
| **Batch size** | 8 – 128, categorical powers of 2 | `suggest_categorical("bs", [16, 32, 64])` | Interacts with LR; keep the range narrow and memory-feasible |
| **Optimiser** | Adam / AdamW / SGD+momentum | `suggest_categorical("opt", ["adamw", "sgd"])` — **optional** | Adds a strong conditional branch; include only if budget allows |
| **Dense units in head** | 64 – 1024 | `suggest_categorical("units", [128, 256, 512])` | Directly relevant to the fusion head |
| *(XGBoost arm)* `max_depth` | 3 – 10 | `suggest_int(3, 8)` | Primary complexity control |
| *(XGBoost arm)* `n_estimators` | 100 – 2000 | Use early stopping instead of tuning this | Avoids wasting trials on a parameter early stopping already handles |
| *(XGBoost arm)* `learning_rate` | 0.01 – 0.3, log | `suggest_float(0.01, 0.3, log=True)` | — |
| *(XGBoost arm)* `subsample`, `colsample_bytree` | 0.5 – 1.0 | `suggest_float(0.6, 1.0)` each | Critical given ~2000–4000 embedding dimensions |
| *(XGBoost arm)* `reg_lambda`, `reg_alpha` | 1e-3 – 10, log | `suggest_float(1e-3, 10, log=True)` | — |

**Objective:** validation ROC-AUC, maximised. **Never the test set.**

**Pruner:** `MedianPruner(n_startup_trials=5, n_warmup_steps=5)`. ASHA (`SuccessiveHalvingPruner`) is stronger per the original paper, but MedianPruner is more predictable at our small trial counts (30–50) where ASHA's asynchronous halving has little to work with.

**Budget:** 30–50 trials. Log every trial. Report the number of trials, the sampler, the pruner, the seed, and the search space in the paper. **An untuned baseline compared against a tuned model is not a comparison** — see §18.

### 12.3 Why optimise only the strongest model

[Our inference, grounded in cost arithmetic and fairness]

**The cost argument.** A single CXR training run across four architectures × 3 seeds is already 12 runs. Adding a 40-trial Optuna study to each of four models is 160 additional (partially pruned) runs. For a time-boxed UROP on shared GPU resources, that is not feasible, and cutting seeds or epochs to afford it would damage the more important result.

**The fairness argument — and this is the subtle one.** Tuning *all four* models with a fixed budget does not make the comparison fairer; it can make it *less* interpretable. Different architectures have differently-shaped loss landscapes, so an identical 40-trial budget explores each search space to a different degree, and the ranking then partly reflects "which architecture is easiest to tune in 40 trials." The cleanest design is:

- **Stage 1 (architecture comparison):** all four models trained under an **identical fixed protocol**. The claim is "under this protocol, architecture X ranks highest." This is a clean, attributable statement.
- **Stage 2 (optimisation):** take the Stage-1 winner and the hybrid, and tune **those two** with an identical Optuna budget. The claim is "tuning improved the best baseline by Δ AUC" and "tuning improved the hybrid by Δ AUC."

**What we must not do:** compare a tuned hybrid against an untuned baseline and attribute the difference to fusion. If the hybrid gets 40 Optuna trials, **the best single baseline must get 40 Optuna trials too.** This is non-negotiable for the fusion claim to mean anything.

### 12.4 BOHB — what it is, and why we are excluding it

**What it is.** BOHB (Falkner, Klein & Hutter, ICML 2018, F7) combines Bayesian optimisation with Hyperband. Hyperband is a multi-fidelity bandit method: it evaluates many configurations on a small budget (few epochs, subset of data), keeps the top fraction, and successively halves toward larger budgets. Hyperband has excellent *anytime* performance but, because it samples configurations at random, "for larger budgets does not perform much better than random search." Pure Bayesian optimisation is the mirror image — slow to start, but converges well given time. BOHB replaces Hyperband's random sampling with **kernel-density-estimator-guided** sampling, getting Hyperband's fast start *and* BO's convergence. The authors report it consistently outperforming both across toy functions, SVMs, feed-forward and Bayesian neural nets, RL, and CNNs.

**Why we are excluding it — honestly:**

1. **Its advantage materialises at large budgets.** BOHB's KDE model needs a substantial number of observations across fidelity levels before its guidance beats random sampling. At our 30–50 trial budget, we are operating in exactly the regime where BOHB is closest to plain Hyperband, and Optuna's TPE+MedianPruner already captures most of the available benefit.
2. **Optuna already provides multi-fidelity pruning.** `SuccessiveHalvingPruner` implements ASHA, the asynchronous successive-halving algorithm that underpins Hyperband's budget allocation. The gap between "TPE + ASHA in Optuna" and "BOHB" is narrow, and it is not the gap our project is about.
3. **Tooling cost.** BOHB's reference implementation is HpBandSter, a separate framework with its own worker/master architecture. Integrating a second HPO framework is engineering effort spent on a variable that is not our research question.
4. **It is not the thing under test.** Our contribution is a controlled comparison of architectures, fusion, and classifier heads. HPO is instrumentation, not the subject. Adding a second optimiser would expand scope without addressing any stated gap.

**What we will write in the paper:** "We used Optuna's TPE sampler with median pruning. Multi-fidelity methods such as BOHB (Falkner et al., 2018), which combine Bayesian optimisation with Hyperband's successive-halving budget allocation, are a stronger choice at large trial budgets; we did not adopt BOHB because our per-trial cost restricted us to 30–50 trials, a regime in which its KDE-guided sampling has limited opportunity to outperform TPE with pruning. Comparing HPO strategies was outside the scope of this study."

That sentence justifies the scope reduction. It does not pretend BOHB doesn't exist.

---

## 13. Explainable AI

### 13.1 Grad-CAM

**What it is.** Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017, F8) produces a coarse localisation map highlighting the image regions most influential for a particular class score.

**How it works, conceptually.**

1. Forward-pass the image, obtain the class score `y^c` (pre-softmax logit) and the feature maps `A^k` of a chosen convolutional layer.
2. Backpropagate `y^c` to that layer to get gradients `∂y^c / ∂A^k`.
3. **Global-average-pool those gradients spatially** to get one importance weight `α^c_k` per feature-map channel. Intuitively: "how much does raising the activation of channel *k* anywhere raise the class score?"
4. Take the weighted sum `Σ_k α^c_k · A^k` over channels.
5. Apply **ReLU** — keeping only features with a *positive* influence on the class.
6. Upsample the resulting low-resolution map to input size and overlay.

**Which layer.** The **last convolutional layer before global pooling**. This is the standard choice because it is the deepest layer that still retains spatial structure — the trade-off point between semantic abstraction (deeper is better) and spatial resolution (shallower is better). Concretely for our models: `layer4` output for ResNet152, `features.denseblock4`/`features.norm5` for DenseNet201, the final `features` block for EfficientNetV2 and MobileNetV3.

**What the heatmap represents.** A **class-discriminative, gradient-derived, positively-rectified, spatially-coarse attribution over the final convolutional feature grid**. For a 224×224 input and a 7×7 final grid, each heatmap "pixel" covers a 32×32 input region. It shows **where evidence for the class was concentrated in that feature grid** — not which pixels are causally necessary, and not what the model "believes."

**Strengths:**

- Class-discriminative (unlike plain CAM-free saliency).
- No architectural modification required — unlike original CAM, which needed a GAP-before-softmax structure.
- Cheap: one forward and one backward pass.
- Works uniformly across all four of our architectures, which matters for a comparative study.
- Ubiquitous in medical imaging, so our figures will be immediately legible to reviewers.

**Weaknesses and common failure modes:**

| Failure mode | Description |
| --- | --- |
| **Coarse resolution** | A 7×7 grid upsampled to 224×224 cannot delineate a boundary. Fine localisation claims are unsupportable |
| **ReLU discards negative evidence** | Regions that *argue against* the class are invisible |
| **Gradient saturation** | When the class score saturates, gradients approach zero and the map becomes noisy or empty |
| **Poor multi-instance handling** | With several disjoint regions of evidence, Grad-CAM tends to highlight one — a key motivation for Grad-CAM++ |
| **Highlights context, not the object** | The map may fall on correlated context (borders, text markers, support devices) rather than the anatomy |
| **Architecture-dependent appearance** | Different final-layer grid sizes across our four models give visually different maps for reasons unrelated to what the models learned — a real confound for a comparative study |
| **Insensitivity to model parameters** | Demonstrated by randomisation tests — see §14 |

### 13.2 Other XAI methods (context only)

| Method | Mechanism | Relevance to us |
| --- | --- | --- |
| **Grad-CAM++** (F9) | Replaces the globally-averaged gradient weight with a **pixel-wise weighted combination of positive partial derivatives**, improving localisation when multiple instances of a class are present and giving better coverage of extended objects | **Worth running as a secondary map** — cheap, and the cardiac silhouette is an extended object, exactly the case Grad-CAM++ was designed to improve |
| **LayerCAM** | Weights activations at a *per-spatial-location* level, allowing usable maps from **shallower** layers and hence finer resolution | Optional; useful if the 7×7 grid proves too coarse to say anything |
| **SHAP** (F14) | Shapley-value feature attributions with game-theoretic consistency guarantees | **Future work.** Practical on the XGBoost arm (`TreeSHAP` over embedding dimensions) but the resulting attributions are over abstract features, not pixels, so they are hard to interpret clinically |
| **LIME** (F15) | Fits a local interpretable surrogate around a single prediction using perturbed samples | **Future work.** On images this requires superpixel segmentation, which introduces yet another arbitrary choice; also slow per-image |
| **Saliency / vanilla gradients** | `∂y^c/∂x` directly | Baseline comparator only; known to be noisy and among the methods that fail sanity checks |
| **Integrated Gradients / SmoothGrad** | Path-integrated or noise-averaged gradients | Included in the Arun et al. evaluation; all evaluated methods failed at least one trustworthiness criterion |

**Confirmation from the field:** the 2025 systematic review of XAI in radiological cardiovascular imaging (D19) found Grad-CAM, SHAP, LIME, and saliency maps to be the dominant methods, with Grad-CAM most prevalent — and identified rigorous validation of explanations as the field's persistent shortfall.

**Our position:** Grad-CAM is our core method; Grad-CAM++ is a cheap secondary; SHAP and LIME are explicitly listed as future work and will not be implemented in this stage.

---

## 14. XAI Reliability

> **This section exists because the single easiest way to produce a bad medical-AI paper is to show four attractive heatmaps and call it validation.**

### 14.1 What the evidence actually says

**Arun et al. 2021 (M16), *Radiology: Artificial Intelligence*** is the definitive empirical study for our context. Using two large public **chest radiograph** datasets with pixel-level or bounding-box ground truth (SIIM-ACR Pneumothorax Segmentation, RSNA Pneumonia Detection), the authors evaluated **eight** widely-used saliency methods — including Grad-CAM and guided Grad-CAM — against four criteria for trustworthiness:

1. **Localisation utility** — does the map overlap the ground-truth region better than a baseline?
2. **Sensitivity to model weight randomisation** — does the map change when the trained weights are destroyed? (If not, the map is not explaining *the model*.)
3. **Repeatability** — do independently-trained models of the **same** architecture produce the same map?
4. **Reproducibility** — do models of **different** architectures produce the same map?

Findings:

- **All eight methods failed at least one criterion.**
- Only XRAI passed the localisation-utility test on both datasets — **and it failed the randomisation test on both.**
- **No saliency method was more repeatable or reproducible than a plain localisation network.**
- The authors' conclusion: saliency maps for high-risk medical applications are "problematic" and require additional validation before clinical application.

**Adebayo et al. 2018 (M17)** established the underlying result: several popular saliency methods are essentially **insensitive to both model parameters and training labels** — they act closer to edge detectors than to explanations. A map that looks identical under randomised weights is explaining the image, not the model.

**Ghassemi, Oakden-Rayner & Beam 2021 (M18)**, *Lancet Digital Health*, extends this to the clinical argument: the case that XAI will engender clinician trust, provide decision transparency, and mitigate bias is, for **patient-level** decision support, "a false hope." Current explainability methods are descriptive, not causal or mechanistic.

**Farina et al. 2023 (D18)** reach the same conclusion from the cardiology side: visualisation analysis "can partially help explain the AI models' decisions," but "it is still not possible to completely understand how the predictions are generated."

### 14.2 The specific failure modes we must watch for

| Failure mode | How it manifests on CXR |
| --- | --- |
| **Localisation error** | Heatmap on lung fields or diaphragm rather than the cardiac silhouette |
| **Instability across seeds** | Two identically-trained models produce visibly different maps for the same image (directly measured by Arun et al.'s repeatability criterion) |
| **Sensitivity to preprocessing** | Changing resize interpolation or CLAHE settings shifts the map |
| **Spurious correlations / shortcuts** | DeGrave et al. (M15) showed CXR models select shortcuts over signal; Zech et al. (M6) showed CNNs identify the source hospital with ~99.95% accuracy |
| **Attention to text markers, borders, laterality tokens** | Burned-in "PORTABLE"/"AP" markers, rotation indicators, and image borders are notorious CXR shortcuts |
| **Attention to support devices** | Pacemakers, ICD leads, sternal wires and central lines co-occur strongly with cardiac disease. A model attending to a pacemaker is predicting *treatment history*, not heart size. CheXpert has a `Support Devices` label — **we can and should test this directly** |
| **Plausible but unfaithful** | The map lands on the heart because the heart is the most salient structure in every chest radiograph, regardless of what the model used |

The last point deserves emphasis. Studies D1, D4, D5, and D6 all report saliency concentrating on anatomically sensible cardiac structures. That is reassuring — and it is also **exactly what you would expect even from a model using the wrong cues**, because the cardiac shadow is a high-contrast, high-gradient region in every frontal chest radiograph. **Anatomical plausibility of the heatmap is weak evidence.**

### 14.3 Practical checks we can perform within our scope

Four checks, ordered by cost. All are cheap; the first two are near-free.

**Check 1 — Model-weight randomisation (sanity check; ~1 hour) [MUST DO]**

Following Adebayo et al. Take the trained best model, produce Grad-CAM maps for N=20 fixed test images. Then progressively randomise the weights from the top layer downward, regenerating maps at each stage. Report SSIM between the original and each randomised map.

> *Pass criterion:* SSIM should **degrade substantially** as weights are randomised. If the map survives randomisation, Grad-CAM is not explaining our model, and we must say so.

**Check 2 — Repeatability across seeds (~free, uses runs we already have) [MUST DO]**

We are training 3 seeds per model anyway. For the same N=20 test images, compute pairwise SSIM between Grad-CAM maps from seed 42, seed 1337, and seed 2024 of the **same architecture**.

> *Report:* mean ± SD pairwise SSIM. This is Arun et al.'s repeatability criterion, reproduced on our task at essentially zero marginal cost.

**Check 3 — Cross-architecture reproducibility (~free) [SHOULD DO]**

Same 20 images, compute SSIM between maps from ResNet152 vs DenseNet201 vs EfficientNetV2 vs MobileNetV3. Caveat the comparison for differing final-grid resolutions (resample all to a common grid first).

> *Interpretation:* low cross-architecture SSIM means "the explanation" is architecture-dependent, and no single heatmap should be treated as *the* reason the prediction was made.

**Check 4 — Quantitative localisation against ground truth (~1 day, conditional) [DO IF VinDr-CXR IS AVAILABLE]**

This is the strongest available check and the reason VinDr-CXR is worth acquiring as a secondary asset. VinDr-CXR provides **radiologist-drawn bounding boxes for cardiomegaly**. Run the CheXpert-trained model on VinDr-CXR cardiomegaly-positive images and compute, for each:

- **Pointing game accuracy**: does the heatmap's peak fall inside the ground-truth box?
- **IoU** between the thresholded heatmap and the box.
- **AUPRC** of heatmap intensity as a pixel-level detector of box membership (Arun et al.'s metric).
- **Compare against two baselines**: a centre-of-image Gaussian prior, and the average heatmap across the dataset. If Grad-CAM does not beat "always point at the middle of the chest," it is providing no localisation information.

**Check 5 — Support-device confound probe (~2 hours) [SHOULD DO]**

CheXpert labels `Support Devices`. Stratify test-set performance by that label. If AUC is markedly higher on device-positive images, the model may be partly reading treatment history. Additionally, inspect Grad-CAM on device-positive false positives — if the heat lands on a pacemaker or sternal wires, we have found a shortcut and that is a **result worth reporting**.

### 14.4 How we must write about Grad-CAM

**Permitted claims:**

- "Grad-CAM attributions were predominantly localised to the cardiac silhouette region."
- "Grad-CAM maps showed *X* mean pairwise SSIM across independently-trained seeds, indicating [high/low] repeatability."
- "Inspection of Grad-CAM on false positives revealed a recurring pattern of attribution to [support devices / image borders], suggesting a potential shortcut."
- "Grad-CAM was used as a **debugging and failure-analysis tool**, not as evidence of clinical validity."

**Forbidden claims:**

- "Grad-CAM confirms the model is focusing on clinically relevant features." (Confirms nothing.)
- "The explanations demonstrate the model is trustworthy / clinically reliable / safe."
- "Grad-CAM validates our model."
- Any figure presenting only successful, attractive heatmaps. **Every Grad-CAM figure in our report must include at least one false positive and one false negative**, selected by a pre-stated rule (e.g. highest-confidence error of each type) rather than by eye.

---

## 15. Research Gap

Four candidate gaps, honestly assessed. Two are strong enough to build the project on; two are secondary contributions.

### Gap 1 — Absence of controlled cross-family architectural comparison for cardiomegaly `[PRIMARY]`

**Supporting literature:** D10, D15, M8, and the broader cardiomegaly literature (Kim 2021, Sarpotdar 2022, Ribeiro 2023, EfficientNetB7 cardiomegaly work, Cardio-XAttentionNet 2023).

**What existing work already does.** Many papers compare several CNNs on cardiomegaly. D15 compares VGG16, ResNet50, InceptionV3, DenseNet121, DenseNet201, AlexNet and ViTs. Cardio-XAttentionNet reports DenseNet121 0.89, ResNet50 0.87, DenseNet201 0.86, InceptionResNetV2 0.85. Ke et al. (M8) compared 16 architectures on CheXpert — but across all 5 competition tasks, not cardiomegaly specifically, and without fusion or classical-head arms.

**What remains unresolved.** These comparisons are almost never **protocol-controlled**. Papers vary datasets, splits, preprocessing, augmentation, epochs, LR, and metrics simultaneously, then attribute the resulting ranking to architecture. Ke et al. is the honourable exception and it explicitly warns that model *family* matters more than size — a finding **not yet tested on the cardiomegaly task in isolation**, and not tested with the modern EfficientNetV2 / MobileNetV3 families. Additionally, most papers report a single run, so architectural differences smaller than seed variance are reported as findings.

**How our project addresses it.** Four architectural families, one frozen dataset split, one preprocessing pipeline, one augmentation policy, one training protocol, one stopping rule, three seeds, DeLong-tested differences. Any observed ranking is then attributable to architecture-under-a-fixed-protocol — a claim we can actually defend.

**Novelty verdict:** **Controlled replication and extension.** Not novel methodology. It extends Ke et al.'s design to a cardiac-specific single-label task with two architecture families they did not include. Honest framing: *"we test whether Ke et al.'s family-over-size finding holds for cardiomegaly."*

### Gap 2 — Feature complementarity is asserted, never demonstrated `[PRIMARY]`

**Supporting literature:** D5 (soft-voting ensemble best), D15 (stacked meta-classifier best), M10 (5-CNN concat → SVM, 99.4%), M13 (multi-layer fusion), **against** D16 (fusion added nothing).

**What existing work already does.** Fusion and ensembling of CNNs demonstrably improves reported metrics on cardiac and thoracic CXR tasks. The gains are real in several studies.

**What remains unresolved.** Three things:

1. **Capacity is never controlled.** No paper in this survey ran the self-concatenation control (`[f_A ; f_A]`) that separates "two models carry complementary information" from "a wider head with more parameters fits better."
2. **Feature-level fusion is rarely compared against the near-free late-fusion baseline.** If probability averaging (D5's approach) matches feature concatenation, the added complexity is unjustified.
3. **Complementarity is never measured directly.** Papers show the fused model scores higher; none report a representational-similarity measure (CKA/CCA) or a prediction-disagreement analysis explaining *why*.

**How our project addresses it.** The five-arm control design in §10.4 (A1–A5), plus CKA between the top-2 embeddings, plus McNemar on prediction disagreement, plus linear probes on each embedding and the concatenation.

**Novelty verdict:** **Genuinely under-explored, and the strongest contribution available to us.** The self-concatenation control is cheap, obvious in hindsight, and — as far as this survey found — absent from the cardiomegaly fusion literature. This is a small but real methodological contribution, not novelty theatre.

### Gap 3 — Neural vs classical heads are compared under unequal conditions `[SECONDARY]`

**Supporting literature:** M11 (deep features → PSO-optimised XGBoost, 98.71%), M12 (flatten → XGBoost improved accuracy), M10 (concat → SVM, 99.4%), D16 (2-feature XGBoost ≈ ResNet-50).

**What existing work already does.** A substantial sub-literature swaps a CNN's dense head for XGBoost/SVM/RF and reports gains.

**What remains unresolved.** The comparisons are structurally unequal. M11's XGBoost received particle-swarm optimisation; the neural comparator did not. Papers routinely compare XGBoost-on-frozen-features against an end-to-end-fine-tuned CNN, conflating "different head" with "different training regime." Meanwhile the one directly-relevant cardiomegaly data point (D16) found **no meaningful difference**. The literature's consensus and its best-controlled data point disagree.

**How our project addresses it.** Same frozen embeddings, same validation split, same Optuna budget, same threshold-selection procedure, same seeds, DeLong test — with the structural bias favouring the softmax head stated explicitly.

**Novelty verdict:** **Controlled replication.** Not novel. But given the disagreement between the enthusiastic sub-literature and D16, a clean controlled answer on cardiomegaly has real value, and a null result is a publishable outcome.

### Gap 4 — Grad-CAM is presented without falsification in cardiac CXR work `[SECONDARY]`

**Supporting literature:** D19 (systematic review: rigorous XAI validation remains rare in cardiovascular imaging), M16 (all eight saliency methods failed ≥1 trustworthiness criterion on CXR), M17, M18. Against: D1, D4, D5, D6 all present saliency maps as supportive evidence without falsification tests.

**What existing work already does.** Cardiac CXR studies routinely include saliency figures and report that attention fell on plausible anatomy.

**What remains unresolved.** Arun et al. established the falsification battery in 2021, on chest radiographs, and the cardiac CXR literature has largely not adopted it. The 2025 XAI systematic review confirms validation is still the field's shortfall.

**How our project addresses it.** Run Checks 1–3 from §14.3 as standard (randomisation SSIM, seed repeatability SSIM, cross-architecture reproducibility SSIM), plus Check 5 (support-device confound probe), plus Check 4 against VinDr-CXR boxes if credentialing completes.

**Novelty verdict:** **Application of an existing methodology to a domain that hasn't adopted it.** Not novel method. But applying Arun et al.'s criteria to cardiomegaly models is a concrete, useful, and cheap contribution — and it is the section most likely to produce an unexpected finding.

### 15.5 The honest overall claim

> This project makes no methodological or architectural novelty claim. It is a **controlled comparative study and partial replication** that (a) tests whether architecture-family effects reported on general CXR benchmarks hold for cardiomegaly under a fixed protocol, (b) introduces a capacity-matched control that distinguishes genuine feature complementarity from added capacity in CNN fusion, (c) equalises the tuning budget in a neural-vs-tree classifier-head comparison that the existing literature conducts unequally, and (d) subjects Grad-CAM to published falsification tests that the cardiac chest-radiograph literature has not yet adopted. Its value is rigour and negative-result tolerance, not novelty.

That paragraph should go, more or less verbatim, into the paper.

---

## 16. Proposed Experimental Design

### 16.1 Pipeline

```
                       CheXpert (frontal only, Cardiomegaly binary)
                                        |
                        [S0] Patient-level split, frozen to CSV
                                        |
                        [S1] Fixed preprocessing + augmentation
                                        |
        +---------------+---------------+---------------+
        |               |               |               |
   ResNet152      DenseNet201    EffNetV2-S     MobileNetV3-L      [S2] Four baselines
        |               |               |               |
        +---------------+---------------+---------------+
                                        |
                        [S3] Model comparison (3 seeds, DeLong)
                                        |
                        [S4] Feature analysis on top-2 (UMAP/PCA/CKA/probe)
                                        |
                        [S5] Hybrid: concat(top-2) -> MLP head
                             + controls A2/A3/A5
                                        |
                        [S6] Optuna (identical budget: best baseline AND hybrid)
                                        |
                        [S7] Softmax head vs XGBoost head on SAME frozen embeddings
                                        |
                        [S8] Grad-CAM + reliability checks (randomisation, seeds,
                             cross-arch, support-device probe, VinDr boxes if avail.)
                                        |
                        [S9] Final analysis + honest limitations
```

### 16.2 Stage-by-stage specification

| Stage | Why it exists | Variable under test | Held constant | Metric | What result would be meaningful |
| --- | --- | --- | --- | --- | --- |
| **S0 — Split** | Everything downstream is meaningless without leakage control (M6, M2) | None — infrastructure | — | Verify zero patient overlap; report label prevalence per split | A split CSV committed to the repo, with an automated assertion that `set(train.patient) ∩ set(test.patient) == ∅` |
| **S1 — Preprocessing** | Preprocessing is the largest silent confound between "comparable" models | None — infrastructure | — | — | One `transforms.py`, imported by every experiment. Ablations only after S3 |
| **S2 — Four baselines** | Establish per-family performance under one protocol; test Ke et al. (M8) on cardiomegaly | **Architecture family** | Split, preprocessing, augmentation, optimiser, LR, batch size, loss, epochs, patience, seeds | Val ROC-AUC for selection; test ROC-AUC, PR-AUC, Sens@Spec0.90 | A ranking with **non-overlapping seed intervals**. If all four land within ±0.01 AUC, the honest finding is *"architecture family did not matter for this task at this scale"* — which directly contradicts M8's family-over-size claim on our task and is worth reporting |
| **S3 — Comparison** | Turn four numbers into a defensible statement | — | — | DeLong test on pairwise test AUC; mean ± SD over 3 seeds | A statistically-supported ranking, or an explicit statement that differences are within noise |
| **S4 — Feature analysis** | Generate and *test* the complementarity hypothesis before building on it | Representational structure of the top-2 | Same test embeddings, same UMAP seed | Linear-probe AUC; PCA explained variance; **CKA(A,B)**; prediction disagreement rate; McNemar | **Low CKA + high disagreement + comparable individual probe AUCs** ⇒ genuine complementarity ⇒ fusion is worth trying. **High CKA + low disagreement** ⇒ predict fusion will fail, and say so *before* running S5 |
| **S5 — Hybrid + controls** | Test whether fusion helps, and whether any gain is complementarity or capacity | **Fusion of two feature spaces** | Backbones frozen at their S2 checkpoints; same head architecture across A2–A5; same split, preprocessing, seeds | Test ROC-AUC; DeLong vs A1/A2/A3/A5 | **A4 > A3** (beats self-concatenation) **and A4 > A5** (beats free late fusion) ⇒ real complementarity. **A4 ≈ A3** ⇒ the gain is capacity, not fusion, and we say so |
| **S6 — Optuna** | Establish how much of any gap is tuning rather than design | **Hyperparameters** | Identical `n_trials`, sampler (TPE), pruner (Median), objective (val AUC), seed, search space *shape* | Best val AUC per study; test AUC of the best config; Δ vs untuned | **Both** the best baseline and the hybrid tuned with the same budget. If tuning closes the baseline-hybrid gap, the fusion claim collapses — that is a finding |
| **S7 — Head comparison** | Answer the softmax-vs-XGBoost question fairly | **Classifier head only** | Same frozen embeddings, same split, same Optuna budget, same threshold procedure | Test ROC-AUC, PR-AUC; DeLong | A difference exceeding seed variance in either direction. A null is expected (D16) and fully reportable |
| **S8 — Grad-CAM + checks** | Inspect failures; test whether the explanations are trustworthy at all | Explanation stability | Same 20 fixed test images throughout, chosen by a pre-stated rule | SSIM (randomisation, seed-repeatability, cross-architecture); pointing-game/IoU/AUPRC vs VinDr boxes; AUC stratified by `Support Devices` | **Substantial SSIM degradation under weight randomisation** = Grad-CAM is at least explaining our model. **High seed-repeatability SSIM** = the explanation is stable. Either result is reportable; the failure case is more interesting |
| **S9 — Final analysis** | Assemble, and state limits | — | — | — | A limitations section that names label noise, single-centre data, absent external validation, and the cardiomegaly-≠-heart-disease gap **without hedging** |

### 16.3 Realistic compute budget

| Stage | Runs | Notes |
| --- | ---: | --- |
| S2 baselines | 4 models × 3 seeds = **12** | The bulk of wall-clock time |
| S5 hybrid + controls | 5 arms × 3 seeds = **15** (backbones frozen ⇒ cheap; only heads train) | Cache embeddings once, reuse for all arms |
| S6 Optuna | 2 studies × ~40 trials, most pruned ≈ **2 studies** | Pruning is what makes this affordable (F6) |
| S7 heads | 2 heads × 3 seeds = **6** (on cached embeddings, minutes) | Near-free |
| S8 XAI | 20 images × several configurations | Near-free |
| Ablations (A1–A3 in §7.4) | 3 runs on best baseline | Optional |

**Total heavy training: ~12–15 full runs.** Everything else operates on cached embeddings. This is achievable within a UROP timeline.

---

## 17. Experimental Fairness

### 17.1 Must be identical across all compared models

| Factor | Enforcement mechanism |
| --- | --- |
| **Dataset split** | One CSV on disk with `patient_id, image_path, label, split`. Loaded by every script. Never regenerated |
| **Preprocessing** | One `transforms.py` module with `get_train_transforms()` / `get_eval_transforms()`. No per-model overrides |
| **Augmentation** | Same functions, same parameters, same probabilities |
| **Evaluation set** | The same test rows, evaluated once per model from the best-validation checkpoint |
| **Metrics + threshold policy** | One `evaluate.py`. Thresholds selected on validation by the same rule for every model |
| **Optimiser and schedule** | AdamW, same base LR, same cosine schedule, same weight decay |
| **Batch size** | One value, chosen to fit the most memory-hungry model |
| **Loss and class weighting** | `BCEWithLogitsLoss` with the same `pos_weight`, computed once from the training split |
| **Max epochs, early stopping metric, patience, min_delta** | Identical |
| **Checkpoint selection rule** | Best validation ROC-AUC, always |
| **Seeds** | The same three seeds `{42, 1337, 2024}` for every model. Seed set for Python, NumPy, and Torch (CPU + CUDA) |
| **Tuning budget** | If any model gets Optuna, its direct comparator gets an identical budget |
| **Hardware/software** | Same GPU type, same library versions, recorded in `requirements.txt` and reported |

### 17.2 May legitimately differ — with justification

| Factor | Why it must differ | How to keep it fair |
| --- | --- | --- |
| **Embedding dimensionality** | ResNet152 → 2048, DenseNet201 → 1920, EfficientNetV2-S → 1280, MobileNetV3-L → 1280. This is intrinsic to the architectures | Unavoidable. Report the dimensions. For the fusion head, do not add per-model projection layers unless applied uniformly |
| **Parameter count / FLOPs** | Intrinsic — and it is part of what we are comparing | Report params, FLOPs, and wall-clock training time per model as first-class results, not footnotes. **Accuracy-per-parameter is a legitimate secondary comparison** (cf. M8's parameter-efficiency analysis) |
| **Final conv grid size** | Affects Grad-CAM resolution | Resample all Grad-CAM maps to a common grid before any cross-architecture SSIM comparison, and state that this was done |
| **Native input resolution** | EfficientNetV2 was designed with progressive resizing and specific resolutions | **Force 224×224 for all four and disable progressive learning.** Note in limitations that this may under-serve EfficientNetV2 |
| **Batch-norm behaviour under small batches** | Different normalisation layer placements react differently | Cannot be equalised; document it |
| **Optimal learning rate** | Genuinely architecture-dependent | Either fix one LR (Option A, §6.4) or give all four the **identical three-point LR grid** (Option B). Never hand-tune one model more than another |
| **XGBoost hyperparameters** | A tree ensemble and an MLP have disjoint hyperparameter spaces | Equalise the **search budget** (`n_trials`), not the parameter names |

### 17.3 The fairness statement for the paper

> All models were trained on an identical patient-level dataset split, with identical preprocessing, augmentation, optimiser, learning-rate schedule, batch size, loss function with identical class weighting, maximum epoch count, early-stopping criterion, and checkpoint-selection rule, using the same three random seeds. Factors that necessarily differ between architectures — embedding dimensionality, parameter count, FLOPs, and final feature-grid resolution — are reported alongside performance rather than controlled away. Where hyperparameter optimisation was applied, every directly-compared model received an identical trial budget, sampler, pruner, and objective.

---

## 18. Things We Must Not Fuck Up

Ranked by how likely each is to invalidate the entire project.

### Tier 1 — Would invalidate every number in the paper

**1. Image-level splitting instead of patient-level splitting.**
CheXpert averages ~3.4 studies per patient. A random image split puts the same person's chest in both train and test. Every metric inflates, the architecture ranking becomes noise, and none of it is recoverable after the fact. **Mitigation:** split by `patient_id`, write an automated assertion that train/val/test patient sets are disjoint, and run it in CI. This is a five-line check that protects the whole project.

**2. Tuning on the test set.**
Every threshold, every early-stopping decision, every Optuna objective must use **validation** data. The test set is touched once, at the end, per model. **Mitigation:** keep the test split in a separate file that the training code physically cannot load.

**3. Comparing models trained under different conditions.**
Different epoch counts, different LRs, different batch sizes, different augmentation — and then calling the result an architecture comparison. **Mitigation:** the fairness table in §17.1, enforced by a single shared config file rather than by discipline.

**4. Comparing a tuned model against an untuned one.**
If the hybrid gets 40 Optuna trials and the baseline gets none, the hybrid will win and the result means nothing. **Mitigation:** any Optuna study on one arm obliges an identical study on its comparator.

### Tier 2 — Would seriously undermine credibility

**5. Reporting only the best run.**
Deep-learning seed variance on a few-thousand-image binary task can easily exceed 0.01 AUC. Reporting the best of three seeds and comparing it to another model's best of three manufactures differences from noise. **Mitigation:** mean ± SD over three fixed seeds, always. Report the SD even when it is embarrassing.

**6. Using accuracy alone, or headlining accuracy.**
See §8.2. **Mitigation:** ROC-AUC primary, PR-AUC and Sens@Spec0.90 co-primary; accuracy appears in a table, never in the abstract.

**7. Ignoring class imbalance.**
Reporting 88% accuracy on a dataset with 82% negatives. **Mitigation:** report prevalence in every split; use `pos_weight`; report PR-AUC.

**8. Using weak labels without discussing them.**
CheXpert training labels come from an NLP labeler over reports. Presenting our AUC as diagnostic accuracy misrepresents what was measured. **Mitigation:** state it in the abstract, the methods, and the limitations. Report the uncertainty-label policy and its ablation.

**9. Inconsistent preprocessing between train and inference.**
Forgetting to disable augmentation at eval, or applying different normalisation at test time. Silently degrades everything and is hard to spot. **Mitigation:** separate `get_train_transforms()` / `get_eval_transforms()`; unit-test that the eval transform is deterministic by running it twice and asserting equality.

**10. Cherry-picking Grad-CAM images.**
Selecting the four prettiest heatmaps. **Mitigation:** pre-state the selection rule (e.g. "the two highest-confidence true positives, the two highest-confidence false positives, and the two highest-confidence false negatives"), fix the image IDs before looking at the maps, and use the same six images for every model.

**11. Merging datasets to inflate sample size.**
Zech et al. showed CNNs identify the source hospital with ~99.95% accuracy and calibrate to site prevalence. Merging CheXpert + NIH + VinDr for training (as D15 does) risks the model learning "this looks like a VinDr image, and VinDr has more cardiomegaly." **Mitigation:** train on one dataset. Use a second dataset only as a held-out external test, and report it as such.

### Tier 3 — Would overclaim

**12. Claiming clinical validity from a benchmark result.**
No external validation, no prospective evaluation, no clinician-in-the-loop study, single centre. **Mitigation:** the word "clinical" should not appear near our own results, only in the discussion of what would be required next.

**13. Claiming Grad-CAM proves trustworthiness.**
See §14.4. **Mitigation:** the permitted/forbidden claim lists.

**14. Overclaiming novelty.**
Presenting a controlled replication as a novel method. Reviewers who know M8, D15, and M16 will notice immediately, and it damages everything else in the paper. **Mitigation:** the honest-claim paragraph in §15.5.

**15. Calling it "heart disease detection."**
We are classifying cardiomegaly. **Mitigation:** the title, abstract, and every results caption say "cardiomegaly." One sentence in the introduction explains the relationship to heart disease, with a citation to the weak CTR–LVEF correlation.

**16. Comparing our numbers to incompatible published results.**
Putting our CheXpert cardiomegaly AUC in a table next to D5's aortic-stenosis AUC, or next to a 99% accuracy from a merged COVID dataset. **Mitigation:** any comparison table must include dataset, split policy, and label provenance columns — which will make most comparisons visibly invalid, as it should.

**17. Not reporting negative results.**
If fusion doesn't help, if XGBoost doesn't help, if the four architectures tie — those are the findings. Quietly dropping the arms that "didn't work" is the single most common form of scientific dishonesty in student ML projects. **Mitigation:** decide the full arm list now, in this document, and report every arm regardless of outcome.

---

## 19. Recommended Methodology for Our UROP

Concrete, actionable, Stage-2-ready.

### 1. Dataset
**CheXpert**, downsampled release. Frontal views only (PA + AP), adults. Record PA/AP as a metadata column for the confound analysis. **Backup/secondary: VinDr-CXR** (begin PhysioNet credentialing and CITI training **immediately**, in parallel — it is off the critical path but only if started now).

### 2. Target labels
Binary `Cardiomegaly` ∈ {0, 1}. Uncertain (−1) → 0 in the primary analysis; U-Ones as a single reported ablation. `Support Devices` retained as a metadata column for the shortcut probe. Optional stretch: add `Enlarged Cardiomediastinum` as a second task — **not recommended for the core comparison**.

### 3. Train / validation / test strategy
Patient-level 70/15/15, stratified on the label at patient level, seed 42, **written once to `data/splits.csv` and committed**. Automated disjointness assertion in the test suite. Validation for all selection and tuning. Test evaluated once per model.

### 4. Image preprocessing
Frontal only → grayscale float32 [0,1] → **pad to square (preserves the cardiothoracic ratio)** → resize 224×224 bilinear → replicate to 3 channels → ImageNet mean/std normalisation. Identical for train, val, test.

### 5. Augmentation
`RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1))` + `ColorJitter(brightness=0.10, contrast=0.10)`. Train split only.
**Explicitly excluded: horizontal flip, vertical flip, shear, elastic deformation, MixUp, CutMix, random erasing.** Horizontal flip is excluded on anatomical grounds (heart laterality) and is run as a single ablation to test that reasoning.

### 6. Four CNN models
`ResNet152`, `DenseNet201`, `EfficientNetV2-S`, `MobileNetV3-Large` — all `torchvision` ImageNet weights, classifier replaced by a single-logit head. Progressive resizing disabled for EfficientNetV2.

### 7. Transfer learning strategy
Two-phase: (1) freeze backbone, train head 3 epochs @ LR 1e-3; (2) unfreeze all, train @ LR 1e-4 with cosine annealing. AdamW, `weight_decay=1e-4`. Batch size fixed at the largest power of two that fits DenseNet201. `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)`. Max 40 epochs, early stopping on validation ROC-AUC, patience 10, `min_delta=0.001`. Best-val-AUC checkpoint. Seeds {42, 1337, 2024}. **No gradual unfreezing** (breaks cross-architecture equivalence).

### 8. Evaluation metrics
**Primary:** test ROC-AUC (mean ± SD over 3 seeds, 95% bootstrap CI). **Co-primary:** PR-AUC; sensitivity at fixed specificity 0.90. **Secondary:** accuracy, precision, recall, specificity, F1 at a validation-selected threshold. **Diagnostic:** confusion matrix, ROC and PR curves. **Statistical:** DeLong test for all pairwise AUC comparisons. Also report parameters, FLOPs, and training wall-clock per model.

### 9. Feature extraction layer
**Post-global-average-pool, pre-classifier** activations. Dimensions: ResNet152 2048, DenseNet201 1920, EfficientNetV2-S 1280, MobileNetV3-L 1280. Extracted in `model.eval()` with `torch.no_grad()`, deterministic transforms, no augmentation. Cached to `.npy` for train/val/test.

### 10. Feature visualisation
**UMAP** (`n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`, `random_state=42`) as primary, with PCA cumulative explained variance at 2/10/50 components as a companion. Four colourings per model: true label, predicted label, correct/incorrect, PA-vs-AP view. **Every UMAP figure paired with a linear-probe AUC** and a caption disclaiming the interpretability of cluster size and inter-cluster distance. **CKA** between the top-2 models' embeddings. **t-SNE omitted.**

### 11. Hybrid architecture
`concat(f_top1, f_top2)` → `Dropout(p)` → `Dense(256, ReLU)` → `Dense(1)`, backbones frozen at their S2 checkpoints.
**Mandatory controls:** A2 (best single CNN with a parameter-matched head), **A3 (self-concatenation `[f_A ; f_A]` — the capacity control)**, A5 (late fusion by probability averaging). Report all five arms.

### 12. Optuna search space
`TPESampler(seed=42)` + `MedianPruner(n_startup_trials=5, n_warmup_steps=5)`, objective = validation ROC-AUC (maximise), `n_trials=40`, RDB-backed study so an interrupted session resumes.

```python
lr        = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
wd        = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
dropout   = trial.suggest_float("dropout", 0.0, 0.5)
units     = trial.suggest_categorical("units", [128, 256, 512])
batch     = trial.suggest_categorical("batch_size", [16, 32, 64])
```

Applied to **exactly two** models with identical budgets: the best S2 baseline, and the hybrid. Report `n_trials`, pruned-trial count, sampler, pruner, seed, and the full search space.

### 13. XGBoost experiment
On the **same cached frozen embeddings** as the softmax head. Optional PCA to 256 components (with `StandardScaler` fit on train only) — **report whether PCA was used**. `xgboost` with `early_stopping_rounds=50` on the validation embeddings. Identical Optuna budget (`n_trials=40`) to the softmax head, searching `max_depth ∈ [3,8]`, `learning_rate ∈ [0.01,0.3] log`, `subsample ∈ [0.6,1.0]`, `colsample_bytree ∈ [0.6,1.0]`, `reg_lambda`/`reg_alpha ∈ [1e-3,10] log`, `min_child_weight ∈ [1,10]`. Compare via DeLong. State the structural bias favouring the softmax head.

### 14. Grad-CAM
Target layer = last convolutional block before global pooling, per architecture. Grad-CAM primary, Grad-CAM++ secondary. **Six fixed test images**, selected by a pre-stated rule (2 highest-confidence TP, 2 highest-confidence FP, 2 highest-confidence FN), IDs frozen before inspection, identical across all models.
**Reliability checks (all mandatory except the last):** (1) cascading model-weight randomisation with SSIM; (2) cross-seed repeatability SSIM; (3) cross-architecture reproducibility SSIM on a resampled common grid; (4) `Support Devices`-stratified AUC + heatmap inspection on device-positive false positives; (5) *if VinDr-CXR credentialing completes* — pointing-game accuracy, IoU, and AUPRC against radiologist bounding boxes, benchmarked against a centre-Gaussian baseline.

### 15. Reproducibility strategy

| Element | Commitment |
| --- | --- |
| Split | `data/splits.csv` committed to the repo |
| Seeds | Fixed {42, 1337, 2024}; set for `random`, `numpy`, `torch`, `torch.cuda`; `torch.use_deterministic_algorithms(True)` where feasible |
| Config | A single YAML per experiment; no hardcoded hyperparameters in scripts |
| Environment | Pinned `requirements.txt` + Python/CUDA/GPU model recorded in the paper |
| Logging | Every run logs config hash, git commit SHA, all metrics, and wall-clock time |
| Optuna | RDB-backed study files retained and released |
| Results | Per-seed raw metrics released as CSV, not just aggregates |
| Code | Public repository with a README that reproduces every table and figure from the committed split |
| Negative results | Every arm listed in §16 reported regardless of outcome |

---

## 20. Evidence Quality Summary

| Conclusion | Evidence label | Basis |
| --- | --- | --- |
| No public CXR dataset has a genuine "heart disease" label; cardiomegaly is the tractable public target | **Strong** | Direct inspection of dataset documentation (M1–M4) + the private-data status of every Tier 2 study (D1, D2, D5–D9) |
| Cardiothoracic ratio / cardiomegaly correlates only weakly with actual cardiac function | **Strong** | Philbin et al. 1998 and Mueller-Lenke et al. 2006, both cited in Farina et al. (D18); reiterated in Ueda et al. (D1) |
| **Dataset choice: CheXpert** | **Moderate / Our inference** | The label, patient-ID, and access facts are Strong; the *weighting* of access latency against label quality is our judgement call |
| ChestX-ray14 labels are unreliable | **Strong** | Radiologist audit with quantified PPV shortfall (M5) |
| ImageNet rank does not predict CXR performance | **Strong** | Ke et al. on 16 architectures (M8) |
| Model family matters more than size within a family | **Moderate** | Single strong study (M8), and only for models without pretraining |
| ImageNet pretraining helps convergence more than final accuracy | **Strong** | Two independent studies agreeing (M7, M8) |
| **Model selection: four families as a comparison design** | **Moderate** | Justified by M8; the specific choice of these four is Our inference |
| Horizontal flip is inappropriate for cardiomegaly | **Our inference** | Anatomically grounded (cardiac laterality) but **not directly tested in the literature** — which is why we run it as an ablation |
| CLAHE improves CXR classification | **Limited** | Used and reported positively in several studies (M14 and others) but rarely ablated. Treated as an ablation, not a default |
| ROC-AUC + PR-AUC + sensitivity-at-specificity is the right metric set | **Strong** | Universal in the [DR] literature; imbalance argument is standard; the AAC review's 0.22-sensitivity example is a concrete demonstration |
| **Hybrid architectures: fusion can improve cardiac CXR performance** | **Moderate** | Two directly-relevant positives (D5 ensemble, D15 stacking) and several [MR] positives (M10, M13) — **against one directly-relevant null (D16)**. Capacity is uncontrolled in all of them |
| **Feature concatenation is the right fusion choice for us** | **Our inference** | Simplest method with precedent; attention fusion has no supporting evidence on this task |
| **XGBoost on CNN embeddings improves over a softmax head** | **Limited, and contested** | Several [MR] positives under uncontrolled comparisons (M10–M12) vs one directly-relevant null (D16). **Expect no difference** |
| Optuna TPE + pruning is an appropriate HPO choice at our budget | **Strong** | Akiba et al.'s own pruning experiments (F6) |
| Excluding BOHB is justified at 30–50 trials | **Moderate / Our inference** | Grounded in BOHB's documented budget-dependent advantage (F7), but the specific threshold is our judgement |
| **Grad-CAM does not establish trustworthiness** | **Strong** | Arun et al. — all eight methods failed ≥1 criterion, on chest radiographs (M16); Adebayo et al. (M17); Ghassemi et al. (M18); confirmed as a field-wide shortfall by the 2025 systematic review (D19) |
| Grad-CAM is still useful as a debugging/failure-inspection tool | **Moderate / Our inference** | Not contradicted by the above; supported by the shortcut-detection literature (M6, M15) where saliency did surface real confounds |
| Merging datasets for training risks site confounding | **Strong** | Zech et al. — 99.95% hospital identification (M6) |
| Anatomy-aware (segmentation/CTR) methods outperform image-level classification for cardiomegaly | **Moderate** | One strong direct study (D10, AUC 0.978 with ~100× fewer annotations) plus supporting reproducibility evidence (D14, ICC>0.98) |
| **Our research gap (controlled comparison + capacity-controlled fusion test)** | **Our inference** | The *absence* of the self-concatenation control in the surveyed literature is an argument from absence; it is possible we missed a paper that runs it |

---

## 21. Decisions Requiring Human Confirmation

These are genuinely open and should go to the supervisor before Stage 2 begins.

| # | Decision | Options | Our lean | Why it needs a human |
| --- | --- | --- | --- | --- |
| 1 | **Reframe the project title** from "heart disease classification" to "cardiomegaly classification" | (a) Reframe fully; (b) keep the title, reframe in the abstract; (c) keep as-is | **(a)** | This changes the stated project title. Academic-supervisor call, not a technical one |
| 2 | **CheXpert vs MIMIC-CXR-JPG** | If the team or supervisor already holds PhysioNet credentials, MIMIC's **official reference splits** make it the better primary choice | CheXpert, unless credentials already exist | Depends on facts we don't have |
| 3 | **Start VinDr-CXR credentialing now?** | 1–3 weeks; unlocks quantitative Grad-CAM localisation and an external-validation check | **Yes — start immediately, in parallel** | Requires a supervisor reference for PhysioNet credentialing |
| 4 | **CheXpert test-set availability** | The 500-study test set was historically withheld for the leaderboard; test labels have since been distributed via Stanford AIMI | **Verify before designing the eval protocol** | A factual check on current availability. If unavailable, we carve our own test split from training patients and say so |
| 5 | **LR policy: single fixed LR (Option A) vs identical 3-point grid (Option B)** | §6.4 | **Option B if compute allows** | Compute budget decision |
| 6 | **Is a null result acceptable as the project outcome?** | Fusion may not help; XGBoost may not help; the four models may tie | We believe **yes**, and have designed for it | This is the most important conversation to have *before* running anything. If the supervisor expects a positive result, the project's incentives are wrong from day one |
| 7 | **Multi-label extension** (`Cardiomegaly` + `Enlarged Cardiomediastinum`) | Adds scope; complicates every metric and fusion analysis | **No** for the core study | Scope decision |
| 8 | **Compute allocation** | ~12–15 full training runs + 2 Optuna studies + cached-embedding work | — | Determines whether Option B, the ablations, and the VinDr external check are affordable |
| 9 | **Is a comparison against the segmentation/CTR paradigm expected?** | D10 shows segmentation substantially outperforms image-level classification for cardiomegaly | Discuss it in related work; **do not implement** | If the supervisor wants it implemented, scope changes materially |

---

## 22. Bibliography

Formatted consistently; DOIs given where verified during this survey. Where a record is a conference **abstract** rather than a full paper, this is stated.

### Directly relevant — cardiovascular inference from chest radiographs

1. Ueda, D., Matsumoto, T., Ehara, S., Yamamoto, A., Walston, S.L., Ito, A., Shimono, T., Shiba, M., Takeshita, T., Fukuda, D., et al. (2023). Artificial intelligence-based model to classify cardiac functions from chest radiographs: a multi-institutional, retrospective model development and validation study. *The Lancet Digital Health*, 5(8), e525–e533. DOI: 10.1016/S2589-7500(23)00107-3
2. Hsiang, C.W., Lin, C., Liu, W.C., Lin, C.S., Chang, W.C., Hsu, H.H., Huang, G.S., Lou, Y.S., Lee, C.C., Wang, C.H., et al. (2022). Detection of Left Ventricular Systolic Dysfunction Using an Artificial Intelligence-Enabled Chest X-ray. *Canadian Journal of Cardiology*, 38(6), 763–773. DOI: 10.1016/j.cjca.2021.12.019
3. Matsumoto, T., Kodera, S., Shinohara, H., Kiyosue, A., Higashikuni, Y., Akazawa, H., Komuro, I. (2020). Diagnosing heart failure from chest X-ray images using deep learning. *European Heart Journal*, 41(Suppl 2), 1201. DOI: 10.1093/ehjci/ehaa946.1201 — **conference abstract**
4. Matsumoto, T., Ehara, S., Walston, S.L., Mitsuyama, Y., Miki, Y., Ueda, D. (2022). Artificial intelligence-based detection of atrial fibrillation from chest radiographs. *European Radiology*, 32(9), 5890–5897. DOI: 10.1007/s00330-022-08752-0
5. Ueda, D., Yamamoto, A., Ehara, S., Iwata, S., Abo, K., Walston, S.L., Matsumoto, T., Shimazaki, A., Yoshiyama, M., Miki, Y. (2022). Artificial intelligence-based detection of aortic stenosis from chest radiographs. *European Heart Journal – Digital Health*, 3(1), 20–28. DOI: 10.1093/ehjdh/ztab102
6. Ueda, D., Ehara, S., Yamamoto, A., Iwata, S., Abo, K., Walston, S.L., Matsumoto, T., Shimazaki, A., Yoshiyama, M., Miki, Y. (2022). Development and Validation of Artificial Intelligence-based Method for Diagnosis of Mitral Regurgitation from Chest Radiographs. *Radiology: Artificial Intelligence*, 4(2), e210221. DOI: 10.1148/ryai.210221
7. Kusunose, K., Hirata, Y., Tsuji, T., Kotoku, J., Sata, M. (2020). Deep learning to predict elevated pulmonary artery pressure in patients with suspected pulmonary hypertension using standard chest X-ray. *Scientific Reports*, 10, 19311. DOI: 10.1038/s41598-020-76359-w
8. D'Ancona, G., Massussi, M., Savardi, M., Signoroni, A., Di Bacco, L., Farina, D., Metra, M., Maroldi, R., Muneretto, C., Ince, H., et al. (2023). Deep learning to detect significant coronary artery disease from plain chest radiographs AI4CAD. *International Journal of Cardiology*, 370, 435–441. DOI: 10.1016/j.ijcard.2022.10.154
9. Kamel, P.I., Yi, P.H., Sair, H.I., Lin, C.T. (2021). Prediction of Coronary Artery Calcium and Cardiovascular Risk on Chest Radiographs Using Deep Learning. *Radiology: Cardiothoracic Imaging*, 3(3), e200486. DOI: 10.1148/ryct.2021200486
10. Sogancioglu, E., Murphy, K., Calli, E., Scholten, E.T., Schalekamp, S., van Ginneken, B. (2020). Cardiomegaly Detection on Chest Radiographs: Segmentation Versus Classification. *IEEE Access*, 8, 94631–94642. DOI: 10.1109/ACCESS.2020.2995567
11. Lu, M.T., Ivanov, A., Mayrhofer, T., Hosny, A., Aerts, H.J.W.L., Hoffmann, U. (2019). Deep Learning to Assess Long-term Mortality From Chest Radiographs. *JAMA Network Open*, 2(7), e197416. DOI: 10.1001/jamanetworkopen.2019.7416
12. Weiss, J., Raghu, V.K., Bontempi, D., Christiani, D.C., Mak, R.H., Lu, M.T., et al. (2024). Deep Learning to Estimate Cardiovascular Risk From Chest Radiographs: A Risk Prediction Study. *Annals of Internal Medicine*, 177(4). DOI: 10.7326/M23-1898
13. Ieki, H., Ito, K., Saji, M., Kawakami, R., Nagatomo, Y., Takada, K., Kariyasu, T., Machida, H., Koyama, S., Yoshida, H., et al. (2022). Deep learning-based age estimation from chest X-rays indicates cardiovascular prognosis. *Communications Medicine*, 2, 159. DOI: 10.1038/s43856-022-00220-6
14. Kim, C., Lee, G., Oh, H., Jeong, G., Kim, S.W., Chun, E.J., Kim, Y.H., Lee, J.G., Yang, D.H. (2022). A deep learning-based automatic analysis of cardiovascular borders on chest radiographs of valvular heart disease: development/external validation. *European Radiology*, 32(3), 1558–1569. DOI: 10.1007/s00330-021-08296-9
15. CELM: An Ensemble Deep Learning Model for Early Cardiomegaly Diagnosis in Chest Radiography (2025). *Diagnostics* (MDPI). PMID: 40647601; PMC12249172. — full author list to be completed from the published record before citing in the paper
16. Multimodal Cardiomegaly Classification with Image-Derived Digital Biomarkers (2022). In *Medical Image Understanding and Analysis (MIUA)*, Lecture Notes in Computer Science. Springer. DOI: 10.1007/978-3-031-12053-4_2 — full author list to be completed
17. Hirata, Y., Kusunose, K., Yamada, H., Tsuji, T., Fujimori, K., Kotoku, J., Sata, M. (2020). Deep Learning for Detection of Elevated Pulmonary Artery Wedge Pressure Using Standard Chest X-ray. *Circulation*, 142(Suppl 3), A13157. DOI: 10.1161/circ.142.suppl_3.13157 — **conference abstract**
18. Farina, J.M., Pereyra, M., Mahmoud, A.K., Scalia, I.G., Abbas, M.T., Chao, C.J., Barry, T., Ayoub, C., Banerjee, I., Arsanjani, R. (2023). Artificial Intelligence-Based Prediction of Cardiovascular Diseases from Chest Radiography. *Journal of Imaging*, 9(11), 236. DOI: 10.3390/jimaging9110236
19. Explainable Artificial Intelligence in Radiological Cardiovascular Imaging — A Systematic Review (2025). *Diagnostics*, 15(11), 1399. DOI: 10.3390/diagnostics15111399 — full author list to be completed
20. Measurement of Cardiothoracic Ratio on Chest X-rays Using Artificial Intelligence — A Systematic Review and Meta-Analysis (2024). *Journal of Clinical Medicine*, 13, 4659. PROSPERO CRD42023437459. PMC11355006 — full author list to be completed
21. Lee, D.K., Kim, J.H., Oh, J., Kim, T.H., Yoon, M.S., Im, D.J., Chung, J.H., Byun, H. (2022). Detection of acute thoracic aortic dissection based on plain chest radiography and a residual neural network (ResNet). *Scientific Reports*, 12, 21884. DOI: 10.1038/s41598-022-26486-3
22. Liu, W.T., Lin, C.S., Tsao, T.P., Lee, C.C., Cheng, C.C., Chen, J.T., Tsai, C.S., Lin, W.S., Lin, C. (2022). A Deep-Learning Algorithm-Enhanced System Integrating Electrocardiograms and Chest X-rays for Diagnosing Aortic Dissection. *Canadian Journal of Cardiology*, 38(1), 160–168. DOI: 10.1016/j.cjca.2021.09.028
23. Philbin, E.F., Garg, R., Danisa, K., et al. (1998). The relationship between cardiothoracic ratio and left ventricular ejection fraction in congestive heart failure. *Archives of Internal Medicine*, 158(5), 501–506. DOI: 10.1001/archinte.158.5.501
24. Mueller-Lenke, N., Rudez, J., Staub, D., et al. (2006). Use of chest radiography in the emergency diagnosis of acute congestive heart failure. *Heart*, 92(5), 695–696. DOI: 10.1136/hrt.2005.074583
25. Artificial Intelligence for Detecting Aortic Arch Calcification on Chest Radiographs: A Systematic Review (2025/2026). *Diagnostics* (MDPI). PROSPERO CRD420251208627; PMC12839748 — full author list to be completed

### Datasets

26. Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., Seekins, J., Mong, D.A., Halabi, S.S., Sandberg, J.K., Jones, R., Larson, D.B., Langlotz, C.P., Patel, B.N., Lungren, M.P., Ng, A.Y. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 590–597. DOI: 10.1609/aaai.v33i01.3301590. arXiv:1901.07031
27. Johnson, A.E.W., Pollard, T.J., Greenbaum, N.R., Lungren, M.P., Deng, C., Peng, Y., Lu, Z., Mark, R.G., Berkowitz, S.J., Horng, S. (2019). MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs. arXiv:1901.07042. PhysioNet: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
28. Johnson, A.E.W., et al. (2019). MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. *Scientific Data*, 6, 317. DOI: 10.1038/s41597-019-0322-0
29. Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., Summers, R.M. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *CVPR 2017*, 2097–2106. DOI: 10.1109/CVPR.2017.369
30. Nguyen, H.Q., Lam, K., Le, L.T., Pham, H.H., Tran, D.Q., Nguyen, D.B., Le, D.D., Pham, C.M., Tong, H.T.T., Dinh, D.H., et al. (2022). VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. *Scientific Data*, 9, 429. DOI: 10.1038/s41597-022-01498-w. arXiv:2012.15029
31. Nguyen, H.Q., Pham, H.H., Le, L.T., Dao, M., Lam, K. (2021). VinDr-CXR: An open dataset of chest X-rays with radiologist annotations (version 1.0.0). *PhysioNet*. DOI: 10.13026/3akn-b287
32. Bustos, A., Pertusa, A., Salinas, J.-M., de la Iglesia-Vayá, M. (2020). PadChest: A large chest x-ray image dataset with multi-label annotated reports. *Medical Image Analysis*, 66, 101797. arXiv:1901.07441

### Dataset quality, generalisation, and confounding

33. Oakden-Rayner, L. (2020). Exploring Large-scale Public Medical Image Datasets. *Academic Radiology*, 27(1), 106–112. DOI: 10.1016/j.acra.2019.10.006
34. Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J., Oermann, E.K. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. *PLOS Medicine*, 15(11), e1002683. DOI: 10.1371/journal.pmed.1002683
35. Oakden-Rayner, L., Dunnmon, J., Carneiro, G., Ré, C. (2020). Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. *Proceedings of the ACM Conference on Health, Inference, and Learning (CHIL)*, 151–159. DOI: 10.1145/3368555.3384468
36. DeGrave, A.J., Janizek, J.D., Lee, S.-I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence*, 3, 610–619. DOI: 10.1038/s42256-021-00338-7
37. Packhäuser, K., Gündel, S., Münster, N., Syben, C., Christlein, V., Maier, A. (2022). Deep learning-based patient re-identification is able to exploit the biometric nature of medical chest X-ray data. *Scientific Reports*, 12, 14851. DOI: 10.1038/s41598-022-19045-3

### Transfer learning and architecture selection

38. Raghu, M., Zhang, C., Kleinberg, J., Bengio, S. (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. *Advances in Neural Information Processing Systems (NeurIPS)*, 32. arXiv:1902.07208
39. Ke, A., Ellsworth, W., Banerjee, O., Ng, A.Y., Rajpurkar, P. (2021). CheXtransfer: Performance and Parameter Efficiency of ImageNet Models for Chest X-Ray Interpretation. *Proceedings of the ACM Conference on Health, Inference, and Learning (CHIL '21)*, 116–124. DOI: 10.1145/3450439.3451867. arXiv:2101.06871
40. Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M.P., Ng, A.Y. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv:1711.05225

### Architectures

41. He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*, 770–778. DOI: 10.1109/CVPR.2016.90. arXiv:1512.03385
42. Huang, G., Liu, Z., van der Maaten, L., Weinberger, K.Q. (2017). Densely Connected Convolutional Networks. *CVPR 2017*, 4700–4708. DOI: 10.1109/CVPR.2017.243. arXiv:1608.06993
43. Tan, M., Le, Q.V. (2021). EfficientNetV2: Smaller Models and Faster Training. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, 10096–10106. arXiv:2104.00298
44. Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V., Adam, H. (2019). Searching for MobileNetV3. *ICCV 2019*, 1314–1324. arXiv:1905.02244

### Fusion and CNN + classical classifiers

45. Fusion-Extracted Features by Deep Networks for Improved COVID-19 Classification with Chest X-ray Radiography (2023). *Diagnostics* / PMC10218019 — full author list to be completed
46. Automatic method for classifying COVID-19 patients based on chest X-ray images, using deep features and PSO-optimized XGBoost (2021). *Expert Systems with Applications* / PMID 34177133 — full author list to be completed
47. Hybrid CNN and XGBoost Model Tuned by Modified Arithmetic Optimization Algorithm for COVID-19 Early Diagnostics from X-ray Images (2022). *Electronics*, 11(22), 3798. DOI: 10.3390/electronics11223798
48. MultiFusionNet: Multilayer Multimodal Fusion of Deep Neural Networks for Chest X-Ray Image Classification (2024). arXiv:2401.00728
49. Improving Tree-Based Lung Disease Classification from Chest X-Ray Images Using Deep Feature Representations (2026). *Bioengineering*, 13(3), 267. DOI: 10.3390/bioengineering13030267
50. Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. DOI: 10.1145/2939672.2939785. arXiv:1603.02754

### Preprocessing

51. An optimized EfficientNetB0 framework with CLAHE-based preprocessing for accurate multi-class chest X-ray classification (2026). *Scientific Reports*, 16, 10811. DOI: 10.1038/s41598-026-42492-1
52. Pre-processing methods in chest X-ray image classification (2022). *PLOS ONE*, 17(4), e0265949. DOI: 10.1371/journal.pone.0265949

### Hyperparameter optimisation

53. Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2623–2631. DOI: 10.1145/3292500.3330701. arXiv:1907.10902
54. Falkner, S., Klein, A., Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80, 1437–1446. arXiv:1807.01774
55. Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research*, 13, 281–305

### Explainable AI

56. Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*, 618–626. DOI: 10.1109/ICCV.2017.74. arXiv:1610.02391
57. Chattopadhay, A., Sarkar, A., Howlader, P., Balasubramanian, V.N. (2018). Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks. *IEEE Winter Conference on Applications of Computer Vision (WACV)*, 839–847. DOI: 10.1109/WACV.2018.00097. arXiv:1710.11063
58. Arun, N., Gaw, N., Singh, P., Chang, K., Aggarwal, M., Chen, B., Hoebel, K., Gupta, S., Patel, J., Gidwani, M., Adebayo, J., Li, M.D., Kalpathy-Cramer, J. (2021). Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging. *Radiology: Artificial Intelligence*, 3(6), e200267. DOI: 10.1148/ryai.2021200267. arXiv:2008.02766
59. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B. (2018). Sanity Checks for Saliency Maps. *Advances in Neural Information Processing Systems (NeurIPS)*, 31, 9505–9515
60. Ghassemi, M., Oakden-Rayner, L., Beam, A.L. (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health*, 3(11), e745–e750. DOI: 10.1016/S2589-7500(21)00208-9
61. Lundberg, S.M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774
62. Ribeiro, M.T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144. DOI: 10.1145/2939672.2939778
63. Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215. DOI: 10.1038/s42256-019-0048-x

### Representation analysis

64. van der Maaten, L., Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605
65. McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426
66. Wattenberg, M., Viégas, F., Johnson, I. (2016). How to Use t-SNE Effectively. *Distill*. DOI: 10.23915/distill.00002
67. Kobak, D., Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. *Nature Communications*, 10, 5416. DOI: 10.1038/s41467-019-13056-x

### Bibliography completeness note

Entries 15, 16, 19, 20, 25, 45, and 46 are cited by DOI/PMID/PMC identifier with author lists to be completed from the published record before any of this material is used in a submitted manuscript. **Every factual claim attributed to these sources in this review comes from the abstract or full text retrieved during the survey; none is invented.** No citation in this document was generated from memory without a corresponding retrieved record.

---

## 23. Final Quality-Control Record

| QC step | Status | Note |
| --- | --- | --- |
| 1. Every citation checked | **Partial** | All entries verified against a retrieved record; seven entries need full author lists completed (see note above) |
| 2. Unsupported claims removed | **Done** | Remaining inferences are explicitly labelled **[Our inference]** |
| 3. Contradictions between papers identified | **Done** | Three flagged and discussed openly: (a) fusion helps (D5, D15, M10) vs fusion adds nothing (D16); (b) XGBoost-on-features gives large gains (M11, M12) vs no meaningful difference (D16); (c) ImageNet pretraining gives little benefit (M7) vs a statistically significant boost (M8) |
| 4. Direct evidence separated from inference | **Done** | §20 evidence-quality table |
| 5. Dataset facts verified against official sources | **Done** | VinDr-CXR verified directly against the PhysioNet record; CheXpert, MIMIC-CXR-JPG, and ChestX-ray14 verified against the original dataset papers |
| 6. Architecture claims verified against original papers | **Done** | ResNet, DenseNet, EfficientNetV2, MobileNetV3 claims are drawn from the original publications |
| 7. Grad-CAM claims verified against foundational/review literature | **Done** | Selvaraju et al. for mechanism; Arun et al., Adebayo et al., Ghassemi et al., and the 2025 systematic review for limitations |
| 8. Recommendations compatible with the reduced scope | **Done** | ~12–15 heavy training runs; everything else on cached embeddings |
| 9. Methodology completable in a UROP timeline | **Done, with one dependency** | VinDr-CXR credentialing is the only long-lead item and it is deliberately off the critical path |
| 10. No implementation code modified | **Done** | This document is the only deliverable of Stage 1 |
