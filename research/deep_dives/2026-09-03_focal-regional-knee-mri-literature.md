# Focal-vs-regional trade in knee-MRI classification models — literature answer to Q-CF

**Question (Claude, Q-CF).** In the project's 182 members, component 1 loads positively on
Effusion/Synovitis (regional fluid) and negatively on Baker's/Medial Meniscus/MCL
(focal structures). Is this focal/regional trade documented in the literature, and
which architectural or training properties break it: input resolution,
localization-before-classification, slice density/through-plane, objective/loss
weighting, or pretraining corpus?

**Short answer.** The trade is visible in almost every published per-finding AUC table,
but it is not an invariant law: several groups report that the gap can be closed or
reversed with the right inductive biases. The strongest evidence points to
(1) explicit localization/ROI cropping, (2) 3D/through-plane and multi-view feature
aggregation, and (3) class-aware loss weighting. Input resolution and pretraining
corpus are real knobs, but the published ablations are thinner and the effects are
more task-dependent.

---

## 1. The trade is real in public benchmarks

| Study | Dataset / model | Focal-structure AUC | Regional-fluid AUC |
|---|---|---|---|
| Bien et al., PLOS Medicine 2018 (MRNet) | MRNet, internal validation | Meniscus tear **0.847** | General abnormality **0.937** |
| Nebelung et al., Eur Radiol 2025 (Res3DKneeModel) | Internal 3,121 studies, 23 conditions | Medial meniscus tear 0.88 / Lateral 0.84; cartilage pathologies 0.74–0.85 | Effusion **0.91**, Baker's cyst **0.93**, MCL tear **0.95** |
| Astuto et al., Radiol AI 2021 | 1,435 high-res 3D CUBE studies | Meniscus horns **0.93**, cartilage **0.93**, ACL **0.90** | BME (regional marrow signal) **0.83** |

*Sources:* [Bien et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/),
[Nebelung et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746/),
[Astuto et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/).

The direction is not always the same: in MRNet and Res3DKneeModel, regional-fluid
findings are easier; in Astuto's ROI-based pipeline, the focal meniscus and cartilage
classifiers match or exceed ACL/BME. What changes across the rows is the model
architecture and the use of anatomy-aware cropping, not the data.

A radiology reading detail from MRNet is also consistent with the project PCA:
logistic regression over per-plane scores found **axial PD most useful for
abnormalities and meniscal tears**, while **coronal T1 was most useful for ACL
tears** ([Bien et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)).
So the same image set already contains a per-finding "best view" axis.

---

## 2. Architectural / training properties that break the trade

### 2.1 Localization-before-classification — strongest lever for focal findings

Astuto et al. segment the knee into 11 ROIs (cartilage compartments, meniscus horns,
ACL) with a V-Net, crop each ROI, and run a dedicated 3D CNN classifier per ROI.
The result: meniscus AUC **0.93** and cartilage AUC **0.93** on a holdout set
([Astuto et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/)).
They explicitly note that meniscus detection IoU is the weakest part of the
segmentation pipeline (0.49–0.61) because menisci are small, yet classification
performance is high once the crop is focused.

Güngör et al. use a two-stage YOLOv8 → EfficientNetV2 pipeline for meniscus tears:
YOLOv8 localizes the meniscus, then EfficientNetV2 classifies the cropped ROI at
**480 × 480**. With weighted cross-entropy for class imbalance, they report AUC
**0.97 (sagittal)** and **0.98 (coronal)** on a 642-patient test set
([Güngör et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11792105/)).

**Take-away:** focal-structure members can catch up to regional-fluid members if the
model is given an explicit anatomical window.

### 2.2 Slice density / through-plane aggregation — second-strongest lever

The same project data are 3D, so the question is how the slice dimension is used.

* Sun et al. (KneeXNet, 2025) compare a vanilla 2D CNN with a graph CNN that models
  spatial dependencies across slices. On MRNet, the 2D CNN reaches AUC 0.889 for
  meniscus tears; KneeXNet reaches **0.968** — a +0.079 gain, larger than the gains
  for abnormality (+0.062) or ACL (+0.076)
  ([Sun et al. 2025](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full)).
* Li et al. (2023) show that a 3D weighted multi-view network improves ACL detection
  as more views are fused: best single-view AUC **91.78%**, multi-view **92.86%**
  on MRNet-v1.0 ([Li et al. 2023](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)).
* Nebelung et al. add **residual 3D blocks** inside a ResNet18 backbone and feed the
  resulting slice tokens into a transformer. The model scores AUC ≥ 0.85 on 8 of 23
  conditions and ≥ 0.75 on 18, including focal MCL tear (0.95) and meniscus tear
  (0.88 internal) ([Nebelung et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746/)).
* Qiu et al. (CoPAS, Nat Commun 2024) use co-plane attention across MRI sequences to
  classify 12 abnormalities; code is released under **Apache 2.0**
  ([zqiuak/CoPAS](https://github.com/zqiuak/CoPAS)).

**Take-away:** 2D slice-by-slice processing handicaps focal findings more than
regional-fluid findings. Explicit 3D or cross-slice graph/attention aggregation is
where meniscus/ACL/MCL gains are found.

### 2.3 Input resolution — thin direct evidence, but plausible

No knee-MRI paper we found runs a clean in-dataset resolution ablation (e.g. 224 vs
256 vs 384 vs 512). The relevant data points are indirect:

* MRNet and most follow-ups resize to **256 × 256**.
* Güngör et al. use **480 × 480** for meniscus ROI classification and reach AUC
  0.97–0.98, but this is confounded by the YOLOv8 localization stage and
  EfficientNetV2 architecture.
* Astuto et al. acquire **0.5 mm isotropic 3D CUBE** data and resample to 512 × 512,
  which likely helps the small meniscus and cartilage compartments, but again is
  confounded with the ROI pipeline.

**Take-away:** resolution is a candidate, but its independent contribution for focal
vs regional findings in knee MRI is not quantified in the literature. A controlled
ablation is a genuine experiment, not a literature answer.

### 2.4 Objective / loss weighting — helps rare focal labels

* MRNet's original training uses **inverse-class-frequency weighting** per binary
  task ([Bien et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)).
* Azcona et al. (2020) find that a tuned augmentation probability per task/plane
  helps meniscus performance, and also use class reweighting
  ([Azcona et al. 2020](https://arxiv.org/pdf/2010.01947.pdf)).
* Güngör et al. explicitly add **weighted cross-entropy** for meniscus class
  imbalance and reach AUC 0.97–0.98
  ([Güngör et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11792105/)).
* SKID (self-supervised pretraining on MR videos) notes that the MRNet downstream
  task is class-imbalanced and reports that self-supervised features reduce bias
  without explicit reweighting ([SKID, arXiv 2104.10481](https://ar5iv.labs.arxiv.org/html/2104.10481)).

**Take-away:** focal findings (meniscus, Baker's, MCL) are often lower-prevalence.
Reweighting or a margin/AUC objective is a documented way to prevent them from being
swamped by high-prevalence regional-fluid labels.

### 2.5 Pretraining corpus — real but task-dependent

* **ImageNet transfer** is the dominant baseline. Azcona et al. note that even
  though natural images are visually different, ImageNet weights regularize the
  small MRNet dataset and outperform training from scratch
  ([Azcona et al. 2020](https://arxiv.org/pdf/2010.01947.pdf)).
* **RadImageNet** (1.35M radiologist-labeled images) was proposed as a better
  starting point ([Mei et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)),
  but a follow-up study on ACL tear detection could **not** confirm superiority:
  their best ACL model (AUC **0.9969**) was ImageNet-initialized, and DeLong tests
  favored ImageNet significantly. They note RadImageNet's benefit may be limited to
  anatomical regions actually present in its pretraining data
  ([Frees et al. 2024](https://arxiv.org/html/2508.17567v2)).
* **Knee-specific pretraining:** SKID shows that a self-supervised encoder
  pretrained on the larger MRNet corpus transfers better (AUC 0.761) than one
  pretrained on the smaller KneeMRI corpus (AUC 0.741)
  ([SKID](https://ar5iv.labs.arxiv.org/html/2104.10481)).
* **OrthoFoundation-L** is a recent knee-specific foundation model (DINOv3-L,
  trained on 1.25M knee images, MIT license, weights on HuggingFace). It is a
  drop-in trunk candidate for our competition and has been tried in the project
  already. MRI-CORE (Apache 2.0) is a SAM-based general MRI feature extractor.

**Take-away:** pretraining corpus matters, but "medical > natural" is not automatic.
A corpus that matches the anatomy and modality helps; otherwise ImageNet remains
competitive.

---

## 3. What this means for the off-axis member question

The project PCA (Effusion/Synovitis vs Baker's/Medial Meniscus/MCL) is consistent
with published per-finding AUC patterns, but the literature says it is not a hard
trade-off. The most promising levers to test are:

1. **Localization/ROI cropping** for the focal structures (Baker's, meniscus, MCL).
2. **3D / cross-slice / multi-view aggregation** instead of pure 2D slice scoring.
3. **Per-label loss weighting or AUC-margin objectives** so rare focal labels are
   not drowned out by high-prevalence regional-fluid labels.
4. **Resolution ablation** (e.g. 336, 448, 480) specifically on focal-structure
   members, keeping architecture constant.
5. **Pretraining corpus ablation** comparing ImageNet, a general medical FM
   (MRI-CORE/Prism), and a knee-specific FM (OrthoFoundation-L) on the same
   downstream heads and labels.

---

## 4. Licenses / redistribution notes for named checkpoints

| Checkpoint / code | License / availability | Note |
|---|---|---|
| MRNet dataset | Stanford Research Use Agreement | weights not openly redistributed |
| MRNet original code | Supporting Information with paper | check PLOS terms before reuse |
| CoPAS (Qiu et al.) | **Apache 2.0** ([GitHub](https://github.com/zqiuak/CoPAS)) | code only; weights not provided |
| Res3DKneeModel | GitHub available, no license file found at time of search | verify before use |
| KneeXNet (Sun et al.) | Frontiers CC BY article; code availability not confirmed | no public weights found |
| Güngör et al. meniscus model | no public code/weights found | method only |
| Astuto et al. pipeline | no public weights found | method only |
| MRI-CORE | **Apache 2.0** | general MRI feature extractor |
| OrthoFoundation-L | **MIT** (HuggingFace) | knee-specific DINOv3-L, 1.25M images |
| RadImageNet | research weights available; verify non-commercial / redistribution terms | mixed-modality radiology pretraining |

---

## 5. Sources

1. Bien N et al. "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: development and retrospective validation of MRNet." PLOS Medicine 2018. https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/
2. Azcona D et al. "A Comparative Study of Existing and New Deep Learning Methods for Detecting Knee Injuries using the MRNet Dataset." IDSTA / arXiv 2020. https://arxiv.org/pdf/2010.01947.pdf
3. Nebelung S et al. "Comprehensive deep learning-assisted multi-condition analysis of knee MRI studies improves resident radiologist performance." Eur Radiol 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC13035746/
4. Astuto B et al. "Automatic Deep Learning–assisted Detection and Grading of Abnormalities in Knee MRI Studies." Radiol Artif Intell 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/
5. Güngör E et al. "Achieving high accuracy in meniscus tear detection using advanced deep learning models with a relatively small data set." Knee Surg Relat Res 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11792105/
6. Sun J et al. "Leveraging spatial dependencies and multi-scale features for automated knee injury detection on MRI diagnosis." Front Bioeng Biotechnol 2025. https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full
7. Li F et al. "Automated diagnosis of anterior cruciate ligament via a weighted multi-view network." Front Bioeng Biotechnol 2023. https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full
8. Qiu Z et al. "Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities." Nat Commun 2024. https://doi.org/10.1038/s41467-024-51888-4 ; code https://github.com/zqiuak/CoPAS
9. Frees D et al. "Towards Optimal Convolutional Transfer Learning Architectures for Breast Lesion Classification and ACL Tear Detection." arXiv 2024. https://arxiv.org/html/2508.17567v2
10. Bhattacharya S et al. "SKID: Self-Supervised Learning for Knee Injury Diagnosis from MRI Data." arXiv 2021. https://ar5iv.labs.arxiv.org/html/2104.10481
11. Mei X et al. "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning." Radiology 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/
