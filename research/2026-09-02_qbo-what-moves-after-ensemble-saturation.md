# What moves a medical-imaging score after ensemble saturation — Q-BO

Context: Claude's Q-BO (`.agent-runtime/inbox/read/2026-09-02T1936Z-claude-what-moves-after-the-ensemble-saturates.md`).
Our position is public 0.944 / rank 33 / leader 0.952; the remaining gap is ~0.008 and the
leaderboard ruler resolves ~0.0015. The seating, member-quality, and stack axes are measured
as closed. The question is what the published record shows actually moved a medical-imaging
leaderboard once member quality and ensembling are both saturated.

The four candidates requested were:

1. Test-time augmentation and multi-view decoding.
2. Label / target engineering.
3. Input geometry (slice count, resolution, crop, band).
4. Something structurally different (segmentation, detection stage, external labelled data).

## Short answer: ranked by measured effect size

| rank | candidate | cleanest published delta | source | provenance |
|---|---|---|---|---|
| 1 | **External labelled data / domain adaptation** (the structurally-different category) | ACL AUC **0.824 → 0.911 (+0.087)** by retraining MRNet on external labelled data; **+3.5%** transfer MRNet → KneeMRI with contrastive domain adaptation | Bien et al. 2018; Liu et al. 2024 | peer-reviewed paper |
| 2 | **Input geometry — ROI crop and slice sampling** | **+0.056** mean AUC from ROI crop (CoPAS); **+0.018** fold-0 OOF from adjacent vs equal-spaced slices | Qiu et al. 2024; competitor forum ablation | paper + forum post |
| 3 | **Label / target engineering** | **+0.04** AUROC for **+0.15** report-labeler F1 (CheXpert); **>+2%** AUC for AUC-margin loss under label noise; mixed-format positional+categorical labels **+AUC/+8% specificity/+4% sensitivity** on knee MRI | Jain et al. 2021; Ihler & Kuhnke 2023; Kornreich et al. 2022 | peer-reviewed paper |
| 4 | **TTA / multi-view decoding** | Multi-view fusion **+1.0%** ACL AUC; per-plane combination **+2–3%** on MRNet; standard TTA can **hurt** over ensembles | Li et al. 2023; Azcona et al. 2020; Medeiros 2026 | paper |

The only structurally-different move with a large *published* number is **external labelled data**.
Segmentation-conditioned and detection-stage pipelines exist, but the knee-MRI papers that use them
either do not report a clean delta against a non-structural baseline or report smaller gains than
the crop/slice axis.

## 1. External labelled data / domain adaptation — largest published move

Bien et al. (*Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development
and retrospective validation of MRNet*, PLOS Medicine 2018) report the cleanest large external-data
number. MRNet trained only on Stanford sagittal T2-weighted series achieved AUC **0.824** on an
external validation set of 183 exams. After fine-tuning on **labelled data from the external site**,
AUC rose to **0.911** — a **+0.087** delta. This is the single largest documented move in the knee-MRI
literature that is not a model-architecture or ensemble change.

Liu et al. (*Domain-Adaptive Framework for ACL Injury Diagnosis Utilizing Contrastive Learning
Techniques*, Electronics 2024) give a cross-dataset transfer-learning ablation: transfer from MRNet
to KneeMRI improved AUC by **3.5%**, and KneeMRI → MRNet improved by **2.5%** (0.845 → 0.870).
The absolute numbers are lower, but the delta is explicitly attributed to domain adaptation rather
than more data.

**Reading for this project.** The repo already has an external-corpus seat axis and it read flat
(one, two, four and six external seats all scored the same). So the *published* effect is large,
but the project's own measurement says the available external corpora do not currently move the
score. The effect in Bien et al. may depend on having high-quality *external labels*, whereas the
repo's external rows are mostly report-derived or single-label.

## 2. Input geometry — the best-documented remaining lever

The largest clean input-side number is the **ROI crop** in CoPAS (Qiu et al., *Learning co-plane
attention across MRI sequences for diagnosing twelve types of knee abnormalities*, Nature
Communications 2024, PMCID PMC11368947). Supplementary Table 10 reports:

| configuration | mean AUC |
|---|---|
| MRNet baseline, ROI-cropped | **0.781** |
| baseline without ROI crop | 0.725 |
| Δ crop | **+0.056** |

This is a single-model delta on the same task and data. The localizer is a coarse meniscus
segmentation used only to find the ROI centre; the crop removes irrelevant background so the same
pixel budget is spent inside the joint.

**Slice sampling** has a smaller but clean number from a competitor forum ablation (topic 737597,
captured in `research/2026-08-26_the-ablation-table-a-competitor-published.md`):

| geometry | their fold-0 OOF |
|---|---|
| 288 px, 140 mm crop, **9 adjacent** slices | **0.7815** |
| 288 px, 140 mm crop, **9 equal-spaced across 24** | 0.7635 |
| Δ adjacent vs equal-spaced | **+0.018** |

**In-plane resolution** has no clean knee-MRI ablation. The available point estimates all suggest
saturation at or below the project's current range: MRNet used 256 × 256 inputs and reached ACL
AUC ~0.97; CoPAS used 224 × 224 × 24; Vuskov et al. (*Comprehensive deep learning-assisted
multi-condition analysis of knee MRI studies*, European Radiology 2026) resizes clinical stacks to
**256 × 256 × 32** and reaches median AUC 0.85 on 23 conditions. A competitor forum post claimed
"slice geometry mattered more than resolution", but its comparison confounded resolution with crop
(192 px no crop vs 288 px with crop), so it cannot be priced as a resolution effect.

**Reading for this project.** The repo already crops to a 130 mm physical FOV, so the ROI-crop delta
is likely already captured. The genuinely unread cell is **slice coverage**: the public frontier
sweep found top notebooks using **64 slices** per study, while repo arms use 12–20 slices because
of VRAM (`research/2026-08-31_public-frontier-sweep.md`). A `stack224` arm was written to test more
slices at lower resolution; this is the only input-side direction the literature points to that has
not been priced here.

## 3. Label / target engineering — real, but smaller than the gap

Jain et al. (*Effect of Radiology Report Labeler Quality on Deep Learning Models for Chest X-Ray
Interpretation*, arXiv:2104.00793, 2021) is the key scaling result. On CheXpert, improving the
report-labeler average F1 from **0.53 to 0.68** translated into a **+0.04 weighted AUROC** gain in
the downstream classifier. This sets a realistic ceiling: even a large labeler improvement moves
the metric by a few points of AUC, not by the full 0.008 gap.

For knee MRI specifically, Astuto et al. (*Automatic deep learning-assisted detection and grading of
abnormalities in knee MRI studies*, Radiology: Artificial Intelligence 2021) reports linear-weighted
Cohen κ between attending radiologists of **0.55–0.72** across tissues. The repo's own consensus
labels already reach ~0.90 macro AUC on the 58 gold studies
(`research/2026-09-02_test_labels_report_derived_or_radiologist_reviewed_qao.md`), so the remaining
label-quality headroom is likely small and hard to measure.

Two additional signals:

- Ihler & Kuhnke (*AUC margin loss for limited, imbalanced and noisy medical image diagnosis — a
case study on CheXpert5000*, Current Directions in Biomedical Engineering 2023) report that an AUC
margin loss beats BCE by **>2% AUC** under label noise in a 5k-image chest-X-ray task. The effect is
modality-adjacent, not knee-specific.
- Kornreich et al. (*Combining mixed-format labels for AI-based pathology detection pipeline in a
large-scale knee MRI study*, MICCAI 2022) augment categorical labels with positional point labels in
a two-stage detect-then-classify pipeline and report an **AUC increase**, plus average **+8%
specificity and +4% sensitivity** over a baseline that discards the positional labels. This is the
cleanest knee-MRI evidence that richer targets can help, but it requires positional annotations that
the competition data do not provide.

**Reading for this project.** The label side is worth continuing (masked BCE on partial MRNet/Rijeka
labels, AUC-margin or agreement-aware smoothing), but the published ceiling suggests it will not
close the 0.008 gap on its own.

## 4. TTA / multi-view decoding — small or negative over an ensemble

Li et al. (*Automated diagnosis of anterior cruciate ligament via a weighted multi-view network*,
Frontiers in Bioengineering and Biotechnology 2023) report a **four-view network** for ACL diagnosis
with AUC **0.9700**, vs the best single view at **0.9600** — a **+1.0%** multi-view gain. Azcona et
al. (*A Comparative Study of Existing and New Deep Learning Methods for Detecting Knee Injuries
using the MRNet Dataset*, arXiv:2010.01947, 2020) found that combining per-plane predictions gave
roughly **+2–3%** over single planes for ACL and meniscus, while early concatenation of all planes
was worse than separate-plane models.

However, **standard test-time augmentation is not a free lunch once an ensemble is already in
place**. Medeiros (*I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical
Image Classification*, arXiv:2604.09697, 2026) evaluated standard TTA on MedMNIST v2 and found that
it usually *hurts*, with accuracy drops up to **31.6 percentage points** and only one modest **+1.6%**
gain. The repo's own `research/2026-09-01_past_ensemble_plateau_mechanisms.md` notes that TTA gains
over an already-diverse ensemble are typically a few parts per thousand, not the half-percent
needed here.

**Reading for this project.** Multi-view decoding is already largely spent: the repo routes each
study into six plane/contrast slots and attends over them with `SlotHead`. More aggressive TTA is
unlikely to rescue a fixed ensemble.

## 5. Structural options outside the four

Two structurally different pipelines are worth flagging, even though their evidence is thinner than
the four above:

- **Segmentation-conditioned classification.** CoPAS uses a learned meniscus segmentation only to
centre the ROI crop (+0.056 AUC). Astuto et al. build a full segmentation → detection →
classification hierarchy for cartilage, meniscus, bone marrow and ACL; they report AUCs of
0.83–0.93 but do not include a clean ablation against an end-to-end model trained on the same
images. Kornreich et al. (MICCAI 2022) do report a gain from adding positional labels, but the gain
is smaller than the CoPAS crop gain and requires extra annotation.
- **Better adaptation of 2D foundation models to 3D/volumetric inputs.** Liu et al. (*Revisiting 2D
Foundation Models for Scalable 3D Medical Image Classification*, arXiv:2512.12887, 2025) show that
replacing naive average pooling of frozen 2D FM slice embeddings with a query-based slice-attention
adapter improves AUC by **~0.11** on a broad 3D medical benchmark. This is not knee-specific, but it
is a reminder that "input geometry" is not only pixel count — it is also how slices are fused. The
repo already uses learned slice aggregation; the remaining question is whether a different
adaptation (e.g. LoRA on the trunk, or a volume-level query) can extract more from the same pixels.

## Take-away for the plateau

The honest ranking, by measured published effect size, is:

1. **External labelled data / domain adaptation** — the only move with a published +0.08-class delta.
   The project's own external-corpus axis is already flat, so this is more of a benchmark fact than
   a live lever.
2. **Input geometry** — ROI crop (+0.056) and slice sampling (+0.018) have clean numbers. The
   project is already cropped; the unread cell is **slice coverage** (more slices at lower
   resolution).
3. **Label / target engineering** — real but bounded; ceiling looks like a few points of AUC unless
   positional or adjudicated labels become available.
4. **TTA / multi-view decoding** — multi-view can give +1–3% on single models, but the repo already
   fuses planes, and standard TTA can hurt over an ensemble.

If the goal is to close 0.008 on the current board, the most plausible published lever that has not
been fully priced here is **reading more slices** (input geometry), not a label rewrite or TTA
campaign.

## Sources

- Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development
  and retrospective validation of MRNet," PLOS Medicine 2018.
  [PMC6258509](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- Liu et al., "Domain-Adaptive Framework for ACL Injury Diagnosis Utilizing Contrastive Learning
  Techniques," Electronics 2024.
  [MDPI](https://www.mdpi.com/2079-9292/13/16/3211)
- Qiu et al., "Learning co-plane attention across MRI sequences for diagnosing twelve types of knee
  abnormalities," Nature Communications 2024.
  [PMC11368947](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)
- Vuskov et al., "Comprehensive deep learning-assisted multi-condition analysis of knee MRI studies
  improves resident radiologist performance," European Radiology 2026.
  [Springer](https://link.springer.com/article/10.1007/s00330-025-12052-8)
- Jain et al., "Effect of Radiology Report Labeler Quality on Deep Learning Models for Chest X-Ray
  Interpretation," arXiv:2104.00793, 2021.
  [arXiv](https://arxiv.org/abs/2104.00793)
- Ihler & Kuhnke, "AUC margin loss for limited, imbalanced and noisy medical image diagnosis — a
  case study on CheXpert5000," Current Directions in Biomedical Engineering 2023.
  [de Gruyter](https://www.degruyter.com/document/doi/10.1515/cdbme-2023-1165/pdf)
- Kornreich et al., "Combining mixed-format labels for AI-based pathology detection pipeline in a
  large-scale knee MRI study," MICCAI 2022.
  [MICCAI](https://conferences.miccai.org/2022/papers/098-Paper0779.html)
- Li et al., "Automated diagnosis of anterior cruciate ligament via a weighted multi-view network,"
  Frontiers in Bioengineering and Biotechnology 2023.
  [Frontiers](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2023.1268543/full)
- Azcona et al., "A Comparative Study of Existing and New Deep Learning Methods for Detecting Knee
  Injuries using the MRNet Dataset," arXiv:2010.01947, 2020.
  [arXiv](https://arxiv.org/pdf/2010.01947)
- Medeiros, "I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts Medical Image
  Classification," arXiv:2604.09697, 2026.
  [arXiv](https://arxiv.org/abs/2604.09697)
- Astuto et al., "Automatic deep learning-assisted detection and grading of abnormalities in knee
  MRI studies," Radiology: Artificial Intelligence 2021.
  [PMC8166108](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/)
- Liu et al., "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification,"
  arXiv:2512.12887, 2025.
  [arXiv](https://arxiv.org/abs/2512.12887)
- Repo prior art:
  `research/2026-09-01_input_side_knee_mri_gains.md`,
  `research/2026-09-01_past_ensemble_plateau_mechanisms.md`,
  `research/2026-09-02_label_noise_pseudo_label_partial_labels_literature.md`,
  `research/2026-08-31_public-frontier-sweep.md`,
  `research/2026-08-26_the-ablation-table-a-competitor-published.md`.
