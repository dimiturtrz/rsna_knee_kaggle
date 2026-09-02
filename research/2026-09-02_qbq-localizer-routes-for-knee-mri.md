# Q-BQ — stage-one knee-MRI localizer routes that need no new annotation

Question from Claude (2026-09-02): what concrete, downloadable weights can serve as a
stage-one localizer for our sagittal/coronal/axial PD/T2 knee MRI, given that the
competition ships only study-level labels and we cannot annotate spatial labels?

## Executive ranking

| rank | route | repo | licence | weights today? | fit to our sequences | recommendation |
|---|---|---|---|---|---|---|
| 1 | **MRAnnotator** (knee) | https://github.com/lzl199704/MRAnnotator | CC-BY-NC 4.0 | ✅ Google Drive `1bqBmvq84ImOZMDtsXWIyMSvE0m-gDzBL` | **sagittal PD, PD FS, T2, T2 FS** — same clinical family | **Act on tonight.** Bone segmentation defines a stable crop box. |
| 2 | **CartiMorph-Toolbox** | https://github.com/YongchengYAO/CartiMorph-Toolbox | CC-BY-NC 4.0 | ✅ checkpoints + OAIZIB-CM data (HF/Zenodo/GDrive) | OAI 3D DESS — domain shift to our 2D TSE | Most anatomically complete, but needs resampling/reorientation. |
| 3 | **SKM-TEA model zoo** | https://github.com/StanfordMIMI/skm-tea | code MIT; weights not explicitly licensed | ✅ HuggingFace `arjundd/skm-tea-models` | **qDESS echoes only** | Poor transfer to PD/T2; useful if we later add qDESS. |
| 4 | **IWOAI 2019 / DOSMA** | https://github.com/denizlab/2019_IWOAI_Challenge, https://github.com/ad12/DOSMA | GNU AGPLv3 / GPL v3 | ✅ cartilage weights exist | OAI DESS; TF 1.13 / IBM Power path | Not recommended — licence and stack are hostile. |
| 5 | attention-rollout / atlas / weak MIL | various | n/a | ❌ no actionable checkpoint | n/a | Thin or no measured evidence for knee MRI stage-one. |

## 1. Pretrained knee segmentation weights

### 1.1 MRAnnotator — the most actionable route

MRAnnotator (Chen et al., *Radiology: Artificial Intelligence* 2025, also arXiv
2403.16613) releases nnU-Net V2 models for whole-body MRI plus a dedicated
**Dataset002 Shoulder/Knee** model.

- **What it outputs:** femur, patella, fibula, tibia masks on sagittal knee MRI.
  Bones-only is enough to define a crop box (e.g. bounding box of all bone masks
  plus margin).
- **Sequences it was trained on:** sagittal PD, PD FS, T2, T2 FS — the same
  Turbo-Spin-Echo family the RSNA competition uses.
- **Weights:** Google Drive file id `1bqBmvq84ImOZMDtsXWIyMSvE0m-gDzBL`, linked
  from the GitHub README.
- **Licence:** CC-BY-NC 4.0. Non-commercial; a Kaggle research/non-profit
  submission is likely permitted, but the Kaggle rules should be checked before
  a final submission.
- **Inference:** standard nnUNetv2_predict one-liner once the checkpoint is
  placed in the nnU-Net results folder.

This is the only route with downloadable weights, a permissive-enough licence,
and training data that overlaps our acquisition family.

### 1.2 CartiMorph-Toolbox — most complete anatomy, but domain shift

CartiMorph-Toolbox (Yao et al., 2024) packages nnU-Net segmentation + VoxelMorph
registration around the **CLAIR-Knee-103R** template.

- **Outputs:** 5-class bone/cartilage mask + 20-region cartilage atlas after
  registration.
- **Training data:** OAI 3D DESS (dual-echo steady-state), which is 3D and has
  different contrast from our 2D TSE PD/T2 FS slices.
- **Weights:** available via HuggingFace, Zenodo and Google Drive from the repo.
- **Licence:** CC-BY-NC 4.0.
- **Caveat:** the whole pipeline is built around reorientation, resampling and
  central cropping to the DESS space. Transfer to our clinical 2D sequences is
  non-trivial and would need a validation pass.

### 1.3 SKM-TEA — strong model zoo, wrong sequence

Stanford SKM-TEA (Desai et al.) ships segmentation baselines on HuggingFace.

- **Outputs:** cartilage/bone/meniscus/pathology segmentation on qDESS.
- **Sequence fit:** qDESS echoes (E1/E2/RSS) only. qDESS is a 3D spoiled
  gradient-echo sequence with very different contrast from PD/T2 TSE.
- **Licence:** code is MIT; HuggingFace weights do not carry an explicit licence
  statement.

Transfer to our sagittal/coronal/axial PD/T2 is expected to be poor.

### 1.4 IWOAI 2019 / DOSMA — not recommended

IWOAI 2019 released pretrained cartilage segmentation weights for OAI DESS.
DOSMA wraps them. Both are GPL-family (AGPLv3 / GPLv3), and the IWOAI code path
is TensorFlow 1.13 with an IBM Power heritage. The licence is less friendly for
a Kaggle kernel and the domain mismatch is the same DESS → TSE problem.

### 1.5 Medical Segmentation Decathlon / nnU-Net knee weights

A generic “knee” task appears in some Decathlon-derived leaderboards, but no
widely-cited, downloadable knee-MRI checkpoint with clear licensing was located
in this search. Most references resolve back to OAI, IWOAI or SKM-TEA data.

## 2. Unsupervised / self-supervised localization

Attention rollout, grad-CAM and similar attribution methods are common in knee
MRI interpretability papers, but none were found that use them as a stage-one
crop mechanism with a reported downstream classification delta. They remain
visualization techniques, not proven localizers.

## 3. Anatomical priors without learning

The closest published resource is the **CLAIR-Knee-103R** template inside
CartiMorph-Toolbox. A pure intensity/geometry rule for finding the joint line
from PD/T2 knee MRI was not found with a reported hit rate. Given MRAnnotator
exists, inventing a rule is not the first move.

## 4. Weak localization from study labels

Multiple-instance learning (MIL) over slices is well studied in medical imaging,
but no top competition solution in the surveyed set used it as stage one, and no
knee-MRI paper was found that reports a measured crop-quality gain from MIL
alone. The 9/9 staged-pipeline signal is therefore not easily reproducible with
our existing labels.

## 5. How much the 9/9 staged-pipeline figure is driven by spatial supervision

The staged pipelines in the surveyed top solutions are partly enabled by the
competition data:

- RSNA 2023 abdominal trauma — shipped **segmentation masks**.
- RSNA 2024 lumbar spine — shipped **label coordinates**.
- RSNA 2022 cervical spine — shipped **segmentation masks + bounding boxes**.

Where spatial supervision was available, stage-one detectors were trained
natively. Where it was not, the top solutions still staged, but usually around a
pretrained anatomy model or a coarse prior. This weakens the raw 9/9 transfer to
our setting, but does not eliminate it: MRAnnotator gives us a pretrained
anatomy model without new labels.

## Recommendation

1. **Tonight:** download MRAnnotator Dataset002, run inference on a few of our
   sagittal studies, and compute the bounding box of {femur, tibia, patella,
   fibula}. Replace the fixed 130 mm centre crop with that box (+ margin) and
   measure the holdout AUC delta on one existing member.
2. **If MRAnnotator fails to transfer:** try CartiMorph-Toolbox as the more
   complete but higher-friction fallback.
3. **File a bead** to track the MRAnnotator experiment and the licence check.

## Sources

- MRAnnotator: https://github.com/lzl199704/MRAnnotator
- CartiMorph-Toolbox: https://github.com/YongchengYAO/CartiMorph-Toolbox
- SKM-TEA: https://github.com/StanfordMIMI/skm-tea
- IWOAI 2019: https://github.com/denizlab/2019_IWOAI_Challenge
- DOSMA: https://github.com/ad12/DOSMA
