# fastMRI continuation arithmetic

Question from Claude (2026-09-02 01:25Z): before spending GPU-hours continuing a
DINO-style trunk on our local fastMRI knee slices, what is the published evidence
for (1) small-corpus continued SSL, (2) the minimum viable recipe, (3) the right
k-space-to-image preprocessing, and (4) whether a cheaper/better corpus exists?

## Verdict in one sentence

At the single-coil scale we actually own (~35k slices), there is **no direct
published evidence of a gain over a strong DINOv2/DINOv3 init** and there **is
direct anecdotal evidence that naive continuation can destroy the init**; the
only positive precedents are at 50k–1.25M images and rely on safeguards against
forgetting. fastMRI is worth *having*, but it should not be the first or only
continuation corpus if `ortho_dense` clears.

## How much fastMRI we actually have

The single-coil knee track is much smaller than the number usually quoted:

- **973 train volumes, 34,742 slices**; 199 validation volumes, 7,135 slices
  ([Pyramid Convolutional RNN for MRI Reconstruction](https://arxiv.org/pdf/1912.00543v5),
  [Benchmarking Neural Network Training Algorithms](https://arxiv.org/pdf/2306.07179.pdf)).
- OrthoFoundation reports fastMRI contributed **300,000 knee MRI slices** to its
  1.25M-image corpus ([ytrsk/OrthoFoundation](https://github.com/ytrsk/OrthoFoundation)),
  i.e. they used a superset / multi-sequence view of the data. Our 78 GB tar is
  the single-coil release, so the effective slice count is roughly the 35k figure.

## 1. Does small-corpus continuation beat the base checkpoint or damage it?

**It can do either.** The direction depends on whether the continuation is
protected against catastrophic forgetting.

- **Honest negative:** a DINOv2 maintainer reports that naively continuing
  pretraining on a small domain dataset collapses KNN accuracy from ~80% to
  **<10% after only 20–30k images** ([facebookresearch/dinov2#204](https://github.com/facebookresearch/dinov2/issues/204)).
  This is the exact failure mode Claude is worried about.
- **Positive with safeguards:** Block-Expanded DINORET continues DINOv2 on
  retinal fundus photographs using block expansion and self-supervised DINO
  training, mitigates catastrophic forgetting, and outperforms domain-specific
  RETFound on several tasks ([BE-DINORET](https://arxiv.org/abs/2409.17332)).
  The retinal corpus there is on the order of 75k–160k images (EyePACS ~88k,
  AIROGS/DDR subsets), so the data scale is larger than our 35k slices.
- **Low-data in-domain SSL can beat general pretraining:** Konstantakos et al.
  survey the low-data SSL regime and find that, for domain-specific downstream
  tasks, **in-domain SSL pretraining in the 50k–300k image range can outperform
  large-scale general pretraining** ([Self-supervised visual learning in the low-data regime](https://arxiv.org/abs/2404.17202)).
  That puts the empirical floor for "maybe it pays" at roughly **50k images**,
  above our single-coil count.

**Reading:** 35k single-coil slices sits *below* the smallest published positive
regime and *inside* the published failure regime if done naively. The sign is
negative unless the run is explicitly designed to avoid forgetting.

## 2. Minimum viable recipe and GPU-hours

The nearest public recipe is OrthoFoundation:

- Backbone: **DINOv3-L** (selected over DINOv2-L by transferability estimators).
- Objective: DINO-style student-teacher self-distillation (local + global crops,
  EMA teacher).
- Hardware: **4 × NVIDIA A100**, CUDA 12.5.
- Optimizer: AdamW, cosine schedule, **batch size 32**
  ([ytrsk/OrthoFoundation](https://github.com/ytrsk/OrthoFoundation)).

Scaled to one **RTX 5090 (32 GB)** and 35k slices:

- ViT-L will not run batch-32 on a 32 GB card at 224² unless gradient checkpointing
  or smaller effective batch is used; a realistic local batch is **8–16**, with
  gradient accumulation to a global batch of 32–64.
- 35k slices ≈ 1,100 steps per epoch at batch 32. DINO-style SSL needs many
  epochs; a useful minimum is likely **50–100 epochs** (50k–100k steps).
- Rough estimate: **25–100 GPU-hours for ViT-L**, with ViT-B/S cutting the cost
  by 2–4× but also reducing the chance of closing a gap that a strong init did
  not already close.

**Important:** the original DINOv2 ViT-L was distilled from a ViT-G teacher and
consumed **~8,000 A100-hours** end-to-end ([DINOv2 MODEL_CARD.md](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)).
A 35k-slice continuation is tiny by comparison, but the recipe still needs the
DINO heads, EMA teacher, multi-crop augmentation, and a learning-rate schedule
small enough to not overwrite the init.

## 3. k-space to image: standard and mismatch risk

- The fastMRI single-coil release ships a field
  `reconstruction_rss`: **root-sum-of-squares** reconstruction of the multi-coil
  k-space, cropped to 320×320
  ([fastMRI-Reconstruction README](https://github.com/TianqGuo/FastMRI-Reconstruction/blob/master/fastmri/data/README.md)).
- Plain zero-filled IFFT of under-sampled k-space produces aliased images and is
  *not* the standard input for downstream classification; RSS is.
- **Distribution mismatch remains:** fastMRI RSS images are still coronal
  PD/PD-FS from NYU clinical protocols, pre-processed differently from the RSNA
  DICOMs this project serves. OrthoFoundation used fastMRI as one corpus among
  many and presumably absorbed that mismatch; a model continued *only* on
  fastMRI and then served on RSNA may overfit the fastMRI preprocessing.

## 4. Cheaper / better corpora

| corpus | size | notes |
| :--- | :--- | :--- |
| **fastMRI single-coil (ours)** | ~35k slices / 973 volumes | Already on disk; smallest of the options. |
| **MRNet** | 1,370 exams × 3 planes ≈ 50k–100k slices | Already knee-injury oriented, but smaller than fastMRI+multi-coil and has report labels that may overlap awkwardly with our own label extraction. |
| **SKM-TEA** | 155 patients, ~25k slices, 3D qDESS | Small, but 3D isotropic and includes segmentations; 1.6 TB raw k-space. |
| **OAI** | ~4,796 subjects, >45,000 MRI scans, 160-slice DESS | **Largest public knee MRI corpus** and closest to OA/RSNA on prevalence and age; OAI contributed ~200k MRI slices to OrthoFoundation ([ytrsk/OrthoFoundation](https://github.com/ytrsk/OrthoFoundation)). Requires a data-use agreement and is much larger than the others. |

**Ranking by decision impact:**

1. **(1) Small-corpus sign.** If the answer were "35k is enough", the rest is
   engineering. The literature says it is probably *not* enough without strong
   safeguards, and naive continuation may be worse than the base checkpoint.
2. **(4) Alternative corpora.** OAI dominates on size and clinical match; if a
   continuation is armed, it should start there or combine OAI + fastMRI,
   not fastMRI alone.
3. **(3) Preprocessing.** RSS is the standard, but the mismatch with RSNA DICOMs
   is a real risk and argues against training on fastMRI in isolation.
4. **(2) Recipe / cost.** Feasible on one RTX 5090, but only after (1) and (4)
   say the corpus is worth the hours.

## Implications for `ortho_dense`

`ortho_dense` tests whether DINO-style knee-specific continuation pays at all
(OrthoFoundation-style). If it fails, 35k fastMRI cannot rescue the idea. If it
succeeds, the follow-up should not be "continue on 35k fastMRI" but "assemble the
largest in-domain knee corpus we can access, with fastMRI as one slice source and
OAI as the main volume source, and use the same protected-continuation recipe
OrthoFoundation used."

## Citations

- fastMRI single-coil counts: [Pyramid Convolutional RNN](https://arxiv.org/pdf/1912.00543v5), [Benchmarking NN Training](https://arxiv.org/pdf/2306.07179.pdf)
- OrthoFoundation corpus and recipe: [ytrsk/OrthoFoundation](https://github.com/ytrsk/OrthoFoundation)
- DINOv2 small-corpus forgetting: [facebookresearch/dinov2#204](https://github.com/facebookresearch/dinov2/issues/204)
- BE-DINORET / block expansion: [arXiv:2409.17332](https://arxiv.org/abs/2409.17332)
- Low-data in-domain SSL: [Konstantakos et al., 2025 / arXiv:2404.17202](https://arxiv.org/abs/2404.17202)
- DINOv2 compute budget: [MODEL_CARD.md](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md)
- fastMRI RSS reconstructions: [fastMRI-Reconstruction README](https://github.com/TianqGuo/FastMRI-Reconstruction/blob/master/fastmri/data/README.md)
- OAI scale: [3DReasonKnee / arXiv:2510.20967](https://arxiv.org/html/2510.20967v1)
