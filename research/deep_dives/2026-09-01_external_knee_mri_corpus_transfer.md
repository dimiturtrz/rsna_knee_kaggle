# External knee-MRI corpus transfer — literature survey

Question: is there published evidence that adding an external knee-MRI corpus to training moves multi-label AUC on a held-out knee-MRI test set?

## TL;DR

Evidence exists, but it is almost entirely **same-label SSL transfer** or **closely related labels**. A clean knee→knee, partial-overlap, multi-label AUC delta is **not published**.

## Papers reviewed

### Zhang et al. 2024 — Self-Supervised Learning for Medical Image Data with Anatomy-Oriented Imaging Planes

* Source: [arXiv:2403.16499](https://arxiv.org/abs/2403.16499)
* External corpus: fastMRI, 10,000 unlabeled knee MRI exams.
* Target: Stanford knee-MRI diagnosis (abnormality, ACL tear, meniscal tear) — same labels as MRNet.
* Transfer form: SSL substrate only (anatomy-plane pretext tasks).
* Numbers (mean AUC on sagittal T2W fat-saturated knee MRI, 800 target subjects):
  * Train from scratch: 0.562
  * ImageNet init: 0.824
  * In-domain SSL (MTL): 0.819
  * In-domain SSL (Rel. loc.): 0.815
  * In-domain SimCLR: 0.800
* Interpretation: large same-label gain over scratch, but the external corpus does not beat a generic ImageNet supervised init; it matches it. The delta that prices the corpus is therefore against scratch, not against a strong init.

### Manna et al. — SKID: Self-Supervised Learning for Knee Injury Diagnosis from MRI Data

* Source: [arXiv:2104.10481](https://arxiv.org/abs/2104.10481)
* Source corpus: MRNet (1,370 clips).
* Target: KneeMRI dataset (917 volumes), ligament-condition labels: healthy / partially injured / completely ruptured.
* Transfer form: SSL substrate only.
* Numbers:
  * MRNet-pretrained encoder → KneeMRI binary classification AUC: 0.761
  * KneeMRI self-pretrained encoder → same task AUC: 0.741
  * Delta: **+0.020 AUC** for using MRNet as external source.
* Caveats: no train-from-scratch baseline; target labels overlap conceptually with MRNet's ACL label but are not identical; binary, not multi-label.

### Takaya et al. 2025 — ModAn-MulSupCon

* Source: [arXiv:2508.18613](https://arxiv.org/abs/2508.18613)
* External corpus: miniRIN, 16,222 mixed-modality/mixed-anatomy images from RadImageNet.
* Target: MRNet-ACL binary classification — same label.
* Transfer form: supervised contrastive pretraining, then fine-tuning.
* Numbers: MRNet-ACL AUC 0.964, beating SimCLR/ImageNet baselines.
* Interpretation: external (non-knee) corpus helps on a knee label, but it is same-label transfer.

### Qiu et al. 2025 — PRISM: Large-scale Multi-sequence Pretraining for Generalizable MRI Analysis

* Source: [arXiv:2508.07165](https://arxiv.org/abs/2508.07165)
* External corpus: 336,476 multi-sequence MRI volumes from 34 datasets, including knee.
* Target: MRNet (abnormality, ACL, meniscal tear) — same labels.
* Transfer form: SSL substrate (multi-sequence MRI foundation model).
* Numbers:
  * MRNet accuracy: 77.78% (Swin-B scratch) → 80.05% (PRISM), +2.27%.
  * Private-Knee_ext accuracy: 69.54% → 71.82%, +2.28%.
* Interpretation: same-label accuracy gains; AUC not reported in the extracted text.

## Summary by transfer form

| form | evidence |
| --- | --- |
| (a) pretrain-then-finetune | ModAn-MulSupCon (non-knee source, same knee label) |
| (b) joint multi-task auxiliary heads | none found |
| (c) pooled training with relabeling | none found |
| (d) SSL substrate only | Zhang et al. fastMRI→MRNet; SKID MRNet→KneeMRI; PRISM multi-sequence→MRNet |

## Honest null

No published study prices a **knee→knee, partial-label-overlap, multi-label AUC delta**. The only cross-target knee result (SKID) is small (+0.020 AUC) and lacks a scratch control. All larger gains are same-label.

Nearest adjacent modality with a measured cross-corpus AUC delta: chest CT → +0.031 (from project Q-I).
