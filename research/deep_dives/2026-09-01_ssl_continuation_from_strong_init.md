# SSL continuation from a strong initialization — literature survey

Question: does continuing self-supervised pretraining of an already-strong general encoder (DINOv2/DINOv3/MAE/CLIP) on an in-domain unlabeled medical corpus beat using the same encoder directly?

## TL;DR

Yes, it can pay, but gains are **modest, scale-dependent, and distribution-sensitive**. All positive evidence is **objective-matched** (DINO-family continued with DINO-family losses, CLIP continued with CLIP/UniCL). No knee-MRI-specific measurement was found.

## Papers reviewed

### Ilse et al. 2025 — Data Scaling Laws for Radiology Foundation Models

* Source: [arXiv:2509.12818](https://arxiv.org/abs/2509.12818)
* Setting: continual pretraining of two open-weights CXR encoders on nested subsets (30k–3.5M) of a single-institution chest X-ray corpus, identical compute and evaluation protocols.
* Models:
  * MI2 (CLIP/UniCL-style, ~500k–3.8M pretraining pairs).
  * RAD-DINO (DINOv2-style, ~840k CXRs).
* Findings (in-domain):
  * MI2 surpasses open-weights with **30k** in-domain samples.
  * RAD-DINO surpasses open-weights with **100k** in-domain samples.
  * At 3.5M, MI2 beats open-weights on all 20 finding tasks; rib-fracture AUPRC improves by **+6%**.
  * At 3.5M, RAD-DINO significantly outperforms open-weights on lines-and-tubes segmentation.
* Findings (out-of-domain VinDR holdout):
  * RAD-DINO shows **reversed scaling**: more in-domain SSL makes it worse than the open-weights checkpoint.
  * MI2 needs **>>100k** samples to beat open-weights on VinDR.
* Interpretation: continuation works on in-domain targets but can hurt under distribution shift / catastrophic forgetting.

### Li et al. 2025 — MedDINOv3

* Source: [arXiv:2509.02379](https://arxiv.org/abs/2509.02379)
* Setting: initialize DINOv3 ViT-B from LVD-1689M, continue SSL on CT-3M (3.87M axial CT slices) with a DINOv2-style objective.
* Numbers on AMOS22 abdominal-organ segmentation (DSC):
  * DINOv3 init, no CT SSL: 85.51%
  * + Stage 1 DINOv2-style SSL on CT-3M: 86.58% (+1.07%)
  * + Stage 2 gram anchoring: 86.54% (no gain)
  * + Stage 3 high-resolution adaptation: 87.38% (total +1.87% over no CT SSL)
* Interpretation: objective-matched continuation on a large CT corpus buys about +1–2 percentage points DSC. The pretext objective stays in the DINO family.

### Chhatkuli et al. 2025 — GLARE

* Source: [arXiv:2509.17816](https://arxiv.org/abs/2509.17816)
* Setting: continual SSL with frozen backbone + lightweight adapters, initialized from DINO or UDI.
* Numbers (semantic segmentation mIoU):
  * ADE20k: UDI baseline → UDI→GLARE, **+0.4 mIoU**.
  * LoveDA (out-of-domain satellite): **+0.6 mIoU**.
* Classification (Derm7pt, COVIDx): DINO→GLARE consistently outperforms DINO alone.
* Interpretation: parameter-efficient continuation helps modestly; small/out-of-domain data can be detrimental.

### Juodelyte et al. 2023 — Revisiting Hidden Representations in Transfer Learning

* Source: [arXiv:2302.08272](https://arxiv.org/abs/2302.08272)
* Setting: supervised ImageNet vs. RadImageNet initialization on seven medical classification tasks.
* Finding: ImageNet-pretrained ResNet50 often outperforms RadImageNet after fine-tuning.
* Interpretation: medical-domain pretraining is not automatic; domain similarity and task matter.

## Corpus-size threshold

| corpus size | what the literature shows |
| --- | --- |
| ~1k exams (fastMRI singlecoil) | below all measured thresholds |
| 30k in-domain images | MI2 (CLIP-style) beats open-weights on in-domain CXR |
| 100k in-domain images | RAD-DINO (DINO-style) beats open-weights on in-domain CXR |
| 1M–3.5M images | reliable gains on classification and segmentation |
| 3.87M CT slices | +1.07% DSC from DINO-family continuation |

fastMRI at ~1,000 exams is two orders of magnitude below the threshold where positive effects are measured.

## Objective matching

All positive results continue the strong init with an objective from the same family:

* RAD-DINO (DINOv2) → DINOv2 losses.
* MI2 (CLIP/UniCL) → UniCL.
* MedDINOv3 (DINOv3) → DINOv2-style self-distillation.
* GLARE (DINO) → DINO-like global consistency.

No study was found that continues DINOv2/DINOv3 under a *different* SSL objective (MAE, SimCLR, BYOL) and reports a downstream gain.

## Honest negatives

* **Out-of-domain CXR (VinDR):** RAD-DINO continual SSL underperforms the open-weights checkpoint; MI2 needs >>100k samples.
* **Small / out-of-domain data in GLARE:** continuation can be detrimental.
* **ImageNet vs. RadImageNet:** supervised medical-domain pretraining does not always beat generic pretraining.

## Conclusion for knee MRI

There is **no published knee-MRI continuation experiment**. The nearest positives (CXR +6% AUPRC at 3.5M, CT +1.07% DSC at 3.87M slices) require corpus sizes far larger than fastMRI singlecoil. The honest prior is that continuation of DINOv2/DINOv3 on ~1,000 unlabeled knee-MRI exams is unlikely to move the metric, and a cross-objective continuation is completely unproven.
