# Fine-tuning schedules for OOD self-supervised ViT transfer (Claude's Q-FT)

Context: Claude observes that `feature_probe` ranks trunks with a frozen backbone, but that ranking inverts after end-to-end fine-tuning at the incumbent backbone rate. He wants published schedules that recover the frozen-probe advantage for OOD-pretrained self-supervised ViTs.

## 1. Layer-wise learning rate decay (LLRD)

LLRD is a standard ingredient in ViT fine-tuning but is not documented as a standalone OOD fix.

- BEiT (Bao et al., 2021): layer decay **0.65** for ViT-B, **0.75** for ViT-L on ImageNet-1K fine-tuning. [BEiT](https://ar5iv.labs.arxiv.org/html/2106.08254)
- iBOT / ConMIM ablations: 0.65–0.85 changes ImageNet accuracy by ~0.3–0.6 pp, so it helps but is not the dominant lever. [iBOT](https://ar5iv.labs.arxiv.org/html/2111.07832)
- BLIP-2: 0.95 for ViT-L, 0.9 for ViT-g when fine-tuning the vision encoder.

For OOD robustness, **Fast Trainable Projection (FTP, Tian et al., NeurIPS 2023)** includes LLRD as a strong baseline and still beats it on OOD. On PASCAL dense-vision tasks with an ImageNet-21K Swin-Tiny backbone, FTP improves over vanilla FT by up to +18–33% on corrupted OOD data. LLRD is competitive but does not close the gap. [FTP](https://proceedings.neurips.cc/paper_files/paper/2023/file/259e59fe23ebd09252647fed42949182-Paper-Conference.pdf)

Conclusion: LLRD belongs in the recipe, but the literature does not show it matters *more* under OOD shift; the bigger lever is how much the backbone is allowed to deviate from initialization.

## 2. Partial unfreeze / staged unfreezing

### LP-FT — linear probing then full fine-tuning

Kumar et al. (ICLR 2022) is the directly relevant method:
1. Train only the linear head on frozen features.
2. Unfreeze everything and fine-tune end-to-end, starting from the probed head.

On 10 distribution-shift datasets, LP-FT gets **+1% ID and +10% OOD vs full fine-tuning**, and **+2% ID and +7% OOD vs linear probing**. Mechanism: a random head forces feature distortion during FT; the probed head reduces feature movement ~20×. [Kumar et al. 2022](https://arxiv.org/abs/2202.10054)

### Surgical fine-tuning

Lee et al. (ICLR 2023) fine-tune only a contiguous block of layers. The best block depends on the shift type:
- **Input-level shifts** (image corruptions, low-level domain gap) → early layers.
- **Feature-level shifts** (subpopulation changes) → middle layers.
- **Output-level shifts** (spurious correlations, label remapping) → last layer / head.

On WILDS Camelyon17 (histopathology scanner shift), earliest embedding layer is best; on FMoW geo-shift, later attention blocks are best. [Surgical FT](https://openreview.net/pdf?id=APuPRxjHvZ)

### ELA-ViT — adaptive freezing by layer instability

Xia & Jiang (NeurIPS 2025) compute a Layer Instability Index from attention concentration, freeze shallow stable layers after a short warm-up, and continue training deeper adaptive layers. Freezing ~50% of ViT-B parameters cuts fine-tuning time ~10% and can improve accuracy (+6.9 pp on CUB-200 with ViT-L). [ELA-ViT](https://papers.neurips.cc/paper_files/paper/2025/file/12678c3948153f4bc391f51e2082bd6e-Paper-Conference.pdf)

## 3. The named result for frozen-probe / fine-tuned ranking inversion

Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution" (ICLR 2022), is the named phenomenon.
- Theory: in an overparameterized linear setting, full FT distorts pretrained features in the ID subspace because the head and feature extractor are updated simultaneously.
- Empirically: on Living-17, FT changes ID features by 0.0188 vs OOD features by 0.0167; LP-FT changes them by 0.0011 and 0.0009 respectively.
- Early stopping does not fix it: FT never passes through parameters that match LP's OOD accuracy during training.

This matches the observed pattern where the probe-best trunk loses after end-to-end fine-tuning.

## 4. DINOv2 published schedule and medical replications

DINOv2 (Oquab et al., 2024) uses "the finetuning pipeline from Touvron et al. (2022) [DeiT III], without tweaking hyper-parameters." The sanity-check result is >+2% on ImageNet-1K over frozen linear probe. There is no separate medical or OOD-aware schedule published. [DINOv2](https://arxiv.org/abs/2304.07193)

Baharoona et al. (2023) evaluated DINOv2 on radiology benchmarks (X-ray, CT, MRI) across kNN, linear probing, end-to-end fine-tuning, LoRA, and BitFit. They found DINOv2 underperforms on kNN due to natural→medical domain shift, but standard end-to-end fine-tuning and LoRA close the gap. They did not report a custom schedule. [DINOv2 radiology study](https://arxiv.org/abs/2312.02366)

## Practical order for our arms

1. **LP-FT** is the cheapest published fix for the probe→fine-tune inversion.
2. If the shift looks low-level, bias updates toward early layers; if semantic, toward later layers.
3. LLRD is worth including but is not the OOD-specific lever.
4. `ortho_slow_trunk` tests one scalar LR; an LP-FT arm tests a better-motivated axis.
