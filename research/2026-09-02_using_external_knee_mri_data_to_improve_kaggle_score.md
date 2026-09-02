# Using external knee-MRI data to improve the Kaggle score — research deep dive

2026-09-02, Kimi. This note builds on `research/2026-09-02_external_data_format_mismatch_models_training.md` and asks a sharper question: *what is the published evidence that external knee-MRI corpora improve a multi-label AUC score on a target knee-MRI task, and what training mechanics actually deliver the gain?*

## TL;DR

No public corpus matches the RSNA competition's exact 12-label space, so the score cannot come from "more of the same labels." The plausible levers are:

1. **Partial-label multi-task pretraining** on overlapping findings (ACL, meniscus, MCL, effusion).
2. **Pseudo-labeling / self-distillation** from a strong teacher on unlabeled images (KMAR-50K, OAI, SKM-TEA).
3. **Domain-invariant training** to absorb scanner/protocol differences.

The literature strongly supports the first lever, gives conditional support to the second, and warns that the third is only useful when paired with enough data and anti-forgetting safeguards.

**Recommended first experiment:** Scheme B — merged multi-task partial-label pretraining on MRNet + fastMRI+ + Rijeka + RSNA, with masked loss, external oversampling, LP-FT, L2-SP and MixStyle-style augmentation. Expected mean-AUC gain **+0.005 to +0.012**, with ACL/meniscus columns potentially **+0.010 to +0.020**.

## Why external data can move the score

The RSNA model's binding constraint is not training-set size in isolation; it is that every encoder was pretrained on natural images and then asked to generalise to clinical knee MRI. External knee-MRI corpora attack three known failure modes:

- **Domain gap:** natural-image pretraining does not teach scanner/contrast/sequence statistics of knee MRI.
- **Partial supervision:** even when labels do not match exactly, thousands of additional ACL/meniscus/effusion labels tighten the decision boundary for those columns.
- **Unlabeled volume:** large unlabeled corpora (OAI in particular) can update the encoder toward a knee-domain manifold without corrupting the target-task head.

## Mechanism 1: learning from partial / heterogeneous labels across datasets

This is the best-supported path for the project because every external corpus is only partially labeled relative to RSNA.

### Evidence

- **DeepLesion universal lesion detection (Yan et al., 2021).** Four CT datasets with heterogeneous and partial labels were combined in a multi-task detector with missing-annotation mining. The result was a **+49% relative improvement in average sensitivity** over the prior state-of-the-art. The key insight is that missing labels were treated as unknown rather than negative, and cross-dataset knowledge transfer filled some of the gaps. ([arXiv:2009.02577](https://arxiv.org/abs/2009.02577), [PubMed](https://pubmed.ncbi.nlm.nih.gov/33370236/))

- **PSSNet for partially supervised segmentation (Liu et al., 2024).** A single segmentation model was trained on multiple datasets where each dataset only annotated a subset of classes. PSSNet reached **93.9% mean F-score** on a fully-labeled test set, compared with **94.9% for a fully supervised upper bound**, by using adversarial self-training and a cross-task attention module. The gap to fully supervised is small, which is the right mental model for the RSNA partial-label problem. ([Pattern Recognition](https://www.sciencedirect.com/science/article/abs/pii/S003132032400387X))

- **DAMTL — multi-task learning with inconsistent labels (Feng & Chen, 2022).** For tasks whose label sets partially overlap or do not overlap at all, the authors pre-train a large auxiliary network that contains all classes, then use soft masks to extract task-specific sub-networks. DAMTL outperformed Cross-Stitch, NDDR-CNN and MTAL on twelve public datasets. The mapping to knee MRI: **RSNA itself can play the role of the auxiliary big task**, because it contains all twelve labels, while MRNet/fastMRI+/Rijeka are the smaller partially-overlapping tasks. ([arXiv:2201.02305](https://arxiv.org/abs/2201.02305))

- **Heterogeneous Semantic Transfer (Pu et al., 2022 / IJCV 2024).** For multi-label image recognition with partial labels, HST improves mAP by **0.4–3.3 points** over prior art by transferring semantics from known to unknown labels within and across images. This is the formal justification for masking missing RSNA labels while still allowing the model to learn about them from co-occurrence. ([arXiv:2205.11131](https://arxiv.org/abs/2205.11131))

- **CXR-MultiTaskNet (Reddy et al., 2025).** A unified chest-X-ray framework jointly trained for disease localization and classification with shared representations. Multi-task heads on overlapping pathology labels is a recurring successful pattern in medical imaging. ([PMC12399747](https://pmc.ncbi.nlm.nih.gov/articles/PMC12399747/))

### What this means for RSNA

The honest label overlap is concentrated in ACL, meniscus, MCL and effusion. Training shared-trunk + task-specific heads with masked loss is not a hack; it is the standard solution for exactly this problem setting. The ~3,300 external studies (MRNet + fastMRI+ + Rijeka) should act like a large auxiliary task that improves the trunk and the overlapping columns.

## Mechanism 2: domain adaptation / domain-invariant training

All external corpora differ from RSNA in scanner, sequence or protocol. The literature says the gap is real but can be reduced.

### Evidence

- **Domain Adaptation for Medical Image Analysis: A Survey (Guan et al., 2022).** Systematically documents that MRI domain shift caused by different scanners, protocols and cohorts degrades test error proportionally to the distribution gap. The survey categories include supervised, semi-supervised and unsupervised deep adaptation, with feature-alignment and adversarial methods as the most studied. ([PMC9011180](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/))

- **MixStyle / Freq-MixStyle.** The audio project (`D:/work/ai/training/research/data_preparation_best_practices.md`, `domain_generalization.md`) treats Freq-MixStyle as the cheapest, highest-ROI domain-gap tool for spectrograms. The image analogue — mixing instance-normalisation statistics between training samples — has been shown in general computer vision to synthesise novel domains and improve cross-domain generalisation. The knee-MRI application is to mix per-channel statistics between RSNA and external slices during training.

- **Domain randomisation.** `D:/work/ai/training/research/domain_generalization.md` argues for *over-randomising* training conditions beyond realistic ranges so the model cannot memorise source-specific artifacts. For MRI this means aggressive intensity/contrast/noise/affine augmentation that includes but exceeds the expected scanner variation.

- **BatchNorm is a domain-shift liability.** The same audio notes show that BatchNorm stores source-domain running statistics that actively hurt under distribution shift; GroupNorm/LayerNorm avoid this. Most of the project's large-trunk members are ViTs with LayerNorm, which is already the safer choice.

### What this means for RSNA

Do not rely on "the domains are close enough." Explicit domain mixing (MixStyle on activations, strong augmentation, external validation) is part of the recipe, not an afterthought.

## Mechanism 3: continual SSL / adapters / anti-forgetting

Unlabeled corpora (KMAR-50K, fastMRI images, SKM-TEA, OAI) can only help through representation learning. The literature is mixed: it helps at scale, but small corpora can collapse a strong init.

### Evidence

- **Data scaling laws for radiology foundation models (Ilse et al., 2025).** Continual SSL of RAD-DINO on chest X-ray needed roughly **100k in-domain images** to beat the open-weights checkpoint; a CLIP-style model needed ~30k. At 3.5M images, RAD-DINO improved rib-fracture AUPRC by **+6%**. The takeaway: without OAI, the project's ~122k unlabeled slices sit near the lower bound; with OAI's ~200k slices the case is stronger. ([arXiv:2509.12818](https://arxiv.org/abs/2509.12818))

- **MedDINOv3 (Li et al., 2025).** Continuing DINOv3 with DINO-family losses on 3.87M CT slices improved downstream abdominal-organ segmentation by **+1.07% DSC**; adding high-resolution adaptation brought the total gain to **+1.87%**. Objective-matched continuation is the positive precedent. ([arXiv:2509.02379](https://arxiv.org/abs/2509.02379))

- **BE-DINORET (2024).** Block-expansion continued DINOv2 on retinal fundus photographs and outperformed domain-specific RETFound on several tasks. This is the strongest published argument for adapter/block-expansion style continuation on a medical domain. ([arXiv:2409.17332](https://arxiv.org/abs/2409.17332))

- **GLARE (Chhatkuli et al., 2025).** Continual SSL with frozen backbone + lightweight adapters improved semantic segmentation by **+0.4–0.6 mIoU** and consistently helped classification. This matches the adapter-based Scheme A/Scheme C design. ([arXiv:2509.17816](https://arxiv.org/abs/2509.17816))

- **Continual self-supervised learning on chest CT (Tasai et al., 2025).** A three-stage CSSL method with a rehearsal buffer, mixup and feature distillation outperformed simultaneous multi-domain pretraining on COVID-19 CT classification. The lesson for knee MRI: **sequencing domains and preventing forgetting beats dumping all images into one SSL pot**. ([arXiv:2501.04217](https://arxiv.org/abs/2501.04217))

- **DINOv2 maintainer report.** Naive continuation on a small domain dataset collapsed KNN accuracy from ~80% to **<10% after only 20–30k images**. This is the failure mode the project must avoid with adapters, L2-SP and low LR. ([facebookresearch/dinov2#204](https://github.com/facebookresearch/dinov2/issues/204))

- **L2-SP / weight proximity.** `D:/work/ai/training/research/fine_tuning_pretrained_audio.md` recommends `lambda * ||theta - theta_pretrained||^2` as the simplest anti-forgetting regulariser. It is equivalent to a Gaussian prior centred on the pretrained weights and should be combined with adapters/LoRA.

### What this means for RSNA

Pure SSL on KMAR/fastMRI/SKM-TEA is unlikely to beat the DINOv2/DINOv3 init unless it is protected against forgetting. OAI is the only corpus large enough to be a clear positive bet. Sequencing + rehearsal + distillation (as in Tasai et al.) is a better recipe than one-shot mixing.

## Mechanism 4: pseudo-labeling and distillation

For unlabeled corpora, generating soft labels from a strong teacher is often more data-efficient than pure SSL.

### Evidence

- **CLCP-MT (Yang et al., 2025).** A semi-supervised medical classification framework with contrastive learning and class-adaptive pseudo-labeling. With only **20% labeled and 80% unlabeled** data, it achieved **+10.38% F1 on ISIC2018** and **+2.64% AUC on ChestX-ray14** over baselines. Class-adaptive thresholds matter because head/tail classes need different confidence cut-offs. ([PMC12563532](https://pmc.ncbi.nlm.nih.gov/articles/PMC12563532/))

- **Report-labeler quality (Jain et al., 2021).** Better pseudo-labels (from higher-quality report labelers) produce better image-classification models. The analog for knee MRI: the champion ensemble is a higher-quality teacher than a single external model, because it already scores ~0.94 mean AUC on the target distribution. ([arXiv:2104.00793](https://arxiv.org/abs/2104.00793))

- **Audio-project pseudo-labeling.** `D:/work/ai/training/research/distillation_from_pretrained.md` shows that using BEATs to label unlabeled real recordings, then training the small model on mixed synthetic + pseudo-labeled real data, is the practical path when direct distillation fails due to capacity gaps. The same logic applies: run the champion ensemble on KMAR/OAI/SKM-TEA to create soft targets, then train a student on RSNA + pseudo-labeled external images.

### What this means for RSNA

Pseudo-labeling is the honest way KMAR-50K (no labels) and OAI/SKM-TEA images can supervise all twelve RSNA findings. The teacher quality is the project-controlled variable; the champion ensemble is the best available teacher.

## Mechanism 5: direct transfer from existing knee-MRI datasets

There is direct evidence that knee-MRI knowledge transfers across datasets, and that transfer learning dominates training from scratch.

### Evidence

- **MRNet (Bien et al., 2018).** On internal Stanford validation, MRNet achieved AUCs of **0.937** (abnormality), **0.965** (ACL) and **0.847** (meniscus). On the Rijeka external set, a Stanford-only model achieved **0.824 ACL AUC** with no additional training; a model trained on the rest of the Rijeka data reached **0.911**. This demonstrates both cross-dataset transfer and the value of training on the target distribution. ([PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699))

- **OAI knee OA transfer learning (Yeoh et al., 2023).** Using 3D DESS volumes from OAI, transfer learning from ImageNet gave **ResNet18 AUC 0.945** versus **0.763 from scratch**; ResNet34 transfer gave **0.925** versus **0.763**. The gain from a strong pretrained init on a knee-MRI task is large. ([PMC10136763](https://pmc.ncbi.nlm.nih.gov/articles/PMC10136763/))

- **Self-supervised vs supervised transfer (Anton et al., 2022).** ImageNet-pretrained self-supervised models were more generalisable than supervised counterparts across eight medical datasets, scoring up to **10% better**. In-domain SSL outperformed by **>20% on in-domain tasks** but lost accuracy on out-of-domain tasks. The implication: continuation should be domain-matched (knee MRI) and evaluation should include external validation. ([MDPI J. Imaging](https://www.mdpi.com/2313-433X/8/12/320))

- **Supervised transfer learning at scale (Azizi et al., 2021).** On large medical targets, ImageNet-21k supervised pretraining can match or exceed medical-domain pretraining. For smaller targets, in-domain pretraining is more valuable. RSNA is a medium-size target; knee-domain pretraining is therefore plausibly useful. ([arXiv:2101.05913](https://arxiv.org/abs/2101.05913))

### What this means for RSNA

The MRNet and OAI results are not guarantees, but they remove the objection that "knee MRI does not transfer." The project has the right modality and anatomy; the question is execution.

## Volume accounting

| corpus | usable now | slices (estimate) | label overlap with RSNA | licence / access |
| :-- | :-- | :-- | :-- | :-- |
| **RSNA** | 4,407 studies | ~52,872 cells | all 12 findings | competition data |
| **MRNet** | 1,370 exams (1,250 decoded) | ~50k–100k across 3 planes | abnormal, ACL, meniscus | permitted (RUA) |
| **fastMRI+** | 1,172 studies on disk | ~35k train + 7k val (single-coil RSS) | ACL graded, MCL, effusion, meniscus (no med/lat) | images: NYU DUA; annotations: MIT |
| **fastMRI+ (full DICOM)** | 10,012 exams if approved | ~300k slices | same as above | images: NYU DUA |
| **KMAR-50K** | 1,444 scans / 1,190 patients | 62,506 slices | none | CC BY 4.0 |
| **SKM-TEA** | 155 patients | ~25,000 slices | meniscus type, ACL full-thickness, effusion, cartilage | Stanford DUA, NC |
| **OAI** | not yet downloaded | ~200k slices used by OrthoFoundation; >45,000 scans total | MOAKS on ~600 knees: meniscus damage, effusion-synovitis, cartilage, BML; KL grade | NDA DUC, no redistribution, 12-month deletion |
| **Rijeka / KneeMRI** | 736 exams (of 917) | unknown | ACL severity only | mirror licence unverified |

The labelled external overlap pool is roughly **3,300 studies**; the unlabeled on-disk pool is roughly **122,000 slices**.

## Three unified score-improvement schemes

### Scheme B: merged supervised multi-task pretraining on overlapping labels (start here)

**What it is.** Shared trunk + separate heads for RSNA, MRNet, fastMRI+ and Rijeka. Masked binary cross-entropy: only supervise labels that exist for each sample.

**Honest mappings.**
- ACL: MRNet abnormal/ACL → binary; fastMRI+ low-grade → ~0.28, high-grade → ~0.94 under `Targets.pair`; Rijeka severity → existing mapping.
- Meniscus: MRNet only; drop fastMRI+ "Meniscus Tear" because it does not split medial/lateral.
- MCL: fastMRI+ low-mod grade → existing mapping.
- Effusion: fastMRI+ "Joint Effusion" → effusion.
- Contusion: fastMRI+ pools fracture/contusion/dislocation; drop or recover via box location.
- Cartilage / OA / synovitis / Baker's: no honest external source.

**Training mechanics.**
- **LP-FT:** freeze trunk, converge heads, then unfreeze adapters/last blocks with 10-100× lower LR (`D:/work/ai`).
- **Oversample external batches 3-5×** and increase external proportion in the last 2-3 epochs.
- **L2-SP** to keep the trunk near the DINO init.
- **MixStyle-style activation mixing** + strong intensity augmentation.
- **External validation split** from MRNet/fastMRI+/Rijeka for early stopping.
- **Label smoothing 0.05-0.1** for noisy external labels.

**Literature support.** DeepLesion (+49% sensitivity from partial labels), PSSNet (near-fully-supervised F-score), DAMTL (best on inconsistent labels), HST (+0.4–3.3 mAP).

**Expected score impact.** Mean AUC **+0.005 to +0.012**; ACL/meniscus columns **+0.010 to +0.020**. Cost: ~10-30 GPU-hours.

### Scheme A: merged unlabeled SSL continuation

**What it is.** Pool KMAR-50K, fastMRI single-coil, SKM-TEA and eventually OAI images; continue DINOv2/DINOv3 with DINO-family losses, ignoring labels.

**Safeguards.** Adapters/LoRA or block expansion, L2-SP, small LR, layer-wise decay. Sequence domains (Tasai et al.) and use a rehearsal buffer if multiple domains are added incrementally.

**Pseudo-label variant.** Run the champion ensemble on the unlabeled images to generate soft targets, then train a student on RSNA + pseudo-labeled external images. This is supported by CLCP-MT (+10.38% F1 / +2.64% AUC with 80% unlabeled data) and the audio project's BEATs pseudo-labeling recipe.

**Expected score impact.** Without OAI: **+0.002 to +0.005** (below most measured positive thresholds). With OAI: **+0.010 to +0.015**. Cost: ~25-100 GPU-hours (current pool), much larger with OAI.

### Scheme C: two-stage adapter pipeline

**What it is.** Stage 1 = Scheme A (adapter SSL on unlabeled pool). Stage 2 = Scheme B (multi-task heads on labelled corpora), keeping adapters and freezing/low-rating the original trunk.

**Literature support.** BE-DINORET/GLARE (adapter continuation works), Tasai et al. (sequencing domains beats simultaneous mixing), LP-FT (preserves pretrained features).

**Expected score impact.** Highest ceiling, highest variance: **+0.010 to +0.020** mean AUC if OAI is in Stage 1, otherwise lower. Cost: ~35-130 GPU-hours.

## Practical recommendations

1. **Run Scheme B first.** It is the cheapest, best-supported experiment and tests the core hypothesis that partial-label overlap transfers.
2. **Use LP-FT, not full fine-tuning.** Freeze the trunk, converge the heads, then unfreeze adapters/last blocks with very low LR.
3. **Oversample external data 3-5×** so the ~3,300 external studies are not drowned by RSNA.
4. **Hold out an external-only validation split** and early-stop on it.
5. **Use L2-SP + MixStyle + strong augmentation** to protect the init and absorb scanner shift.
6. **If Scheme B is flat, do not spend GPU-hours on Scheme A without OAI.**
7. **If Scheme B is positive, add Scheme A with OAI or the pseudo-label variant.**
8. **Scheme C is the capstone.** Only run it after A and B independently read positive.

## What not to do

- **Do not treat missing labels as negative.** That is the DeepLesion failure mode and the fastest way to manufacture a false gain.
- **Do not full-fine-tune a ViT-L on ~3,300 external studies.** The trainable-parameter budget (~165k) forces adapters/LoRA.
- **Do not blindly map fastMRI+ meniscus or contusion labels** when the anatomy does not split cleanly.
- **Do not rely on SSL without anti-forgetting safeguards.** The DINOv2 maintainer report is the cautionary tale.

## Sources

- Yan et al., "Learning from Multiple Datasets with Heterogeneous and Partial Labels for Universal Lesion Detection in CT", IEEE TMI 2021. ([arXiv](https://arxiv.org/abs/2009.02577), [PubMed](https://pubmed.ncbi.nlm.nih.gov/33370236/))
- Liu et al., "Many birds, one stone: Medical image segmentation with multiple partially labeled datasets", Pattern Recognition 2024. ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S003132032400387X))
- Feng & Chen, "Learning Multi-Tasks with Inconsistent Labels by using Auxiliary Big Task", arXiv 2022. ([arXiv](https://arxiv.org/abs/2201.02305))
- Pu et al., "Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels", IJCV 2024. ([arXiv](https://arxiv.org/abs/2205.11131))
- Reddy et al., "CXR-MultiTaskNet", 2025. ([PMC12399747](https://pmc.ncbi.nlm.nih.gov/articles/PMC12399747/))
- Guan et al., "Domain Adaptation for Medical Image Analysis: A Survey", IEEE TBME 2022. ([PMC9011180](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/))
- Ilse et al., "Data Scaling Laws for Radiology Foundation Models", arXiv 2025. ([arXiv](https://arxiv.org/abs/2509.12818))
- Li et al., "MedDINOv3", arXiv 2025. ([arXiv](https://arxiv.org/abs/2509.02379))
- BE-DINORET, arXiv 2024. ([arXiv](https://arxiv.org/abs/2409.17332))
- Chhatkuli et al., "GLARE", arXiv 2025. ([arXiv](https://arxiv.org/abs/2509.17816))
- Tasai et al., "Continual Self-Supervised Learning Considering Medical Domain Knowledge in Chest CT Images", arXiv 2025. ([arXiv](https://arxiv.org/abs/2501.04217))
- Yang et al., "Unlabeled Insight, Labeled Boost", 2025. ([PMC12563532](https://pmc.ncbi.nlm.nih.gov/articles/PMC12563532/))
- Jain et al., "Effect of Radiology Report Labeler Quality on Deep Learning Models", 2021. ([arXiv](https://arxiv.org/abs/2104.00793))
- Bien et al., "Deep-learning-assisted diagnosis for knee MRI", PLOS Medicine 2018. ([PLOS](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699))
- Yeoh et al., "Transfer learning-assisted 3D deep learning models for knee osteoarthritis detection", 2023. ([PMC10136763](https://pmc.ncbi.nlm.nih.gov/articles/PMC10136763/))
- Anton et al., "How Well Do Self-Supervised Models Transfer to Medical Imaging?", J. Imaging 2022. ([MDPI](https://www.mdpi.com/2313-433X/8/12/320))
- Azizi et al., "Supervised Transfer Learning at Scale for Medical Imaging", arXiv 2021. ([arXiv](https://arxiv.org/abs/2101.05913))
- `D:/work/ai/training/research/data_preparation_best_practices.md` — mixed training, oversampling, late-epoch emphasis, validation design.
- `D:/work/ai/training/research/domain_generalization.md` — domain randomisation, MixStyle, normalisation choices.
- `D:/work/ai/training/research/joint_vs_frozen_training.md` — LP-FT, gradual unfreezing.
- `D:/work/ai/training/research/fine_tuning_pretrained_audio.md` — L2-SP, LoRA, parameter budgets.
- `D:/work/ai/training/research/distillation_from_pretrained.md` — pseudo-labeling with a strong teacher.
