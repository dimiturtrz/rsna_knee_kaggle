# Leveraging external knee-MRI data when the format/labels do not match

The project has access to several external knee-MRI corpora, but none share the exact format or label space of the RSNA competition. The question is: what model architectures and training recipes turn mismatched external data into Kaggle-score improvements?

## The mismatch problem

External corpora differ along at least four axes:

| axis | RSNA | fastMRI+ | KMAR-50K | MRNet | OAI | SKM-TEA |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Labels | 12 multi-label findings | image-read box annotations | none / clean-artifact pairs | ACL/meniscus/abnormality | OA severity, no acute findings | 3D segmentations |
| Reader | report extractor | single radiologist | n/a | report-derived | research-grade | research-grade |
| Acquisition | sagittal/coronal clinical MRI | multi-coil k-space | unknown clinical | sagittal T2, coronal T1, axial PD | 3D DESS | 3D qDESS |
| Scale | 4,407 studies | 1,172 volumes | 1,444 scans | 1,370 exams | ~4,796 subjects | 155 patients |

The right recipe depends on which axes mismatch and how much the external data can influence the target model without corrupting it.

## Strategy map

### 1. Self-supervised pretraining on unlabeled external images

**When to use it:** the external data has no labels, or labels are so different that supervised learning would waste data. The model learns anatomy and acquisition statistics from pixels alone.

**Evidence:**

- VanBerlo et al. 2024 survey: self-supervised pretraining generally improves downstream medical task performance compared to full supervision, especially when labeled data is scarce ([PMC10998380](https://pmc.ncbi.nlm.nih.gov/articles/PMC10998380/)).
- Azizi et al. 2021 ("Supervised Transfer Learning at Scale for Medical Imaging"): larger pretraining scale helps medical→medical transfer; ImageNet-21k pretraining can match or exceed medical-domain pretraining on large targets ([arXiv:2101.05913](https://arxiv.org/abs/2101.05913)).
- Konstantakos et al. 2025: in-domain SSL in the 50k–300k image range can outperform large-scale general pretraining on domain-specific downstream tasks ([arXiv:2404.17202](https://arxiv.org/abs/2404.17202)).

**Fit for this project:**

- **KMAR-50K** is unlabeled; SSL pretraining is its only honest use. However, the project's prior SSL-as-init arms collapsed (`mae_matched` 0.6735, `encoder_adapt_thawed` 0.7162, `encoder_adapt_lower` 0.7309), so SSL pretraining from scratch on this corpus is unlikely to beat a strong DINOv2/DINOv3 init unless it is very large and the recipe avoids catastrophic forgetting.
- **fastMRI+** can be used without its box labels: treat the 1,172 volumes as unlabeled images for SSL pretraining. The scale (~35k slices if single-coil) sits near the lower bound where SSL can help.
- **OAI** is the largest unlabeled option and the closest clinical match; it is the strongest candidate for SSL pretraining.

**Models and training:**

- Use the same SSL objective already in the project (DINO/DINOv3 student-teacher with EMA, multi-crop).
- Continual pretraining from a DINOv2/DINOv3 checkpoint is safer than training from scratch, but needs safeguards against catastrophic forgetting (small LR, layer-wise LR decay, block expansion, or LoRA-style adapters).
- The project already has `ortho_dense` as the test of this idea; if it fails, no smaller corpus can rescue it.

### 2. Continual pretraining with task-specific adapters

**When to use it:** you want to absorb external data into the trunk without overwriting the strong init.

**Evidence:**

- BE-DINORET (arXiv:2409.17332) continues DINOv2 on retinal fundus photographs with block expansion and self-supervised DINO training, outperforming domain-specific RETFound on several tasks. The corpus is 75k–160k images, larger than our single-coil fastMRI slice count.
- DINOv2 maintainer report: naive continuation on a small domain dataset collapses KNN accuracy from ~80% to <10% after 20–30k images ([facebookresearch/dinov2#204](https://github.com/facebookresearch/dinov2/issues/204)).

**Models and training:**

- **Adapter-based fine-tuning**: freeze most of the trunk, train small adapter modules (serial or parallel) on external data. This preserves the strong init and adds domain knowledge with minimal parameters.
- **LoRA / scaled attention adapters**: parameter-efficient, easy to merge or drop at inference.
- **Block expansion**: add new residual/attention blocks and train only them, as in BE-DINORET.

**Fit for this project:** adapters are the safest way to use fastMRI+/KMAR-50K/OAI without the catastrophic-forgetting risk of full continued pretraining. The cost is lower and the risk is bounded.

### 3. Multi-task learning with heterogeneous labels

**When to use it:** external data has labels that partially overlap with the target task (e.g., MRNet has ACL and meniscus labels; fastMRI+ has meniscus and bone-marrow-edema boxes).

**Evidence:**

- Kamnitsas et al., "Learning from Partially Overlapping Labels: Image Segmentation under Annotation Shift" (arXiv:2107.05938): combining semi-supervised learning with an adaptive cross-entropy loss can exploit heterogeneously annotated data and substantially improve segmentation accuracy over baselines.
- "Methods for the frugal labeler: Multi-class semantic segmentation on heterogeneous labels" (PMC8824336): proposes a method to train on datasets with heterogeneous, pixel-wise labels without discarding them.

**Models and training:**

- **Shared trunk, task-specific heads**: a common encoder feeds separate heads for the target labels and each external label set.
- **Adaptive / masked loss**: only backpropagate on the labels that exist for each sample. Missing labels are ignored, not treated as negative.
- **Heterogeneous semantic transfer** (Pu et al. 2022, arXiv:2205.11131): for multi-label partial labels, use intra-image and cross-image semantic correlations to generate pseudo-labels for unknown labels. This can fill in missing labels across datasets.

**Fit for this project:**

- **MRNet** (ACL/meniscus/abnormality) → shared trunk with a head for MRNet labels and the RSNA labels. The ACL and meniscus labels overlap directly.
- **fastMRI+** (box annotations) → object-detection or weakly-supervised classification head. The boxes can be used as weak supervision for the 12 RSNA findings where pathology classes overlap.
- **SKM-TEA** (segmentations) → segmentation head. Segmentations can provide anatomical pretraining but are unlikely to improve classification directly unless the segmentation task forces the encoder to learn relevant anatomy.

### 4. Partial-label learning

**When to use it:** each external sample is only partially annotated (e.g., fastMRI+ labels some pathologies but not all 12 RSNA findings; MRNet labels only three findings).

**Evidence:**

- Pu et al., "Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels" (IJCV 2024 / arXiv:2205.11131): HST framework improves mAP by 0.4–3.3 points over prior art on COCO, Visual Genome, and Pascal VOC by transferring knowledge from known to unknown labels.
- "Learning from Multiple Datasets with Heterogeneous and Partial Labels for Universal Lesion Detection in CT" (DeepLesion): shows how to combine datasets where lesion types are partially labeled.

**Models and training:**

- Treat missing labels as "unknown" rather than negative.
- Use label co-occurrence (e.g., meniscus tear often co-occurs with effusion) to impute missing labels.
- Train with masked binary cross-entropy: mask the loss for labels not present in the sample.

**Fit for this project:** every external corpus is a partial-label dataset relative to RSNA. This is the cleanest formalism.

### 5. Domain adaptation

**When to use it:** the external data comes from a different scanner/protocol (e.g., fastMRI NYU protocol vs RSNA clinical DICOMs; OAI 3D DESS vs RSNA 2D slices).

**Evidence:**

- Guan et al., "Domain Adaptation for Medical Image Analysis: A Survey" (PMC9011180): surveys adversarial, self-supervised, and feature-alignment methods for medical domain adaptation.
- Adversarial domain adaptation can align feature distributions between source and target, but it is finicky and adds training complexity.

**Models and training:**

- **Feature-level alignment**: train a domain discriminator on trunk features; the trunk learns domain-invariant representations.
- **Self-supervised domain adaptation**: use rotation prediction, contrastive learning, or masked autoencoding on both domains with domain-invariant heads.
- **Simplest practical option**: if the domains are close, just mix the external data into pretraining/fine-tuning and rely on data augmentation and batch normalization to absorb the shift.

**Fit for this project:** domain adaptation is most relevant for fastMRI+ and OAI, which have different acquisition protocols. For MRNet and SKM-TEA, the domain gap may be smaller but sample size is smaller.

### 6. Pseudo-labeling / distillation from external models

**When to use it:** you have a model trained on external data and want its predictions to guide the target model.

**Evidence:**

- Jain et al. 2021 ("Effect of Radiology Report Labeler Quality on Deep Learning Models for Chest X-Ray Interpretation"): better report labelers produce better image-classification models when the image model is trained on those labels. The gain comes from higher-quality pseudo-labels.
- Knowledge distillation from a stronger teacher to a student can transfer inductive biases.

**Models and training:**

- Train a teacher on the external corpus (e.g., a detection model on fastMRI+ boxes).
- Use the teacher's predictions on the RSNA training set as soft pseudo-labels or per-cell weights.
- This is the "text as privileged information" mechanism from Q-AC, but with an image teacher instead of a text teacher.

**Fit for this project:** train a fastMRI+ detection/weak-classification teacher and use its RSNA predictions as auxiliary supervision. The circularity risk is lower than text-teacher because the teacher sees images, not reports.

### 7. Data augmentation derived from external data

**When to use it:** the external data captures a nuisance variation (e.g., motion artifacts) that you want the model to be robust to.

**Evidence:**

- Shaw et al. 2020: k-space model of motion artifacts can learn to remove them; the same model can generate realistic artifacts for augmentation.
- DIMA (2025): diffusion model generates realistic motion artifacts on clean images.

**Models and training:**

- Learn an artifact-corruption operator from KMAR-50K's clean-artifact pairs.
- Apply the learned corruption as train-time augmentation.

**Fit for this project:** limited. The hidden test is same-distribution and presumably clean; teaching robustness to artifacts is only useful if the test set contains them. The project's collapsed SSL-as-init arms suggest nuisance pretexts do not help here.

## Ranking by expected impact and risk

| strategy | best corpus | expected gain | risk |
| :-- | :-- | :-- | :-- |
| **SSL pretraining on OAI** | OAI | highest (largest, closest match) | catastrophic forgetting if done naively; use adapters |
| **Multi-task / partial-label with MRNet** | MRNet | medium (direct label overlap) | MRNet is small; license check needed |
| **Adapter-based continual pretraining on fastMRI+** | fastMRI+ | low–medium | domain shift; small scale |
| **Pseudo-labeling from fastMRI+ teacher** | fastMRI+ | low–medium | teacher quality uncertain |
| **Segmentation pretext on SKM-TEA** | SKM-TEA | low | very small dataset |
| **Motion-artifact augmentation from KMAR-50K** | KMAR-50K | negligible unless test is artifact-ridden | no clean-test gain precedent |

## Practical recommendation

1. **Do not chase format-matched labels.** The external corpora will never match RSNA's 12 labels exactly. Use them as unlabeled or partially-labeled pretraining data instead.
2. **Start with adapters.** Freeze the DINOv2/DINOv3 trunk and train small adapter modules on OAI + fastMRI+. This is the lowest-risk way to absorb external domain knowledge.
3. **Add multi-task heads for overlapping labels.** MRNet's ACL/meniscus labels and fastMRI+'s pathology boxes can supervise task-specific heads while a shared trunk learns representations useful for RSNA.
4. **Use masked loss for missing labels.** Every external sample is partially labeled relative to RSNA; do not treat missing labels as negative.
5. **Only use motion-artifact augmentation if the test set is known to contain artifacts.** Same-distribution clean-test robustness is not a score lever.

## Merged training schemes

The external-data programme should not be one recipe per corpus. The corpora are too small for that. Three unified schemes absorb multiple datasets at once and keep the experimental surface small.

### What `D:/work/ai` says about merging heterogeneous data

The audio project has already worked through the same design space (synthetic + real + pretrained backbone, mismatched labels, domain shift). Its research notes give concrete recipes that map directly to the knee-MRI problem:

- **LP-FT is the default for preserving a strong init.** `training/research/fine_tuning_pretrained_audio.md` shows that linear-probing first, then fine-tuning with a 10-100× lower backbone LR, gives ~10% better OOD generalisation than standard fine-tuning because the randomly initialised head no longer distorts pretrained features early in training. For knee MRI this means: freeze the DINOv2/DINOv3 trunk, converge the RSNA head (and any external heads), then unfreeze with a very small LR.

- **Oversample the minority external set.** `training/research/data_preparation_best_practices.md` recommends 3-5× oversampling of real clips when mixing with synthetic, plus late-epoch emphasis (8-10× in the last epochs). RSNA is ~4,400 studies; the honest labelled external pool is ~3,300 studies. Without oversampling the external signal is diluted; with 3-4× sampling each external epoch contributes comparable gradient to RSNA.

- **Domain randomisation beats careful matching.** `training/research/domain_generalization.md` argues for over-randomising training conditions so the model cannot memorise source-specific artifacts. Applied to MRI: randomise intensity scale, contrast, noise, slice thickness, and mild affine deformation beyond the natural range, so scanner/protocol differences become just another variation.

- **MixStyle-style normalisation mixing closes domain gaps.** `training/research/data_preparation_best_practices.md` and `domain_generalization.md` both use Freq-MixStyle (mix per-frequency mean/std between samples) to synthesise novel acoustic domains. The image analogue is **MixStyle on channel/activation statistics** — mix the instance-normalisation statistics of two MRI slices during training so the network sees synthetic "scanner domains" without extra data.

- **Pseudo-labeling unlabeled data with a strong teacher works at small scale.** `training/research/distillation_from_pretrained.md` uses BEATs to label real recordings, then trains the small model on mixed synthetic + pseudo-labeled real data. For knee MRI: run the champion ensemble on KMAR-50K / OAI / SKM-TEA images to generate soft pseudo-labels, then train a student on RSNA + pseudo-labeled external images. This is the only honest way KMAR-50K (no labels) can supervise the 12 findings.

- **L2-SP keeps the backbone near the init.** `training/research/fine_tuning_pretrained_audio.md` recommends `lambda * ||theta - theta_pretrained||^2` as the simplest way to prevent catastrophic forgetting. Combine it with adapters/LoRA so the trunk only moves when external data really forces it.

- **Validation must include the target domain.** `training/research/data_preparation_best_practices.md` replaces pure-synthetic validation with a 70/30 synth/real split and early-stops on the real portion. For knee MRI, hold out a small external-only validation set (e.g., 10-20% of MRNet/fastMRI+) and use it as the early-stopping signal, not just RSNA validation loss.

- **Trainable-parameter budget matters.** `training/research/fine_tuning_pretrained_audio.md` keeps trainable params under ~1:50 vs. samples. With ~3,300 external studies, that is ~165k trainable parameters — far below a full ViT-L. Adapters, LoRA, or partial unfreezing are not optional for the large-trunk members.

These lessons do not replace the medical literature; they give the *training mechanics* that the medical surveys rarely specify.

### Volume accounting

| corpus | usable now | slices (best estimate) | label overlap with RSNA | licence / access |
| :-- | :-- | :-- | :-- | :-- |
| **RSNA** | 4,407 studies | ~52,872 cells (6-cell/study proxy) | all 12 findings | competition data |
| **MRNet** | 1,370 exams (1,250 decoded) | ~50k–100k slices across 3 planes | abnormal, ACL, meniscus | permitted (RUA, non-redistributable) |
| **fastMRI+** | 1,172 studies on disk (single-coil: 973 train + 199 val vols) | ~35k train + 7k val slices (single-coil RSS) | ACL graded, MCL, effusion, meniscus (no med/lat), contusion/fracture pooled | images: NYU DUA; annotations: MIT |
| **fastMRI+ (full DICOM)** | 10,012 exams if DUA approved | ~300k slices (OrthoFoundation used ~300k from fastMRI) | same as above | images: NYU DUA |
| **KMAR-50K** | 1,444 scans / 1,190 patients | 62,506 slices | none | CC BY 4.0 |
| **SKM-TEA** | 155 patients | ~25,000 slices | meniscus type, ACL full-thickness, effusion, cartilage grades | Stanford DUA, NC clause |
| **OAI** | not yet downloaded | ~200k MRI slices used by OrthoFoundation; >45,000 scans in the full cohort | MOAKS on ~600 knees: meniscus damage, effusion-synovitis, cartilage, BML; KL grade | NDA DUC, no redistribution, 12-month deletion |
| **Rijeka / KneeMRI** | 736 exams (of 917 published) | unknown | ACL severity only | mirror licence unverified |

The labelled external pool that honestly overlaps RSNA is roughly **3,300 studies** (MRNet + fastMRI+ + Rijeka). The unlabelled pool already on disk is roughly **122,000 slices** (KMAR + fastMRI single-coil + SKM-TEA). OAI dominates on volume but is not downloaded and carries the heaviest access terms.

### Scheme A: merged unlabeled SSL continuation

**What it is.** Pool every image that does not need a label—KMAR-50K, fastMRI single-coil, SKM-TEA, and eventually OAI and the MRNet images without labels—and continue a DINOv2/DINOv3 trunk with the same DINO-family objective. Labels are ignored.

**Why it is one scheme.** One SSL objective, one dataloader that reads multiple corpora, one continuation run. The differences between corpora become data augmentation.

**Training mechanics from `D:/work/ai`.**
- **LP-FT style two-stage training**: first linear-probe any heads on the frozen trunk, then continue the trunk with a 10-100× lower LR than the head. This prevents the random head from distorting pretrained features early on.
- **Parameter budget**: with ~3,300 effective external studies, keep trainable parameters under ~165k. Use LoRA or serial adapters on the ViT blocks; do not full-fine-tune a ViT-L.
- **Domain randomisation + MixStyle**: randomise intensity, contrast, noise, and mild affine deformation beyond the natural range; mix instance-normalisation statistics between slices to synthesise scanner/protocol domains. The audio project calls the frequency analogue Freq-MixStyle and treats it as the cheapest domain-gap tool.
- **L2-SP regularisation**: add `lambda * ||theta - theta_pretrained||^2` to keep the trunk near the strong init. This is the simplest anti-forgetting safeguard.
- **External validation**: hold out 10-20% of the unlabeled pool (or a small MRNet/fastMRI+ split) and early-stop on that domain, not just on RSNA validation loss.

**Pseudo-label variant.** KMAR-50K has no labels, so SSL is its only honest path. A stronger alternative is to run the champion ensemble on KMAR/OAI/SKM-TEA images to generate soft pseudo-labels, then train a student on RSNA + pseudo-labeled external images. This is the image-domain analogue of `distillation_from_pretrained.md` using BEATs to label real recordings.

**Expected Kaggle impact.** Honest prior, from `research/deep_dives/2026-09-02_fastmri_continuation_arithmetic.md`:
- Without OAI, ~122k slices sits near the low end of the positive regime (50k–300k images). Mean-AUC gain likely **+0.002 to +0.005**.
- With OAI, the merged unlabeled pool approaches the scale where OrthoFoundation/PRISM saw gains (+1–2% on segmentation/accuracy). Mean-AUC gain could reach **+0.010 to +0.015**, but there is no knee-specific measurement.

**Cost.** ~25–100 GPU-hours on a 32 GB card for the current pool; much larger with OAI.

### Scheme B: merged supervised multi-task pretraining on overlapping labels

**What it is.** Train a shared trunk with separate heads for RSNA, MRNet, fastMRI+, and Rijeka. Each sample only supervises the labels that actually exist for that corpus. Missing labels are masked, not treated as negative.

**Why it is one scheme.** The partial-label formalism is the same for every external corpus: every external sample is partially labelled relative to RSNA. A single masked-loss training loop handles MRNet (3 labels), fastMRI+ (graded ACL/MCL/effusion/contusion), and Rijeka (ACL severity).

**Honest mappings.** Only map labels that survive the denominator trap (fastMRI+ ships a "no findings" study list; negatives must come from that list):
- ACL: MRNet abnormal/ACL → binary; fastMRI+ low-grade → ~0.28, high-grade → ~0.94 under `Targets.pair`; Rijeka severity → existing mapping.
- Meniscus: MRNet meniscus tear only; fastMRI+ "Meniscus Tear" does not split medial/lateral, so **drop it** to avoid teaching medial as lateral.
- MCL: fastMRI+ low-mod grade → existing mapping.
- Effusion: fastMRI+ "Joint Effusion" → effusion.
- Contusion: fastMRI+ pools fracture/contusion/dislocation; **drop or recover via box location**, not map blindly.
- Cartilage/OA/synovitis/Baker's: no honest external source.

**Training mechanics from `D:/work/ai`.**
- **LP-FT**: Phase 1 trains only the task heads on a frozen trunk; Phase 2 unfreezes adapters/LoRA and the last few blocks with very low LR.
- **Oversample external data 3-5×** so the ~3,300 external studies are not drowned by 4,407 RSNA studies. In the last 2-3 epochs raise the external proportion further (late-epoch real emphasis).
- **Curriculum external-data late**: start with mostly RSNA, then ramp external data as the heads converge, so the scarce external signal is not wasted on random gradients.
- **MixStyle on activations** and strong intensity augmentation to make scanner differences just another domain.
- **L2-SP + AdamW** with weight decay 1e-2 on heads, 1e-3 on unfrozen trunk.
- **External validation split** from MRNet/fastMRI+/Rijeka used for early stopping; do not rely solely on RSNA validation.
- **Label smoothing 0.05-0.1** because external labels are noisier than RSNA competition labels.

**Expected Kaggle impact.** The labelled external volume is ~3,300 studies. The data-dose curve from the ledger prices a doubling at roughly +0.004–0.010 mean AUC depending on column. Because the overlap is concentrated in ACL/meniscus/effusion/MCL, those columns should move more than the mean. A plausible range is **+0.005 to +0.012 mean AUC**, with ACL/meniscus potentially +0.010–0.020. The honest null is the MRNet-only experiment in `knee-ombx`.

**Cost.** ~10–30 GPU-hours; cheaper than SSL continuation because labels make the signal denser.

### Scheme C: two-stage adapter pipeline

**What it is.** Stage 1: Scheme A (adapter-based SSL continuation on the merged unlabeled pool). Stage 2: Scheme B (multi-task heads on the labelled corpora), keeping the adapters and freezing or low-rating the original trunk.

**Why it is one scheme.** It uses every corpus in a single pipeline: unlabelled data improves domain statistics through adapters; labelled data then teaches the shared heads. The adapter module is the only new architectural object.

**Training mechanics from `D:/work/ai`.**
- Treat the trunk as a pretrained audio backbone: keep it frozen or in eval mode for BN-equivalent layers; let adapters/LoRA carry domain knowledge.
- Stage 1 uses DINO-family SSL on the mixed unlabeled pool with L2-SP and MixStyle.
- Stage 2 is LP-FT: linear-probe the multi-task heads on the frozen Stage-1 trunk, then unfreeze only adapters/last blocks with 10-100× lower LR.
- Use pseudo-labeling in Stage 2 for KMAR-50K / OAI / SKM-TEA so the unlabeled images also contribute to the 12 RSNA columns.

**Expected Kaggle impact.** Highest ceiling, highest variance. If Stage A produces a better knee-domain init without forgetting, Stage B benefits from both more data and a better starting point. Plausible mean-AUC gain **+0.010 to +0.020** if OAI is in Stage A, lower without it. The risk is that either stage collapses and debugging is harder.

**Cost.** ~35–130 GPU-hours.

### Ranking by expected score impact and risk

| scheme | corpora used | expected mean-AUC gain | risk | first experiment |
| :-- | :-- | :-- | :-- | :-- |
| **B** — multi-task overlap | MRNet + fastMRI+ + Rijeka + RSNA | +0.005 to +0.012 | label-mapping/domain-shift errors; denominator trap | **start here** |
| **A** — merged SSL (no OAI) | KMAR + fastMRI + SKM-TEA | +0.002 to +0.005 | catastrophic forgetting; small corpus | after B reads |
| **A** — merged SSL (with OAI) | + OAI | +0.010 to +0.015 | access paperwork; distribution shift | only if B is positive |
| **C** — two-stage adapter | all of the above | +0.010 to +0.020 | compounding failures; most GPU-hours | last |

### Practical recommendation

1. **Run Scheme B first.** It is the cheapest, uses the most honest labels, and tests the core hypothesis: does partial-label overlap transfer to RSNA AUC? MRNet is already permitted and on disk; fastMRI+ annotations are MIT-licensed and already public; Rijeka is on disk. The arm can be built without new access paperwork.
2. **Implement it like the audio LP-FT recipe:** freeze the trunk, converge the heads (Phase 1), then unfreeze adapters/last blocks with 10-100× lower LR (Phase 2). Use L2-SP and MixStyle-style augmentation from the start.
3. **Oversample external batches 3-5× and use late-epoch emphasis**, exactly as the audio project does with real recordings. Otherwise the ~3,300 external studies are diluted by RSNA.
4. **Hold out an external-only validation split** and early-stop on it. RSNA validation alone will hide domain-shift failures.
5. **If Scheme B is flat, do not spend GPU-hours on Scheme A without OAI.** A 122k-slice unlabeled pool is below the threshold where SSL continuation reliably beats a strong init.
6. **If Scheme B is positive, add Scheme A with OAI as the next step.** OAI is the only corpus large enough to move the encoder meaningfully; KMAR/fastMRI/SKM-TEA alone are supporting cast. Consider the pseudo-label variant for KMAR/OAI/SKM-TEA.
7. **Scheme C is the capstone pipeline.** Only arm it after both A and B have independent positive readings.
8. **Use masked loss everywhere.** Every external sample is partially labelled; treating missing labels as negative is the fastest way to manufacture a false gain.
9. **Keep trainable params under ~165k for the external-data stages.** Full ViT-L fine-tuning is not viable on ~3,300 external studies; adapters/LoRA are mandatory.

## Sources

### Local project sources (merging mechanics)

- `D:/work/ai/training/research/data_preparation_best_practices.md` — mixed training, oversampling, late-epoch real emphasis, validation design.
- `D:/work/ai/training/research/domain_generalization.md` — domain randomisation, MixStyle, normalisation choices, regularisation stack.
- `D:/work/ai/training/research/joint_vs_frozen_training.md` — LP-FT, gradual unfreezing, catastrophic forgetting.
- `D:/work/ai/training/research/fine_tuning_pretrained_audio.md` — LP-FT protocol, L2-SP, LoRA, trainable-parameter budget, domain adaptation tricks.
- `D:/work/ai/training/research/distillation_from_pretrained.md` — pseudo-labeling with a strong teacher, logit vs. feature distillation, capacity ratios.
- `D:/work/ai/training/research/self_supervised_small_scale.md` — SSL viability at small scale, VICReg/Barlow Twins, projector overhead.

### Medical literature sources

- Kamnitsas et al., "Learning from Partially Overlapping Labels: Image Segmentation under Annotation Shift", arXiv:2107.05938. ([arXiv](https://arxiv.org/abs/2107.05938))
- Pu et al., "Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels", IJCV 2024 / arXiv:2205.11131. ([arXiv](https://arxiv.org/abs/2205.11131))
- VanBerlo et al., "A survey of the impact of self-supervised pretraining for medical image analysis", 2024. ([PMC10998380](https://pmc.ncbi.nlm.nih.gov/articles/PMC10998380/))
- Azizi et al., "Supervised Transfer Learning at Scale for Medical Imaging", arXiv:2101.05913. ([arXiv](https://arxiv.org/abs/2101.05913))
- Konstantakos et al., "Self-supervised visual learning in the low-data regime", arXiv:2404.17202. ([arXiv](https://arxiv.org/abs/2404.17202))
- BE-DINORET, arXiv:2409.17332. ([arXiv](https://arxiv.org/abs/2409.17332))
- Guan et al., "Domain Adaptation for Medical Image Analysis: A Survey", PMC9011180. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9011180/))
- Jain et al., "Effect of Radiology Report Labeler Quality on Deep Learning Models for Chest X-Ray Interpretation", arXiv:2104.00793. ([arXiv](https://arxiv.org/abs/2104.00793))
- CLIMB: Data Foundations for Large Scale Multimodal Clinical Foundation Models, arXiv:2503.07667v2. ([arXiv](https://arxiv.org/html/2503.07667v2))
- Universal Lesion Detection in CT from multiple datasets with heterogeneous and partial labels, arXiv:2009.02577. ([arXiv](https://arxiv.org/html/2009.02577v3))
