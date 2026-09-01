# KMAR-50K as an encoder-transfer asset for RSNA knee MRI classification — research note

> Context: the source letter was not present at the expected inbox paths, so this note answers the three questions stated in the prompt directly.

## 1. Self-supervised / motion-artifact-pair pretraining on unlabelled knee corpus → labelled knee classification

**No published evidence specifically uses motion-artifact pairs for this transfer.** The KMAR-50K paper itself is an artifact-removal benchmark; it does not train or evaluate a classifier.

- **KMAR-50K (Xi et al., *Scientific Data* 2025)** released 1,190 patients, 1,444 paired MRI sequences, 62,506 slices, with paired NIfTI files (corrupted + rescan-ground-truth) and radiologist quality scores. Its downstream evaluation is DDPM / EDSR / U-Net for motion-artifact removal (PSNR/SSIM/NRMSE), not classification [[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12241304/)] [[Mendeley](https://data.mendeley.com/datasets/xw7mrg7ntg)].
- The closest **knee-MRI SSL → classification** results are from two other papers:
  - **SKID (Manna et al., arXiv 2022)** pre-trains a jigsaw-puzzle encoder on the unlabelled MRNet knee-video frames, then freezes the encoder and trains a ConvLSTM classifier. Validation AUCs: abnormality 0.904, ACL 0.893, meniscus 0.810, mean 0.869. The fully supervised MRNet baseline on the same data reaches 0.944 / 0.915 / 0.822, mean 0.894 — so SSL lags supervised by about **0.025 mean AUC** [[arXiv v3 PDF](https://arxiv.org/pdf/2104.10481v3.pdf)].
  - **Zhang et al., *Medical Image Analysis* 2024** pre-train on ~10,000 unlabelled fastMRI knee exams using anatomy-oriented plane-regression pretext tasks, then fine-tune on Stanford knee diagnosis. With 80 labelled subjects, ImageNet pretraining gives mean AUC 0.682 and their best SSL gives 0.755 (+0.073); with 200 subjects, 0.737 → 0.783 (+0.046); with 400 subjects, 0.799 → 0.801 (+0.002); with 800 subjects, 0.824 → 0.819 (−0.005) [[arXiv](https://arxiv.org/abs/2403.16499)] [[ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2403.16499)].

The published lever for knee-MRI SSL over a strong supervised/generic baseline is therefore **not a 1.5%-class gain**. It is roughly **+0.00 to +0.05 AUC**, concentrated in the low-label regime, and it can be negative at full label budget.

## 2. Paired-scan structure: is artifact-pair pretraining a known win over generic SSL?

**No.** Motion-artifact pairs are a restoration asset, not a classification pretraining signal.

- Paired corrupted/clean data are standard for **denoising, super-resolution, and motion-correction** (e.g., fastMRI, MR-ART, KMAR-50K), but the learned objective is image-to-image reconstruction, not pathology discrimination.
- I found no paper showing that artifact-pair pretraining beats SimCLR, DINO, MAE, or BYOL on a downstream medical **classification** task.
- Zhang et al. explicitly show that generic pretext tasks can *hurt* transfer to knee diagnosis: slice-ordering SSL fine-tuned from ImageNet underperforms plain ImageNet pretraining on several data budgets, and MAE/BYOL/SimCLR from ImageNet are markedly worse than the same methods pre-trained in-domain. The transfer benefit comes from **anatomy-aligned pretexts**, not from artifact-pair reconstruction.
- For classification, the artifact-pair objective is likely a distraction: the encoder learns to map motion corruption to a clean template, which is not the same as learning features that separate ACL tear, meniscal tear, and normal studies.

## 3. Honest null: can a 62k-slice domain-specific corpus add value on top of DINOv2/DINOv3?

The literature says the bar is high and the gain is conditional.

- **DINOv2 on radiology:** Baharoon et al. benchmark DINOv2 (142M natural images) across X-ray, CT, and MRI classification and segmentation, and find it outperforms self-supervised and weakly-supervised methods and is competitive with supervised ImageNet-21k models [[arXiv](https://arxiv.org/abs/2312.02366)].
- **DINOv3 on medical vision:** Chen et al. show DINOv3 sets a strong off-the-shelf baseline, beating medical-specific models like BiomedCLIP and CT-CLIP on several tasks, though it fails in deeply specialized domains (histopathology WSIs, electron microscopy, PET) [[arXiv](https://arxiv.org/abs/2509.06467)].
- **Domain-specific pretraining only helps when it closely matches the target modality.** Krones (2026) systematically compared ImageNet → chest X-ray, chest CT, and retina OCT SSL, and found that only pretraining on the same modality improved downstream performance; mismatched medical modalities could underperform ImageNet [[arXiv](https://arxiv.org/abs/2605.08819)].
- **Scale gap matters.** KMAR-50K is ~62,000 slices. DINOv2 saw 142M images; DINOv3 saw ~1.7B. Even the fastMRI experiments above use ~10,000 exams and still only beat ImageNet in the low-label regime.
- A retinal-imaging datapoint supports domain-specific wins in some medical sub-fields: RETFound-DINOv2 (trained on ~1.6M retinal images) outperforms generalist DINOv2 and DINOv3 on retinal tasks [[arXiv](https://arxiv.org/pdf/2509.03421)], but that domain has far more in-domain data and very different image statistics than knee MRI.

**Honest null:** KMAR-50K pretraining might yield a small gain (+0.00 to +0.03 AUC) if our labelled RSNA set is small, but it is unlikely to beat simply fine-tuning DINOv2/DINOv3 harder on our own labelled data. With thousands of labelled studies, the expected effect is probably **±0.005 AUC**, not a 1.5% lever.

## 4. What KMAR-50K actually is

Xi et al. built KMAR-50K for **motion-artifact removal benchmarking**, not for downstream diagnostic pretraining. It provides:
- 1,190 patients, 1,444 paired scans, 62,506 slices;
- Coronal, sagittal, and transverse views;
- PD-TSE, PD-FSE, T1-TSE/T1-FSE, T2-TSE/T2-FSE sequences;
- 1.5T (87%) and 3.0T (13%), almost all Siemens, single-center;
- Preprocessed NIfTI files (N4-corrected, normalized, registered) plus radiologist subjective scores;
- **No finding labels, no classification downstream experiments.**

## Bottom-line recommendation

KMAR-50K is a **restoration dataset**, not an encoder-transfer asset for knee classification. There is no published evidence that motion-artifact-pair pretraining helps knee MRI diagnosis, and the honest null against a DINOv2/DINOv3 backbone is high. If we have limited labels, a cheap pilot (DINOv2 + KMAR-50K SSL pretraining vs. DINOv2 fine-tuned on RSNA) could be informative, but the expected upside is modest and likely negative if it displaces supervised fine-tuning compute. We should prioritise our own labelled RSNA data, architecture, and ensemble design over acquiring this corpus for encoder pretraining.

## Sources

- Xi et al., KMAR-50K, *Scientific Data* 2025 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12241304/
- KMAR-50K Mendeley Data — https://data.mendeley.com/datasets/xw7mrg7ntg
- Manna et al., SKID: Self-Supervised Learning for Knee Injury Diagnosis from MRI Data, arXiv 2022 — https://arxiv.org/abs/2104.10481
- Manna et al., SKID PDF with AUC table (v3) — https://arxiv.org/pdf/2104.10481v3.pdf
- Zhang et al., Self-Supervised Learning for Medical Image Data with Anatomy-Oriented Imaging Planes, *Medical Image Analysis* 2024 — https://arxiv.org/abs/2403.16499
- Zhang et al., ar5iv HTML with Table 5 knee results — https://ar5iv.labs.arxiv.org/html/2403.16499
- Baharoon et al., Evaluating General Purpose Vision Foundation Models for Medical Image Analysis: DINOv2 on Radiology Benchmarks, arXiv 2023/2024 — https://arxiv.org/abs/2312.02366
- Chen et al., Does DINOv3 Set a New Medical Vision Standard?, arXiv 2025/2026 — https://arxiv.org/abs/2509.06467
- Krones, From pre-training to downstream performance: Does domain-specific pre-training make sense?, arXiv 2026 — https://arxiv.org/abs/2605.08819
- RETFound-DINOv2 comparison with DINOv2/DINOv3 — https://arxiv.org/pdf/2509.03421
