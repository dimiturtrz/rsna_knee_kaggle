# Q-DP — BiomedCLIP vision trunk vs RadioDino for knee MRI trunk

**Date:** 2026-09-04  
**Ask:** With one ~40-minute arm slot and a 1.5% absolute bar, choose between BiomedCLIP's vision trunk and RadioDino as a medical trunk for volumetric knee MRI classification. Need published evidence on MRI/musculoskeletal transfer, RadioDino's pretraining mix, any head-to-head, and evidence for small-patch ViT on localised findings.

## Summary

- **BiomedCLIP's vision tower has no published MRI or musculoskeletal evaluation.** Its literature is figure-caption and chest-X-ray.
- **RadioDino (Snarci et al.) is trained on RadImageNet**, a 1.35M-image radiology dataset with CT, MRI, and ultrasound. Study-level counts show MRI is the dominant modality (~70% of studies) and knee is one of the explicitly listed MRI regions.
- **The only published BiomedCLIP-vs-DINO-on-radiology head-to-head is on chest X-ray** (Microsoft RAD-DINO paper), where DINO wins on dense/localised tasks and segmentation.
- **Small-patch ViTs have direct evidence** for fine/localised findings in medical imaging (Dehghan et al. 2025).
- **Recommendation:** RadioDino (RadImageNet-trained) earns the arm; s8 is a defensible second arm on patch-size mechanism.

## Candidate details

### BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)

- Architecture: ViT-B/16, 768-wide, 224 px.
- Pretraining: Contrastive (CLIP-style) on 15M biomedical **figure-caption pairs** from PMC-15M.
- Published radiology evaluation:
  - BiomedCLIP paper: biomedical figure-caption benchmarks, some chest-X-ray tasks via PMC-CLIP-style filtering.
  - *"Interpreting Biomedical VLMs on High-Imbalance Out-of-Distributions: An Insight into BiomedCLIP on Radiology"* ([arXiv:2506.14136v1](https://arxiv.org/html/2506.14136v1)): evaluated on **IU-Xray chest X-rays only**.
- **No published vision-tower evaluation on MRI or musculoskeletal imaging.**

### RadioDino / RAD-DINO — two different models with similar names

| Model | Source | Pretraining corpus | MRI? | Knee? | Published downstream numbers |
|-------|--------|-------------------|------|-------|------------------------------|
| **Microsoft RAD-DINO** | [arXiv:2401.10815](https://arxiv.org/pdf/2401.10815) | Multi-CXR: MIMIC-CXR, CheXpert, PadChest, BRAX, ChestX-ray8, etc. | No | No | VinDr-CXR, CANDID-PTX, RSNA Pneumonia, MIMIC-CXR report generation, segmentation. Beats BiomedCLIP. |
| **Snarci RadioDINO** | [GitHub](https://github.com/Snarci/Radio-DINO), *Computers in Biology and Medicine* 2025 | **RadImageNet** — 1.35M CT/MRI/US images, 11 anatomic regions | Yes, dominant | Explicitly listed | MedMNIST, BUSI, RadImageNet kNN. No knee-MRI downstream number found. |

### RadImageNet corpus

Source: Mei et al., *"RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"* ([PMC9530758](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)).

- 1.35 million annotated CT, MRI, and ultrasound images from 131,872 patients.
- Study-level modality split: 203,341 total studies = **52,691 CT + 142,422 MRI + 8,228 ultrasound** → MRI is **~70% of studies**.
- MRI anatomic regions explicitly include: shoulder, **knee**, ankle, foot, spine, hip, brain.
- The paper demonstrates RadImageNet CNNs outperform ImageNet CNNs on knee MRI ACL/meniscus tasks (+4.8% and +4.5% AUC), confirming radiology pretraining transfers to knee MRI.
- Exact image-level modality proportions are in supplement Table E1; the main text gives the study-level split and the region list.

## Head-to-head evidence

Microsoft RAD-DINO vs BiomedCLIP on chest X-ray (from [arXiv:2401.10815](https://arxiv.org/pdf/2401.10815)):

- **Image classification:** RAD-DINO matches or beats BiomedCLIP on VinDr-CXR, CANDID-PTX, RSNA Pneumonia.
- **Semantic segmentation:** RAD-DINO wins clearly; CLIP-style patch embeddings are noted to be weaker for dense tasks.
- **Report generation (MIMIC-CXR):** RAD-DINO wins on lexical and clinical metrics.

**No published head-to-head on MRI.**

## Small-patch ViT mechanism

Dehghan et al., *"Effect of Patch Size on Fine-Tuning Vision Transformers in Two-Dimensional and Three-Dimensional Medical Image Classification"* ([arXiv:2602.18614v1](https://arxiv.org/html/2602.18614v1)):

- Evaluated patch sizes 1, 2, 4, 14, 28 on 12 medical datasets.
- Smaller patches consistently outperformed larger ones.
- Gains up to +12.78% balanced accuracy (2D) and +23.78% (3D).
- Supports the hypothesis that small patches help fine, localised findings — exactly the project's task.

## Conclusion

For the one medical-trunk arm, **RadioDino (RadImageNet-trained Snarci et al.) is the evidence-backed choice**: its pretraining corpus contains MRI of the knee, and the project needs a different feature family, not a chest-X-ray figure encoder. **RadioDino-s8 is a defensible second arm** because small-patch ViTs have direct medical-imaging evidence for localised findings. BiomedCLIP's vision tower should only be considered if the project runs its own knee-MRI probe, because no published transfer evidence exists.
