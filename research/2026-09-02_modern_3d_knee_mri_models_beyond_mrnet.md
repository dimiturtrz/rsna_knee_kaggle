# Modern 3D knee MRI models beyond MRNet

MRNet (Bien et al., PLOS Medicine 2018) is now seven years old. It uses an ImageNet-pretrained 2D AlexNet to encode each slice and max-pools across slices. Since then, the field has moved toward true 3D encoders, vision transformers, and self-supervised/foundation pretraining. This note surveys what is available and whether any of it is actionable for the RSNA Kaggle pipeline.

## Modern 3D architectures for knee MRI

### MR-Transformer (Zhang et al., 2024/2025)
- **What it is:** A 3D vision transformer adapted from the ImageNet-pretrained DeiT-Ti backbone for total knee replacement (TKR) prediction from OAI/MOST knee MRI.
- **Key idea:** Replicate 2D ImageNet position embeddings across slices, then fine-tune on 3D volumes. This preserves the pretraining while modeling 3D spatial correlation and long-range dependencies.
- **Performance:** AUC 0.89-0.91 on OAI contrasts, 0.82 on MOST contrasts. Statistically better than a 3D CNN (TSE) and a 3D transformer trained from scratch (3DMeT); mostly comparable to MRNet, with a significant win on MOST COR STIR (MR-Transformer 0.82 vs MRNet 0.78, p=0.010).
- **Limitations:** Task-specific to TKR, not multi-label abnormality classification. Quadratic attention cost on 3D volumes is heavy (a 36×512×512 volume becomes ~37k patches). Code and weights are open-source at https://github.com/denizlab/MR-Transformer.
- **Relevance to us:** The architecture is a plausible template for building a 3D member, but it would need retraining on our labels and likely a downsampled volume to fit GPU memory.

### Berrimi et al. (2024) — semi-supervised multi-view 3D CNN for knee OA
- **What it is:** A semi-supervised multi-view MRI network with a 3D CNN for knee osteoarthritis detection.
- **Key idea:** Combine coronal, sagittal, and axial 3D volumes with a 3D CNN and use unlabeled data via a consistency/self-training loss.
- **Relevance to us:** Demonstrates that 3D CNNs with semi-supervised learning can work on knee OA. However, no public weights, and the task is binary OA detection rather than our multi-label setting.

### 3DMeT (Wang et al., MLMI 2021)
- **What it is:** A 3D medical image transformer with a CNN teacher for knee cartilage defect assessment.
- **Performance:** In the MR-Transformer paper, 3DMeT is the weakest baseline (AUC 0.63-0.82 depending on contrast), suggesting that training a 3D transformer from scratch on small knee datasets is brittle.
- **Relevance to us:** A warning that 3D transformers without strong pretraining underperform.

### OrthoDiffusion (Lan et al., 2026) — game changer for 3D
- **What it is:** A unified diffusion-based foundation model for musculoskeletal MRI interpretation. Three orientation-specific 3D diffusion backbones (sagittal, coronal, axial) are pretrained on 15,948 unlabeled knee MRI examinations, then used to extract bottleneck features for multi-label diagnosis and segmentation.
- **Performance:** Macro-average AUROC **0.908** on an 8-class knee abnormality diagnosis task across eight institutions. Strong label efficiency (10% labels retains high precision) and cross-scanner robustness (1.5T vs 3.0T).
- **Weights:** Available from a Kaggle dataset: [OrthoDiffusion Official Knee MRI Foundation](https://www.kaggle.com/datasets/prvsiyan/orthodiffusion-official-knee-mri-foundation). Code is on GitHub: https://github.com/lt-0123/OrthoDiffusion.
- **License:** Verify on Kaggle; the paper does not state a license.
- **Relevance to us:** This is the first public, knee-specific, multi-label-diagnosis 3D foundation model with weights. It is now the primary candidate for a 3D member. The main work is feature extraction (select timestep/block, run U-Net, pool, fuse planes) and adapting the head to our 12 labels.

## 3D foundation / segmentation models that touch knee MRI

### SegmentAnyBone (Gu et al., 2025)
- **What it is:** The first vision foundation model for musculoskeletal MRI, focused on bone segmentation at any body location.
- **Relevance to us:** Segmentation, not classification. Could be useful as an auxiliary task (e.g., bone/joint-region masks as additional inputs), but not a drop-in classifier.

### VISTA3D (CVPR 2025)
- **What it is:** A unified 3D segmentation foundation model that distills 2D pretrained backbones for 3D zero-shot segmentation.
- **Relevance to us:** General 3D segmentation, not knee-specific classification. Could generate anatomical masks for localization, but would need significant adaptation.

### SAM-Med3D (Wang et al., ECCV 2024)
- **What it is:** A general-purpose 3D segmentation model for volumetric medical images.
- **Relevance to us:** Same as above — segmentation backbone, not a classification model. Useful for auxiliary segmentation tasks if we want to add anatomical priors.

### Hoyer et al. (2026, npj Digital Medicine)
- **What it is:** A clinical study using promptable foundation segmenters (SAM, SAM2, MedSAM) fine-tuned on heterogeneous MSK MRI to produce standardized biomarkers (cartilage, bone, soft tissue). Demonstrates a triage cascade and 48-month TKR/OA-incident prediction.
- **Relevance to us:** Shows that foundation-model segmentation can produce clinically useful MSK biomarkers, but the pathway is segmentation → measurements → risk models, not direct classification.

## What about a 3D path for our competition?

### The case for it
- 3D encoders can model spatial continuity across slices, which 2D slice models ignore.
- **OrthoDiffusion** is the first public, knee-specific, large-scale 3D foundation model with weights and a reported multi-label diagnosis score (macro AUROC 0.908 on 8 abnormalities).
- External 3D data (fastMRI knee volumes, MRNet .npy stacks, Rijeka .pck volumes, SKM-TEA DICOMs) could in principle support pretraining or fine-tuning a 3D member.

### The case against it
- **Inference cost.** Diffusion-based feature extraction requires running a 3D U-Net forward at selected timesteps; it is not a single encoder pass.
- **Data format mismatch.** The competition and our current pipeline are slice-oriented. Moving to full 3D volumes requires a new dataloader, resampling, and patch strategy.
- **License/terms.** The Kaggle weights must be checked for competition-use permission.

## Recommendations

1. **OrthoDiffusion is now the primary 3D bet.** Download the Kaggle weights, verify license, and run a smoke experiment as the first 3D member. If it reads positive, it becomes a candidate for the ensemble and may even replace or augment the 2D trunk.

2. **If OrthoDiffusion is blocked, MR-Transformer is the fallback.** It is open-source and ImageNet-pretrained, but task-specific to TKR and uses a tiny DeiT-Ti backbone.

3. **Auxiliary segmentation remains a parallel path.** Foundation segmenters (SAM-Med3D, SegmentAnyBone, VISTA3D) could provide joint-region masks or cartilage/meniscus segmentations as extra input channels to our 2D classifier.

## Update 2026-09-02

A 2026 OAI-based 3D segmentation+classification paper (Moorthy et al., *3D CSFA-UNet*, [PMC12916792](https://pmc.ncbi.nlm.nih.gov/articles/PMC12916792/)) reports 99.15% accuracy on KL grading and strong segmentation Dice on 507 OAI-ZIB DESS volumes. The pipeline is segmentation-first, uses custom attention/feature-selection/spiking-transformer modules, and evaluates on osteoarthritis severity rather than our 12 abnormality labels. No public weights are mentioned. It does not change the conclusion below: there is still no public, knee-specific, multi-label 3D classification foundation model ready to drop into our ensemble.

## Bottom line

OrthoDiffusion changes the 3D calculus: for the first time there is a public, knee-specific, multi-label-diagnosis 3D foundation model with weights. The first action is to smoke-test it. If it transfers to the RSNA 12-label task, it is both a strong member and a benchmark for any 3D architecture we might train ourselves. If it does not transfer or is license-blocked, the 3D lane reverts to a diversification bet rather than a main lever.
