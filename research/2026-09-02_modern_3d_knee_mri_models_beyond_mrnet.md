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
- MR-Transformer shows that a 3D ViT with ImageNet pretraining can match or beat MRNet on some knee MRI tasks.
- External 3D data (fastMRI knee volumes, MRNet .npy stacks, Rijeka .pck volumes, SKM-TEA DICOMs) could in principle support pretraining or fine-tuning a 3D member.

### The case against it
- **No public, knee-specific, large-scale 3D foundation model with weights.** The closest is MR-Transformer, which is task-specific to TKR and uses a tiny DeiT-Ti backbone.
- **Compute cost.** Full 3D attention on knee MRI volumes is expensive; MR-Transformer authors explicitly warn about quadratic complexity.
- **Data format mismatch.** The competition and our current pipeline are slice-oriented. Moving to full 3D volumes requires a new dataloader, resampling, and patch strategy.
- **Unclear headroom.** MR-Transformer's gains over MRNet are real but modest (+0.04 AUC on one contrast, parity elsewhere). In our multi-label setting, the marginal gain from a single 3D member may be small compared with the engineering cost.

## Recommendations

1. **Do not pursue a full 3D volumetric model as the main lever.** The strongest actionable model for our pipeline remains OrthoFoundation-L (2D, knee-specific, DINOv3-L, MIT weights). A 3D member is a diversification bet, not a primary path to 1.5%.

2. **If we want a 3D member anyway, the cheapest experiment is:**
   - Start from MR-Transformer (open-source, ImageNet-pretrained DeiT-Ti).
   - Replace the TKR head with our multi-label head.
   - Train on downsampled 3D volumes from our RSNA data or from MRNet/fastMRI.
   - Use it as one seat in the ensemble, expecting modest diversity gain at significant compute cost.

3. **Auxiliary segmentation is more promising than 3D classification.** Foundation segmenters (SAM-Med3D, SegmentAnyBone, VISTA3D) could provide joint-region masks or cartilage/meniscus segmentations as extra input channels to our 2D classifier. This keeps the compute manageable while injecting 3D anatomical structure.

4. **Watch ISMRM 2025 abstract "Self-supervised Pretraining on OAI data for 3D Knee MRI Analysis."** If a pretrained 3D knee encoder emerges from that work, it would change the calculus.

## Bottom line

MRNet has modern successors (MR-Transformer, Berrimi's 3D CNN), but none offer a public, knee-specific, large-scale 3D foundation model ready for our multi-label task. A 3D member could add diversity, but the cost/benefit is worse than continuing to exploit strong 2D backbones (OrthoFoundation-L / DINOv3-L) and external 2D pretraining data. Use 3D as an auxiliary or secondary axis, not the main bet.
