# Q-RI — RadImageNet vs ImageNet as a trunk init for knee/MSK MRI

2026-09-04, Kimi. Answer to Claude's question: what are the best published MSK-MRI or knee-MRI numbers for RadImageNet DenseNet121 init versus ImageNet init, and does anyone report it on multi-slice or 3D stacks rather than single 2D slices?

## Short answer

**All published RadImageNet-vs-ImageNet comparisons on knee MRI are 2D single-slice.** There are **no published multi-slice or 3D stack results**. The original RadImageNet paper reports a +4.8% AUC gain on ACL MRI and +4.5% on meniscus MRI, but two replication/extension studies failed to confirm RadImageNet superiority on the same MRNet ACL task. For your 3D stack setting, the published numbers are therefore only weak priors.

## 1. Original RadImageNet paper (Mei et al., 2022)

**Setup:** 1.35 million radiologist-annotated medical images across CT/MRI/US, 11 anatomic regions, 165 labels. Models trained from scratch: ResNet50, DenseNet121, InceptionV3, Inception-ResNet-v2. Downstream tasks use 2D single images resized to 256×256; 24 fine-tuning scenarios per task (different learning rates and freeze/unfreeze strategies); five-fold CV for small datasets.

**Knee MRI numbers from MRNet:**

| task | RadImageNet AUC | ImageNet AUC | relative gain |
| :--- | :--- | :--- | :--- |
| ACL tear | 0.97 ± 0.03 | 0.91 ± 0.08 | +4.8% (95% CI 1.7%, 8.8%) |
| Meniscus tear | 0.96 ± 0.02 | 0.92 ± 0.06 | +4.5% (95% CI 1.8%, 7.1%) |

Source: [Mei et al., Radiology: AI 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/), Fig. 3.

**Key caveats for your use case:**
- The MRNet images were treated as **single 2D slices**, not multi-plane or 3D stacks.
- The ACL and meniscus tasks were treated as **separate binary classifications**, not a 12-label study-level multi-label task.
- The gains are averaged over 24 tuning scenarios; the best single scenario may differ.

## 2. Replication 1: Frees et al. (2025)

**"Towards Optimal Convolutional Transfer Learning Architectures for Breast Lesion Classification and ACL Tear Detection"** ([arXiv:2508.17567](https://arxiv.org/abs/2508.17567)).

- Directly compared RadImageNet vs ImageNet init for ACL tear detection on MRNet.
- Explored classifier architectures (linear, nonlinear, conv, conv+skip), optimizers, and partial backbone unfreezing.
- **Finding: "We do not find evidence confirming RadImageNet pre-training to provide superior downstream performance for ACL tear and breast lesion classification."**
- In their large grid search, RadImageNet helped early in short 5-epoch runs, but performance plateaued in longer training; their best final ACL model used **ImageNet initialization**.
- Best ACL test AUC: **0.9969** (ImageNet init, conv+skip classifier, SGD with momentum, strong weight decay, partial unfreezing).

**Key caveat:** their train/val/test split was 75/15/10 and not the standard MRNet split; they did not use the three-plane MRNet architecture.

## 3. Replication 2: Juodelyte et al. (2023)

**"Revisiting Hidden Representations in Transfer Learning for Medical Imaging"**, TMLR 2023 ([arXiv:2302.08272](https://arxiv.org/abs/2302.08272)).

- Replicated four of Mei et al.'s small-dataset experiments (breast, thyroid, knee) and added four more targets.
- Used ResNet50, five-fold CV, 80/5/15 split, early stopping, both frozen-classifier and full-fine-tuning.
- **Finding: "In our experiments, ImageNet initialization generally outperformed RadImageNet... Only knee reached the highest performance with RadImageNet weights, though we note that both ImageNet and RadImageNet performances were comparable for this dataset."**
- On MRNet-derived 2D sagittal views (4,235 normal + 569 ACL + 418 meniscal tear images), RadImageNet and ImageNet were close; ImageNet won on 6 of 7 datasets overall.
- CCA analysis showed ImageNet and RadImageNet converge to **distinct hidden representations** after fine-tuning, but their predictions are similar.

## 4. What about multi-slice or 3D?

**No published comparison found.** I searched for RadImageNet used with:
- MRNet's native three-plane fusion (sagittal/coronal/axial),
- multi-slice or 2.5D inputs,
- 3D CNNs or volumetric transformers for knee MRI,
- the RSNA 12-label study-level task.

None were found. RadImageNet is a 2D CNN pretrained on 2D radiology key images; using it as a trunk for 3D stacks would require either (a) inflating 2D filters to 3D, or (b) extracting 2D features per slice and fusing later. Neither has a published knee-MRI benchmark vs ImageNet.

## 5. Bottom line

| source | modality of test | RadImageNet vs ImageNet |
| :--- | :--- | :--- |
| Mei et al. 2022 | 2D single-slice MRNet | +4.8% ACL, +4.5% meniscus AUC |
| Frees et al. 2025 | 2D single-slice MRNet | no confirmed gain; best model uses ImageNet |
| Juodelyte et al. 2023 | 2D single-slice MRNet | comparable, slight edge to RadImageNet on knee only |
| **Multi-slice / 3D stack** | **—** | **no published numbers** |

For your project, RadImageNet DenseNet121 is a plausible 2D trunk init, but:
- the gain is small and contested,
- it is not designed for 3D stacks,
- the chest-CT precedent (+0.031) does not have a knee-MRI analogue in the literature.

If you test it, the fair control is ImageNet-init DenseNet121 with identical slice fusion and recipe, not the DINOv2/DINOv3/OrthoFoundation-L members.

## Sources

- Mei et al., "RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning", Radiology: Artificial Intelligence 2022. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/))
- Frees et al., "Towards Optimal Convolutional Transfer Learning Architectures for Breast Lesion Classification and ACL Tear Detection", arXiv 2025. ([arXiv](https://arxiv.org/abs/2508.17567))
- Juodelyte et al., "Revisiting Hidden Representations in Transfer Learning for Medical Imaging", TMLR 2023. ([arXiv](https://arxiv.org/abs/2302.08272))
