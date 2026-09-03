# ROI-crop gain against strong pretrained trunks in knee MRI — literature answer to Q-CG

**Question (Claude, Q-CG).** Is the ROI/localization gain in knee-MRI classification an
*information* effect (the crop raises the tear's share of the input for any backbone) or a
*compensation* effect (small CNNs need the crop; large pretrained transformers do not)?
Specifically:

1. Is there a knee-MRI ROI ablation where the backbone is a large pretrained transformer or
   foundation model (DINOv2/v3, SAM-derived, MRI-CORE, OrthoFoundation, etc.) and that reports
   whole-image **and** ROI-cropped numbers for the same backbone?
2. Does the ROI gain shrink as the backbone gets stronger?
3. What do Astuto et al. say about why meniscus classification AUC stays high despite poor
   segmentation IoU?
4. Where does the crop enter: input pixels vs. ROI-pooled features from a whole-image forward pass?
5. What are the SKM-TEA licence terms for a classifier trained using SKM-TEA-derived crop windows —
   does the RUA reach the classifier weights if they are re-hosted as a Kaggle dataset?

**Short answer.** No published knee-MRI study compares whole-image vs. ROI-cropped input for
DINOv2/v3, SAM-derived, MRI-CORE, OrthoFoundation-L, or any comparably large pretrained transformer.
The ROI/localization gains that *are* documented are for ResNet/EfficientNet/3D CNN backbones, so
the literature does **not** separate the information and compensation explanations for a ViT-L/14-scale
trunk. Astuto et al.'s low meniscus-IoU/high-AUC pattern suggests the crop is doing something
coarser than exact segmentation, which leans toward information but is still confounded with backbone
capacity. SKM-TEA's Research Use Agreement is non-commercial and prohibits redistribution outside
Stanford's existing process; re-hosting SKM-TEA-derived crop windows or classifier weights on Kaggle
would almost certainly violate it.

---

## 1. Published ROI/localization gains are CNN-era evidence

| Study | Backbone | Localization | Gain reported | Whole-image baseline for same backbone? |
|---|---|---|---|---|
| **Astuto et al., Radiol AI 2021** | 3D CNN classifiers | V-Net segmentation → bounding-box crop of 11 ROIs | Meniscus AUC 0.93, cartilage 0.93 | **No** — entire pipeline is ROI-cropped |
| **Xie et al. / KAMRNet, Lancet eClinicalMedicine 2025** | ResNet-based 2.5D CNN | 3D U-Net meniscus segmentation → 256×256 meniscus-centred crop + attention-guided object localization module (AOLM) | Removing AOLM drops per-abnormality AUC from 0.730–0.966 to 0.687–0.953 | **No** — all runs use the crop |
| **Güngör et al., Knee Surg Relat Res 2024** | EfficientNetV2 | YOLOv8 meniscus localization → 480×480 ROI classification | Meniscus tear AUC 0.97–0.98 | **No** — method is crop-only |
| **Yiu et al., arXiv 2025** | ResNet50, InceptionV3, ViT | ROI *detection* with Grad-CAM/saliency | ResNet50 AUC 0.8184 vs. ViT 0.74 on MRNet sagittal meniscus | **No** — compares architectures, not whole-image vs. crop |

*Sources:* [Astuto et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/),
[Xie et al. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/),
[Güngör et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11792105/),
[Yiu et al. 2025](https://arxiv.org/html/2508.14151v1).

The common thread is that the gain is measured relative to *removing* a localization module or
attention block, not relative to feeding the whole image to the same foundation-model backbone.
These are strong hints that localization helps, but they do not answer Claude's question for a
DINOv2/v3-scale trunk.

---

## 2. Foundation-model backbones: no knee-MRI ROI ablation found

A second, targeted sweep for knee-MROI papers that compare whole-image and cropped input for
DINOv2, DINOv3, SAM/MedSAM/SAM2, MRI-CORE, OrthoFoundation-L, or other vision foundation models
returned **no direct evidence**.

The closest indirect data points are:

* **Müller-Franzes et al. / Medical Slice Transformer, Sci Rep 2025** applies a **frozen/fine-tuned
  DINOv2** to individual MRNet sagittal slices and aggregates slice tokens with a transformer.
  On MRNet meniscus tear detection it reaches AUC **0.85 ± 0.04**, well above a 3D ResNet50 baseline
  (0.69 ± 0.05). Freezing DINOv2 drops knee-MRI AUC to 0.79 (non-significant), while it cripples
  breast MRI and chest CT. The MRNet preprocessing crops each slice to **150 × 150** and resamples
  to 224×224, but the paper does **not** compare this crop against a whole-image DINOv2 forward pass
  ([MST paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)).
* **Yiu et al. 2025** explicitly find that on MRNet-scale data a **CNN transfer-learning backbone
  outperforms a ViT** (ResNet50 AUC 0.8184 vs. ViT 0.74 for sagittal meniscus). Their conclusion is
  that transformers underfit without larger or domain-specific pretraining
  ([Yiu et al. 2025](https://arxiv.org/html/2508.14151v1)). Again, this is an architecture
  comparison, not a crop ablation.
* **OrthoFoundation-L** (DINOv3-L trained on 1.25M knee images, MIT weights) and **MRI-CORE**
  (SAM-based general MRI extractor, Apache 2.0) are actionable drop-in trunks, but no published
  knee-MRI paper ablates whole-image vs. ROI-cropped input for either.

**Take-away:** the experiment Claude wants — whole-image vs. ROI-cropped DINOv2/v3 on the same knee
MRI labels — appears to be **novel** in the published literature.

---

## 3. Does the ROI gain shrink with backbone strength?

The literature cannot answer this directly for knee MRI. In natural-image work, large pretrained
transformers often localize well from a full image and show diminishing returns from tight crops,
but that pattern is task- and pretraining-dependent and should not be assumed to transfer to
small-medical-dataset settings.

The indirect signals are mixed:

* **Pro "compensation":** All published knee-MRI localization gains are for ResNet/EfficientNet/3D
  CNNs, i.e. backbones with limited receptive-field priors and capacity. The fact that Yiu et al.
  still need a CNN to beat a ViT on MRNet suggests the dataset is too small for a transformer to
  learn spatial attention from scratch.
* **Pro "information":** Astuto et al. obtain high meniscus classification AUC (0.93) despite the
  *worst* segmentation IoU in their pipeline (0.49–0.61). If the crop were only useful as a precise
  geometric compensation for a small CNN, worse localization should hurt classification more than it
  does.

Until someone runs the controlled ablation on a DINOv2/v3/OrthoFoundation trunk, both explanations
remain live.

---

## 4. Astuto et al. on low meniscus IoU vs. high meniscus AUC

Astuto et al. report 3D bounding-box IoU for the four meniscus horns of **0.49–0.61**, the lowest of
all 11 ROIs, yet the meniscus classifier reaches AUC **0.93**. Their explanation is explicit and
coarse-grained:

> "For the menisci, however, we saw lower values between 0.49 and 0.61 for the intersection over the
> union, which were due to the fact that the bounding box for the meniscus is much smaller than that
> for other ROIs."

([Astuto et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/), Results, "Segmentation and
Detection of Bone, Cartilage, Meniscus, and Ligament Compartments")

They do not claim the crop is geometrically precise; they attribute the low IoU to the small ROI
size. The high downstream AUC implies that even an imprecise crop is enough to remove most
background and let a dedicated 3D CNN learn meniscus-specific features.

**Where the crop enters:** Astuto's pipeline crops **input pixels**. A V-Net produces 11 ROI
segmentations, a bounding box is computed for each segmented structure, and "the segmented ROIs
within the MRI volumes were detected and cropped into smaller subvolumes ... Cropping enabled
focusing on the ROIs and diminished model size and memory requirements during training. Finally, 17
3D convolutional classifiers took the cropped subvolumes as input."

([Astuto et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/), Materials and Methods,
"Data Processing and DL Classification Pipeline")

There is no whole-image forward pass followed by ROI-pooled features; the crop happens before the
classifier sees the data.

---

## 5. SKM-TEA licence terms and derivative weights

### 5.1 What the dataset terms say

SKM-TEA data is distributed through Stanford AIMI, not under an open data licence. The dataset
repository's `DATASET.md` states that downloading requires account registration and acceptance of a
Research Use Agreement. The MRNet description in Yiu et al. (which also uses Stanford AIMI data)
summarizes the restrictions as:

> "Access to the dataset requires acceptance of a Research Use Agreement prohibiting commercial use,
> redistribution, and re-identification."

([Yiu et al. 2025](https://arxiv.org/html/2508.14151v1), Section 1.6)

Stanford AIMI's own FAQ is equally clear:

> "The data is limited to non-commercial (research and education) use only. The data use agreement
> prohibits distribution outside the existing process."

([Stanford AIMI FAQ](https://stanfordaimi.azurewebsites.net/faq))

A separate AIMI dataset page notes that "Trained models and the de-identified dataset are made
available for non-commercial use under a data use agreement."

([Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/datasets/521d0d11-7c3c-45ef-8b99-a49cd70cba2c))

### 5.2 What this means for the project

* **Crop windows derived from SKM-TEA segmentations/labels** are derivative of the dataset. Re-hosting
  them as a Kaggle dataset would be redistribution and would violate the RUA/DUA.
* **Classifier weights trained using SKM-TEA-derived crop windows** are a murkier case legally, but the
  safest reading is that they are also covered by the non-commercial/research restriction and the
  prohibition on distribution outside Stanford's process. Re-hosting them on Kaggle as a dataset for
  a competition kernel would therefore be risky.
* **Using SKM-TEA only to develop a cropping heuristic offline, then training the classifier on the
  competition's own RSNA data** weakens the link, but if the heuristic or any intermediate
  representation is redistributed, the RUA problem returns.
* **Kaggle competitions are non-commercial research**, so *using* the data/weights internally is likely
  fine; *redistributing* them through a Kaggle dataset is not.

**This is not legal advice.** The actual DUA text is shown only at download time; any plan to ship
SKM-TEA-derived artifacts should be reviewed against that text.

---

## 6. What to do next

The only way to settle Claude's question is to run the missing experiment:

1. **Controlled ROI ablation on the same strong trunk.** Take the project's best DINOv2/v3 or
   OrthoFoundation-L member and train two versions: whole-image input and anatomy-aware cropped
   input. Use the same head, same slice aggregation, and same holdout.
2. **Vary backbone capacity in the same ablation.** Include DINOv2-B/14, DINOv2-L/14, and/or
   OrthoFoundation-L. If the ROI gain shrinks as capacity/pretraining grows, the effect is
   compensation; if it is constant or larger for the big trunk, the effect is information.
3. **Use the project's own RSNA labels or a small held-out radiologist subset** for the crop
   supervisor so that SKM-TEA licence concerns do not arise.

Until that experiment exists, the honest literature answer is: **the evidence is consistent with
both explanations, and the decisive ablation has not been published.**

---

## 7. Sources

1. Astuto B et al. "Automatic Deep Learning-assisted Detection and Grading of Abnormalities in Knee
   MRI Studies." Radiol Artif Intell 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/
2. Xie Z et al. "Development and validation of a deep learning system for comprehensive knee MRI
   abnormality classification." Lancet eClinicalMedicine / PMC12508577, 2025.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/
3. Güngör E et al. "Achieving high accuracy in meniscus tear detection using advanced deep learning
   models with a relatively small data set." Knee Surg Relat Res 2024.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11792105/
4. Yiu J et al. "A Systematic Study of Deep Learning Models and xAI Methods for Region-of-Interest
   Detection in MRI Scans." arXiv:2508.14151, 2025. https://arxiv.org/html/2508.14151v1
5. Müller-Franzes G et al. "Medical slice transformer for improved diagnosis and explainability on
   3D medical images with DINOv2." Sci Rep 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/
6. Stanford MIMI. "Stanford Knee MRI Multi-Task Evaluation (SKM-TEA) Dataset." GitHub DATASET.md.
   https://github.com/StanfordMIMI/skm-tea
7. Stanford AIMI Shared Datasets FAQ. https://stanfordaimi.azurewebsites.net/faq
8. Stanford AIMI dataset page (licence note). https://stanfordaimi.azurewebsites.net/datasets/521d0d11-7c3c-45ef-8b99-a49cd70cba2c
