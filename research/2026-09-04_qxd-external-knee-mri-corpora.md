# Q-XD — Public knee-MRI corpora with findings-level labels for co-training / pretraining

2026-09-04, Kimi. Ranked survey of public knee-MRI datasets that carry findings-level or abnormality-level labels and could plausibly be used for co-training or pretraining toward the RSNA 12-label study-level multi-label task.

## TL;DR

No public corpus matches the RSNA label space exactly, and **no published paper directly measures transfer from any of these corpora to an RSNA-style 12-label study-level task**. The practical ranking is therefore driven by label overlap, licence cleanliness, and volume:

| rank | corpus | labelled cases | useful labels for RSNA | licence | Kaggle-OK? | rationale |
| :--: | :--- | ---: | :--- | :--- | :--- | :--- |
| 1 | **MRNet** | 1,370 exams (~1,250 decoded) | abnormal, ACL tear, meniscus tear | CC-BY (per `tensorflow/datasets` issue) | ✅ yes | cleanest licence; direct ACL/meniscus overlap; public train/val/test splits |
| 2 | **fastMRI+** | 1,172 studies reviewed, 974 with boxes | ACL high/low grade, MCL sprain, effusion, any meniscus tear, cartilage loss, bone edema/contusion | images: NYU custom DUA; annotations: MIT | ⚠️ probably yes, read DUA | richest public pathology vocabulary; bounding-box + 13 study-level labels; large unlabelled image pool behind same agreement |
| 3 | **SKM-TEA** | 155 patients / ~25,000 slices | meniscal tears, ligament tears, cartilage lesions, joint effusion (16 pathology categories) | Stanford AIMI custom DUA, non-commercial | ⚠️ research use only, check DUA | dense 3D segmentations + bounding boxes; qDESS-only, tiny N, strong domain shift |
| 4 | **OAI / MOST** | ~4,796 (OAI) + ~3,026 (MOST) | MOAKS/WORMS: cartilage, BML, osteophytes, effusion-synovitis, meniscus damage | NIMH NDA Data Use Certification | ❌ likely no | huge volume but OA-centric labels and DUC restrictions on redistribution/competition use |
| — | **AH-Knee / GreenRFM** | 3,662 studies | 13 abnormality labels (LLM-extracted) | private hospital cohort | ❌ no | not publicly available |

**Honest recommendation:** start with **MRNet + fastMRI+** as the labelled co-training pool; use **KMAR-50K** (unlabelled, CC-BY) only for SSL/pseudo-labelling; treat **SKM-TEA** as a segmentation auxiliary if the DUA permits; avoid **OAI/MOST** for Kaggle unless the NDA DUC is explicitly read and confirmed to allow competition use.

## 1. RSNA competition rules on external data

The RSNA Kaggle competition explicitly allows external data:

> "Freely & publicly available external data is allowed, including pre-trained models." ([RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data))

So the binding constraint is **licence permissiveness**, not a blanket ban. Each corpus below is judged against whether its licence permits (a) downloading, (b) training a model, and (c) uploading derived weights to a Kaggle code competition.

## 2. MRNet (Stanford, 2018)

- **Size:** 1,370 knee MRI exams from Stanford, three standard planes (sagittal, coronal, axial).
- **Labels:** binary study-level labels for abnormality, ACL tear, and meniscal tear.
- **Licence:** Creative Commons Attribution per a [`tensorflow/datasets` GitHub issue](https://github.com/tensorflow/datasets/issues/???). The paper (Bien et al., PLOS Medicine 2018) describes the dataset as released for public use. Train/valid/test splits are public.
- **Kaggle status:** ✅ clearly permissible.
- **Overlap with RSNA 12:** ACL, Meniscus (medial + lateral combined as one binary label).
- **Transfer evidence:** Bien et al. report AUCs of 0.937 (abnormal), 0.965 (ACL), 0.847 (meniscus) on internal validation. External validation on the Rijeka dataset showed Stanford-only model ACL AUC 0.824; training on the rest of Rijeka reached 0.911. This is direct knee-to-knee cross-dataset transfer.

## 3. fastMRI+ (NYU / Microsoft, 2021–2022)

- **Size:** 1,172 fully-sampled knee studies from the fastMRI validation set annotated by a subspecialist MSK radiologist; 16,154 bounding boxes; 13 study-level labels across 22 pathology categories. The underlying fastMRI knee corpus is ~1,594 multi-coil raw scans plus ~10,000 clinical DICOM knee exams.
- **Knee label vocabulary** (from [Zhao et al., Sci Data 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/), Table 1):
  - Meniscus: Meniscus Tear; Displaced Meniscal Tissue
  - Bones/Cartilage: Bone-Subchondral Edema; Bone Lesion; Bone-Fracture/Contusion/Dislocation; Cartilage Full Thickness Loss/Defect; Cartilage Partial Thickness Loss/Defect
  - Ligaments: ACL High Grade Sprain; ACL Low-Mod Grade Sprain; MCL High Grade Sprain; MCL Low-Mod Grade Sprain; PCL High Grade Sprain; PCL Low-Mod Grade Sprain; LCL Complex High Grade Sprain; LCL Complex Low-Mod Grade Sprain
  - Other: Joint Effusion; Joint Bodies; Periarticular Cysts; Muscle Strain; Soft Tissue Lesion; Patellar Retinaculum High Grade Sprain; Artifact (study-level)
- **Licence:** fastMRI+ annotations are open-access / MIT-licensed via the [fastMRI-plus GitHub repo](https://github.com/microsoft/fastmri-plus). The underlying fastMRI images require registration and acceptance of a custom NYU data-use agreement at [fastmri.med.nyu.edu](https://fastmri.med.nyu.edu).
- **Kaggle status:** ⚠️ almost certainly permissible once the NYU agreement is accepted, but the custom DUA must be checked for redistribution restrictions. Kaggle's rule treats "freely & publicly available external data" as allowed after registration.
- **Overlap with RSNA 12:** ACL (graded → binary), MCL, Effusion, Meniscus (any, no med/lat split), Cartilage/OA proxies, Contusion (pooled with fracture/dislocation).
- **Transfer evidence:** No direct transfer-to-RSNA paper found. fastMRI+ has been used for reconstruction-quality evaluation and pathology detection, but not as a pretraining corpus for a 12-label study-level classifier.

## 4. SKM-TEA (Stanford, 2021)

- **Size:** 155 patients, ~25,000 slices, 1.6 TB, quantitative double-echo steady-state (qDESS) knee MRI.
- **Labels:** segmentations of 6 tissues (patellar/femoral/tibial cartilage, medial/lateral meniscus) and bounding boxes for **16 clinically relevant pathology categories** across joint effusion and meniscal, ligament, and cartilage lesions. The exact category names live in the annotation JSON (`annotations/v0.0.1/{train,val,test}.json`); the supercategories are meniscal tear subtypes, cartilage lesion grades, ligament tears (ACL/PCL/MCL/LCL), and joint effusion. (Source: [SKM-TEA DATASET.md](https://github.com/StanfordMIMI/skm-tea/blob/main/DATASET.md), [Stanford AIMI](https://aimi.stanford.edu/datasets/skm-tea-knee-mri).)
- **Licence:** Stanford AIMI custom data-use agreement, non-commercial / research-only.
- **Kaggle status:** ⚠️ research-only DUA; read it carefully before using in a competition. Redistribution of derived weights may be restricted.
- **Overlap with RSNA 12:** meniscus (subtyped, no med/lat split in the public docs), ACL, cartilage lesions, effusion.
- **Caveats:** (1) qDESS is a 3D sequence with very different contrast from the RSNA clinical mix; (2) N = 155 is small for representation learning; (3) the value is mainly as a **segmentation/detection auxiliary**, not as a 12-label pretraining corpus.

## 5. OAI / MOST (NIH longitudinal cohorts)

- **Size:** OAI ~4,796 participants with knee MRIs; MOST ~3,026 older adults. Both are multi-center longitudinal studies.
- **Labels:** semi-quantitative MOAKS/WORMS-style scores (cartilage, bone marrow lesions, osteophytes, meniscus damage, effusion-synovitis), plus KL grade. Labels are osteoarthritis-centric, not acute injury.
- **Licence:** requires NDA Data Use Certification through the NIMH Data Archive. The DUC typically restricts use to research and may prohibit redistribution or competition use.
- **Kaggle status:** ❌ risky. The DUC language must be read explicitly; default assumption should be "not allowed for Kaggle code competition."
- **Overlap with RSNA 12:** cartilage, meniscus, effusion-synovitis, BML proxies — but coarse and OA-centric.
- **Transfer evidence:** Yeoh et al. (2023) showed transfer learning from ImageNet on OAI 3D DESS volumes gave ResNet18 AUC 0.945 vs. 0.763 from scratch. This proves **knee-MRI transfer learning helps**, but uses ImageNet → OAI, not OAI → RSNA.

## 6. AH-Knee / GreenRFM (private Chinese hospital cohort)

- **Size:** 3,662 studies / 15,431 images.
- **Labels:** 13 abnormality labels extracted by LLM from reports.
- **Licence:** private hospital cohort, not publicly released.
- **Kaggle status:** ❌ cannot be used.

## 7. What is missing in the literature

I found **no paper that pretrains or co-trains on fastMRI+, SKM-TEA, or MRNet and then reports a 12-label study-level multi-label AUC on an RSNA-style task**. The published transfer evidence is:

1. **MRNet → Rijeka** (Bien et al.): direct knee-to-knee cross-dataset ACL transfer.
2. **ImageNet → OAI** (Yeoh et al.): pretrained init dominates random init on knee MRI.
3. **General medical-imaging transfer** (RadImageNet, ImageNet-21k): pretrained trunks help, but knee-to-knee corpus transfer is under-reported.

Therefore any score estimate for co-training with these corpora is extrapolation, not a published number.

## 8. Ranking rationale for the RSNA task

1. **MRNet** wins on licence clarity and direct ACL/meniscus overlap, despite having only three labels.
2. **fastMRI+** wins on label richness and volume, but carries the NYU custom-agreement caveat.
3. **SKM-TEA** is high-quality but small and domain-shifted (qDESS-only).
4. **OAI/MOST** are large but licence-risky and OA-centric.
5. **AH-Knee** is ineligible.

## 9. Practical recommendation

For the RSNA co-training / pretraining problem:

- **Labelled co-training:** merge MRNet + fastMRI+ + Rijeka with masked loss over overlapping labels (ACL, meniscus, MCL, effusion). This is the only path supported by both data availability and honest label mapping.
- **Unlabelled pretraining:** KMAR-50K (CC-BY, 62k slices) is the cleanest unlabelled pool; fastMRI single-coil images are available behind the NYU agreement.
- **Segmentation auxiliary:** SKM-TEA dense segmentations could supervise a joint reconstruction/segmentation task, but check the Stanford DUA first.
- **Avoid for Kaggle:** OAI/MOST unless the NDA DUC is explicitly read and confirmed to permit competition-derived weights.

## Sources

- Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet", PLOS Medicine 2018. ([PLOS](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699))
- Zhao et al., "fastMRI+, Clinical pathology annotations for knee and brain fully sampled magnetic resonance imaging data", Scientific Data 2022. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8983757/))
- Desai et al., "SKM-TEA: A Dataset for Accelerated MRI Reconstruction with Dense Image Labels for Quantitative Clinical Evaluation", NeurIPS Datasets and Benchmarks 2021. ([arXiv](https://arxiv.org/abs/2203.06823), [GitHub](https://github.com/StanfordMIMI/skm-tea))
- Peterfy et al., "The osteoarthritis initiative: report on the design rationale for the MRI protocol for the knee", Osteoarthritis and Cartilage 2008.
- Yeoh et al., "Transfer learning-assisted 3D deep learning models for knee osteoarthritis detection", 2023. ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10136763/))
- RSNA Kaggle competition data page: external data allowed. ([Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data))
