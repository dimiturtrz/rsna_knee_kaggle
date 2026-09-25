# Pathology-specific SOTA for knee MRI (2025–2026)

The user asked to break the SOTA search down by pathology rather than looking for one model that
covers everything. This note surveys the best published work per RSNA finding, with the practical
filter: are weights public, and could the model plausibly improve our 12-label score?

## RSNA 12 label → pathology groups

| RSNA finding | Pathology group | External data with honest mapping |
|---|---|---|
| ACL | Ligament | MRNet, fastMRI+, Rijeka severity, OrthoDiffusion |
| Medial Meniscus | Meniscus | MRNet (any meniscus), fastMRI+ (any), OrthoDiffusion |
| Lateral Meniscus | Meniscus | MRNet (any), fastMRI+ (any), OrthoDiffusion |
| MCL | Ligament | fastMRI+, OrthoDiffusion |
| Effusion | Fluid | fastMRI+, OrthoDiffusion |
| Synovitis | Synovium | None found |
| Medial OA / Lateral OA / PF OA | Osteoarthritis / cartilage | OAI (KL grades), fastMRI+ cartilage boxes, no direct RSNA mapping |
| Baker's cyst | Cyst | None found |
| Bone Contusion | Bone marrow lesion | fastMRI+ pooled with fracture/contusion; no clean mapping |
| Fracture | Bone | fastMRI+ fracture/contusion boxes |

## 1. ACL — strongest single-pathology literature

- **Xue et al., *Scientific Reports* 2024** — "Approaching expert-level accuracy for
differentiating ACL tear types on MRI with deep learning." 862 patients, sagittal PDW-SPAIR.
ACL segmentation U-Net (ACL-DNet, Dice 0.98) + VGG16-based status classifier with radiomics
and clinical factors (ACL-SNet). AUC 0.99 on six-class status discrimination (intact + five
Sherman tear types), comparable to a senior radiologist (AUC 0.96) and above junior readers.
  - **Weights/data:** not public; single-centre hospital dataset.
  - **Relevance to RSNA:** RSNA only asks binary ACL, which is easier than tear-type
    discrimination. The paper confirms that ACL is a solved-ish localisation task with a good
    segmentation + radiomics pipeline, but it does not give us a drop-in model.

- **MRNet (Bien et al., *PLOS Medicine* 2018)** — internal validation ACL AUC **0.965**, external
Rijeka validation **0.911** after Rijeka fine-tuning. Still the strongest public ACL number.

- **OrthoDiffusion (2026)** — reports ACL injury as one of its 8 diagnosis labels; macro AUROC
0.908 across all eight. The only public knee-specific model that includes ACL.

**Bottom line for ACL.** No public ACL-only foundation model with weights. MRNet and
OrthoDiffusion are the actionable sources; a task-specific ACL segmentation auxiliary could
help, but the gains would be concentrated in the already-strong ACL column.

## 2. Meniscus — segmentation is ahead of classification

- **Li et al., *European Radiology* 2022** — "Automated meniscus segmentation and tear detection
of knee MRI with a 3D mask-RCNN." Arthroscopy-confirmed tears, 3D Mask R-CNN for segmentation
+ tear detection. Strong for medial/lateral meniscus tear localisation.
  - **Weights/data:** not public.

- **Şimşek et al., *Applied Sciences* 2025** — "Automatic meniscus segmentation using
YOLO-based deep learning models with ensemble methods in knee MRI images." YOLO segmentation
ensemble for meniscus, public dataset flavour.
  - **Weights:** GitHub-style project; not a foundation model.

- **MRNet** — meniscus tear AUC **0.847** internal, but does not distinguish medial vs lateral.

- **OrthoDiffusion** — includes medial and lateral meniscus tears as two of its eight labels.

**Bottom line for meniscus.** Segmentation models exist but are task-specific and weight-less.
OrthoDiffusion is again the only public multi-label model with meniscus seats. A meniscus
segmentation auxiliary (e.g., SegmentAnyBone-style) is plausible but would need labelled masks
on RSNA or an external corpus.

## 3. Osteoarthritis / cartilage — dominated by radiographs and OAI

- **Panfilov et al., *J Orthop Res* 2022** — deep-learning-based segmentation of knee MRI for
subregional cartilage morphological assessment (OAI data). Correlation with OA progression
r > 0.93 for cartilage volume.
  - **Weights:** not public.

- **Panwar et al., *Scientific Reports* 2024** — ResNet50 + DeepStack for MRI-based KOA severity,
reported accuracy 99.71%.
  - **Weights:** not public.

- **Alyami, *Curr Med Imaging* 2024** — EfficientNet-B3 + ResNeXt-101 ensemble on OAI for KL
grade prediction, 93% validation accuracy.
  - **Weights:** not public.

- **Calivà et al., *Nat Rev Rheumatol* 2022** — review of AI applied to MRI in OA; concludes
that MRI-based DL can quantify cartilage, bone marrow lesions and meniscal pathology, but
clinical translation is limited by small datasets and lack of standardisation.

**Bottom line for OA/cartilage.** The OA literature is mostly radiograph-based KL grading or
OAI MRI segmentation. Neither maps cleanly to the RSNA "Osteoarthritis" columns, which are
likely report-derived and not KL-graded. There is no public cartilage/OA foundation model with
weights that would directly improve those columns.

## 4. Bone marrow lesions / contusion / fracture — weak public coverage

- **Maarek et al., *Clinical Imaging* 2025** — "Enhancing knee MRI bone marrow lesion detection
with artificial intelligence: an external validation study." AI-assisted BME detection improves
sensitivity/specificity for general radiologists.
  - **Weights/data:** not public.

- **fastMRI+** — "Bone-Fracture/Contusion/dislocation" and "Bone-Subchondral edema" boxes exist,
but are pooled and location-agnostic; honest mapping to RSNA Fracture or Contusion is weak.

**Bottom line.** The Fracture and Contusion columns are among RSNA's weakest. External data with
clean labels is scarce; the best near-term lever is probably the internal label-correction work
(`blend_hybrid_fx.csv`) rather than a foreign model.

## 5. Synovitis / Baker's cyst — no actionable public models

No public deep-learning models specifically for knee MRI synovitis or Baker's cyst classification
were found in this search. These are likely the most under-served columns by external data.

## 6. Comparison with the general knee models already on the table

| Model | Labels covered (of RSNA 12) | Weights | Kaggle-shippable? |
|---|---|---|---|
| MRNet | abnormal, ACL, any meniscus (3-ish) | requires registration | Yes per rule 2.6(a) |
| Rijeka kneeMRI | ACL severity only | CC BY-NC-ND 4.0 | Licence question |
| fastMRI+ | ACL, MCL, meniscus-any, effusion, cartilage, bone-fracture/contusion/bone-edema | fastMRI DUA | No |
| OrthoDiffusion | ACL, medial meniscus, lateral meniscus, MCL, effusion (5 direct) | ✅ HuggingFace / Kaggle | Kaggle licence unclear |
| OrthoFoundation-L | ACL, meniscus, cartilage, MCL, PCL, LCL, etc. | ✅ GitHub | MIT code; weights? |

The pathology-specific literature does not change this table. Every actionable external source is
already captured by one of the general knee models above.

## 7. Implications for the final score

- **No hidden pathology-specific foundation model exists.** The best published per-pathology work
  is on private hospital datasets without released weights.

- **The overlap map is the constraint.** External labels honestly map to only ~5 of the 12 RSNA
  labels (ACL, meniscus-any, MCL, effusion, possibly cartilage/fracture weakly). Synovitis,
  Baker's, contusion and the three OA columns have no clean external supervision.

- **The largest gains are therefore capped by the per-label denominator.** Even a perfect
  external model for ACL/meniscus/MCL/effusion can only move the mean-AUC by roughly the column
  fraction times the column improvement. The `merged_blend_s2026` result (+0.0063 mean, +0.0196
  lateral meniscus, +0.0161 ACL) is consistent with this ceiling.

- **OrthoDiffusion remains the only untested public model that covers multiple pathologies.** Its
  3D diffusion backbone and five direct label overlaps make it the highest-ceiling single bet,
  even though only 5/8 of its labels map.

## 8. Recommendations

1. **Stop searching for pathology-specific SOTA models; the cupboard is bare.** The actionable
   frontier is general knee models (OrthoDiffusion) and segmentation auxiliaries, not
   per-pathology classifiers.

2. **If a pathology must be attacked directly, use segmentation not classification.**
   SegmentAnyBone / SAM-Med3D / SegmentAnything-style models can produce joint-region or
   meniscus/cartilage masks that feed the 2D classifier as extra channels. This attacks OA and
   meniscus columns indirectly.

3. **For Synovitis and Baker's**, the only honest lever is internal: the report-label
   correction/refinement programme and the 58-study gold set.

4. **For Fracture**, the already-verified `blend_hybrid_fx.csv` correction is the best evidence
   the project has; an external fracture model is unlikely to beat it because the mapping is
   noisy.
