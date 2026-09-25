# `train_series.csv` metadata as a predictive signal axis (Q-AA)

2026-09-02, Kimi. Audit of whether the per-series acquisition metadata in `train_series.csv` (`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`) can be used as a signal axis for the RSNA knee-abnormality competition.

## TL;DR

The metadata is **available at inference time** and **not a leak**, but it is **mostly redundant with what the pipeline already uses**. Of the three fields, `Anatomical_Plane` is constant (every study has Axial, Coronal and Sagittal), `Fat_Suppression` equals `Fluid_Sensitive` on every row, and `Fluid_Sensitive` is already used upstream to route series to slots. The literature shows metadata conditioning helps when the metadata carries *non-redundant* information (tumor type, multi-field sequence+plane), but is flat/inconclusive when the metadata is an image-inferable contrast flag. The original FiLM/concat seat should be **declined**; the only remaining metadata sub-axis is study-level series count / fluid-sensitive series count, and there is no direct evidence for it.

## What the CSVs actually contain

`train_series.csv` and `test_series.csv` have exactly the same schema:

| Column | Values | Meaning |
|---|---|---|
| `StudyInstanceUID` | string | study identifier |
| `SeriesInstanceUID` | string | series identifier |
| `Fluid_Sensitive` | 0 / 1 | sequence emphasizes fluid signal |
| `Fat_Suppression` | 0 / 1 | fat-suppression applied |
| `Anatomical_Plane` | Sagittal / Axial / Coronal | acquisition plane |

Counts:

- `train_series.csv`: 24,371 series across 4,407 studies.
- `test_series.csv`: 15 series across 3 studies (small public test set).
- No missing values in any metadata field.
- `train.csv`: 4,407 studies, all with a free-text `Report`, but only **58 studies have any of the 12 abnormality labels filled in** (and those 58 have all 12 labels filled in).

## Correction: only one of the three fields is informative, and it is already used

`kneemri/evaluation/study_series.py:8-11` documents what the data actually says:

- `Anatomical_Plane` is **constant at the study level**: all 4,407 training studies carry Axial, Coronal and Sagittal series. It is therefore not a discriminative signal for study-level abnormality classification.
- `Fat_Suppression` equals `Fluid_Sensitive` on **every one of the 24,371 rows**. `StudySeries.redundant()` asserts this.
- `Fluid_Sensitive` is the only remaining bit, and it is already consumed upstream:
  - `kneemri/evaluation/raptor_feed.py:263,267` routes series with `row["Anatomical_Plane"] == slot.plane` and `int(row.get("Fluid_Sensitive", 0)) == slot.fluid_sensitive`.
  - `kneemri/kaggle/slot_fill.py:40` builds the routing map offline from `train_series.csv`.
  - `kneemri/kaggle/submission.py:346` builds the same map at inference.

So a FiLM/concat seat conditioning on these three fields would re-inject a bit the sampler has already used to choose which slices the trunk sees. The meaningful question is not "condition on Fluid_Sensitive" but "is there any residual signal in *how many* series a study has, or *how many* are fluid-sensitive?" Those counts are computed in `StudySeries.counts()` and `.fluid()` but are not used as model inputs.

## Leakage check

The metadata is **not a leak**: `test_series.csv` exists and contains the same three predictive fields for every test series. Using it is no different from using DICOM orientation or sequence tags that the test set also carries.

The only leakage-shaped concern is if the *presence* of a series type were correlated with the underlying pathology because the ordering clinician selects sequences based on indication. That is well documented in protocol-selection literature, but here every study in the competition already has the same metadata columns, and the fields describe *which* sequence each series is, not *whether* it was ordered. So conditioning on the fields is safe; conditioning on "which series were present in this study" would need the same caution discussed in `research/deep_dives/2026-09-01_missing_sequences_knee_mri.md`.

## Literature evidence

### 1. Lemay 2021 — when metadata helps and when it does not

[Benefits of Linear Conditioning with Metadata for Image Segmentation](https://proceedings.mlr.press/v143/lemay21a/lemay21a.pdf) (Lemay et al., MIDL 2021) is the most important reference for the corrected question. It conditions a 2D U-Net with FiLM on metadata and reports:

- **+5.1% Dice** on spinal-cord tumor segmentation when the metadata is **tumor type** (astrocytoma / ependymoma / hemangioblastoma) — a property not directly readable from a single slice and not used to select input slices.
- The authors explicitly discuss the case where metadata does **not** help: citing Vincent et al. 2020, they note that conditioning on MRI contrast (T2-weighted vs T2star-weighted) gave **inconclusive gains** because "the modality-related features might already be encoded in the regular U-Net, therefore the metadata added to FiLM is not informative enough".

This maps directly to our case. `Fluid_Sensitive` is an image-inferable contrast flag that is already used to route slices; it matches the Vincent et al. scenario, not the tumor-type scenario.

### 2. Meta-D — FiLM/cross-attention conditioning on sequence + plane metadata

[Meta-D: Metadata-Aware Architectures for Brain Tumor Analysis and Missing-Modality Segmentation](https://arxiv.org/html/2603.04811v1) (Kim et al., arXiv 2026) is the closest direct analog. It treats MRI sequence type and anatomical plane as categorical metadata and conditions a 2D CNN via **Progressive FiLM** and a 3D transformer via **metadata-driven cross-attention**.

Reported effect sizes:

- 2D brain-tumor classification: adding both sequence and plane metadata improved F1 by up to **+2.62% absolute** over an image-only ResNet-18.
- Permutation test (randomized metadata at inference) caused accuracy drops up to **-10.28%**, confirming the model actively uses the metadata rather than learning it from pixels.
- Mean FiLM scaling magnitude |γ| ≈ **0.20**, i.e., the network rescales intermediate features by ~20% based on metadata.
- 3D missing-modality tumor segmentation: replacing spatial self-attention with metadata-keyed cross-attention improved tumor Dice by up to **+5.12% absolute** under extreme missingness, while cutting parameters by **24.1%**.

Mechanism: embed metadata strings → MLP predicts per-channel γ/β → modulate backbone feature maps. In 3D, image tokens query a fixed metadata dictionary, and a deterministic mask zeros attention to missing sequences.

### 2. Frontiers 2021 — MRI sequence identification from DICOM metadata

[Liang et al., *Frontiers in Neuroinformatics* 2021](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2021.622951/full) trained a random forest on 18 DICOM header features to classify MRI sequence type across 34,691 scans from three multi-site programs.

Key finding: with ~800 training examples per sequence type, classification accuracy plateaued above **0.999**; with 1,200 per type, accuracy reached **99.99%** (36 errors in 442,326 predictions). The top features were `Scanning Sequence`, `Sequence Variant`, `Scan Options`, `Sequence Name`, `Repetition Time`, and `Echo Time`.

Translation: MRI metadata is a *reliable* descriptor of sequence contrast and geometry. In this competition the metadata is already cleaned and reduced to three high-level flags, so the model does not need to learn to read DICOM headers.

### 3. Dohmen et al. 2026 — end-to-end DICOM series classification with image + sparse metadata

[Revisiting Integration of Image and Metadata for DICOM Series Classification](https://arxiv.org/html/2602.23833v1) (Dohmen et al., arXiv 2026) is the most relevant engineering reference because it fuses image and metadata for *series-level* classification, handles missing metadata, and benchmarks against image-only and 3D baselines.

Architecture:

- 2.5D visual encoder (DenseNet121 on equidistant slices + cross-slice attention).
- Sparse Metadata Encoder (SME): each observed DICOM tag is a (feature-index, value) pair; feature embeddings are modulated by a FiLM value-network, then averaged over observed tags. No imputation.
- Bi-directional Cross-modal Attention (BCA): image tokens attend to metadata tokens and vice-versa, then a learned pooler aggregates slices.

Results on Duke Liver MRI (13 classes):

| Method | Weighted F1 |
|---|---|
| 2D image-only | 92.90% |
| 3D image-only | 93.51% |
| Metadata-only (XGBoost) | 74.71% |
| Joint concat + zero imputation | 93.51% |
| Joint concat + learned imputation | 93.55% |
| **Joint SME + BCA (ours)** | **96.66%** |

The metadata-only model is far weaker than the image model, but the joint model beats the best image-only/3D baseline by **~3 percentage points**. The SME/BCA components are responsible for the extra gain, not simple concatenation.

### 4. Public baseline usage

The Kaggle notebook [RSNA Knee: Format-Correct Baseline](https://www.kaggle.com/code/jacobegarcia/rsna-knee-format-correct-baseline) by jacobegarcia reads `test_series.csv` metadata to label each series by anatomical plane, fluid sensitivity, and fat suppression, confirming the competition baseline treats these fields as part of the input pipeline.

## Mechanisms applicable here

Given the correction above, the only unused metadata-derived quantities are:

1. **Study-level series count** (`StudySeries.counts()`) — how many series the study has.
2. **Study-level fluid-sensitive series count** (`StudySeries.fluid()`) — how many of those series are fluid-sensitive.
3. **Per-series Fluid_Sensitive** — already used to route slices to slots.

Conditioning on per-series `Fluid_Sensitive`, `Fat_Suppression` or `Anatomical_Plane` is re-injecting information already in the pipeline. A study-level count vector could in principle carry signal (e.g., complex injuries are scanned with more sequences), but there is no published precedent for this specific feature in knee MRI.

## Caveats

- **The three fields collapse to one bit plus a constant.** After deduplication, the only per-series signal is `Fluid_Sensitive`, and it is already a routing input.
- **Lemay 2021 warns exactly against this case.** When metadata is an image-inferable contrast property, FiLM gains are inconclusive because the trunk already encodes it.
- **Meta-D's gain is from multi-field conditioning.** Sequence + plane together resolve both contrast ambiguity and geometric variance; plane is constant here and sequence is already routed.
- **Study-level counts are speculative.** They are not used upstream, but also not supported by literature the project has located.

## Verdict and recommended next step

**Decline the original metadata-conditioning seat (knee-bc1m).** The per-series fields are either redundant with the slot-routing pipeline or constant across studies, and the literature precedent for conditioning on an already-used contrast flag is flat.

The only remaining sub-hypothesis is whether **study-level series count or fluid-sensitive count** predicts abnormality. That is a different, weaker claim with no direct evidence. If someone wants to test it, the cost is one cheap feature-engineering seat, but it should not be confused with the Meta-D/Dohmen literature.

Bead `knee-bc1m` should be closed as declined; a new bead can be filed if the count hypothesis is worth a smoke run.
