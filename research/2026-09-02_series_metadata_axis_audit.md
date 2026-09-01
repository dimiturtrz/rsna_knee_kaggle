# `train_series.csv` metadata as a predictive signal axis (Q-AA)

2026-09-02, Kimi. Audit of whether the per-series acquisition metadata in `train_series.csv` (`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`) can be used as a signal axis for the RSNA knee-abnormality competition.

## TL;DR

Yes. The metadata is **available at inference time** (`test_series.csv` ships the same columns), is **complete and clean** (no missing values), and the literature shows that conditioning a vision model on analogous MRI metadata gives measurable gains. The recommended move is a cheap metadata-conditioning seat, not a metadata-only classifier, because only 58 of 4,407 training studies carry labels.

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

## Leakage check

The metadata is **not a leak**: `test_series.csv` exists and contains the same three predictive fields for every test series. Using it is no different from using DICOM orientation or sequence tags that the test set also carries.

The only leakage-shaped concern is if the *presence* of a series type were correlated with the underlying pathology because the ordering clinician selects sequences based on indication. That is well documented in protocol-selection literature, but here every study in the competition already has the same metadata columns, and the fields describe *which* sequence each series is, not *whether* it was ordered. So conditioning on the fields is safe; conditioning on "which series were present in this study" would need the same caution discussed in `research/2026-09-01_missing_sequences_knee_mri.md`.

## Literature evidence

### 1. Meta-D — FiLM/cross-attention conditioning on sequence + plane metadata

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

The project already has a whole-stack, per-series feature pipeline. Three natural ways to add metadata, from cheapest to most complex:

1. **Late concatenation** — embed the three fields (2 binary + 3-class plane) into a small vector and concatenate to the pooled series embedding before the label heads. Cost: one embedding table + a few fully-connected weights.
2. **FiLM conditioning** — metadata embedding → MLP → per-channel γ/β applied to backbone feature maps. This is what Meta-D showed +2.6% F1 for. Cost: a lightweight MLP; no extra inference memory for the backbone.
3. **Cross-attention** — image tokens as queries, metadata tokens as keys/values. This is what Dohmen et al. used for +3 pp over image-only, but it adds a transformer block and is only worthwhile if the gain is shown in a seat test.

Because the fields are *per-series*, not per-slice, late concatenation or series-level FiLM is the natural first step. Per-slice metadata would only make sense if the project moves to per-slice token inputs.

## Caveats

- **Metadata is coarsely quantized.** There are only 3 effective bits of information per series (2 binary + 3 categories). The image encoder may already learn to infer these properties from pixels, so the marginal gain may be smaller than the literature's +2–3 pp.
- **Labels are scarce.** A metadata-only model cannot be trained on 58 labeled studies. Metadata must condition a vision model trained on image–label pairs.
- **Plane/sequence may already be encoded in the slot design.** The repo already treats sagittal/coronal/axial series differently via `extended_slots`. If the existing architecture effectively uses plane, late concatenation may duplicate that signal.
- **No missing metadata** means the sparsity-aware machinery from Dohmen et al. is not required, but it is a nice robustness feature if future data arrives with blank fields.

## Verdict and recommended next step

The metadata axis is **real, safe, and under-tested in this repo**. The cheapest proof is a distinct-and-strong seat that adds a metadata embedding (late concat or FiLM) to one of the current whole-stack members and compares it to the image-only control. If that seat clears the project's seat threshold, the mechanism can be rolled into the broader ensemble.

A bead has been filed to track the implementation seat: `knee-bc1m`.
