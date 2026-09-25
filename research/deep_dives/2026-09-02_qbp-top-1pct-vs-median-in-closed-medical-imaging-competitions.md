# What separates the top 1% from the median in closed medical-imaging competitions — Q-BP

Context: Claude's Q-BP (`.agent-runtime/inbox/read/2026-09-02T1952Z-claude-qbp-what-separates-top-1pct-from-median.md`).
Position: 0.944 public, rank 33 / 2919, leader 0.952. Every axis priced in this repo — member
quality, ensemble seating, decode geometry, precision, pooling operator — has read flat. The
question is what closed-competition write-ups show that top-1% solutions did that median solutions
did not.

Scope: RSNA-Kaggle 2022 cervical-spine fracture, 2023 abdominal trauma, 2024 lumbar-spine
degenerative classification, plus SIIM-ISIC 2020 as a non-RSNA reference. "Median" means the
public starter/baseline notebooks and silver/bronze write-ups that treat the task as a single-stage
image-level classification problem.

## Short answer

The top 1% are not distinguished by a single trick. They are distinguished by a **staged pipeline
that localizes first and classifies second**, combined with **metric-aware late fusion** and
**label-noise handling** that median solutions do not implement. Ensemble size and compute budget
are necessary but not sufficient: the same large ensembles used by top teams would be wasted on
single-stage inputs.

Ranked by how consistently they appear in top solutions and are absent from median ones:

| rank | difference | category | frequency in top write-ups | absent from median? |
|---|---|---|---|---|
| 1 | **Two-stage localize-then-classify** (segmentation/keypoint/heatmap → crop → classifier) | pipeline | 9/9 reviewed top solutions | yes — median is single-stage |
| 2 | **RNN / attention slice aggregation + metric-aware late fusion** | modelling / fitting | 7/9 | yes — median pools slices with mean/max |
| 3 | **Pseudo-labeling / confident learning / label denoising** | pipeline / fitting | 4/9 | yes — median uses provided labels only |
| 4 | **Auxiliary pixel-level / segmentation supervision** | pipeline | 4/9 | partly — median may use provided masks, but not as auxiliary loss |
| 5 | **Task-structure exploitation** (level/organ/severity relationships, metric-shaped losses) | exploitation | 5/9 | yes — median treats labels as independent binary |
| 6 | **Large diverse ensemble + TTA-like ensemble** | modelling / fitting | 9/9 | partly — median ensembles are smaller and less diverse |
| 7 | **External / public data augmentation** | pipeline | 2/9 | yes — median uses competition data only |

The honest bottom line: the top-and-median gap is **mostly pipeline architecture (staged + metric-aware aggregation)**, not a single modelling or exploitation move.

## 1. Two-stage localize-then-classify — the strongest separator

Every top write-up I could access uses some form of **stage-1 localization / detection / segmentation
followed by stage-2 classification**.

- **RSNA 2022 Cervical Spine, 1st** (Qishen Ha): stage 1 trains a 3D U-Net on 87 segmentation masks to
  locate each vertebra; stage 2 crops the vertebra and runs a 2.5D CNN + LSTM classifier on 15 slices.
  [Zenn summary](https://zenn.dev/kabupen/articles/da4a2d180e9e62)
- **RSNA 2022, 3rd** (Darragh): ignores the provided bounding boxes, reconstructs per-slice bboxes from
  segmentation masks, crops with the largest per-study bbox, then CNN + LSTM.
  [Zenn summary](https://zenn.dev/kabupen/articles/391a9742a667b6)
- **RSNA 2023 Abdominal Trauma, 1st** (Nischaydnk): stage 1 is a 3D segmentation model that crops
  liver/spleen/kidney; stage 2 is a 2.5D CNN + RNN. Auxiliary segmentation loss is added to the
  classification encoder.
  [GitHub](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)
- **RSNA 2023, 2nd** (Theo Viel): 3D organ segmentation produces crops for kidney/liver/spleen; a 2D
  CNN + RNN processes full frames; a final RNN aggregates probabilities with organ-conditioned
  pooling.
  [GitHub](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma)
- **RSNA 2024 Lumbar Spine, 1st** (Avengers): 2-stage, predicting disc-level severity from cropped
  sagittal sequences.
  [Kaggle write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)
- **RSNA 2024, 2nd** (Brendan Artley / IanPan-Kevin-Yuji-Bartley): stage 1 predicts ten disc
  coordinates from sagittal T1/T2; stage 2 classifies each cropped sequence. Localization accuracy
  within 5 pixels ~99.1%.
  [GitHub](https://github.com/brendanartley/RSNA-2024-Competition)
- **RSNA 2024, 3rd** (Moyasii / SonySpine): keypoint detectors for sagittal and axial planes in stage
  1; center and side classifiers in stage 2. Iterative pseudo-labels generated from stage-2 OOF
  predictions.
  [GitHub](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub)
- **RSNA 2024, 5th** (siwooyong): heatmap-based detection with Gaussian-expanding labels in stage 1;
  2.5D CNN + RNN with a class-relationship module in stage 2.
  [GitHub](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification)
- **RSNA 2024, 7th** (lhwcv): keypoint models (2D and 3D) in stage 1; 3D ROI crop classifiers in
  stage 2, fusing three views in feature space.
  [GitHub](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine)

The median public notebook is typically a single-stage model that feeds the whole study or a fixed
crop into a 2D/2.5D CNN. The top-1% investment in stage 1 is the most consistent separator.

## 2. RNN / attention slice aggregation + metric-aware late fusion

Top solutions rarely reduce slice features with simple mean/max pooling. They use recurrent or
attention mechanisms and often condition the final logits on organ/level.

- **RSNA 2022 1st**: LSTM over 15 slice embeddings (bidirectional, 2-layer).
- **RSNA 2023 2nd**: a heavily customized LSTM that concatenates segmentation probabilities,
  classification probabilities, and organ-conditioned pooled features; independent per-organ logits
  see only the relevant organ's crop and pooled features. The model is trained to optimize the
  competition metric directly.
- **RSNA 2024 2nd**: LSTM + attention pooling over slice embeddings; manifold mixup on embeddings.
- **RSNA 2024 5th**: level-wise RNN modeling — instead of predicting 25 independent labels, the
  model first models the 5 labels within each spinal level, exploiting the fact that co-occurring
  stenoses are level-local.

This is both a **modelling** choice (RNN/attention) and a **fitting** choice (the final aggregator is
trained on the metric). Median solutions usually global-average-pool and use a single BCE head.

## 3. Pseudo-labeling / confident learning / label denoising

Top teams treat the provided labels as noisy and augment them.

- **RSNA 2024 2nd**: best submission used `0.5 * loss_with_labels + 0.5 * loss_with_pseudolabels`.
  Pseudo-labels came from an out-of-fold team ensemble; confident learning removed noisy training
  points where the ensemble strongly disagreed with the label.
- **RSNA 2024 3rd**: creates `pseudo_label_v1` and `pseudo_label_v3` from stage-2 OOF predictions and
  retrains additional models on the mixed targets.
- **RSNA 2024 5th**: used a public dataset to improve stage-1 detection; this is weaker pseudo-label
  evidence but still label-augmentation.

Median solutions train on the provided labels without denoising. This is a low-card move that is
under-used outside the top.

## 4. Auxiliary pixel-level / segmentation supervision

When segmentation masks are available, top teams use them as auxiliary signal, not only as crops.

- **RSNA 2023 1st**: adds a U-Net / 2D-CNN segmentation head to the classification encoder; dice loss
  weighted at 0.125. Reports this gave **+0.01 to +0.03** per model.
- **RSNA 2022 1st/3rd**: segmentation masks drive cropping; no auxiliary-loss number is reported,
  but the masks are central to the pipeline.
- **RSNA 2024**: stage 1 itself is a localization/heatmap task, which is a form of pixel-level
  pretraining for stage 2.

Median notebooks may load segmentation masks but usually only for cropping or ignore them.

## 5. Task-structure exploitation

Top solutions explicitly encode the structure of the label space and the competition metric.

- **RSNA 2024 lumbar**: labels are ordinal severity grades for three conditions across five disc
  levels. The 5th-place team models **only the 5 classes within the same level** with an RNN,
  exploiting co-occurrence at a level. The same team uses a two-step train: first maximize AUC
  without weighted loss, then freeze the backbone and finetune the head with weighted loss + any
  loss to optimize the competition metric.
- **RSNA 2023 trauma**: the 2nd-place RNN uses independent per-organ logits with organ-conditioned
  feature access, mirroring the multi-organ task structure.
- **RSNA 2022 cervical**: per-vertebra binary fracture labels are predicted with per-vertebra
  classifiers after localization.

Median solutions flatten all labels into a single multi-label vector and optimize BCE. Encoding the
task structure is the clearest **exploitation** difference.

## 6. Large diverse ensembles + TTA-like inference

This is the one difference that *is* about size and budget, but it is not enough by itself.

- **RSNA 2024 5th**: "various backbone ensemble" across stage 1 and stage 2 plus a "tta-like
  ensemble" where stage 2 is run for each stage 1 model and results are ensembled.
- **RSNA 2023 2nd**: final ensemble is 3 × 2D models + 8 × 2.5D crop models.
- **RSNA 2023 1st**: ensemble of multiple CoaT and EfficientNetV2 models trained on 4 folds.
- **RSNA 2024 2nd**: nine rotation TTA at inference.

The key point is that these ensembles sit **on top of a diverse pipeline** (different localizers,
crops, slice counts, backbones). A median solution that simply averages five copies of the same
single-stage notebook does not get the same benefit.

## 7. External / public data

Least frequent, but present.

- **RSNA 2024 5th**: leveraged a public dataset for stage-1 detection improvement.
- **RSNA 2023**: the 1st-place preprocessing uses Theo Viel's preprocessed data, but the main gain
  is from the staged pipeline and auxiliary loss, not from external labels.

External data is not a universal top-1% separator in these three RSNA competitions.

## What is *not* a separator

- **Single-stage architecture choice alone.** Median and top solutions both use EfficientNet,
  ConvNeXt, ResNet, etc. The same backbones appear in both groups.
- **Plain TTA.** Top teams use TTA, but median solutions also often use TTA. TTA is not the gap.
- **Standard data augmentation.** Flip/rotate/scale/color jitter are common everywhere. The gap is
  in *localization*, *task-structure modeling*, and *label handling*.

## Reading for our plateau

Claude's note also corrected the Q-BO answer: **slice coverage is not unread**. `stack224` and
`cover_labelwise` (36 slices) already ran on 2026-08-31 and were flat (−0.0033 and −0.0014 vs
`base224`). So the one remaining input-geometry lever from Q-BO is closed.

That leaves the repo in the same situation as the closed competitions: the remaining gap is most
likely to come from **pipeline structure** (staged localization/classification, metric-aware
aggregation, label denoising) rather than from more models, more resolution, or more slices.

## Sources

- Qishen Ha, RSNA 2022 Cervical Spine Fracture Detection — 1st place solution write-up.
  [Kaggle](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution)
  (summary: [Zenn](https://zenn.dev/kabupen/articles/da4a2d180e9e62))
- Darragh, RSNA 2022 — 3rd place solution.
  [Zenn summary](https://zenn.dev/kabupen/articles/391a9742a667b6)
- Nischaydnk et al., RSNA 2023 Abdominal Trauma Detection — 1st place solution.
  [GitHub](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)
- Theo Viel, RSNA 2023 Abdominal Trauma Detection — 2nd place solution.
  [GitHub](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma)
- Avengers, RSNA 2024 Lumbar Spine Degenerative Classification — 1st place solution.
  [Kaggle write-up](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)
- Brendan Artley, RSNA 2024 Lumbar Spine — 2nd place (IanPan-Kevin-Yuji-Bartley team).
  [GitHub](https://github.com/brendanartley/RSNA-2024-Competition)
- Moyasii / SonySpine, RSNA 2024 Lumbar Spine — 3rd place solution.
  [GitHub](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub)
- siwooyong, RSNA 2024 Lumbar Spine — 5th place solution.
  [GitHub](https://github.com/siwooyong/RSNA-2024-Lumbar-Spine-Degenerative-Classification)
- lhwcv, RSNA 2024 Lumbar Spine — 7th place solution.
  [GitHub](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine)
- Repo prior art:
  `research/deep_dives/2026-09-02_qbo-what-moves-after-ensemble-saturation.md`,
  `research/deep_dives/2026-09-01_past_ensemble_plateau_mechanisms.md`.
