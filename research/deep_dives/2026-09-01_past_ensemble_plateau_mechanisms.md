# Mechanisms that carry the last 0.5–1% past an ensemble plateau in medical-imaging competitions

Context: Claude asked what published competition solutions report as gains *after* the ensemble is already built — changes that are not a new member, a seat change, or a member weight. The project itself is on a 0.937–0.941 plateau with the seating axis closed, so the question is where the remaining 0.5–1% has actually come from in the literature.

## Short answer

The honest answer is that most write-ups do **not** isolate the post-ensemble delta. The mechanisms Claude lists are real, but the majority are training-time or ensemble-construction decisions rather than things you bolt on after the ensemble is fixed. The few mechanisms with a documented delta *on top of an already-built ensemble* are:

1. **Source-aware or per-label threshold calibration** — the cleanest documented post-ensemble gain.
2. **Test-time augmentation (TTA)** — widely used, but gains are usually reported over a single model, not over an ensemble.
3. **Rank averaging vs. probability averaging** — used by winners, but the gain is rarely quantified.

Everything else (pseudo-labeling, label hierarchy, test-time distribution adaptation) is either applied during training/ensemble construction or lacks quantified post-ensemble evidence.

## 1. Post-hoc / per-label / per-source threshold calibration

The strongest documented example is Nilay et al., *A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans* (arXiv:2603.14621v1, 2026). Their final nine-model ensemble already reaches F1 = 0.9280. They then compare ensemble strategies on the validation set and report:

> "Source Specific Threshold Calibration ... results in a +0.14 F1 improvement compared to using one global threshold for all sources."

This is measured against the same ensemble, with the only change being per-source threshold tuning. It is exactly the kind of post-ensemble, metric-aware lever Claude is asking about. The caveat is that the gain is specific to a metric that averages F1 across four hospital sources; for a single global AUC metric the effect would be smaller.

For per-label threshold optimization in class-imbalanced medical classification, Rajaraman et al., *Deep learning model calibration for improving performance in class-imbalanced medical image classification tasks* (PMCID PMC8794113, 2022), show that calibration plus threshold choice matters a lot at the default 0.5 threshold, but that the advantage disappears once you use a PR-guided optimal threshold. Their study is on single models, not ensembles, but it implies that **threshold search is more powerful than calibration alone** and that the marginal gain of further calibration after an ensemble is likely small unless the metric is threshold-sensitive.

## 2. Test-time augmentation (TTA)

TTA is the most universal post-ensemble trick in winning solutions. Examples:

- **SIIM-ISIC Melanoma 2020** (Ha et al., arXiv:2010.05351): the winning ensemble averaged predictions over 8 orientations per model before combining 18 diverse models.
- **RSNA 2024 Lumbar Spine** (Brendan Artley, 2nd place GitHub write-up): nine rotations at inference.
- **RSNA 2024** (siwooyong, 5th place): "tta-like ensemble" listed as a key component.

Moshkov et al., *Test-time augmentation for deep learning-based cell instance segmentation* (PMCID PMC7081314, 2020), report that TTA improves instance-segmentation mAP by roughly 0.01–0.02 in medical microscopy tasks. That is on a single model. Over an already diverse ensemble, the marginal gain of TTA is usually smaller — commonly cited as a few parts per thousand rather than a full half-percent, though almost always positive for flippable/rotatable medical images.

## 3. Ensemble-level objective: rank vs probability averaging

The SIIM-ISIC 2020 winning solution used **rank averaging** across 18 models instead of probability averaging, on the grounds that it is more robust to overconfident models. This is a post-ensemble choice, but the write-up does **not** report the delta versus probability averaging. It is an example of the pattern Claude flagged: a widely copied trick whose gain is asserted but not measured.

Müller et al., *An Analysis on Ensemble Learning optimized Medical Image Classification* (arXiv:2201.11440, 2022), systematically compare 12 pooling functions for combining medical-image predictions. Their main finding is that **simple statistical pooling (mean, median) is usually equal or better than learning-based stacking** on held-out medical data. This cuts against spending much effort on sophisticated ensemble-level objectives once a sensible average is already in place.

## 4. Pseudo-labeling / confident learning / label denoising

The RSNA 2024 2nd-place write-up (Brendan Artley) describes using an out-of-fold team ensemble to generate pseudo labels, then training with `0.5 * loss_with_labels + 0.5 * loss_with_pseudolabels` and removing noisy labels via confident learning. This is a powerful technique, but it is **training-time**, not something applied after the ensemble is frozen. It cannot rescue a fixed ensemble.

## 5. Label-space structure (hierarchy, mutual exclusion, co-occurrence)

RSNA 2024 Lumbar Spine labels are ordinal severity grades (normal / moderate / severe) for spinal canal stenosis, subarticular stenosis, and neural foraminal narrowing. Top solutions often treat this with task-specific heads or losses, but this is again model-training or architecture design, not a post-ensemble output constraint. I found no published competition solution that reports a 0.5–1% gain from enforcing label hierarchy on top of an already-built ensemble.

## 6. Test-time distribution adaptation

Adapting normalization statistics or pseudo-labeling the test set is rarely reported in top medical-imaging competition write-ups. Many competitions forbid test-set training, and the risk of test-set overfitting is high. Tent-style test-time entropy minimization is an active research area (see Wang et al., ICLR 2021; Nado et al., 2020), but it is not a documented source of the last percent in the competition solutions I could access.

## 7. Exploiting the metric's shape

Nilay et al.'s per-source threshold calibration is the clearest example of metric-aware post-processing carrying a large delta. Beyond that, competition solutions rarely write down the marginal gain of metric-aware losses once the ensemble is fixed. Most metric-aware choices (e.g., optimizing AUC surrogate losses, ordinal regression for severity labels) are made during training.

## Take-aways for the plateau

- The single most evidence-backed post-ensemble lever is **per-label or per-source threshold optimization** on out-of-fold predictions. It requires only a holdout set and a metric that is threshold-sensitive.
- **TTA** is cheap and almost always helps, but expect a small marginal gain over an ensemble that already contains diverse members.
- **Rank averaging / learned ensemble weights** are commonly used but rarely shown to beat a simple mean once the ensemble is strong; Müller et al. suggest simple means are already near-optimal.
- Pseudo-labeling, label hierarchy, and test-time adaptation are training-time levers, not fixes for a frozen ensemble.
- The literature supports a sober view: after the seating axis is closed, the remaining 0.5–1% is more likely to come from calibration/threshold tuning and small TTA gains than from any unmeasured "trick."

## Sources

- Nilay et al., *A Heterogeneous Ensemble for Multi-Center COVID-19 Classification from Chest CT Scans*, arXiv:2603.14621v1, 2026.
- Rajaraman et al., *Deep learning model calibration for improving performance in class-imbalanced medical image classification tasks*, PMCID PMC8794113, 2022.
- Ha et al., *Identifying Melanoma Images using EfficientNet Ensemble: Winning Solution to the SIIM-ISIC Melanoma Classification Challenge*, arXiv:2010.05351, 2020.
- Moshkov et al., *Test-time augmentation for deep learning-based cell instance segmentation*, PMCID PMC7081314, 2020.
- Müller et al., *An Analysis on Ensemble Learning optimized Medical Image Classification*, arXiv:2201.11440, 2022.
- Brendan Artley, *RSNA 2024 Lumbar Spine Degenerative Classification — 2nd place solution*, GitHub, 2024.
