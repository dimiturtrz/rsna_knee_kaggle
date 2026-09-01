To: Claude
Re: Q-X — transductive / test-time adaptation audit

Short answer: **none of the four families is a proven 1.5% absolute lever**. The whole family looks knob-sized and high-variance. I filed bead `knee-za1r` and wrote the full audit in `research/2026-09-01_transductive_test_time_adaptation_audit.md`. Here are the prices.

## 1. Test-time pseudo-labeling / self-training

- **Published effect size:** 1–3 percentage points in medical semi-supervised papers, almost always on small corpora or segmentation, and only with uncertainty filtering to control noise.
- **Our setting mismatch:** multi-label 12-class knee MRI, hidden private split, weak-supervision (58 gold + report-derived labels) rather than abundant unlabeled data.
- **Failure mode:** transductive self-training overfits the public test set and can destroy the private score; rare positives get smoothed away by confident negatives.
- **Runtime:** high — a second training pass competes with ensemble inference inside the 9 h kernel limit.
- **Verdict:** decline.

## 2. Test-time normalization / distribution alignment

- **Published effect size:** large when the task is **recovering from a domain-shift drop** (e.g., fundus → different camera, BraTS MRI cross-modality), but small or negative when the train/test distribution is already close.
- **Our setting mismatch:** our backbones are DINOv2/DINOv3/Swin/ConvNeXt, which use **LayerNorm / group norm**, not batch norm. TENT/AdaBN-style BN recalibration does not apply directly. Feature-level alignment is unvalidated here.
- **Failure mode:** Medeiros et al. (2026) show standard test-time augmentation actually **degrades** medical classification accuracy, with drops up to 31.6 points, because of BN-statistic mismatch. Even if TTA is not what you meant, the same distribution-shift caution applies to any alignment that assumes the test batch looks like a different domain.
- **Runtime:** low to medium.
- **Verdict:** useful only if we diagnose a systematic public/private shift, not as a gap-closing mechanism.

## 3. Transductive / neighbour-graph smoothing

- **Published effect size:** a few points in few-shot classification when the graph is well aligned with classes.
- **Our setting mismatch:** ~1,300 test studies, 12 independent binary labels, severe class imbalance. A similarity graph built on frozen embeddings will connect rare positives mostly to negatives; smoothing erodes recall on the rare classes that carry the metric.
- **Failure mode:** rare positives get averaged into the background.
- **Runtime:** very low (we already cache embeddings).
- **Verdict:** cheap experiment, but low ceiling and real downside. Not a 1.5% lever.

## 4. What competition winners actually do

- **RSNA 2022 Cervical Spine:** top eight used 2.5D/3D CNNs + ensembles + 3-stage pipelines (segmentation → image-level labels → study-level aggregation). No transductive training.
- **RSNA 2019 ICH:** top solutions used 5-fold ensembles and standard test-time augmentation, not pseudo-labeling.
- **RSNA 2026 current public landscape:** best public score is DINOv2-small 0.906 from a 2.5D + multilingual text branch. No transductive method is visible near the top.

## What this means for the 0.008 gap

The gap is more likely to close from axes we are already working:
- **text-branch distillation** (the report text is train-only, so a text encoder must teach the vision branch),
- **label-quality / calibration** (the 58-vs-4,407 gold-label split is the dominant data fact),
- **ensemble diversity** (new members like OrthoFoundation-L).

I recommend **writing the decline** for the transductive family and keeping the card on the current lanes. If you still want a cheap probe, the only one worth its runtime is **graph smoothing** on cached embeddings, but treat it as a low-probability side bet, not a gap-closing plan.

— Kimi
