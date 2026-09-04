# Q-EG — what is actually known about multi-label noise correction?

Claude asked four specific things about correcting noisy labels in a multi-label
setting.  This note is a literature scan targeted at those four questions.

## Sources checked

- **BoMD** — Chen et al., *Benchmarking Medical Vision Language Models:
  Are We on the Right Track?* / *Beware of Medical Image-Text Memorization*,
  ICCV 2023 / related medical-VLM work.  The only prevalence-related figure is
  a histogram of label distributions before/after re-labelling (Figure 4).
- **HLC** — Xia et al., *Holistic Label Correction for Noisy Multi-Label
  Classification*, ICCV 2023.  Code at
  [xiaoboxia/HLC](https://github.com/xiaoboxia/HLC).
- **"Deep Active Re-Labeling"** (2026) — notes limited opportunity for
  improvement when the label noise rate is low.
- **"On the Effects of Different Types of Label Noise in Multi-Label
  Classification"** (Burgert et al., arXiv 2022) — evaluates synthetic noise
  rates from 10% to 70%.
- **"From Label Smoothing to Label Relaxation"** (AAAI 2021) and
  **"Structural Label Smoothing"** (Li et al., AISTATS 2020) — discuss how
  smoothing changes the label distribution, not how to preserve it.

## Answer to each question

### 1. Per-label prevalence change beside per-label accuracy?

**Not reported.**  BoMD shows a histogram (Figure 4) of how often each label
appears before and after re-labelling, but it does not give a per-label
table of prevalence shift.  HLC reports mAP / OF1 / CF1 at the dataset level
only.  In both cases the reader can see whether labels were added or removed
in aggregate, but cannot read off `Δp` per class.

### 2. Cross-head transfer when one label is corrected and the rest are reported?

**Not measured.**  HLC corrects all labels jointly via a GCN that models label
dependence.  There is no ablation that fixes label A and measures the effect
on labels B..L while the trunk is shared.  Shared-trunk multi-label models
obviously couple gradients, but the size of the cross-head spillover from a
single-label correction is an open empirical quantity.

### 3. Does expected gain scale with noise rate, and is there a published curve?

**Yes, but only at high noise.**  HLC tests symmetric noise at 30%, 40%, 50%
and pairflip noise at 20%, 30%, 40%.  Burgert et al. extend the range to
10%-70%.  The general trend is that correction methods separate from the
baseline more as noise rate rises.  At very low noise rates the gains shrink;
"Deep Active Re-Labeling" explicitly says there is "limited opportunity for
improvement" when noise is rare.

Our detector is targeting **3.65% of cells** — far below the 10-20% floor of
the published curves.  That does not mean correction is useless, but it does
mean the published gains are not directly transferable.

### 4. Is there a standard mean-preserving corrector?

**No.**  The usual tools — label smoothing, bootstrapping (soft targets),
self-distillation — deliberately change the marginal distribution.  A
prevalence-preserving correction appears to be a custom requirement, not a
standard method.  The closest conceptual relative is calibration, but
calibration adjusts confidence scores rather than hard labels.

## Bottom line

- Questions 1, 2, and 4 appear to be **genuinely unaddressed** in the
  multi-label noise-correction literature.
- Question 3 has a clear answer: **gains scale with noise rate**, and the
  measured regime starts around 10-20% synthetic noise, well above the
  3.65% cell-level rate we are correcting.
- If the new arm deliberately changes marginals, that is a design choice
  without a direct precedent to cite — it should be evaluated on its own
  board contribution rather than justified by an external standard.
