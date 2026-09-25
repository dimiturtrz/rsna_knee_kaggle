# Q-EF — How far below the incumbents can an ensemble member be and still pay?

Claude's Q-EF (2026-09-04T1640Z) asks about the boundary at which a weaker-but-diverse ensemble member stops helping. The trigger is `radiodino_s16` at ~0.8105 AUC against `band_dense` at 0.8578 — a 4.7-point gap — and whether the whole RadioDino trunk lane deserves seats or only rows.

## The source of the "~5 point" rule

The only direct measurement in the project notes is the NVIDIA safety-diversity paper (Filipiuk & Singh, IJCAI-ECAI AISafety 2022). They ensemble pairs of ImageNet-21k models on ImageNet-C and report:

- MLP-Mixer **Large** is **5 percentage points** less accurate than MLP-Mixer **Base**.
- Despite that, "MLP-Mixer L ensembles perform slightly better than all MLP-Mixer B ensembles."
- But they also note that the Large and Base Mixer are not actually very different in CE-diversity terms — the Large's apparent diversity comes mostly from its higher error rate, not from genuinely different predictions.
- Their conclusion: "High diversity does not translate to an improved performance of their ensemble ... a discrepancy in accuracies causes [diversity and ensemble potential] to misalign." ([NVIDIA paper](https://ceur-ws.org/Vol-3215/16.pdf))

## Answers to the four sub-questions

### 1. Is the ~5-point boundary absolute or relative?

**Absolute, in the only paper that states it.** The 5 pp is literally 5 accuracy points on ImageNet / ImageNet-C. The authors do not express it as a fraction of headroom, a fraction of the best model's accuracy, or a scale-free quantity. The single-model accuracies in that paper are in the ~70–90% range, so 5 pp is not a tiny fraction, but it is not stated as "5% of the remaining error" either.

For AUC the translation is even less clear. A 5-point AUC gap near 0.86 is a larger fraction of the remaining headroom than a 5-point accuracy gap near 0.90, and AUC compresses errors differently than top-1 accuracy. No published conversion exists.

### 2. Is there a published case where a member 4–5 points below the best still improved the ensemble?

**Yes, but the gains are marginal and the evidence is not leave-one-out.**

- **NVIDIA MLP-Mixer L vs B (5 pp gap):** the L+B ensemble is "slightly better" than B+B. No numeric delta is given.
- **NVIDIA MLP-Mixer Base + ConvNeXt-XL on contrast corruption:** the Mixer Base is "almost 20 p.p. worse than ViT-L" yet the cross-architecture ensemble is "marginally better." This is a much larger gap than 5 pp, but again the improvement is tiny and attributed to the specific corruption, not to a general rule.
- **No named leave-one-out ablation was found** that removes a specific 4–5 pp weaker member and shows the ensemble degrades. The NVIDIA paper reports pair-wise ensemble tables, not leave-one-out studies.

So the boundary is a **tendency, not a wall**: weak members can still nudge the ensemble up, but the nudge is small and depends on whether the weakness comes with genuine prediction differences.

### 3. Does the boundary move with ensemble size?

**Theory says yes, in the direction of your intuition, but with a caveat.**

- **Durrant & Lim (AISTATS 2020)** model majority-vote ensemble accuracy with a Polya-Eggenberger distribution parameterized by average member accuracy `p` and pairwise error correlation `r`. As ensemble size `N` grows with `r > 0`, the ensemble accuracy converges to a beta-distribution limit and the marginal gain of adding one more member shrinks. A weak member's damage is capped by `1/N`, but so is the benefit of a decorrelated member unless its errors are genuinely negatively correlated. ([Durrant & Lim](https://proceedings.mlr.press/v108/durrant20a/durrant20a.pdf))
- **Wood et al. (JMLR 2023)** give the general decomposition:

  ```
  expected ensemble risk = average bias + average variance − diversity
  ```

  Adding a member changes all three terms. In a large ensemble, the diversity contribution of one more member is diluted unless it provides strong negative correlation. ([Wood et al.](https://jmlr.org/papers/volume24/23-0041/23-0041.pdf))

So: a weak-but-decorrelated member is *more affordable* in an 8-member mix than in a pair, because its weight is `1/8` and its errors average against seven others — but only if the errors are actually decorrelated. If the weak member is just a noisier version of the same predictor (like MLP-Mixer L vs B), adding it adds noise regardless of `N`.

### 4. Does anything measure this on AUC or with a rank-based combiner?

**Not found.**

- NVIDIA uses softmax probability averaging on ImageNet-C.
- Medical-imaging ensemble studies in the project notes (Thapar et al. COVID CT, Ashraf et al. ChestX-ray14, Nakata et al. liver ultrasound) report accuracy or AUROC with probability/weighted averaging or stacking, not rank averaging.
- The Durrant model is for majority vote (0/1 loss).
- A rank-based combiner is insensitive to calibration in a way that probability averaging is not, so a poorly calibrated weak member could be less harmful under rank averaging. But this is **conjecture** — no measured case was found.

## Implication for `radiodino_s16`

- The 4.7 AUC-point gap is right at the only published absolute boundary (5 pp). There is no scale-free form to tell us whether that boundary is softer or harder at AUC 0.86 than at accuracy 0.90.
- The marginal cases in the literature (MLP-Mixer L, the 20pp contrast case) show that weak members can still help, but the help is marginal and usually comes from genuine cross-architecture differences, not from a weaker variant of the same family.
- With eight champion seats, a weak-but-decorrelated RadioDino seat is more affordable than it would be in a pair, but the gain is likely small unless its errors are genuinely different from the ViT/DINOv2 pool.
- **Practical recommendation:** before committing a seat, measure the leave-one-out delta on a clean holdout. The literature does not supply a reliable threshold; the only safe signal is an empirical one.

## Sources

- Filipiuk & Singh, *Exploring Diversity in Neural Architectures for Safety*, IJCAI-ECAI AISafety 2022. https://ceur-ws.org/Vol-3215/16.pdf
- Ortega et al., *Diversity and Generalization in Neural Network Ensembles*, AISTATS 2022. https://proceedings.mlr.press/v151/ortega22a/ortega22a.pdf
- Wood et al., *A Unified Theory of Diversity in Ensemble Learning*, JMLR 2023. https://jmlr.org/papers/volume24/23-0041/23-0041.pdf
- Lim & Durrant, *A Diversity-aware Model for Majority Vote Ensemble Accuracy*, AISTATS 2020. https://proceedings.mlr.press/v108/durrant20a/durrant20a.pdf
