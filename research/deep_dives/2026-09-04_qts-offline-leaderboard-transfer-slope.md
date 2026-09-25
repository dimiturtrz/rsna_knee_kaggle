# Q-TS — Offline-to-leaderboard transfer slope and top-band saturation

2026-09-04

## Bottom line for Claude

- **A sub-unity offline→public-leaderboard slope is *not* the classic adaptive-overfitting signature.** The landmark benchmark-reuse studies (Recht et al. CIFAR-10.1 / ImageNetV2) find slopes **greater than 1**: improvements on the original test set transfer *more than* linearly to a fresh test set. Your 0.82 slope runs in the opposite direction.
- **The plateau shape you see *is* a named phenomenon, but the literature usually plots it as public→private.** Roelofs et al. (NeurIPS 2019) describe the expected overfitting signature as submissions moving on the public-score axis while stagnating on the private-score axis. Akhtar et al. (ICML 2026) formalize the same compression as **benchmark saturation**: top models become statistically indistinguishable and cluster near an empirical ceiling.
- **Three plausible mechanisms for slope < 1:** (1) the public test set is systematically harder or different in a way that attenuates gains (distribution gap); (2) the public metric has limited headroom / resolution, so real improvements shrink on the board (ceiling/saturation); (3) your offline validation is slightly optimistic and the ensemble-selection process overfits it (adaptive overfitting on our side).
- **The cheapest discriminating experiment is a hard-case reweighting diagnostic on the existing holdout.** Stratify holdout cases by model confidence / difficulty, recompute per-stratum AUC for your entries, and check whether the board slope is recovered on a locally harder stratum. Zero submissions, ~1 GPU-hour of inference if it is not already cached.

## Q-TS-1 — What the classic literature says about slope

### Recht et al. on CIFAR-10 and ImageNet: slope > 1, not < 1

Recht et al., "Do CIFAR-10 Classifiers Generalize to CIFAR-10?" (2018) [[arXiv:1806.00451](https://arxiv.org/abs/1806.00451)] and the follow-up "Do ImageNet Classifiers Generalize to ImageNet?" (ICML 2019) [[PDF](http://proceedings.mlr.press/v97/recht19a/recht19a.pdf)] replicated the original dataset curation to build fresh test sets (CIFAR-10.1 and ImageNetV2). Both papers report a linear relationship between original-test accuracy and new-test accuracy with slope **greater than 1**:

| Dataset | linear fit | interpretation |
|---|---|---|
| CIFAR-10.1 | acc_new = 1.62 · acc_orig − 65.5% | every point of original gain yields ~1.6 points on the new test |
| ImageNetV2 (top-1) | acc_new = 1.11 · acc_orig − 20.2% | every point of original gain yields ~1.1 points on the new test |

The authors explicitly frame this as evidence *against* the standard adaptive-overfitting story: later models do not show diminishing returns on fresh data. Instead, the drop from original to new test is attributed to a **distribution gap** (small, hard-to-replicate differences in curation/annotation) rather than test-set overfitting.

For your data, the direction is reversed: +1 point offline yields +0.82 points on the public board. That rules out "classic test-set adaptivity" as the explanation and points to something specific about the relationship between your local validation and the Kaggle public test.

### Roelofs et al. on Kaggle: little overfitting, but plateau signature is named

Roelofs et al., "A Meta-Analysis of Overfitting in Machine Learning" (NeurIPS 2019) [[PDF](https://proceedings.neurips.cc/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf)] analyzed 120 Kaggle competitions. Their headline result is that adaptive overfitting is small in effect size (typically <1% accuracy). However, they describe the **expected overfitting signature** in the same plateau terms you observe:

> "a plateauing trend where later submissions move on the public-score x-axis but stagnate on the private-score y-axis."

This is usually interpreted as public→private shakeup, but the *shape* — later improvements compress near the top — is the same one in your offline→public table.

### Akhtar et al. on benchmark saturation: the formal vocabulary for your top-band

Akhtar et al., "When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation" (ICML 2026) [[arXiv:2602.16763](https://arxiv.org/abs/2602.16763)] define:

- **Stagnation**: top models are statistically indistinguishable.
- **Saturation**: stagnation *plus* performance near the empirical ceiling.
- **Saturation index**: an uncertainty-aware measure derived from leaderboard spread and evaluation noise.

They find that ~half of 60 LLM benchmarks are saturated, with saturation increasing in benchmark age and exposure. The key actionable point is that saturation can arise from **measurement resolution limits** and **cumulative exposure** even without contamination or a true capability ceiling. Your top five entries span +0.0047 offline but only +0.0005 board — that is exactly the loss of discriminative power they describe.

## Q-TS-2 — Three candidate explanations for slope 0.82

| Mechanism | Expected signature | Why it fits / does not fit |
|---|---|---|
| **A. Distribution gap** — public test is a harder mix | Slope < 1; relative order preserved; variance in per-case gains | Fits the direction. Kaggle medical competitions often have scanner/site/study variation between train and test. |
| **B. Ceiling / saturation** — public metric has limited headroom | Slope → 0 as scores rise; small offline deltas map to near-zero board deltas | Fits the top-band compression in your table. Does not require the test set to be different. |
| **C. Offline validation overfit** — ensemble selection tuned the holdout | Offline scores drift up without true generalization; board stays flat | Possible, but your holdout is frozen and your ensemble members are trained on different folds, so this would be modest. |

The interesting question is whether the right model is **A** or **B**, because the response differs:
- If **A** (distribution gap), the fix is to make the local validation distribution match the public test better (adversarial validation, site-stratified splits, scanner harmonization).
- If **B** (ceiling), the fix is to increase measurement resolution (larger/finer test set, different metric, or accept that leaderboard gains are now expensive and look elsewhere).

## Q-TS-3 — A cheap discriminating experiment

**Goal:** distinguish distribution-gap attenuation from ceiling/saturation using only the existing holdout predictions.

**Procedure:**

1. **Difficulty score per case.** For each study/case in the frozen holdout, compute a difficulty score. Good choices, in order of preference:
   - Ensemble disagreement: variance of predicted probabilities across your ensemble members.
   - Distance from decision threshold: `min(|p − 0.5|)` averaged over labels.
   - Prediction entropy over labels.

2. **Stratify the holdout.** Split the holdout into quintiles by difficulty (Q1 = easiest, Q5 = hardest).

3. **Recompute entry scores per quintile.** For each of your entries (spansolo_v1, holdsolo_v1, peak8, foldfour_v1, holdfive_v1, greedyseat_v1, honest12), compute AUC within each quintile.

4. **Compare slopes.**
   - If the offline→board slope is recovered on **Q5 (hardest)**, the public test likely resembles the hard stratum — a distribution-gap effect.
   - If the slope is most compressed on **Q1 (easiest)**, the public metric is hitting a ceiling where easy cases have no room to improve — a saturation effect.
   - If all quintiles show slope ~1 and only the full-board slope is <1, the effect is more subtle (e.g., case-level weighting mismatch or label noise structure).

**Cost:** zero submissions; only inference/prediction aggregation if holdout predictions are not already cached. If you need to regenerate predictions, it is one forward pass per entry.

**Follow-up if the diagnostic is ambiguous:**

A two-submission probe can break the tie:
- **Variant A**: take your best ensemble and clip all predictions toward 0.5 by a small fixed amount (e.g., `p' = 0.5 + 0.95·(p − 0.5)`).
- **Variant B**: the original ensemble.

If the board score barely drops under the conservative smoothing but the offline score drops proportionally more, the board is in a flat, saturated region. If the board drops roughly in line with offline, the board is still responsive and the slope <1 is more likely a distribution effect.

## Q-TS-4 — What I would tell the user

The 0.82 slope is unusual in the published benchmark-reuse literature, which tends to report slopes >1. The *shape* of your top-band is well documented (Roelofs plateau; Akhtar saturation), but the direction suggests something specific to this competition's validation/test relationship. Run the quintile diagnostic before spending another submission slot on marginal ensemble gains — it will tell you whether the next meaningful lever is validation-set matching or whether the leaderboard is simply saturated at the current resolution.
