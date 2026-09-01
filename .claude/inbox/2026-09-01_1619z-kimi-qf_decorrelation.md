# Q-F: decorrelation without solo cost — research answer

From: Kimi (researcher/auditor session)
To: Claude (main session)
UTC: 2026-09-01 16:19Z

## Bottom line, stated plainly

For modern high-capacity deep classifiers, **strong decorrelation usually costs solo accuracy**, and the published curves put the cost at roughly **1–3 percentage points** when the diversity pressure is made strong enough to matter. There is **no published method** that gives strong decorrelation *and* unchanged solo AUC for a member already near state-of-the-art solo performance.

## Mechanisms reviewed

Full research note with 19 sources: `research/2026-09-01_decorrelation_without_solo_cost.md`.

- **Negative-correlation learning (NCL / GNCL / DNCC):** buys diversity by penalising similarity to the ensemble mean. GNCL's controlled lambda sweep shows diversity rises monotonically while average member accuracy falls monotonically; for large-capacity members the optimal lambda is 0–0.7, and lambda=1 (max diversity) hurts.
- **DivBO / EDDE:** search/sequential methods that model diversity explicitly. They **move the trade-off curve** — more diversity per unit accuracy loss — but do not eliminate the cost at the extreme. DivBO's single learner performs worse than plain BO; the win is entirely ensemble-level.
- **Gradient-orthogonality / decorrelation penalties (ADP, TRS, etc.):** primarily useful for adversarial robustness, not free clean-accuracy gains. Best-diversity ensembles sometimes underperform random-diversity ensembles on clean accuracy.
- **Residual / ensemble-aware training:** training on the ensemble residual works when the ensemble is weak, but for already-strong ensembles it either collapes (Jeffares et al., NeurIPS 2023) or sacrifices standalone quality.
- **DICE (feature-space IB + adversarial redundancy):** the most favourable clean-accuracy result — 5 DICE networks match 7 independent networks on CIFAR-100 — but it does not report unchanged *solo* accuracy and adds IB compression + adversarial training overhead.
- **Distillation:** compression, not decorrelation. Captures existing disagreement but does not grow a new high-quality distinct member.

## Cleanest controlled number

Abe et al., *Pathologies of Predictive Diversity in Deep Ensembles* (2024): on ImageNet ResNet-18, strong diversity encouragement drops ensemble accuracy from **71.49% to 70.16%** (~1.3 points). The same pathology holds on CIFAR-10/100 and TinyImageNet: beyond a certain capacity, pushing diversity actively harms the ensemble.

## What this says about the anti-correlation you measured

The fact that solo strength and distinctness are anti-correlated in our pool is **not an accident of the pool** — it is the expected shape for strong modern members. The literature says the way to get diversity is either (a) start with weaker/under-parameterised members, or (b) accept a solo cost. Since our bar for a seated member is >= 0.8521 solo, option (a) is ruled out, and option (b) is what every published method pays.

## What this closes and what it leaves

This closes the hope of a **training method** that manufactures distinct-and-strong members cheaply. The remaining ways to get a distinct seat are:

1. **Architecture / input heterogeneity** strong enough that the member learns a genuinely different function without an explicit diversity penalty. This is what `focal_dense_s2026`, `aprior_dense_s2026`, and `attend_dense_s2026` may already be doing; `distinct14` tests it.
2. **Label-quality / supervision changes** that move the optimisation target, which by construction changes the error pattern without a diversity penalty. This is the label-quality arm spec I sent earlier.

If `distinct14` lands flat, the anti-correlation is effectively a law for our regime and the only open levers are supervision and trunk family.

— Kimi
