# Cross-vendor domain shift — what a sibling project already measured, and what it prices here

Provenance: this is NOT our measurement. It is the `cardiac-seg` project's, on cardiac cine MRI
(ACDC / M&Ms-1 / M&Ms-2, held-out vendors Canon and GE). Primary source is that repo's
`cardioseg/preprocessing/normalization/README.md`, which carries the numbers and the bead ids
(`cardiac-seg-{jp1,chm,h8k,276,bgc}`). It is filed under `research/` because relative to kneemri it is
somebody else's result, and it is filed at all because it prices two levers that are open here.

## The framing it supplies (standard vocabulary, worth adopting)

Domain shift decomposes (Moreno-Torres 2012) into **covariate shift** (P(x) moves, P(y|x) does not —
the scanner/intensity case), **label/prior shift** (P(y) moves — disease mix per site), and **concept
shift** (P(y|x) moves — annotation protocol). The MRI-specific name for scanner-induced covariate
shift is **acquisition shift**, also scanner / site / vendor effects. Training for *unseen* domains is
**domain generalization**; adapting to a known target is **domain adaptation**.

Three strategy families: **strip** (harmonization / normalization — ComBat, Nyul, N4, z-score),
**diversify** (augmentation / domain randomization), **adapt** (not our setting, we hold the target
out). Their rule, which we should inherit: **per factor pick ONE direction** — stripping and augmenting
the same axis adds back what was just removed.

MRI intensity is uncalibrated: unlike CT Hounsfield units, a pixel value depends on coil gain, recon
auto-scaling and sequence weighting, so the same tissue reads differently across scanners. That is the
physical reason the axis exists at all, and it applies to knee MRI unchanged.

## The five nulls — normalization is not the lever (their measurement)

1. **Nyul histogram standardization: NULL.** 0.857 vs 0.864 cross-vendor Dice, -0.007, inside noise;
   EF unchanged. Built, measured, kept off by default.
2. **Blood-anchored two-point affine harmonization: NULL** despite a real input-level effect — it
   halved cross-vendor tissue-level spread against z-score (myocardium 0.558 -> 0.262) and did not move
   Dice. Run with oracle ground truth, i.e. at the method's upper bound.
3. **The whole normalization grid is FLAT.** instance-norm + {z 0.861, blood 0.862}; no-norm +
   {z 0.870, blood 0.851}. Span 0.019 = single-seed scatter. Their conclusion, which is the sentence to
   carry: **normalization — input OR feature-layer — does not move the model-class ceiling.**
4. **N4 bias correction REGRESSED on their test** (ACDC EF 6.5 -> 7.3%, Canon 11.9 -> 15.0%, GE
   11.3 -> 13.5%). Retained opt-in anyway, with the finite-test caveat stated explicitly: their held-out
   set is bias-mild, so "N4 regressed" means "our test does not reward N4", not "N4 is useless".
5. **Bias-field AUGMENTATION regressed too** (Canon Dice 0.839 -> 0.827, ECE 0.172 -> 0.192,
   error-detection AUPRC 0.600 -> 0.563) — "confidently wrong, not honestly uncertain". Both directions
   of the same factor hurt, which is what makes the duality experiment itself the result.

## The two findings that price OUR open levers

**A. Same-recipe ensembling is structurally blind to distribution shift.** Deep-ensembling bought
nothing on unseen vendors — Canon Dice +0.000, GE +0.006, inside noise. The mechanism is measured, not
asserted: BALD uncertainty decomposition puts only ~31% of Canon uncertainty as epistemic under a weak
TTA estimate, and a 4-seed deep ensemble puts it at ~15-18%. The seeds AGREE, so more of the same
recipe cannot help.

This is an independent, differently-sourced mechanism for **our seven board flats** — ensembles that
gain on our holdout and do not move the board. Our holdout is a random split over our own scanner mix,
so it is in-distribution by construction; the Kaggle test set need not be. An ensemble selected on an
in-distribution ruler is exactly the object their BALD numbers say has no headroom off-distribution.
It does not prove that is what happened to us. It says the mechanism is real, has been measured once,
and is testable here without a card (see the interpretation, and knee-vendor-ruler).

**B. What DID close their gap was MODEL CLASS, not normalization.** nnU-Net at a deliberately weak
setting (50 epochs, 1 fold, 2D) reached Canon EF 2.6% and GE EF 4.3% against their 11.9% / 11.3% — a
4.5x reduction on the held-out vendor. Read carefully, this is not an architecture claim: nnU-Net's
distinguishing property is that it SELF-CONFIGURES its recipe (resampling, per-image z-score,
augmentation) from a dataset fingerprint. So the transferable statement is about recipe planning, not
about a particular network. Our own trunk axis is closed on our numbers (~25 foreign trunks, all below
the natural-image control), and this result does not re-open it.

## What this prices, for us, before any card is spent

- **A scanner/intensity harmonization arm is priced ~0.** Five confirmations, one of them at the
  method's oracle upper bound. Do not build ComBat, Nyul, or a vendor histogram match for kneemri on
  the strength of the shift existing.
- **We already run z-score plus intensity augmentation**, which their finding 3 says captures the
  harmonizable variance. Adding a strip on top of an axis we augment violates their one-direction rule.
- **The finite-test caveat cuts our way too, and is the actionable half.** Their test lacked the
  bias-field axis; ours lacks the vendor axis, because our 909-study holdout is a RANDOM split over the
  same scanner mix as the training set. So we cannot currently measure whether anything we do
  generalizes across scanners — not because the effect is absent, but because the ruler has no such
  axis. Building that ruler is card-free and is the open work.

## What is NOT settled by any of this

Their task is segmentation with a dense oracle-supervised loss; ours is twelve-column study-level
classification scored by mean ROC AUC. Prior shift (their P(y) row) is a different bucket from the
covariate shift these five nulls close, and it is the bucket our own `merged3_corpus` rotation and the
fastMRI+ all-negative census both live in. Nothing above prices per-source conditioning or loss masking
on partially-supervised columns; that question is out with Kimi.
