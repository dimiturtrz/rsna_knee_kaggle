# The holdout is not blind above 0.870 — and the band-to-band slope spread is mostly an artefact

**AMENDED 2026-09-08 09:35Z, and the amendment retracts this file's original mechanism.** It was published
at 09:20Z claiming the compression was SATURATION, concentrated in the easy stratum. That claim does not
survive the first check anyone ran against it (D195 §3, then my own per-band correlation). The finding that
mattered — blindness is refuted — survives untouched, and is the only thing here that should be cited.
The original headline is preserved below under "What was published, and why it was wrong".

2026-09-08. Answers the owner's standing question: *"how do you know that the small reduction (below
noise) of the score for mrnet is not paid for with better generalization? and that the kaggle public
score and final score is the same as your offline benchmark"* — the second half of it. Bead `knee-6j8b4`.

## The fork this settles

`board_transfer` reports one median ratio over all entry pairs. `kneemri/evaluation/transfer_strata.py:17`
states why that shape cannot be planned with: two readings fit the seven fold-0-honest entries equally
well, and they predict **opposite** things about the same next arm.

* **Uniform ~0.8** — another +0.005 of holdout is worth +0.004 of board, and the next arm is worth running.
* **Near 1 low down, ~0 in the band our last five entries sit inside** — the holdout is BLIND above 0.870
  and every plan ending in "improve the offline number" is unfunded.

The module cuts the holdout into difficulty strata and asks the transfer question inside each, because the
three candidate causes of a sub-unity slope disagree about that cut: a DISTRIBUTION GAP recovers the
ordering in the HARD stratum; SATURATION compresses worst in the EASY stratum; NEITHER reads ~1 everywhere.

Difficulty is measured **without the labels** — per-study member disagreement across a fixed reference pool
of 41 members, the union of every member any honest entry seats. A truth-aware ruler would make the hardest
stratum mostly-errors by construction and read as saturation whatever was true.

## The reading

37 rebuildable ensemble entries, 5 bands of ~182 studies, 12 of 12 findings scoreable in every band
(`logs/transfer_strata.log`, run 2026-09-08 09:12Z):

| stratum | studies | disagreement | pair ratio |
|---|---|---|---|
| Q1 (easiest) | 182 | 0.0764 | **+0.42** |
| Q2 | 182 | 0.1014 | +0.58 |
| Q3 | 182 | 0.1147 | **+0.74** |
| Q4 | 182 | 0.1237 | **+0.74** |
| Q5 (hardest) | 181 | 0.1382 | +0.66 |

(The column is `compression()`'s median pair ratio, not a fitted slope; see the amendment below for why the
distinction stopped mattering — both estimators share the same denominator defect.)

Independently, splitting the 443 resolved entry PAIRS at the median hard-share: a gain made on HARD studies
transfers at **+0.71**, one made on EASY studies at **+0.62**.

## What was published, and why it was wrong

The original reading was: *"SATURATION, not a distribution gap. The compression is concentrated exactly
where saturation predicts it — the easy stratum, where every entry is already near ceiling and nothing
separates them."* Two independent checks refuse it.

**1. Saturated entries BUNCH. Q1's are the most spread, and the lowest-scoring.** D195 §3 decomposes member
macro by the same bands: Q1 mean **0.7773**, the LOWEST of the five, with the LARGEST sd (0.0213). At the
entry level the same holds — Q1's 37 offline scores carry sd **0.0145** against Q5's 0.0094. A band where
nothing separates the entries is a band with a small spread near a ceiling. Q1 is the opposite of that on
both counts, so the ceiling mechanism is not merely unproven here, its own signature points the other way.

**2. The band ordering is very nearly the offline-SPREAD ordering.** Both estimators divide by a per-band
offline spread, and that spread is what varies. `transfer_strata.py:233 compression()` reports the median
board-per-holdout ratio over the PAIRS a band resolves; an OLS fit over the same 37 entries is a second,
independent estimator. They agree, and the added columns say why:

| stratum | pair ratio (the tool) | sd(offline) | sd(board) | **OLS r** | OLS slope |
|---|---|---|---|---|---|
| Q1 | +0.42 | 0.0145 | 0.0080 | 0.822 | +0.455 |
| Q2 | +0.58 | 0.0100 | 0.0080 | 0.870 | +0.697 |
| Q3 | +0.74 | 0.0084 | 0.0080 | 0.871 | +0.836 |
| Q4 | +0.74 | 0.0079 | 0.0080 | **0.748** | +0.757 |
| Q5 | +0.66 | 0.0094 | 0.0080 | 0.906 | +0.772 |

Board score is a per-entry constant, so `sd(board)` is identical in every band and the OLS slope is
`r · sd(board)/sd(offline)` — driven by the denominator unless r moves. The scale-free measure is
flat-to-noisy across the difficulty axis: 0.75 to 0.91, with the LOWEST value at Q4, in the MIDDLE. The
"1.8x compression in the easy band" I published is the 1.84x offline-spread ratio (0.0145 / 0.0079) wearing
a mechanism's clothes. The pair-ratio column inherits the same defect by a shorter route — a band whose
per-pair `delta offline` is noisier has a noisier DENOMINATOR, which drags the median ratio down with
nothing about transfer having changed.

**The alternative that does fit is regression dilution.** A noisier per-entry offline estimate inflates
`sd(offline)` and attenuates the slope toward zero, which is exactly the Q1 pattern (widest spread, lowest
mean, second-lowest r). It is not fully established either, and the bound is worth stating: if ALL of Q1's
excess spread over Q5 were noise, the disattenuated `r` would exceed 1, which is impossible — so part of
Q1's spread is real. The honest position is that **the band-dependence of the slope is unexplained, is
consistent with a measurement artefact, and must not be planned against.**

## What survives

**The blindness branch is REFUTED, and this is the whole of the result.** `transfer_strata.py:19` named the
live possibility that the holdout says nothing above 0.870 — which would leave every plan ending in
"improve the offline number" unfunded. Every stratum reads a positive slope; every stratum reads a
correlation of 0.75 or better. There is no band in which offline movement buys nothing.

**Price an offline delta at ~0.7 of face value, uniformly.** The pooled figure stands. What is RETRACTED is
the follow-on rule that a gain should be re-priced by WHERE it landed (the published version said easy-band
gains were worth ~half of mid-band ones). The pair split — hard +0.71 vs easy +0.62 — is the same
comparison at a smaller effect size and carries no interval; it is directional at best and is not
independent evidence for a rule the band table just failed to support.

**This raises, not lowers, the bar on knob-sized arms.** Against a 0.7 slope an offline +0.006 is a board
+0.004 — under the ~1.5% bar, directional only. The owner's standing correction holds unchanged: *little
changes are usually within noise*.

**It does not license a seating operation.** The member axis is closed on the board with seven flats, and
nothing here is evidence about seating.

## The member-side companion (D195, 2026-09-08 09:30Z)

Asked whether a member's hard-band profile is a second axis: **REDUNDANT**, and closed on a mechanism.
`hard_macro@holdout909-Q3Q5/midpoint` vs `macro@holdout909/midpoint`, competent pool (overall >= 0.82,
n=219): pearson **r = 0.9777**, slope 1.243, residual sd 0.0029. The `hardtop` = Q4-Q5 spelling reads
r = 0.9675, slope 1.356, residual sd 0.0038. Split-half reliability of the hard residual is **+0.2493**
(Spearman-Brown +0.3991), so ~60% of the observed residual is measurement noise and the TRUE per-member
residual sd is **0.0020** — against a 0.0062 paired floor. There is no member the profile can promote.
Slope > 1 means the hard bands AMPLIFY the existing ordering rather than reorder it.

**Quote the competent-pool numbers only.** The full-pool r = 0.9816 is inflated by two broken members
(`focus_latmen_s2026` 0.4747, `focus_mcl_s2026` 0.5332) whose enormous residuals replicate perfectly
because being broken is a stable property.

**Spelling.** `transfer_strata.py:269` defines `hard` as `self.strata()[self.bands // 2 :]` — with
bands=5 that is **Q3-Q5**. `hardtop` is reserved for Q4-Q5. My M197 prose said "Q4/Q5" while pointing at
the code's `hard`; the two give different numbers and the verdict is the same on both.

## What it cannot say

The strata are ~180 studies each, so a per-label AUC inside one is wide; the report prints scoreability for
that reason (12 of 12 in every band here, which is the good case). The 37 entries span days and a kit that
changed underneath them — stratifying does not repair that confound, it only asks the confounded question
where the two hypotheses differ. And Q5's dip below Q3/Q4 is a single non-monotonicity at the smallest,
widest-AUC stratum: it is consistent with noise and is **not** read here as a hard-stratum effect of its own.

The residual-scale companion question — how much scatter is left around the fitted line, and whether
multi-site entries sit off it — is the separate reading owed under the same bead and is not answered here.

## Prior art

Recht et al. (arXiv:1806.00451, ICML 2019) measure this quantity on CIFAR-10.1 and ImageNetV2 and report
slopes **above** one (1.62, 1.11), read as evidence against test-set adaptivity. Ours runs the other way, so
adaptive overfitting is not the story. Roelofs et al. (NeurIPS 2019), over 120 Kaggle competitions, describe
the plateau shape directly. Akhtar et al. (ICML 2026, arXiv:2602.16763) name the end state BENCHMARK
SATURATION and separate it from stagnation — models becoming indistinguishable near a ceiling. **That is
the frame this file originally reached for and it does not apply**: saturation requires entries to be
bunched near a ceiling, and the easy band is where ours are most spread and lowest-scoring. Cited here so
the next reader does not reach for it a second time.

**Method note for the next stratified reading.** A slope fitted inside a subset compares a per-subset x
against a whole-entry y, so its denominator is the subset's own spread and it is not comparable across
subsets. Report the CORRELATION alongside every stratified slope, or the spread ordering will be read as a
transfer ordering — which is precisely what happened here, within fifteen minutes of publication.
