# The holdout is not blind above 0.870 — it is compressed, and the compression is saturation

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

| stratum | studies | disagreement | slope |
|---|---|---|---|
| Q1 (easiest) | 182 | 0.0764 | **+0.42** |
| Q2 | 182 | 0.1014 | +0.58 |
| Q3 | 182 | 0.1147 | **+0.74** |
| Q4 | 182 | 0.1237 | **+0.74** |
| Q5 (hardest) | 181 | 0.1382 | +0.66 |

Independently, splitting the 443 resolved entry PAIRS at the median hard-share: a gain made on HARD studies
transfers at **+0.71**, one made on EASY studies at **+0.62**.

## What it says

**SATURATION, not a distribution gap, and not blindness.** The compression is concentrated exactly where
saturation predicts it — the easy stratum, where every entry is already near ceiling and nothing separates
them. A distribution gap would have put the cleanest ordering in Q5; Q5 instead reads 0.66, *below* Q3 and
Q4. The pair split says the same thing from the other side and at a smaller effect size.

**The blindness branch is refuted.** Every stratum reads a positive slope, and three of five read 0.58-0.74.
There is no band in which offline movement buys nothing. So "improve the offline number" remains a funded
plan — at about **0.7 of face value**, not at 1.0 and not at 0.

**The single median understates the mid-band and overstates the easy band.** Quoting one constant reports
the average of two regimes as if it were one, which is the failure `transfer_strata.py:14` names. Where an
arm's gain lands now matters as much as its size.

## What follows for planning

**Price an offline delta at ~0.7, and ask WHERE it landed before pricing it at all.** A +0.005 macro gain
concentrated on easy studies is worth about +0.002 of board; the same +0.005 on the mid-to-hard bands is
worth about +0.0037. That is nearly a factor of two, and it is invisible to the macro alone.

**This raises, not lowers, the bar on knob-sized arms.** The project's significance bar is ~1.5% absolute.
Against a 0.7 slope an offline +0.006 is a board +0.004 — still under the bar, still directional only. The
owner's standing correction holds unchanged: *little changes are usually within noise*. What the stratum cut
adds is that a small change on the EASY band is worth even less than its already-small face value.

**It does not license a seating operation.** The member axis is closed on the board with seven flats, and
nothing here is evidence about seating. It is a pricing rule for readings we already take.

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
SATURATION and separate it from stagnation — models becoming indistinguishable near a ceiling, which is what
a 0.42 slope in the easiest fifth of the holdout looks like.
