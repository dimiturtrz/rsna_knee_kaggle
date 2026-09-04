# The ruler is not blind above 0.870 — and the difficulty story it suggested does not hold

`python -m kneemri.evaluation.transfer_strata`, 2026-09-04. 34 rebuildable `ensemble` entries, every
member scored on its own holdout record, cut into five equal-count difficulty bands by member
disagreement (no labels read). Slope is the median board-delta-per-holdout-delta over the pairs each
band resolved, the same arithmetic `board_transfer` runs over the whole holdout.

    band   studies   disagreement   slope
    Q1  (easiest)  182   0.0771   +0.44
    Q2             182   0.1018   +0.59
    Q3             182   0.1153   +0.81
    Q4             182   0.1246   +0.68
    Q5  (hardest)  181   0.1389   +0.73

## What it settles

The whole-holdout slope of 0.82 was being read as one constant, and the seven-rung honest ladder made
that constant look like it might be **zero** in the top band — which would have meant the ruler was
blind above 0.870 and every plan ending in "improve the offline number" was unfunded. That reading is
wrong, and the module was built to tell the two apart without spending a slot.

The slope is not flat and it is not zero anywhere. It rises from +0.44 on the easiest fifth of the
holdout to +0.68..+0.81 across the hard three-fifths. Against Kimi's Q-TS discriminator
(`research/2026-09-04_qts-offline-leaderboard-transfer-slope.md`), that is the **ceiling-saturation**
signature — compression is worst where the cases are easy — and it is not the distribution-gap
signature, which would have put the recovered slope in Q5 and left the easy bands looking normal.
Nothing here reads near 1.0, so the "case-weighting / label-noise everywhere" branch is also out.

Read plainly: the ruler works everywhere, and one median over all of it reports the average of bands
that are not alike.

## The refutation ran, and it cut the claim down

The obvious next reading — written below as "what would refute it", then run before this file was
committed — asks the question the per-band table is actually being read as: does a pair whose holdout
gain sits in the HARD bands convert more of it into board than a pair whose gain sits easy? Same 34
entries, 348 pairs the holdout resolves, split at the median hard-share:

    gain made HARD transfers +0.74      gain made EASY transfers +0.67

**0.07 apart.** The direction survives; the magnitude does not. A band-slope table spanning +0.44 to
+0.81 sits on top of a pair-level difference of seven hundredths, which means most of that spread is a
fact about WHICH ENTRIES separate in which band, not about where a gain has to be made to count.

So the strong version — "an easy-case gain is worth half its size, a hard-case gain three quarters" —
is **not supported**, and it should not be planned against. What is supported is weaker and still
useful: transfer is broadly ~0.7 wherever the gain is made, mildly better for hard-case gains, and
nowhere near zero. The band table stays in the report because it is the thing the pair test had to be
run against; it is not the finding.

## What it licenses, and what it does not

- It does **not** license "the offline number is dead, stop measuring". Every band transfers.
- It does **not** rescue polish. Calibration, refit and reseating move confident, well-agreed cases —
  Q1 territory — which is the band that transfers at +0.44. This is a second, independent argument for
  what the seven board flats already said about the member axis, and it arrives from the ruler side
  rather than from the board side.
- It does **not**, after the pair test, license picking the next arm by which band it moves. +0.74
  against +0.67 is not a basis for choosing an axis; a method has to earn its place on the size of its
  gain, the way it always did.
- It is consistent with saturation being real and with there still being 0.009 of gap to the leader:
  saturation compresses the *ordering*, it does not put a wall at 0.944.
- The strongest thing it does is negative and it is worth having: the ruler is **not** blind above
  0.870. Every band transfers, the pair-level number is +0.7, and the plan-killing reading — "offline
  gains stop meaning anything up here" — is dead. That was what this cost a morning to find out, and
  it is why the next arm is still worth running.

## What refuted the strong version

Exactly the test written here before it ran: decompose each resolved pair's offline delta across the
bands and ask which band's gain predicts the board delta. It came back +0.74 hard against +0.67 easy
over 348 pairs, which is the artefact answer rather than the mechanism answer. Recorded above. The
band table is kept, the claim built on it is not.

## Caveats that are real

- The bands are ~182 studies. All twelve findings were scoreable in every band, so nothing was dropped,
  but a per-band AUC is wide and the slope is a median over resolved pairs, not a fit.
- Offline numbers here are **recomputed** under `LogitAverage` from the on-disk records, not the ledger's
  quoted numbers. That makes the ruler internally consistent across all 34 entries; it also means these
  offline values are not comparable line-for-line with `comp/board.jsonl`'s `offline` field.
- The entries span days and a kit that changed underneath them. Stratifying does not repair that
  confound; it asks the confounded question in a place where the candidate causes disagree.
