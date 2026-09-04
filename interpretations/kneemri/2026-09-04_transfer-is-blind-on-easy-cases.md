# The holdout is not blind — it is blind on the easy half

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

Read plainly: **a holdout gain that comes from getting easy cases righter is worth about half of what
its size claims on the board. A gain that comes from high-disagreement cases is worth about three
quarters.** Both are positive. The ruler works; it is differently sensitive in different places, and
one median over all of it reports the average of two regimes.

## What it licenses, and what it does not

- It does **not** license "the offline number is dead, stop measuring". Every band transfers.
- It does **not** rescue polish. Calibration, refit and reseating move confident, well-agreed cases —
  Q1 territory — which is the band that transfers at +0.44. This is a second, independent argument for
  what the seven board flats already said about the member axis, and it arrives from the ruler side
  rather than from the board side.
- It **does** argue that the next arm should be picked by which band it moves, not by its whole-holdout
  delta. A method that moves Q4-Q5 buys ~0.75 of its size; the same delta parked in Q1 buys ~0.44.
- It is consistent with saturation being real and with there still being 0.009 of gap to the leader:
  saturation compresses the *ordering*, it does not put a wall at 0.944.

## What would refute it

Decompose each already-scored board pair's offline delta into per-band deltas and ask which band's
delta predicts the pair's board delta. If a pair whose gain sits in Q5 does **not** out-transfer a pair
whose gain sits in Q1, the per-band slopes above are an artefact of which entries happen to separate in
which band rather than a statement about where transfer lives. That reading spends no slot either and
is the obvious next step.

## Caveats that are real

- The bands are ~182 studies. All twelve findings were scoreable in every band, so nothing was dropped,
  but a per-band AUC is wide and the slope is a median over resolved pairs, not a fit.
- Offline numbers here are **recomputed** under `LogitAverage` from the on-disk records, not the ledger's
  quoted numbers. That makes the ruler internally consistent across all 34 entries; it also means these
  offline values are not comparable line-for-line with `comp/board.jsonl`'s `offline` field.
- The entries span days and a kit that changed underneath them. Stratifying does not repair that
  confound; it asks the confounded question in a place where the candidate causes disagree.
