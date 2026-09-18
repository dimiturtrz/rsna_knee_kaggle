# The cut is not what exhausted our ruler — the corpus is

2026-09-18, 00:41Z. Card-free. Instrument: `research/scripts/cut_sweep.py`.

## The question this answers

`headlrsolo_v1` put three members at offline75 0.9045 / 0.9053 / 0.9059 onto the board at 0.937 / 0.941 /
0.937. Their offline spread is **0.0014** against a **0.0062** paired floor: on our own ruler the three best
single members this project owns are a **tie**. Every plan shaped "raise offline75 by X and collect ~1.1X of
board" is therefore unpriced exactly where we ship.

A tie can come from the corpus (909 studies cannot resolve 0.0014) or from the cut (ten of twelve columns are
recut at 0.75, two stay at midpoint, and neither number was ever chosen by how well it predicts the board).
Those two have opposite consequences, and the second one would be cheap to fix.

## What was measured

For each flat cut on a grid, every solo entry's twelve-column macro is recomputed from its kept holdout arrays
and regressed on that entry's board score, over **all ten single-member rows** in `comp/board.jsonl`.
Residual sd is the quantity that matters: it is exactly the error you make when you use that ruler to say
which of two members will score higher.

| ruler | slope | resid sd | offline span |
|---|---|---|---|
| incumbent `ledger75` (mixed 0.75 / midpoint) | +1.072 | **0.0021** | — |
| flat 0.50, 0.60 | +1.312 | 0.0031 | 0.0250 |
| flat 0.68, 0.75, 0.80 | +1.084 | **0.0021** | 0.0302 |
| flat 0.82, 0.90 | +1.048 | 0.0040 | 0.0292 |

## Two readings

**Nothing beats the incumbent.** The best flat cut *ties* it at 0.0021 — inside its own scatter, which by the
instrument's own stated rule is no winner at all. Recutting cannot rescue the top-three tie, so the tie is a
property of the **corpus**, not of the threshold. The lever that remains on the ruler axis is more held-out
studies, and that is expensive in a way a threshold is not.

**The grid collapses to three rulers, not seven.** 0.50 and 0.60 agree exactly; 0.68, 0.75 and 0.80 agree
exactly; 0.82 and 0.90 agree exactly. The label reader has five rungs, so a cut can only ever land in one of
three gaps between them. That is a hard bound on the whole axis: the incumbent had at most two alternatives
available to lose to, and it lost to neither. Any future proposal to "tune the cut" is proposing a choice
among three, and all three have now been measured.

## What this does not say

Ten points and one free parameter. It screens for a cut that is *obviously* better and finds none; it does not
license adopting a tie as an improvement, and it says nothing about a non-flat recut whose rungs are chosen
per column by an argument rather than by a sweep — that remains the only untried shape, and it owes a
mechanism before it owes a run.

## And the corpus cannot be the lever either — 00:52Z, `research/scripts/ruler_ceiling.py`

The section above leaves the corpus as the only remaining way to make our ruler separate the top members.
This prices it. The paired difference between the two highest members on disk — `headlr_ship_s2026` and
`labelwise_depth24_drop40_long_s2026`, the pair the board disagreed about at 0.937 vs 0.941 — is
bootstrapped over studies at a ladder of subsample sizes, 400 draws a rung, under the incumbent mixed ruler.

| n | sd(paired delta) | 2 sd floor |
|---|---|---|
| 200 | 0.0046 | 0.0092 |
| 400 | 0.0035 | 0.0070 |
| 700 | 0.0024 | 0.0048 |
| 909 | 0.0021 | **0.0042** |

The scaling exponent is **−0.525**, against −0.500 for a clean sqrt-n law — so the floor shrinks with the
holdout at very close to the textbook rate, and the extrapolation is honest.

**This pair's own floor is 0.0042, not the repo's general 0.0062**, and the spread to resolve is 0.0014 —
three times short. Extrapolating: **8,246 held-out studies** would be needed. The project has **4,406
labelled studies in total**. Holding out every one of them, with nothing left to train on, still lands at
0.0019 — above 0.0014.

**So the ruler ceiling is structural, not a budget.** Above offline75 ~0.904 no holdout cuttable from this
corpus can rank the members we actually choose between, under any threshold. Three consequences, and they
change what a slot is for:

1. **Stop ranking shipping candidates offline at the top of the line.** An offline delta under ~0.004
   between two of our best members is not weak evidence, it is *no* evidence, and it has been read as
   evidence repeatedly.
2. **The board is the only instrument above 0.904**, so a submission slot is now the scarce resource for
   RANKING, not merely for confirming. That is an argument for spending slots on members that differ by a
   MECHANISM — where the prior does the separating the ruler cannot — and against spending them on knob
   variants, whose offline separation was the only thing ever recommending them.
3. **The offline ruler keeps its job below 0.904**, where its residual sd of 0.0021 against a 0.030 span is
   an excellent instrument. Nothing here retires it as a gate on new families; it retires it as a chooser
   among our own best.
