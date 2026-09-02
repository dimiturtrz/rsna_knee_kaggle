# A learned combiner beats every fixed rule, by less than the floor

2026-09-02. Code: `kneemri/inference/stacked.py` (the rule), `kneemri/evaluation/stack_fit.py` (the
measurement). Commit `eb6ba087`.

## What was measured

A per-label ridge over member logits — one weight per (member, label) — fitted on half the 909 holdout
studies and scored on the other half. The ridge penalty is chosen on a **nested** split inside the fitting
half, so no parameter ever sees a scoring study.

| pool | StackedLogit | LogitAverage | RankAverage | delta over best fixed |
|---|---|---|---|---|
| 6 members | 0.8776 | 0.8760 | 0.8756 | +0.0015 |
| 12 members | 0.8797 | 0.8776 | 0.8773 | +0.0021 |
| 24 members | 0.8766 | 0.8757 | 0.8747 | +0.0009 |

It wins on all three pools. Every delta is under the 0.0062 holdout floor. **Directional, not a finding.**
At the +0.98 board-per-holdout-point slope this is +0.001 to +0.002 on the board, under BOARD_RESOLVABLE
0.0015 — so it does not earn a slot on its own.

## What it says about the axis

`comp/champion.json` names stacking as one of two remaining routes to 0.950, on the assumption that the
project's directional gains ADD. This is the first test of the combiner half of that assumption, and the
answer is that a learned combiner is worth about one directional knob — not a lever.

The mechanism was pre-stated and held. Krogh-Vedelsby: an ensemble's gain over its members is ambiguity,
and a fitted combiner can only redistribute ambiguity, not create it. `competence_axis` reads 81.7-82% of
what separates 105 members as ONE scalar. A pool that is one scalar has almost no ambiguity to
redistribute, so the extra expressiveness of 144 fitted parameters has noise to fit instead. The measured
+0.002 is what that predicts.

**What this does NOT close.** Two readings survive and the local holdout cannot separate them:

- **A — the pool is saturated.** +0.002 is all a combiner can ever buy on a one-scalar pool, and the four
  earlier fixed-rule nulls were never about the rules. Under A the lever is diversity, full stop.
- **B — the holdout is too small.** 144 parameters against ~455 scoring studies, with per-finding
  prevalences down to 5%, may be starving the combiner rather than saturating it. Under B the move is
  fewer parameters (one weight per member, shared across labels), not abandoning the axis.

Separating them needs a published stacking learning curve — held-out gain vs number of fitting studies at
fixed pool size. Asked as Q-BH; 909 studies is all we have, so we cannot run it.

## The bug, because it nearly became a false axis closure

The first version centred the target but not the design and fitted no intercept, so ridge shrank the
weights toward zero about the wrong origin. It read **0.8587 — a 0.0173 LOSS** — and "stacking is refuted
on a correlated pool" was one sentence from being written down.

What caught it was **the in-sample number**: 0.8616, also below the fixed rule's held-out 0.8760. A
combiner that loses on the studies it was fitted on is misfitted, not overfitted — overfitting shows as a
big in-sample win and a held-out loss. Report the in-sample number beside the held-out one; it costs
nothing and it is the only thing that distinguishes a broken fit from a real refutation.

A second, smaller retraction from the same session: the claim that the gain doubles with pool size came
from a penalty picked by reading the scoring half. Under nested selection the scaling vanished (+0.0015 /
+0.0021 / +0.0009 is flat within noise, not a trend).

## The limit on shipping it

Weights can only be fitted where holdout predictions exist, which is fold-0 members. The pool we ship is
six `--fold -1` members with no offline predictions at all. Weights reach the shipped seating only through
the fold-0 twin assumption that `comp/kernels/kneemri-aucvote` states in its own docstring — and
`aucvote_v1`'s board read is the first test of whether twin-derived per-label numbers price the shipped
seats at all. Until that comes back, a stacker cannot be shipped on any better footing than that kernel.
