# The grafted members became a cost

**2026-08-16.** Two submissions returned the same day and, between them, retire a reading the ledger
had carried for six leaderboard entries: that ensemble diversity beats member quality on this corpus.

## The three numbers

| entry | members | passes/study | LB |
|---|---|---|---|
| `kneemri-efficient` v4 | 2, all ours (`strongest(2)`, post-LoRA pool) | 2 | **0.919** |
| `kneemri-trained` v8 | 5, all ours | 1 | **0.911** |
| `kneemri-fusion` v10 | 24 — our 6 plus 18 published CC0 | 1 | 0.898 |

Noise floor on this board is ≈0.0053. Against the 24-member fusion, v4 is **+0.021** and v8 is
**+0.013** — the first clears the 1.5% bar outright, the second sits at about 2.5× the noise floor and
just under the bar.

## What is a result and what is not

**The result: the published CC0 members are now a cost, not a contribution.** Two independently built
all-ours entries — different member counts, different pass budgets, different kernels — both beat the
mixed fusion. That is the claim two arms agree on, and neither of them shares the other's confound.

**Not a result: two members beating five.** v4 − v8 = +0.008, above the noise floor but well under the
bar, and confounded (v4 also spends two passes a study). Read it as directional. It does say the fused
pool was not being carried by member *count*.

## Why the earlier reading was not wrong when it was made

The ledger's "diversity beats member quality" came from entries where our own members were weaker. The
comparison then was: six mediocre members of ours, or those six plus twenty of theirs? Adding theirs
won, repeatedly, and that was the correct call **at that member quality**.

What changed underneath is the pool, not the principle. The post-LoRA members (all-block adaptation,
`cb6c5d1`) are strong enough that rank-averaging them with weaker external predictions dilutes rather
than diversifies. A rank-average has no idea which member is worth listening to; every member gets a
vote, so a member below the pool's average drags the ensemble toward itself. Diversity only pays while
the diverse members are close in quality. They no longer are.

This is the same shape as every other threshold in the project: the mechanism did not reverse, the
operating point moved across it.

## The ruler and the board disagree, and both deltas are small

Local holdout ranks the five all-ours members **above** the pair: 0.8610 vs 0.8581, +0.003. The
leaderboard ranks them **below** it: 0.911 vs 0.919, −0.008. Opposite signs.

Do not build on that. +0.003 is below the noise floor and −0.008 is barely above it; two sub-bar deltas
of opposite sign are what scatter looks like. What it does forbid is using the local holdout to *choose*
between configurations this close — the ruler has not earned that resolution, and 2026-08-14's
"the ruler was five times too small" is the standing precedent for how expensive that mistake is.

## What this does NOT rule out

- **Diversity as a mechanism.** Refuted here is the specific graft: *these* CC0 weights, at *this* member
  quality, under a *rank-average*. A quality-weighted or stacked combiner was never tried, and it is the
  obvious thing that would let a weaker-but-different member contribute without voting equally.
- **Larger all-ours ensembles.** Nothing here says five is the ceiling; it says five mixed-in externals
  cost more than they pay. The untested cell is many members, all ours, post-LoRA.
- **The published members as a floor.** The graft row remains what it always was: the field's floor
  measured on our harness, and the reason we know 0.898 is reachable without training anything.

## Consequence for the plan

The best entry is now **entirely ours** — 0.919, no CC0 weights, at a twelfth of the fusion's forward
passes. That retires the fusion track as the score-carrying line and makes the cheap configuration the
one to build on: whatever is tried next is tried on two-to-five strong own members, where a per-study
pass costs little enough that test-time budget is a free axis.

Gap to the board's leader (0.950) is 0.031.

## Sources

- `comp/SUBMISSIONS.md` rows for 2026-08-15 fusion v10 and 2026-08-16 trained v8 / efficient v4.
- The reading this replaces: `interpretations/kneemri/2026-08-15_crop_centre_coverage.md:77`
  ("diversity has beaten member quality on this corpus six leaderboard entries running") and the
  six-entry settlement cited in `2026-08-13_the-ruler-was-wrong-and-the-witness-earned-its-keep.md`.
- The pool change underneath it: `cb6c5d1` (all-block LoRA adaptation).
