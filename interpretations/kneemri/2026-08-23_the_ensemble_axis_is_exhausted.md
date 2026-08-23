# Every ensemble-side axis is measured flat, and the flatness is in the pool

Four separate questions have been asked about how to combine members better: which rule votes, how many
members sit, whether the members can be chosen per label, and which member takes a given seat. All four
are now measured on the champion seven — the pool this project actually ships — and all four are flat
against the board's own resolution. This document is the closing argument, because the axis has been
re-opened three times on the belief that the instruments were at fault.

## What was measured

`python -m kneemri.evaluation.member_curve` over exactly the seven members `comp/champion.json` pins
(`logs/champion_seven.log`, 909 holdout studies, one split, no TTA). The seven, solo:

| member | solo |
| --- | --- |
| `band_dense_long` | 0.8587 |
| `band_dense` | 0.8578 |
| `grade_ladder` | 0.8565 |
| `through_dense` | 0.8559 |
| `calib_span` | 0.8545 |
| `lr4e-5_ws` | 0.8524 |
| `base_dense` | 0.8512 |

## The four axes

**The combination rule buys +0.0002.**

```
RankAverage            0.8694   (the shipped rule)
LogitAverage           0.8696
QualityWeightedRank    0.8695
```

This is the axis that was re-opened today, and it deserved to be: it had been retired on 2026-08-17 by
comparing a +0.0005 offline delta straight against a board floor, which is the units error `knee-7d1`
later corrected and `comp/champion.json` now forbids by name. Re-priced honestly at 4.67 board points
per holdout point, +0.0002 is +0.0009 board — under one tick.

**And the re-open clause is spent, which is the finding rather than the number.** The argument for
re-testing was the interpretation's own pre-registered condition: *a combiner over members that disagree
far more than these six, which share a recipe lineage*. The champion seven is the most-disagreeing pool
this project owns — the only one holding both the DINOv2-**base** member and the different-pixels
member. On it, LogitAverage moved **less** (+0.0002) than on the six all-ours members it was retired
against (+0.0005). The effect does not grow with disagreement; it shrinks. That is the opposite of the
prediction, so the clause is answered rather than still open.

**Member count is saturated, and the curve says so four steps running.**

```
4 members: 0.8675  +0.0015 over 3
5 members: 0.8684  +0.0009 over 4
6 members: 0.8690  +0.0006 over 5
7 members: 0.8694  +0.0004 over 6
```

The increments are halving. An eighth is worth about +0.0002, which is +0.001 board, and that is the
optimistic read because it assumes the eighth is as good as the seventh. The best SUBSET at each size
says the same thing from the other direction: best-4 scores 0.8696, best-6 0.8695, best-7 0.8694 — the
seventh member is negative on its own roster.

**Per-label selection is negative.** The oracle sits at 0.8643 against the flat average's 0.8694,
headroom **−0.0051**: a ceiling beneath the floor. This reproduces `knee-vjmw` on a different pool, and
no compression factor recovers a negative.

**Seat exchange is flat on the board itself**, which is the only one of the four that did not need an
offline ruler at all. Three single-member displacements of the same seven: `sevenplane` 0.932,
`gradeswap` 0.933, `accumseat` 0.933. Spread 0.001 across the whole axis.

## The finding

**The flatness is in the POOL, not in the instruments.** Three seat rulers were retired this week —
residual distance (0-for-2, and in opposite directions), binary seat value (blind at its own ±0.0015),
ordinal concordance (passed at a +0.011 gap and TIED at a larger +0.014 one) — and each retirement was
read as a reason to go looking for a fourth ruler. The curve above closes that: a pool whose marginal
member is worth +0.0004 and whose best subset peaks at four cannot be rearranged into a result by any
selection rule, however good, because there is nothing left in it to select. A better ruler measuring an
exhausted axis returns a better-justified zero.

That also explains why the board has answered 0.933 three times. It is not the board's noise swallowing
a real gain; it is the gain being +0.001-sized before it ever reaches the board.

## What this does NOT rule out

- **An eighth member from an axis no current member occupies.** Every one of the 63 candidates the seat
  scans ranked was fitted from the same encoder family on the same pixels. The count axis is exhausted
  *for members drawn from that set*; a 3D trunk (`knee-rpkw`) or an externally-pretrained one
  (`knee-ombx`) has never been in the pool, so what is untested is the AXIS, not the count.
- **A stacked combiner fitted across folds** — the other half of the 2026-08-17 clause, and the half
  this measurement did not test. It sits awkwardly with folds being banned, so it is not queued.
- **Nothing about member QUALITY.** Every number here holds the members fixed and varies the
  arrangement. A pool of seven better members would produce a different curve, and that is precisely
  where the argument points next.

## What it redirects to

Two levers remain, and both change what is IN the pool rather than how the pool is read:

- **More data.** `final_fit` (`knee-g8nb`) fits `band_dense`'s exact recipe on all 4,406 studies instead
  of 3,497. It is on the card as this is written. The concave data slope (`quarter_data`, each doubling
  buying ~0.48 of the last) prices the step at ~+0.0029 offline — the largest number left anywhere on
  this board.
- **A different encoder class.** `vjepa_member` (`knee-rpkw`) is armed and has never been run; the
  tubelet trunk is the only architecture in the repo with zero arms in the ledger.

The honest statement of where this leaves the target: **0.933 → 0.940 is seven board ticks, and no
arrangement of the current seven members produces more than one of them.** It has to come from a member
that does not exist yet.
