# A member that reads MCL and Lateral OA PERFECTLY is worth +0.0074, and the gap is +0.0117

*2026-09-06 05:10Z, CPU-only while the card ran `margin_seat_ff`. Zero card, zero slots.*

Four hours ago `pool_ceiling --columns` decomposed the gap to first place by finding and pointed at two
columns worth **+0.0073** of the **+0.0097** gap, and an arm (`lean_weak_columns`, knee-vbj77) went into
the queue behind that arithmetic with a hand-guessed `P(seat) ~ 0.15`. That guess stood on nothing. The
decomposition prices what the STACK's columns are worth; it does not price what a MEMBER shaped that way
adds, because a member does not replace a column — it joins a rank-average and moves it part of the way.

`kneemri/kaggle/column_lift.py` measures the thing the guess was about. It seats a synthetic eighth
member beside the greedy-optimal seven whose ten other columns **are the seated blend's own combined
prediction**, so it agrees with the base perfectly outside the two findings and can earn nothing there.
Its MCL and Lateral OA columns are that same prediction dragged toward truth in rank space at a strength
`s`. Sweeping `s` walks one member from "identical to the blend" to "reads those two findings perfectly".

```
a synthetic eighth member, identical to the blend of 7 except on MCL + Lateral OA:
  strength                MCL        Lateral OA   seat gain
      0.00             0.8238            0.8431   -0.0000
      0.10             0.8912            0.9054   +0.0014
      0.20             0.9429            0.9528   +0.0026
      0.30             0.9761            0.9834   +0.0035
      0.40             0.9960            0.9972   +0.0040
      0.50             1.0000            1.0000   +0.0041
      1.00             1.0000            1.0000   +0.0074
ORACLE CEILING on this axis: +0.0074 against a required +0.0117
```

**The two-column route cannot close the gap on its own, and this is now measured rather than argued.** A
member reading both findings *perfectly* — and otherwise being the blend itself — adds +0.0074 against
the +0.0117 first place costs. Every real member is below that.

## Two things the sweep says that the decomposition could not

**1. The achievable oracle is +0.0041, not +0.0074.** Both the `s=0.50` and the `s=1.00` rows read AUC
1.0000 on both columns and they are worth wildly different amounts. At `s=1` the column is literally the
binary target: every study inside a class is TIED, and `RankAverage` gives a tied column the class
midrank, which is a maximally confident vote. At `s=0.5` the column separates the classes completely but
still carries the blend's within-class ordering. **AUC does not determine the seat.** A real member with
perfect MCL discrimination and any ordinary within-class ordering is worth the smaller number, so the
honest ceiling on this axis is **+0.0041**, and +0.0074 is a bound no trained member can reach.

That is a general fact about this pool's ruler, not about these two columns, and it is worth carrying:
two members with identical per-label AUC can be worth twice as much as each other in the ensemble
depending on how they tie.

**2. The independent check passed.** The decomposition predicted +0.0073 from the stack's own per-label
scores; the seat measurement lands at +0.0074 at the degenerate oracle. Two different arithmetics over
different objects agreeing to the fourth decimal is the confirmation that the earlier reading was not a
bookkeeping error.

## What this does to the queued arm

`lean_weak_columns` keeps its slot on the card — it is the cheapest keep/kill on the loss-reweighting
half, its control is already on disk, and no member this project owns leans that way at all. But its
price is now measured rather than guessed, and it is worse:

- if the reweighting lifts both columns by **+0.03** (a large move for a loss balance knob) the sweep
  puts that near `s≈0.05`, worth roughly **+0.0007** in the ensemble;
- to be worth +0.0041 the member has to reach MCL **1.00** — which is to say it cannot;
- so the arm's honest value is **the keep/kill, and nothing else**. `P(seat over the greedy-optimal
  seven)` should read ~0.02, not 0.15.

**And the strategic consequence is bigger than the arm.** The gap does not have a single-route answer.
The best-shaped column route is capped at +0.0041 achievable; `seat_requirement` already reported that no
selection over 206 members produces +0.0117; the member axis has seven board flats. **First place has to
come from two or three routes summing, and each of them must be priced this way before it takes card
time.** That is the standing rule the project already has — a chain of sub-bar increments against the
ORIGINAL baseline IS a result — and this is the first time the routes have been priced against a common
denominator.

## What would refute this

- **The base is the greedy-optimal seven.** A different seated set changes the marginal value of any
  eighth member. Measured against a worse base, the same synthetic member is worth more — which is why
  the base is the best one available, and why the number is a floor on nothing and a ceiling on the route.
- **The synthetic member is maximally agreeing by construction.** A real member differs on all twelve
  columns and that difference is worth something this one is built to have none of. So a real member at
  the same column quality is worth MORE than the sweep says — the ceiling bounds the column CONTRIBUTION,
  not the member.
- It says nothing about whether the columns are liftable. That is still `lean_weak_columns`'s question.

## Addendum, 05:19Z — the tie-structure lever it exposed is CLOSED, and it cost no card

The sweep above found that AUC does not determine a seat: identical 1.0000 columns were worth +0.0041
and +0.0074 depending only on whether the column was tied inside each class. That is a lever over members
ALREADY OWNED if it generalises, so it was priced immediately.

`SharpenedRank(k)` bends each member's ranks through a logistic centred on the median before averaging —
the flat rule is its own `k → 0` limit (an affine map of every member's ranks averages to an affine map of
the mean, and AUC is invariant under it), and large `k` drives every member toward the two-level vote that
was worth double. It fits nothing, so its honest and fit-on-self readings are identical by construction.
`vote_ceiling` reads it on the split half:

```
over 109 members                    over the greedy-optimal seven
  RankAverage        0.8713           RankAverage        0.8764
  LogitAverage       0.8721 (+0.0008) LogitAverage       0.8766 (+0.0002)
  SharpenedRank(2)   0.8712 (-0.0001) SharpenedRank(2)   0.8764 (-0.0001)
  SharpenedRank(4)   0.8710 (-0.0002) SharpenedRank(4)   0.8762 (-0.0003)
  SharpenedRank(8)   0.8704 (-0.0009) SharpenedRank(8)   0.8755 (-0.0010)
  SharpenedRank(16)  0.8675 (-0.0038) SharpenedRank(16)  0.8740 (-0.0024)
```

**Monotonically negative at both pool sizes.** The lever is closed, and the mechanism says why it had to
be: a tied column is worth more only when the column is RIGHT. Binarising a perfect ordering loses
nothing and buys vote mass; binarising a 0.83 ordering throws away the within-class information that is
the only thing separating it from a coin. The synthetic oracle had the first case and no member here has
it. So the +0.0074 row is confirmed as an artefact of perfection rather than a shape worth copying — which
is the more useful reading of it, because it means the ceiling this document reports is the +0.0041 one.

Incidentally the flat rule reads **0.8764** on the greedy-optimal seven here, the same number
`seat_requirement.DEFAULT_BASE` records from a different tool and a different path. The ruler agrees with
itself.
