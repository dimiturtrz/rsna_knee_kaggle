# The vote is closed; the two columns are under the floor ALONE — two readings in one hour, on CPU

2026-09-07, 05:50–06:00Z. Card-free. Both readings came out of instruments that already existed and
had never been run: `kneemri.evaluation.vote_ceiling` and `kneemri.kaggle.column_lift`.

## 1. The vote axis is worth +0.0008, and we already ship the winner

`vote_ceiling --roster comp/members.roster` reads 109 members over 909 shared studies, fits each
rule's weights on one half and scores it on the other. That split is the whole instrument: a weighted
rule fitted on the studies it is then scored on cannot lose, which is why every vote number this
project owned before today was either a fit-on-self number or a board tick inside a noise band.

```
rule                  honest    vs flat    fit-on-self   overfit gap
LogitAverage          0.8721    +0.0008      0.8721        +0.0000
LabelQualityRank      0.8713    +0.0001      0.8717        +0.0004
RankAverage           0.8713    +0.0000      0.8713        +0.0000
QualityWeightedRank   0.8712    -0.0000      0.8713        +0.0000
SharpenedRank(2)      0.8712    -0.0001      0.8712        +0.0000
SharpenedRank(4)      0.8710    -0.0002      0.8710        +0.0000
MedianLogit           0.8709    -0.0004      0.8709        +0.0000
SharpenedRank(8)      0.8704    -0.0009      0.8704        +0.0000
SharpenedRank(16)     0.8675    -0.0038      0.8675        +0.0000
```

**The best honest rule is `LogitAverage` at +0.0008 over a flat average** — which is the rule the
project already ships (`votefit_v1`). Every alternative is at or below flat, and `SharpenedRank` is
monotonically worse as it sharpens.

`QualityWeightedRank`'s own docstring asked for exactly this read — "it has to be read on a split
half, exactly as per-label selection was, and per-label selection is what failed that test" — and
nothing had ever built it. The answer is that it is flat.

**What this settles.** The vote/weighting axis is closed on a matched offline ruler, not on a board
flat. That distinction matters: the five previous closures of this axis were board readings inside a
noise band, and knee-6j8b4 argues a board flat may be unresolved rather than closed. This one is not
a board reading. It agrees with them anyway.

**The overfit-gap column is the second finding.** It is +0.0000 for every rule except
`LabelQualityRank` (+0.0004). These rules are not overfitting their weights — they simply have nothing
to fit. The members do not differ in a way any of these nine rules can exploit.

## 2. The two-column route cannot close the gap, and column AUC does not determine the seat

`column_lift` seats a synthetic eighth member identical to the blend of seven except on MCL and
Lateral OA, and sweeps how hard those two columns are dragged toward perfect.

```
strength      MCL      Lateral OA    seat gain
 0.00       0.8238      0.8431        -0.0000
 0.10       0.8912      0.9054        +0.0014
 0.20       0.9429      0.9528        +0.0027
 0.30       0.9761      0.9834        +0.0035
 0.40       0.9960      0.9972        +0.0040
 0.50       1.0000      1.0000        +0.0041   <- column AUC saturates
 0.70       1.0000      1.0000        +0.0041
 1.00       1.0000      1.0000        +0.0077   <- oracle bound
```

`interpretations/kneemri/2026-09-06_two-columns-carry-three-quarters-of-the-gap.md` decomposed the gap
to first place and found MCL 0.8283 and Lateral OA 0.8369 against a 0.8765 pool mean, worth **+0.0073
of a +0.0097 gap** if lifted to that mean. That arithmetic was a decomposition of the stack's own
per-label scores, not a measurement of what a member shaped that way would ADD — a member does not
replace the stack's column, it joins a rank-average and moves it part of the way.

Measured, the axis is worth **+0.0041 achievable, +0.0077 as a hard oracle bound**.

**Against what, is the part this document got wrong on its first pass.** It quoted a single required
figure of +0.0117, which is 0.009 board divided by a 0.77 top-tercile OLS slope that was one hour old
and under three open challenges. §4 prices the same climb four ways and every cell lands between
**+0.0092 and +0.0126**. So the honest statement is a RANGE, and the column member covers between a
third and a half of it.

**It is under the floor ALONE, which is not the same as closed.** Increments below the bar compose, and
the running total is against the original baseline; nothing this project currently owns reaches 0.953
on its own. A +0.0041 offline term times the regime's compression is roughly +0.004 to +0.007 board —
it composes with the six-swap (+0.006–0.010 offline) and with `noisy_student` (unread) IF they sit in
different seats. What is refused is the claim that this axis is a gap-closer by itself, and the
hand-priced P(seat) of 0.15 that `lean_weak_columns` was armed behind.

**The mechanism is worth more than the refusal.** Note that AUC saturates at strength 0.50 for +0.0041
while the oracle row reads the same 1.0000 AUC for +0.0077. **Column AUC does not determine the seat.**
A fully dragged column is *binary* — it ties every study inside a class, and a rank-average reads a
tie as a confident vote. A member with identical column AUC but a real within-class ordering is worth
the smaller number. That is a fact about our vote rule, not about MCL, and it means per-column AUC is
the wrong quantity to optimise a member against.

## 3. What both readings jointly say

Two axes that the project was still spending arms and hand-priced P(seat) guesses on are worth
+0.0008 and +0.0041. Against a board gap of 0.010 and a frontier pulling away at −0.00009/day (board
read 2026-09-07T05:51Z: rank 47 of 3245, ours 0.944, leader 0.954), neither is the lever.

Both instruments existed in the tree and had never been run. That is the process finding: the cheapest
audit beat the fastest arm here by roughly eight hours of card, and the cost of running them was ten
minutes of CPU while the card stayed busy on something else.

## 4. The regime fall survives probit — the wall is in transfer, and it is half as tall as it looked

`board_transfer` now carries a `Scale` strategy and a `--scale {raw,probit}` flag. The scale enters at
exactly one place, `BoardPair.ratio()`, so `DELTA_FLOOR` and `RESOLVED` keep the raw-AUC meaning they
were measured with and the pair set is IDENTICAL between readings — 634 orderable pairs, 21 inversions,
the same four resolved ones, on both. A scale change moves the ratios and nothing else, which is what
makes the two comparable.

```
scale     whole record   >= 0.8678 (24 pairs)   >= 0.8695 (4 pairs, direction only)
raw          +1.12            +0.98 [+0.94, +1.08]      +0.82 [+0.71, +0.93]
probit       +1.74            +1.59 [+1.53, +1.76]      +1.34 [+1.18, +1.52]
```

The fall from the whole record to the top cell is **27% raw and 23% probit**. Probit does not linearise
it. Kimi's caveat named the inverting condition explicitly — "if probit linearises it, the conclusion
inverts to pool-side and measured-on-the-wrong-scale" — and it is **not met**. The wall reads as
transfer on both scales.

**The size of the wall does not survive, and that is the bigger correction.** Kimi's answer put 0.953
at offline **0.8957**, +0.0206 above our best-ever entry and +0.031 above our best-ever member — a
number that makes every lever we own hopeless. That came from an OLS slope of 0.788. This module's own
estimator, over the same record and deliberately not OLS (pairwise medians, "which needs no linearity
and is not dragged by the two entries furthest apart"), prices the same climb from 0.944/0.8751:

```
regime floor   scale     holdout needed   climb
   0.8678      raw          0.8843        +0.0092    <- 24 resolved pairs, priceable
   0.8678      probit       0.8858        +0.0107    <- 24 resolved pairs, priceable
   0.8695      raw          0.8861        +0.0110    <- 4 pairs, direction only
   0.8695      probit       0.8877        +0.0126    <- 4 pairs, direction only
```

**+0.0092 to +0.0126, not +0.0206.** Both priceable cells sit at +0.009 to +0.011. The difference is
estimator, not scale: a least-squares line through a six-thousandths-wide tercile whose residual sd is
0.0015 reports a slope with a precision that band cannot carry, and the module's docstring says so in
advance about this exact temptation.

That reprices the whole board. The six-swap alone is +0.006–0.010 offline; with the column member's
+0.0041 and `noisy_student` unread, the target is inside the range of things already on the table
rather than +0.031 beyond the best member we have ever trained.

**The arithmetic is now a method, not a hand calculation** — `required_offline`, behind
`--target/--from-board/--from-offline`, refusing with a zero climb when the regime holds no pair (a
climb FROM the top entry has nothing above it to price with). Every previous version of this sum was
done by hand off a printed constant, and knee-dn4t9 records three that came out wrong.

## 5. What is NOT settled

- **The champion cannot be re-seated offline at all.** `seat_value --against raptorweight_v1` refuses:
  1 of 6 champion members carries a holdout record, and a marginal contribution needs at least 2. Five
  of its seats are `--fold -1` and have seen every holdout study.
- **The pool has no single ruler.** `seat_value` over all 247 candidates with a holdout record refuses
  with `no study is held out by all 247 members`. The per-candidate re-rank against `honest12` is the
  workaround, not a fix.
- The vote reading is on 909 studies of OUR distribution. It says nothing about robustness, and
  knee-avzz's objection to same-distribution rulers applies to it as it does to everything else here.
