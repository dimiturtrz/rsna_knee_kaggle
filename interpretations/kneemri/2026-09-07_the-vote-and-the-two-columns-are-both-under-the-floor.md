# The vote and the two columns are both under the floor — two axes closed in one hour, on CPU

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

Measured, the axis is worth **+0.0041 achievable, +0.0077 as a hard oracle bound, against +0.0117
required**. A member reading both findings perfectly, and otherwise being the blend itself, does not
close the gap. `lean_weak_columns` was armed behind that arithmetic with a hand-priced P(seat) of
0.15; it is not a gap-closer at any P(seat), and on its own it sits below the 0.0062 floor.

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

## 4. What is NOT settled

- **The champion cannot be re-seated offline at all.** `seat_value --against raptorweight_v1` refuses:
  1 of 6 champion members carries a holdout record, and a marginal contribution needs at least 2. Five
  of its seats are `--fold -1` and have seen every holdout study.
- **The pool has no single ruler.** `seat_value` over all 247 candidates with a holdout record refuses
  with `no study is held out by all 247 members`. The per-candidate re-rank against `honest12` is the
  workaround, not a fix.
- The vote reading is on 909 studies of OUR distribution. It says nothing about robustness, and
  knee-avzz's objection to same-distribution rulers applies to it as it does to everything else here.
