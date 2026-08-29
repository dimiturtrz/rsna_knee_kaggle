# The vote has nothing to weight

*2026-08-29 · closes `knee-xsfc` · instrument: `kneemri/evaluation/vote_ceiling.py` · 0 card, 0 slots*

## What was never measured

`QualityWeightedRank` has said in its own docstring, since the day it was written, that it "has to be
read on a split half, exactly as per-label selection was, and per-label selection is what failed that
test." Nobody built that read. So the project shipped `LogitAverage` (`votefit_v1`, one board tick over
`dark6`, inside its own pre-registered flat band) and had never measured what **any** weighting rule is
worth once its fitting is paid for. Every vote number in the ledger was either a fit-on-self number or a
board tick inside a noise band.

## The instrument

Weights come from one half of the holdout studies, the score comes from the other, both directions,
averaged. The halves alternate by position rather than cutting the array in two, because the holdout is
built in corpus order and a contiguous cut would put any ordering structure — site, date, anything —
entirely in one half.

The **fit-on-self** number is reported beside the honest one. Not as a result: as the ruler on the
result. The gap between them is how much of the rule is fitting noise, and it turned out to be the
load-bearing quantity here.

## The reading

Two pool sizes, one conclusion.

**42 members / 909 studies**

| rule | honest | vs flat | fit-on-self | overfit gap |
|---|---|---|---|---|
| `RankAverage` | 0.8711 | — | — | — |
| `QualityWeightedRank` | 0.8712 | +0.0000 | 0.8713 | +0.0001 |
| `LabelQualityRank` | 0.8713 | +0.0002 | 0.8717 | +0.0004 |
| **`LogitAverage`** | **0.8725** | **+0.0014** | — | — |
| `MedianLogit` | 0.8710 | −0.0001 | — | — |

**The six-seat one-decode shipping shape**

| rule | honest | vs flat | overfit gap |
|---|---|---|---|
| `RankAverage` | 0.8738 | — | — |
| `QualityWeightedRank` | 0.8738 | −0.0001 | — |
| `LabelQualityRank` | 0.8739 | +0.0001 | +0.0006 |
| **`LogitAverage`** | **0.8745** | **+0.0006** | — |
| `MedianLogit` | 0.8728 | −0.0010 | — |

Every weighted rule sits under the `+0.0015` floor beneath which `seat_value` has never picked a board
tick, at both sizes.

## Why the overfit gap is the finding

The expected failure mode of `LabelQualityRank` was overfitting: 12 numbers per member is a lot of
parameters to fit on 450 studies, and a rule with enough parameters reproduces the holdout rather than
the corpus. That is not what happened. The gap between grading its own homework and being honest is
**+0.0004** and **+0.0006** — the rule barely beats flat *even when it is allowed to cheat*.

That is a different and stronger statement than "the weights do not transfer." It says there is nothing
in the per-label quality table to find. Which is exactly what `competence_axis` measured independently:
member quality accounts for **81.7%** of the pool's spread. The members are ordered along one axis and
differ from each other mostly in overall strength, not in *which labels* they are good at — so a
per-label weight has no per-label structure to lock onto.

## What follows

**The vote axis is closed. No further vote rule gets written.** That was the pre-registered falsifiable
clause in `knee-xsfc` and it fired.

`LogitAverage` — already shipped — is the family's best at +0.0014 / +0.0006. Its offline margin
corroborates `votefit_v1`'s single board tick instead of adding a lever on top of it. Nothing here
changes what a submission serves.

## What it cannot say

It reads the fold-0 holdout, so `_cf` and `_ff` members are excluded by `RecordMix` rather than by
judgement, and the transfer caveat to the hidden test set is unchanged. Halving the studies roughly
doubles each per-label sampling error; the flat rule is scored on the *identical* halves precisely so
the comparison is paired and that inflation cancels where it can, but a rule separated from flat by
less than the paired floor on a half corpus is separated by nothing. Every number above is well inside
that condition — which is the point: the result is a null, and the instrument was built to make a null
readable rather than to chase a margin.
