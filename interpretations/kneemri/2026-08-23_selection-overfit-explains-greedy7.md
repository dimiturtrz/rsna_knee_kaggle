# Per-label structure is real and anti-exploitable, and the overfit that hides it is twice greedy7's edge — 2026-08-23

## What was run

`kneemri.evaluation.member_curve --paired --device cpu` over six of the champion's seven members
(`band_dense_long`, `band_dense`, `grade_ladder`, `through_dense`, `calib_span`, `base_dense`), fold 0,
split seed 2026, 909 holdout studies. **It cost no card and no compute**: every one of the six already had
its holdout predictions in `holdout_predictions/`, so the run was cache hits plus arithmetic. The solo
scores came back identical to the ledger (band_dense 0.8578, grade_ladder 0.8565, base_dense 0.8512),
which is what says the cache was the right cache.

`final_ws_e12` is absent by construction — a `--fold -1` member has seen every holdout study and can
never enter an offline mean — and `lr4e-5_ws` had no cached predictions.

## The two numbers

    oracle 0.8643   vs plain average 0.8695   vs best single 0.8587   (headroom -0.0052)

    top-1 per label:  in-sample 0.8643   split-half 0.8541
    top-2 per label:  in-sample 0.8708   split-half 0.8642
    top-3 per label:  in-sample 0.8712   split-half 0.8668
    top-4 per label:  in-sample 0.8714   split-half 0.8681
    top-5 per label:  in-sample 0.8706   split-half 0.8690
    top-6 per label:  in-sample 0.8695   split-half 0.8694   (= the flat rank-average)

Two independent statements, and the second is the one worth keeping.

## Per-label selection is dead, and this is a harder close than 2026-08-17's

The earlier oracle put headroom at **+0.0002** over seven same-family members and closed the axis on the
grounds that it was two orders under the floor. This one puts it at **−0.0052**: the ceiling is BELOW the
floor of the thing it is a ceiling over. An oracle that knows, per label, which member scores best on the
very studies it is scored on still LOSES to averaging all six.

The honest curve says the same thing without needing an oracle at all. Fit the selection on half the
studies, score it on the other half, and every restriction costs: 0.8541, 0.8642, 0.8668, 0.8681, 0.8690,
0.8694. Monotone, all the way up to "use everyone equally". There is no k at which picking beats averaging.

Nor do the alternative vote rules move it: `LogitAverage` 0.8696 against `RankAverage` 0.8695, and
`QualityWeightedRank` 0.8694. Three combiners inside 0.0002 of each other.

## It is NOT that the members agree

That was the 2026-08-17 reading — "the members do not disagree per label in a way any selection rule
could exploit" — and it is now too weak. They disagree, significantly, with paired bootstrap intervals
that exclude zero:

| label | delta vs band_dense_long | 95% |
|---|---|---|
| ACL | **−0.0338** | [−0.0560, −0.0137] |
| Medial Meniscus | **+0.0175** | [+0.0058, +0.0307] |
| Medial OA | **−0.0152** | [−0.0278, −0.0034] |
| Fracture | −0.0209 | [−0.0563, +0.0126] |
| MCL | −0.0116 | [−0.0332, +0.0115] |

A 0.034 gap on ACL is five times the corpus noise floor. **The per-label structure is real; what fails is
the attempt to trade on it.** Averaging six members reduces variance by more than per-label specialisation
gains in bias, so the specialised choice loses even when the specialisation is genuine. That is a
different and more useful statement than "the members are the same", because it survives adding a
distant member — a distant member widens the per-label gaps, and widening them does not change which
side of this trade wins.

## The finding: selection overfit is +0.0033, and greedy7's whole edge was +0.0018

    top-4 per label:  in-sample 0.8714   split-half 0.8681      difference +0.0033

That difference is not an ensembling number. It is the price of CHOOSING — the amount by which a rule
that maximises the offline mean over subsets inflates its own estimate on the studies it maximised over.

Now read `greedy7` against it. `greedy7` was fired precisely because it maximised the offline mean: it
scored **0.8712 against the incumbent's 0.8694, an edge of +0.0018**, chosen by the best selection rule
available. It returned **+0.000** on the board and became the fourth tie in a row.

**+0.0018 sits entirely inside +0.0033.** The edge greedy7 was chasing is smaller than the overfit its
own selection procedure generates, so its 0.8712 was not a better ensemble measured well — it was the
same ensemble measured on the studies that flattered it. The board did not fail to resolve greedy7's
improvement; there was no improvement to resolve.

This upgrades the roster axis from RETIRED-BY-OBSERVATION to CLOSED-BY-MECHANISM. Four ties said the
axis was flat. This says why, and it says it offline, in advance, for free.

## What it licenses and what it forbids

**FORBIDS: firing an entry chosen by offline maximisation.** Any offline selection gain under about
0.003 is overfit by construction and must not buy a submission slot. That covers `greedy7`, and it covers
the eighth-seat scan (knee-dfl1, flat at +0.0004 across 59 candidates) and the seat scan that priced 63
candidates at +0.0005 or less — all three were reading selection noise, and the correct reading of a
0.0004 seat edge is not "too small to resolve" but "smaller than the estimator's own bias".

**FORBIDS: further combiner work.** Vote rule, per-label weighting, quality weighting and per-label
selection are now measured together on one honest estimator and span 0.0002. There is nothing there.

**LICENSES: nothing new, and that is the honest part.** This closes doors; it opens none. It does
converge with what the board has said across six entries — arrangement returns +0.000, contents returned
+0.001 — so two independent rulers now agree that the pool's CONTENTS are the only live variable. That
is the value: the offline ruler and the board stopped disagreeing.

## What it does NOT rule out

- **It says nothing about whether a NEW member helps.** Adding a member changes the plain average itself,
  which is a different quantity from the headroom above the plain average. `raddino_dense` (knee-vp8q) is
  unaffected by this result in either direction.
- **It does not close per-label work on the LOSS side.** This measures selection among finished members.
  A model TRAINED to be better on MCL is not a member chosen for MCL, and knee-5pdl's ordinal target is
  untouched.
- **It does not re-close knee-clpy.** That bead asks the same question with `vjepa_warm` in the set, whose
  nearest seated neighbour is 0.8210 against a seated spread topping out at 0.8801. The argument above
  predicts it will close the same way and explains why, but predicting is not measuring, and vjepa_warm's
  predictions are the one thing not in the cache. Downgrade it; do not close it.
- **The estimator itself is a suspect and should be named as one.** Split-half over 909 studies scores
  each half on ~455, so its own noise is larger than the corpus floor. What it can support is the SIGN
  and the rough size of the overfit, which is all that is claimed here. It cannot support "+0.0033" as a
  constant, and no arm should be planned against that digit.
