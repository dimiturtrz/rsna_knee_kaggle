# The two rulers return the same delta — and one permutation seed nearly said otherwise

2026-08-15 · `python -m kneemri.evaluation.ruler_transfer --labels .../labels/report_labels_v2.csv` · no GPU, no
submission · knee-8g9, following knee-7d1

## The question

Every number this project reports scores agreement with LLM-read report labels; the competition scores
agreement with radiologist annotations. `supervision_gap` had already compared the two by LEVEL on the
58 annotated studies — 0.8419 report, 0.8391 annotation, with a fifth of the cells disagreeing — and
knee-7d1 correctly refused to read that as an answer: **compression is a claim about DELTAS between
models, and one ensemble's level agreeing on both rulers says nothing about how a gap transfers.**

So: score two arms twice over the same studies, and compare the two deltas.

## What made it free

The published members were fitted on `report_labels_v2.csv` with no annotation weighting and carry
out-of-fold predictions for every study, so the radiologist annotations are genuinely held out for
them. Arms need no training — combining the first k of them is arithmetic over predictions that already
exist.

## The member ladder cannot be read at n=58

| arms | report | 95% | annotation | 95% |
|---|---|---|---|---|
| 1→2 | +0.0021 | [-0.0087, +0.0142] | +0.0041 | [-0.0072, +0.0153] |
| 2→4 | +0.0043 | [-0.0037, +0.0124] | +0.0054 | [-0.0025, +0.0139] |

Four seeds of one recipe are not far apart, and a ratio computed off a denominator of 0.004 with that
band is a report on the denominator. The arms had to be pushed apart deliberately: blend an arm toward
a permutation of itself, which degrades it by a chosen amount. A ruler does not care how an arm got
worse, only whether it can see that it did.

## The single seed that lied

At the project's default seed, share 0.40 gave **report +0.0430, annotation +0.0730 — ratio 1.70, with
a paired bootstrap interval of [1.07, 3.48] that excludes 1.** That is a clean-looking finding, it
points the way the prior points, and it is roughly the 1.9× compression knee-7d1 inferred from two
leaderboard rows. It would have been written up.

Four more permutation seeds at the same share:

| seed | report | annotation | ratio |
|---|---|---|---|
| 11 | +0.0768 | +0.0688 | 0.90 |
| 101 | +0.0630 | +0.0721 | 1.14 |
| 2026 | +0.0430 | +0.0730 | **1.70** |
| 7777 | +0.0491 | +0.0594 | 1.21 |
| 31337 | +0.0563 | +0.0443 | 0.79 |

The bootstrap interval was honest about the STUDIES and silent about the PERMUTATION, and the
permutation turns out to carry variation of the same size or larger. An interval that excludes 1 while
a second, unmeasured source of variation spans 1 is not evidence — it is the noise floor measured on
the wrong axis.

## The answer

Averaged over five permutation seeds:

| share | report | annotation | ratio | seed range |
|---|---|---|---|---|
| 0.05 | +0.0006 | +0.0017 | 0.97 | [0.97, 0.97] |
| 0.10 | +0.0010 | +0.0032 | 0.98 | [0.75, 1.22] |
| 0.20 | +0.0078 | +0.0114 | 1.17 | [0.75, 1.64] |
| 0.40 | +0.0577 | +0.0635 | 1.15 | [0.79, 1.70] |

**A delta measured against the report labels transfers to the radiologist annotations at a ratio of
about 1, and every seed range spans 1.** The local ruler is biased in LEVEL and appears unbiased in
DELTA, which is the property that matters for choosing what to submit.

## What this rules out, and what it does not

RULED OUT as the main mechanism: that the local-to-leaderboard compression (0.0107 local → 0.020 on the
board) comes from the difference between report labels and radiologist annotations. Both rulers were
applied here to the same predictions on the same studies and returned the same delta. The candidate
`supervision_gap.thresholds` named — our ruler calling 45.5% of cells positive where the radiologist
calls 34.4% — does not, on this measurement, distort deltas.

NOT RULED OUT: that the compression is real and comes from somewhere else (different studies, twenty
members at submission against four out-of-fold, hidden-set composition), or that it is two points of
scatter. This measurement cannot see the leaderboard.

LIMITS, stated: 58 studies; degradation by permutation blends hurts all twelve findings uniformly while
a real recipe change does not, so this prices the ruler's sensitivity and not the per-label structure of
a genuine arm difference; and the published members were fitted on the report labels, so their errors
share structure with the report ruler in a way an unrelated model's would not.

## The process note

The rule this project keeps paying for is *don't read signal into single-seed scatter*. It was applied
here to a permutation seed rather than a training seed — an axis nothing in the harness was watching,
in a module whose whole subject is whether a measurement can be trusted. The instrument now reports the
seed range by construction, because the version that reported one seed's bootstrap band was ready to
publish a 1.70.
