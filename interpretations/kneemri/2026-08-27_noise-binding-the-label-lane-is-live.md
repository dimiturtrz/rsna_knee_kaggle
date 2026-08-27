# The members inherited the reader's confident errors — the label lane is LIVE

**Date:** 2026-08-27 · **Instrument:** `kneemri.evaluation.noise_binding` · **Cost:** zero slots, zero card hours

## The question this settles

`core.data.labels.reader_agreement` measures the label reader's dominant defect: **108 confident false
positives over the 58 annotated studies, all verdict `YES`, 15.8% of every annotated cell** — against 32
misses in the other direction, of which only 8 are a confident `NO`. That is four times the size of the
silence hole `knee-pju4` was originally written about.

A defect in the LABELS is not automatically a ceiling on the MODEL. A member trained on 4406 studies sees
the over-called cells as a minority of noisy targets among many consistent ones, and a fit with enough
regularisation and enough agreeing evidence can average them out — in which case the label lane buys
nothing however large the defect looks in the table. Nothing here had asked which of those two worlds we
are in, and the answer decides what the remaining card time is worth spending on.

## The instrument

Take the cells where reader and radiologist AGREE as the two poles and ask where the DISPUTED cells fall
between them:

    tracking = (mean p(over-called) - mean p(agreed negative))
               / (mean p(agreed positive) - mean p(agreed negative))

Unit-free, so it does not care about a member's absolute calibration. `1` means the member is as confident
on cells the reader INVENTED as on cells that are really positive — it learned the lexicon's mistake.
`0` means it is as unconvinced on the invented positives as on true negatives — it averaged the noise away
from image evidence alone.

It reads checkpoints already on disk through `AnnotatedReadings`, which refuses any member not trained with
`--hold-annotations`, so the annotated cells are held out by construction and the reading is not the
member's own supervision reflected back.

## The reading

All four `--hold-annotations` checkpoints, over the same 58 studies. Census identical across all four: 203
agreed positive, 188 agreed negative, 177 silent, 108 over-called, 8 missed.

| member | over-called tracking | 95% (study bootstrap) | missed |
|---|---|---|---|
| `band_span_hold` | **+0.683** | [0.600, 0.762] | +0.122 |
| `calib_span_hold` | **+0.538** | [0.467, 0.608] | +0.105 |
| `band_dense_hold` | **+0.700** | [0.627, 0.773] | +0.208 |
| `calib_dense_hold` | **+0.550** | [0.470, 0.630] | +0.265 |

## Three things follow

**1. The label lane is LIVE.** Every member is 54–70% of the way to a true positive on cells the
radiologist calls negative, and every interval excludes BOTH poles. The fit did not average the reader's
confident errors away. A better reader is a real lever, not a tidying exercise — which is the licence this
was run to get or to withhold, and it granted it.

**2. `--calibrated-targets` finally has a mechanism.** It cuts tracking by **−0.145** at the span corner and
**−0.150** at the dense corner — the same size twice, at two independent corners, and roughly 2× the CI
half-width. It is the only axis that has moved the board, and this is the first statement of WHY: it is
demonstrably decoupling the members from the reader's invented positives. That also predicts where the
remaining headroom is, since it removes only ~a fifth of a binding that starts near 0.7.

**3. The misses are NOT visible to the model** (0.11–0.27, near the radiologist pole). Consistent with the
24-cell ceiling a perfect silence-filler could reach, and an argument against spending anything on one. The
defect worth attacking is the over-call, not the silence.

## What it does not say

58 studies, 1.3% of the corpus; 108 over-called cells; cells within a study share a report and a
radiologist, so every interval here is bootstrapped over STUDIES. A ratio on this denominator is
DIRECTIONAL: it can tell 1 from 0, it cannot resolve 0.6 from 0.7. Read the −0.145/−0.150 as "roughly a
fifth", never as a constant.

It also says nothing about how much board score a further decoupling buys. Tracking is not AUC, and the
route from one to the other runs through a retrained table — which is the 126–147% pass-through class, the
only mechanism measured ABOVE full pass-through, because a label-table change retrains every seat.

## What it licensed next

`core.data.labels.finding_calibration` (this same day). The pooled `MEASURED_AGREEMENT` map flattens a
37-point per-finding spread — the radiologist agrees with 44.4% of the reader's positive Contusion calls
and 81.2% of its Synovitis calls — and the two findings fail in OPPOSITE directions. One shrunk logit
offset per finding, empirical-Bayes weighted, τ = 0.294 (sd 0.543 logits) surviving the moment subtraction.
See `core/data/labels/finding_calibration.py`, whose docstring also records the harness caveat: the offline
holdout is scored against the REPORT labels, which carry the very defect these offsets correct, so the
offline ruler is biased AGAINST this table and its number is a floor rather than the verdict.
