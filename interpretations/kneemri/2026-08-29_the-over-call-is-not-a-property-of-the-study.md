# The over-call is not a property of the study

*2026-08-29 · instrument: `core/data/labels/overcall_clustering.py` · 0 card, 0 slots*

## The lane this was gating

`noise_binding` measured the label lane LIVE: every member tracks the reader's confident false positives
at +0.54 to +0.70, and `--calibrated-targets` is the only axis that has ever moved that coupling (−0.145
and −0.150 at two independent corners). When `finding_calibration` retired the per-FINDING version, the
ledger recorded what it did *not* rule out, in its own words:

> "a per-STUDY confidence rather than a per-FINDING constant, or supervision that is not the reader at all."

Per-study shrinkage was therefore the last live mechanism in the lane. The obvious way to build it — a
covariate from the report TEXT — was already dead: `ReaderAgreement.signal_report` priced two text flags
and **both came back backwards**, with the cause named (a regex cannot supply an independent weight,
because the reader has already read the same sentence, and its verdict levels ARE its confidence in that
text).

## What was asked instead, and why it is cheap

A per-study weight can only pay if there **is** a per-study latent. If each cell's over-call were an
independent draw at the pooled rate, the per-study counts would be Binomial and any per-study weight
would be fitting sampling noise — exactly the way twelve per-finding scalars did.

So the question is not "which covariate predicts over-call" but "is the over-call over-dispersed relative
to independent cells at all." That needs no covariate, no card and no slot: the variance of the per-study
over-call count over the Binomial variance at the pooled rate, with a bootstrap **over studies** (the unit
the claim is made at — resampling cells would destroy the structure being tested).

Pre-registered before running, in the module's own report: interval above `1.0` licenses an arm; interval
below `1.0` says suspect the census; interval spanning `1.0` closes the lane.

## The reading

```
56 annotated studies, 311 positive calls, 108 over-called (34.7%)
per-study over-call dispersion 1.24 [0.83, 1.65] against 1.00 for independent cells
```

The interval brackets independence. The point estimate leans clustered and cannot be distinguished from
a coin.

## What follows

**The per-study branch of the label lane closes without card.** The third branch fired as written: the
latent a shrinkage would grip is not measurable at this size, so an arm built on it would be fitting the
same noise `finding_calibration` fit — and would arrive with a plausible-looking number attached.

What stays open is the clause the ledger wrote beside it and this instrument says nothing about:
**supervision that is not the reader at all.** The reader's over-call is real (34.7% of its positive
calls are wrong) and it is coupled to every member; it is just not *addressable* by re-weighting the
reader's own output.

## What it cannot say

The denominator is 56 studies — not the 58 the annotation count is usually quoted as, and not the 311
positive calls those studies carry. A ratio resolved to `[0.83, 1.65]` can separate strong clustering
from none and nothing finer; it cannot rank two clustering strengths, and no arm should be designed off
its magnitude. It also measures only the annotated slice, so it describes the studies a human graded, and
generalises to the corpus only as far as that slice does.
