# The residual has no acquisition population, and one competition field is a duplicate — knee-y7v3

Reproduce: `python -m kneemri.evaluation.population_split --records <the twelve honest12 checkpoints>`.
Same pool, same 909 held-out studies and same prevalence-matched operating point as
`2026-09-01_a-quarter-of-the-residual-is-unanimous.md`; this cuts those errors by the study instead of
by the finding.

## What there was to cut on

knee-y7v3 asks for slice count, laterality and scanner. The competition ships none of the last two.
`train_series.csv` is the whole of the acquisition metadata — one row per series, with
`Fluid_Sensitive`, `Fat_Suppression` and `Anatomical_Plane` — and of its three fields:

- **`Anatomical_Plane` is not an axis.** All 4,407 training studies carry all three of Axial, Coronal
  and Sagittal. A cut on plane coverage has exactly one stratum.
- **`Fat_Suppression` is `Fluid_Sensitive`.** The two columns agree on **24,371 of 24,371 rows**. Two of
  the three published fields are one field.
- **Series count is the only real axis**: 3 to 14, median 5.

## The reading

| stratum | studies | positives/study | errors | per study | **per positive** | unanimous |
|---|---|---|---|---|---|---|
| 4 or fewer series | 151 | 2.81 | 205 | 1.358 | **0.482** | 25.4% |
| 5 series | 478 | 3.80 | 1008 | 2.109 | **0.555** | 22.8% |
| 6 series | 144 | 4.02 | 321 | 2.229 | **0.554** | 22.7% |
| 7 or more series | 136 | 3.79 | 258 | 1.897 | **0.501** | 31.4% |
| 2 fluid-sensitive | 78 | 4.18 | 169 | 2.167 | **0.518** | 24.3% |
| 3 fluid-sensitive | 643 | 3.49 | 1203 | 1.871 | **0.536** | 24.1% |
| 4+ fluid-sensitive | 188 | 4.07 | 420 | 2.234 | **0.549** | 25.0% |

**The per-study column is confounded and must not be read.** It rises with series count (1.36 → 2.23),
which looks like "thin acquisitions are harder" and is the opposite of the truth. A longer protocol is
a sicker knee: positives per study go 2.81 → 4.02 across the same bands. The members are not being hurt
by short protocols; short protocols belong to knees with less to find.

**Normalised, there is no population.** Errors per positive is 0.482 / 0.555 / 0.554 / 0.501 across
series count and 0.518 / 0.536 / 0.549 across fluid-sensitive count. The correlation between a study's
series count and its error load is **0.058**. Whatever the pool is failing at, it is not failing at a
kind of study that the competition's acquisition metadata can name.

## What it says

**The data lane closes on this axis.** Oversampling thin acquisitions, reweighting by protocol length,
or rendering short studies differently all target a population that does not exist in the residual. That
is a real closure, not an absence of evidence: the cut was run, at the right denominator, with the
confounder removed, and it came back flat.

**It agrees with the failure-mode cut.** `residual_split` found the errors spread across findings with
a 24.3% unanimous floor and only ~5.4% recoverable by a combiner. This adds that they are also spread
across studies. Two independent cuts of the same residual both say the same thing: our errors are not
concentrated anywhere addressable by reseating or resampling, which is why the remaining lane is a
genuinely different member — the same place `2026-09-01_the-trunks-are-tied-frozen-and-not-fitted.md`
points.

**`Fat_Suppression` should never be used as a feature or a stratum.** It is `Fluid_Sensitive` under
another name. Any model, sampler or analysis that reads both is reading one signal twice.

## What this does NOT establish

- **The 31.4% unanimity in the 7+ band is 136 studies and 258 errors.** It is the one cell that moves,
  and it is one of the two smallest strata; it is not a finding at this n.
- **Nothing here transfers to the hidden test set.** `test_series.csv` carries three studies. The public
  test acquisition metadata is a stub, so the distribution we are scored on cannot be compared to the one
  we trained on by this route. That is knee-5nbv's question and this instrument cannot answer it.
- **"No population" means no population THIS METADATA NAMES.** Scanner, field strength, laterality,
  patient age and referral reason are all absent from the competition data, and any of them could carry
  the structure that series count does not.
- **The operating point is prevalence-matched**, so false negatives equal false positives by construction
  per finding; this cut inherits that and cannot say whether any stratum is over- or under-called.
