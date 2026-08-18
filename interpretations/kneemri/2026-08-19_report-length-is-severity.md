# Report length predicts how many findings a study has, and it is not the reader padding

**2026-08-19.** CPU only, no training, while `band_wide` held the card:
`length_confound` / `length_errors` over the 4,406 studies where a report merges with a read label row,
then over the 57 radiologist-annotated ones.

## What was being checked, and why

The Croatian block found the same day
(`2026-08-17_the-corpus-is-not-english.md`) carries the corpus's most extreme labels — 97.8% of its
studies have at least one positive, against 89.3% for English, and 5.44 positive findings a study
against 3.54. Croatian reports are also long. The obvious confound is that **the reader harvests more
YES out of more text**, in which case 9.2% of the corpus carries systematically inflated labels and a
per-language recalibration is a real "better data handling" lever.

## Length predicts findings everywhere, and strongly

Mean positive findings a study by report-length quartile, computed **inside** each language so language
is held fixed:

| lang | Q1 | Q2 | Q3 | Q4 | Q4−Q1 | rank corr |
|---|---|---|---|---|---|---|
| en | 2.15 | 2.87 | 4.74 | 4.40 | +2.24 | +0.416 |
| es | 1.23 | 2.94 | 2.52 | 5.47 | +4.23 | +0.609 |
| tr | 1.01 | 2.88 | 4.01 | 6.49 | +5.48 | **+0.846** |
| hr | 3.25 | 5.10 | 6.02 | 7.39 | +4.14 | +0.650 |
| el | 1.43 | 3.20 | 4.34 | 6.03 | +4.60 | +0.745 |
| de | 2.51 | 3.18 | 4.56 | 6.28 | +3.77 | +0.607 |
| bg | 1.85 | 2.49 | 3.20 | 4.95 | +3.09 | +0.644 |
| nl | 1.92 | 3.08 | 4.26 | 4.92 | +3.00 | +0.573 |

Universal, and large: a Turkish report in the top length quartile carries 6.5 positive findings against
1.0 in the bottom. **Length is not a nuisance variable here — it is most of the label's variance.**

It also does not explain Croatian. English reports are the LONGER of the two (median 1,277 characters
against 1,172), and restricted to Croatian's own 742–1,692 character band the gap barely moves: **+1.90
unmatched, +1.72 length-matched**. Croatian describes more findings per unit of text, not more text.

## It is real severity, not harvesting

Only ground truth separates "the radiologist wrote more because there WAS more" from "the reader found
more because there was more text to find it in". Over the 57 annotated studies — 684 cells, 311 of them
called positive by the reader — the two sides of that question move differently:

    rank corr(chars, positive calls per study) = +0.506   over 56 studies with any positive call
    rank corr(chars, precision on those calls) = -0.192

| length tercile | studies | chars | calls | radiologist agrees |
|---|---|---|---|---|
| 1 | 24 | 183–1147 | 105 | 67.6% |
| 2 | 18 | 1149–1667 | 108 | 63.9% |
| 3 | 14 | 1814–3891 | 98 | 64.3% |

The volume rises steeply and the precision does not fall with it. The 3.3-point spread across terciles
is **under one standard error** — at ~100 calls and p ≈ 0.65 the binomial SE is 4.8 points, and the
effective n is smaller still because the calls cluster inside 14–24 studies each. The extra findings a
long report yields are about as likely to be true as the findings a short one yields.

## What it changes

**It kills a candidate before it cost an arm.** No per-length and no per-language label recalibration is
warranted: there is no measured inflation to calibrate away. Croatian's positive rate is a real
reporting or population difference, and the correct handling of it is none.

**It leaves one candidate standing, unrun and unpriced.** Report length is a cheap, language-independent,
reader-independent proxy for how much there is to say about a study, correlated with the finding count
at ρ ≈ 0.5 — and the model never sees it. Regressing it as an auxiliary target costs one scalar per
study and reuses the `ReportSupervision` plumbing already in the trainer. Nothing here says it moves the
score; it says the signal is present and free to construct.

## Limits

- **57 annotated studies.** The precision comparison is the weak half of this document: it can exclude a
  large precision collapse with length, not a small decay. A drop of a few points is entirely live.
- The annotated set is not language-balanced, so this says nothing about whether the reader's precision
  differs on Croatian specifically — that block is too thin in ground truth to split.
- "Positive finding" is `level > 0.5` on the reader's twelve columns, i.e. the same midpoint
  `LabelScores` uses. Denominator for the corpus half is 4,406 merged studies of 4,407 reports.
- Length is measured in characters, which is not neutral across languages (German compounds, Greek
  script). Every within-language row above is immune to that; only the cross-language medians are not.

## Sources

- the corpus half: reader levels against `Report` length, per language and length-matched.
- the ground-truth half: `ReaderAgreement.cells` over the 57 annotated studies.
- `2026-08-17_the-corpus-is-not-english.md` (the Croatian block this started from), knee-mxj (the
  severity branch), knee-dbw (replace the rule engine with an actual reader).
