# The medial-over-lateral gap is not prevalence either

**2026-08-18.** `python -m kneemri.evaluation.prevalence_match --labels report_labels_v2.csv
--resamples 300`. Corpus-wide on the published four-seed out-of-fold ruler, 4,407 studies, zero GPU.

knee-yrg measured the two compartment pairs far apart and refuted the laterality-flip explanation; then
knee-1v0n refuted the slice-band one (the band's cut is symmetric, and a symmetric deletion cannot make
an asymmetric deficit). Two candidates were left. This is the cheaper one: the lateral findings carry
about half the positives, so the gap could be sample size rather than anatomy.

## The numbers

| pair | rich | poor | full gap | matched gap, 95% | counts scored at |
|---|---|---|---|---|---|
| meniscus | Medial 0.8602 | Lateral 0.7992 | **+0.0611** | **+0.0610 [+0.0522, +0.0719]** | 1121+ / 2060− |
| OA | Medial 0.8683 | Lateral 0.8246 | **+0.0437** | **+0.0438 [+0.0375, +0.0499]** | 1087+ / 2788− |

Both findings of a pair are subsampled, without replacement, to ONE shared positive and negative count —
the smaller of each side — and differenced within every draw, 300 draws.

**Both intervals exclude zero by a wide margin, and the matched gap equals the full gap to the fourth
decimal.** Prevalence is refuted as an explanation, and refuted cleanly: it does not shrink the gap at
all.

## Why "the rich label at the poor label's counts" is unsatisfiable, and what was measured instead

The obvious design — draw the rich finding down to the poor finding's counts — cannot be run. A finding
with MORE positives has FEWER negatives in a fixed corpus: Medial Meniscus cannot supply the 3,293
negatives its lateral twin has. Taking the smaller of each count gives one population both findings can
be scored on, at one base rate. That is the comparison the question asks for, and the module refuses the
other one rather than silently clamping.

## The scope, stated rather than implied

Matching controls the **measurement** half completely and the **training** half not at all. The medial
head was still fitted on every positive it had, so "twice the gradient signal" remains live and is a
different experiment (it needs the card). What is now closed is the reading that the lateral numbers are
merely noisier: at identical counts they are not noisier, they are lower.

That leaves knee-yrg with one cheap candidate unrun — the in-plane one: the lateral compartment sits
nearer the edge of a 130 mm crop centred on the joint. It is measurable against the crop geometry,
label-free, the same way the band was.

## What it sharpens

knee-dtk already splits Lateral Meniscus as the largest MODEL-side gap of all twelve (report 0.8429,
model 0.6497, −0.1932) — a vision failure rather than a label artefact. This is the corpus-scale version
of the same statement, on 1,121 positives instead of 58 studies, and with the sample-size objection
removed rather than assumed away.

## Sources

- `python -m kneemri.evaluation.prevalence_match` (`PrevalenceMatch.matched`, seeded at 2026).
- knee-yrg (the profile and the flip refutation), knee-1v0n /
  `interpretations/kneemri/2026-08-17_slice_band_coverage.md` (the band refutation), knee-dtk (the
  label-vs-model split).
