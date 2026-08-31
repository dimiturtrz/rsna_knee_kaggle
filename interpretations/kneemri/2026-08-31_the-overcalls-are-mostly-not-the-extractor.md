# The over-calls are mostly not the extractor

`knee-deo5`. Reproduce with `python -m core.data.labels.overcall_audit`; every classified cell and the
span it was read from land in `<processed>/labels/overcall_audit.csv`.

## The precondition, answered first

The report TEXT is on disk and joinable. `raw/train.csv` carries `StudyInstanceUID` and `Report` in the
same file as the radiologist columns: 4,407 rows, 58 annotated, **0 null and 0 empty reports among the
annotated**, 57 joining to `report_labels_v2.csv` — the same 57-study / 684-cell denominator
`reader_agreement` scores. Nothing here substitutes the derived label table for the text.

What the text does NOT carry is a span. `report_labels_v2.csv` holds value, verdict and a confidence that
is a deterministic function of the verdict, and no offset or evidence sentence. So the span every rule is
scored over is this module's RECONSTRUCTION of the sentence the reader used, not the sentence the reader
used. That is the instrument's first-order limit and it is printed with the number.

## The numbers, with both halves of every fraction

684 annotated cells over 57 studies; the reader called 311 positive, the radiologist agreed on 203 and
refused 108. The 108 span 49 studies and 8 languages — 45 of them English, which is why every text
instrument this repo had before (all English-only) could only ever see 42% of them.

Located the finding in **101 of 108 cells (93.5%)**; the 7 misses are Contusion (4) and Medial OA (3).

| category | false positive (n=108) | true positive (n=203) | difference [95% study-clustered] |
|---|---|---|---|
| missed_negation | 12 · 11.1% | 3 · 1.5% | +9.6% [+4.4%, +15.3%] |
| laterality | 2 · 1.9% | 6 · 3.0% | -1.1% [-4.3%, +2.5%] |
| hedge_dropped | 2 · 1.9% | 12 · 5.9% | -4.1% [-8.6%, +0.0%] |
| historical | 1 · 0.9% | 11 · 5.4% | -4.5% [-9.4%, -1.0%] |
| subthreshold | 46 · 42.6% | 48 · 23.6% | +18.9% [+6.6%, +32.5%] |
| genuine | 38 · 35.2% | 106 · 52.2% | -17.0% [-27.6%, -5.5%] |
| unlocated | 7 · 6.5% | 17 · 8.4% | -1.9% [-8.0%, +4.0%] |

The TRUE-POSITIVE column is the control and it is what makes this a measurement. Three of the four
extractor categories fire MORE OFTEN on cells the radiologist AGREED with — they are report style, not
extraction error, and had this been run without the control, `hedge_dropped` and `historical` would have
been read as 3 more extractor failures. Only `missed_negation` separates, and its interval excludes zero.

## The answer: neither lane, and the lane the bead framed is the smaller one

**Clean extractor failure is 16 of 108 (14.8%)** — 12 negation, 2 laterality, 2 hedge. Real, mechanism-
identified, and corpus-wide if fixed. But it is one cell in seven of the defect, not the defect, and 15%
of a 15.8% supervision hole is ~2.3% of annotated cells: nowhere near a 1.5%-absolute lever on its own.

**The majority — subthreshold 46 + genuine 38 = 84 of 108 (77.8%) — is rubric mismatch and real
disagreement.** And `subthreshold` is NOT free to bank: `rubric_gap` BACKED it for meniscus, ACL and
Baker's while REFUSING it for MCL and the size family, and `severity_boundary` retired "read the report
better" for Effusion outright by showing the size qualifier is not even monotone in the gold classes. So
the largest bucket here is one the repo has already, per finding, partly refused.

So: **Rep-GLS-style loss SHARPENING stays contraindicated.** The softening half already shipped is the
correct half. That closure is the finding, and it agrees with the 2026-08-20 manual pilot (1,200 cells,
92.9% agreement, only 6 of 48 contested cells in the over-call direction — "the reader fabricates
DENIALS, never positives"), which is a prior this measurement did not have to overturn.

The negation slice is worth a targeted extractor pass on the corpus, priced as ~15% of the over-call
defect and not as the lane.

## Honest reading of the instrument

Regex over reconstructed spans, in eight languages, hand-built by reading the reports it then scores.
Two things earn it some trust and one does not.

Earned: the **scope rule**. Scoring a negation cue anywhere in the span fired on SIXTEEN of the reader's
confident TRUE positives — every one of them denying a MODIFIER while the finding was asserted (`without
extrusion`, `zonder significant botoedeem`, `без дислокация`, `χωρίς καθίζηση`). Splitting the vocabulary
into forward-scoping DENIALS bounded at the next clause boundary and clause-scoping NORMALITY predicates,
plus a lesion-family test for what the denial must land on, took negation from 18 to 12 on the disputed
cells and 16 to 3 on the agreed ones. Every one of those shapes is pinned in
`tests/unit/core/data/labels/negation_scope.py`.

Also earned: 93.5% located, against 44.1% for the English-only detector this replaces.

NOT earned: the 3 surviving control negations are LOCATION errors, not vocabulary errors — the report
denies the finding in one sentence and asserts it in another, and the locator hands over the first. That
is not fixable without a parse, so the negation count is an over-count by an unknown small amount, which
makes the 14.8% extractor share an upper bound rather than an estimate. And a negation phrased in a way
no pattern covers reads as a plain assertion and lands in `genuine` — a bias TOWARD the answer this
document reaches, stated so the reader can discount it.

A regex negation detector auditing negation errors measures its own failure mode. The control is what
keeps that honest, and the control is why the three style categories were not filed as findings.
