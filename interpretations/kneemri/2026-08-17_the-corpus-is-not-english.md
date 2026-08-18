# The corpus is not English, and a THIRD of it is invisible to the patterns we read it with

**2026-08-17, corrected 2026-08-19.** CPU only, no training, no LLM pass:
`python -m kneemri.labels.report_language` over all 4,407 reports. The census already existed; what it had
never been read for is the size of what our extraction path cannot see.

**The correction.** This document's own Limits section said of the 412 `unknown` reports that "nothing
here says whether those 412 reports are gradeable" — and left it there. Read, they are **Croatian**: 406
of the 412, a full eighth radiology language absent from `SIGNATURES`, and the LONGEST reports in the
corpus (median 1,164 characters against 945 for the classified ones). They grade **86.2%** of the time in
their own words and **0.0%** to English patterns. That is the single largest blind spot the English ruler
had, it was the largest unexplained residue in the table, and leaving it as `unknown` under-reported this
document's headline by eight points. **An unclassified residue is a language you have not named yet.**

| lang | n | share | graded in its own words | graded to English patterns |
|---|---|---|---|---|
| en | 1769 | 40.1% | 64.7% | 64.7% |
| es | 727 | 16.5% | 55.2% | **1.5%** |
| tr | 546 | 12.4% | 80.8% | 51.5% |
| hr | 406 | 9.2% | 86.2% | **0.0%** |
| el | 321 | 7.3% | 75.1% | 24.6% |
| de | 259 | 5.9% | 83.8% | **2.7%** |
| bg | 220 | 5.0% | 61.4% | 30.5% |
| nl | 153 | 3.5% | 80.4% | **2.6%** |
| unknown | 6 | 0.1% | — | — |

## The denominator, stated first

These are shares of the **4,407-report corpus**, not of the cells some pattern happened to locate. That
distinction is the one this project has already paid for once: knee-ttk's negation finding was filed as
"19% of report-YES cells" and corrected to 6.6% when the located subset was divided out properly.

The first consequence falls out of the table's top row. **English is 40.1% of the corpus.** Every
English-pattern measurement here — including the negation defect at z = 3.00 — speaks for two fifths of
the data and is silent about the rest. Not wrong; scoped.

## The number

Reports graded in their own language but invisible to English patterns, summed per language as
`n × (own − english)`:

    es 390 · hr 350 · de 210 · el 162 · tr 160 · nl 119 · bg 68  =  ~1,459 reports  =  33.1% of the corpus

**A third of the corpus carries severity vocabulary that the current extraction path cannot see at all**,
and the loss is concentrated rather than spread. Spanish, Croatian, German and Dutch together are 35.1% of
reports and convert at 1.5%, 0.0%, 2.7% and 2.6% — effectively zero. Turkish is the opposite case
(80.8 → 51.5) because its clinical vocabulary carries enough Latin loanwords to survive an English
pattern, and it is the cheapest language to leave alone.

Croatian is the extreme of the same mechanism: Slavic in Latin script, so no loanword bridges it and no
alphabet check flags it as foreign, which is exactly why it fell through as `unknown` rather than as a
visibly-unreadable block. It converts at 0.0% — a clean zero, the only one in the table.

## What it changes

The severity branch (knee-mxj) has been carried as two upgrades — a better text ENCODER for the auxiliary
embedding target, and a better READER emitting ordinal grades. On this corpus they are not independent:
**both have to be multilingual first and medical second.** The loss is 25.2% of the corpus to *language*,
and only then whatever is lost to reading quality on the part we can already read.

It also prices the LLM pass honestly. The pass buys the es/de/nl quarter, which no regex tier reaches —
which is the same conclusion `report_language.py`'s own docstring reached from the other end when a
per-finding English regex was measured to add nothing beyond the levels already present. The zero-cost
path is closed because the expensive path is the only one that goes where the data is.

## Limits

- ~~`unknown` is 9.3% and grades at 0.0%…~~ **closed 2026-08-19**: it was Croatian, it grades at 86.2%,
  and it is now in the table. The residue is 6 reports (0.1%) and no longer worth a limit.
- The per-language "graded" rate is a report-level boolean — it says a severity marker is present, not
  which of the twelve findings it grades. Attribution is the difficulty the LLM pass exists to buy, and
  this document does not touch it.
- No claim is made that fixing the quarter moves the score. It sizes the population a stronger reader
  reaches; whether better supervision on it clears the 1.5% bar is a training question, unrun.

## Sources

- `python -m kneemri.labels.report_language` (the census; `ReportLanguages.census`).
- knee-mxj (the severity branch this sizes), knee-ttk (the negation defect and its denominator
  correction), knee-dbw (replace the rule engine with an actual reader).
