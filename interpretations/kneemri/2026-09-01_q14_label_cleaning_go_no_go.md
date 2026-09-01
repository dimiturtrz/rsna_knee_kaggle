# Q14 go/no-go: static label cleaning and loss weighting

Bead `knee-t7i5` asks for a written go/no-go on the two remaining sub-lanes of
the label-cleaning question, each with an expected label-flip count over the
4,407 studies, plus an explicit statement that corpus-wide re-extraction is not
the lane.

## Bottom line

- **Corpus-wide re-extraction is NOT the lane.** The over-call audit measured
clean extractor failure at 16/108 disputed cells (14.8%). A corpus-wide scan
flags more negation-like cells, but the same rules fire on agreed cells at a
rate that shows most cues are report style, not extraction error. The residual
extractor failures are a small share of the supervision gap.
- **(a) Targeted negation pass: GO as a measurement, not as the main bet.**
The current scope rule flags **876 positive cells (5.4% of 16,129 positives,
1.66% of the whole label matrix)**. The true-positive control suggests
~1.5% of positives are style false alarms, so the net fixable count is
probably **~700 cells**. The 12 disputed missed-negation cells are the
location-error tail of this population, not a separate systematic shape.
- **(b) Rubric-bearing re-read: GO, gated.** The rubric-gap ruler identifies
**502 backed contradiction cells** in English (40.1% of the corpus). If the
rate is language-flat, the corpus figure is roughly **502 / 0.401 ≈ 1,250
cells**. This is the larger, addressable lane. The 1,098 refused cells under
size rules are NOT targets.

---

## 1. Why corpus-wide re-extraction is not the lane

`core.data.labels.overcall_audit` classified the 108 confident false positives
from the annotated studies:

| category | false positives (n=108) | true positives (n=203) | difference |
|---|---|---|---|
| missed_negation | 12 · 11.1% | 3 · 1.5% | +9.6% |
| laterality | 2 · 1.9% | 6 · 3.0% | −1.1% |
| hedge_dropped | 2 · 1.9% | 12 · 5.9% | −4.0% |
| subthreshold | 46 · 42.6% | 48 · 23.6% | +18.9% |
| genuine | 38 · 35.2% | 106 · 52.2% | −17.0% |

Only `missed_negation` separates with a confidence interval that excludes zero.
`laterality` and `hedge_dropped` fire MORE often on cells the radiologist
agreed with, so they are report style, not extraction error. The clean
extractor-failure share is therefore **16/108 = 14.8%** of the over-call defect,
or about **2.3% of the 684 annotated cells**. That is too small to be the main
lane.

A corpus-wide run of the same classifier over all 16,129 positive cells in
`report_labels_v2.csv` gives:

| category | count | share of positives |
|---|---|---|
| missed_negation | 876 | 5.43% |
| laterality | 298 | 1.85% |
| hedge_dropped | 397 | 2.46% |
| historical | 285 | 1.77% |
| subthreshold | 4,638 | 28.76% |
| genuine | 6,699 | 41.53% |
| unlocated | 2,936 | 18.20% |

The large `missed_negation` count is real, but the control says it is mixed
with style false alarms. On the annotated controls the rule fires on 1.5% of
true positives; applying that as a style-correction to the corpus gives an
estimated **~700 net fixable cells**, not 876. `laterality` and `hedge_dropped`
remain style cues, so a generic "re-extract everything" would flip many cells
that are not actually wrong.

## 2. (a) The 12 missed-negation cells — systematic shape or residue?

The 12 disputed missed-negation cells are the ones the current scope rule still
flags after the forward-scope / clause-bound / lesion-family corrections. The
interpretation notes that the 3 surviving control cases are **location errors**:
the report denies the finding in one sentence and asserts it in another, and
the locator hands over the first sentence. That is a parse / multi-sentence
problem, not a vocabulary gap.

The corpus-wide scan shows negation is common (876 cells), but the fixtures
already cover the systematic shape — prepositional denials of attributes
("without extrusion"), clause-bounded forward scope, and normality predicates
in verb-final languages. The 12 disputed cells are the **location-error tail**
of that distribution, not a new shape that needs its own fixture.

**Go/no-go:** GO for a targeted extractor pass that writes corrected labels
for the scope-flagged cells, but with the expectation modest:

- Upper bound: **876 flips** (1.66% of all label cells).
- Style-corrected estimate: **~700 flips**.
- This is smaller than the rubric-mismatch lane and should be run as a
  side-measurement, not the primary label-cleaning card.

## 3. (b) Rubric mismatch — addressable by giving the reader the rubric?

`core.data.labels.rubric_gap` compares the host's published image-review
criteria against the project's training table (`merged_blend_v1.csv`). It
splits findings into two groups:

**Backed by annotations (contradicted cells are annotated negative):**

| finding | contradicted cells | share of corpus |
|---|---|---|
| Medial Meniscus | 165 | 1.87% |
| Lateral Meniscus | 165 | 1.87% |
| ACL | 98 | 2.22% |
| Baker's | 239 | 5.42% |
| **total backed** | **502** | **~11.4% of English cells** |

**Refused by annotations (subthreshold cells are often annotated positive):**

| finding | contradicted cells | share of corpus |
|---|---|---|
| MCL | 185 | 4.20% |
| Effusion | 524 | 11.89% |
| Medial OA | 123 | 2.79% |
| Lateral OA | 112 | 2.54% |
| PF OA | 154 | 3.49% |
| **total refused** | **1,098** | — |

The tool reads only the 1,768 English reports (40.1% of the corpus). If the
rate is language-flat, the backed count scales to roughly **502 / 0.401 ≈
1,252 cells**. That is the expected flip count from a rubric-bearing re-read.

**Go/no-go:** GO. This is the larger, addressable lane. The re-read should
use the host's twelve criteria plus the global specificity rule, rendered into
`ReadingPrompt.system()`, and must be scored on whether the **502 backed cells**
move. The 1,098 refused cells must NOT be used as success criteria — rewarding
agreement with a ruler the radiologists do not follow would be a perverse
goal.

Gates before banking the result:

1. `kneemri.evaluation.annotated_readings` must not regress.
2. The re-read must move the backed cells; if it does not, it has failed
   loudly rather than silently.

## 4. Loss-weighting note

This question originally bundled label cleaning with loss weighting. The
measurements above say the residual label errors are (1) small-ish for the
extractor lane and (2) material but data-fixable for the rubric lane. Neither
supports a new loss-weighting scheme: loss weighting can only reweight cells
that are already in the table, and the largest defects are cells that should
not be positive in the first place. Fix the table, then re-measure whether any
residual class imbalance still needs a weight.

---

## Numbers on disk

- `core.data.labels.overcall_audit` output and `<processed>/labels/overcall_audit.csv`
  (run 2026-08-31) give the 108-cell classification.
- `core.data.labels.rubric_gap` output (run just now) gives the 502 / 1,098 split.
- The corpus-wide scan in this note was run ad hoc via
  `core.data.labels.overcall_audit.OvercallAudit.classify` over every YES cell
  in `report_labels_v2.csv`.
