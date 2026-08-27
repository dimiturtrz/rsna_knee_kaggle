# The host's rubric against our training table: the OA supervision is the defect

2026-08-27. Tool: `core/data/labels/rubric_boundary.py`. No ground truth, no card, no submission slot.

## What was asked

Discussion topic 733491, host reply: the challenge labels come from a standardized **image** review, not from
the report. Three divergences are named outright — a meniscal tear must be *definite* (intrasubstance
degeneration not reaching the articular surface is negative), marrow-edema wording does not by itself make
contusion positive, and cartilage wording does not by itself make osteoarthritis positive.

Every target this project trains on is derived from the report by an LLM reader whose system prompt says
"say what THE REPORT SAID about it. Do not diagnose, do not infer" — the correct instruction for
transcription and the wrong one for this competition. So: over the corpus, how many cells does the host's
stated rule call negative while our training table calls them positive?

The table measured is `merged_blend_v1.csv` — the one the champion's seats were **fitted** on. The eval
ruler (`report_labels_v2`) is deliberately not the default: a contradiction in the ruler moves what a
holdout number *means*, a contradiction in the training table moves what the model *learned*.

## Both denominators, stated once

1768 English reports of 4407 (**40.1%**), 10 608 cells cross-tabbed. The non-English 59.9% is
**unmeasured** — every share below is a share of the English subset and never of the corpus. This project
has read the first as the second three times in one session; the arithmetic is one line and it is the whole
difference between a P1 and a P2.

## The lead was wrong

| family | contradicted | of its excluded bucket | of the family's English cells |
|---|---|---|---|
| meniscus | 4 | 3.3% | 0.11% of 3536 |
| contusion | 39 | 18.2% | 2.21% of 1768 |
| **osteoarthritis** | **615** | **86.9%** | **11.60% of 5304** |

The host named meniscus first and `2026-08-20_label_contest_pilot_tables.md` had already caught it by hand,
so meniscus is where the search started. Meniscus is nearly clean: four cells. The mass is osteoarthritis —
615 cells whose only OA-family sentence is chondral / cartilage / chondromalacia wording, which the host
says does not determine the label, and which our table calls positive anyway. Fourteen of fourteen sampled
rows were the named case on inspection.

## The count is a floor, and here is the reason

The host's own phrasing of the meniscus case — "degeneration **without** a definite tear" — carries a
negation word, so it bins as a denial rather than as excluded wording and never reaches the contradiction
line. Those cells are still printed (a positive in the denial row is a reader error under any reading), but
they are not counted. A `BOTH` bucket ("degenerative tear reaching the surface") is genuinely positive under
the rule and is likewise excluded. The line is a floor on the disagreement, never an estimate of it.

One false reading was caught before the number was quoted, by printing the *sentences* rather than the
counts: "Articular cartilage of the patellofemoral joint is well-maintained" was sitting in the excluded
bucket. A knee report says negative without a negation word. Adding `intact|preserved|well-maintained|normal`
to the denial guard moved the OA excluded bucket 1095 → 708 while the contradictions moved only 638 → 615 —
the fix removed true negatives, and the headline is robust to it.

## A second, larger defect, and the row that keeps it honest

The OA "no sentence names this column" bucket holds **714 positives of 2604 cells (27.4%)**. Splitting them
by where the value could have come from:

| | cells | share |
|---|---|---|
| spillover — another column of the same study is named **and** positive | 171 | 23.9% |
| global — OA wording present, localised to no compartment ("tricompartmental osteoarthritis") | 67 | 9.4% |
| **unsupported** — no OA-qualifying wording anywhere, no named positive column | **476** | **66.7%** |

The `global` row is why this is not a subtraction. Without it, every unlocalised report reads as an invented
label and the headline becomes the compartment regex's blind spot rather than a fact about the table. It
takes 67 cells. The residue is 476 — **9.0% of the family's English cells** — where the training table's
value is not derived from the report at all.

"Unsupported" means no wording matching the OA *qualifying* set
(`osteoarthr|degenerative joint|osteophyt|joint space narrow|spur`) anywhere in the report. A report carrying
only cartilage wording lands here, and under the host's rule that is the right place for it. Which makes the
two findings one finding seen from two sides: **the reader is producing OA positives from something other
than an OA statement.** 615 contradicted + 476 unsupported = **1091 of 5304 English OA cells (20.6%)**.

## What this does not say

It does not say 1091 cells are wrong. The ground truth is an image review, which can be positive on a report
that says nothing; and `confident224` (−0.0071) already inverted this project's premise that unknown cells
are noise, measuring real signal in them at 0.2875 weight. A confident prior can carry what a report
withholds. What is established is narrower and still load-bearing: for a fifth of the OA supervision, the
value is not what the report says, while every claim this project makes about that table assumes it is.

## Sizing against what the label lane has actually paid

The measured label lane is small. `blendlabels` returned +0.0034 on the mean and +0.0070 on the six
label-bottlenecked findings — both at or under the 0.0062 arm-vs-arm floor. `table_blend` bracketed every
reweighting of the readers we already have at +0.0015 [−0.0069, +0.0121], and concluded that a better label
table has to come from a better **reader**, not a new weighting of the readers here.

This is that reader question, with a population attached. A correction touching ~20% of the cells on three OA
columns is a materially larger perturbation of the training table than any blend recomposition was. It is
still unpriced, and pricing it costs a card arm — which is the peer's tonight. What it does not cost is a
submission slot, and the measurement above is complete without one.

Beads: `knee-rwcy` (the rubric), `knee-312m` (the attribution).
