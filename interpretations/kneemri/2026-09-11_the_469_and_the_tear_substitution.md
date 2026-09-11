# The 469: what a blind hand-adjudication said about our own label table

2026-09-11. Closes the audit opened on 2026-09-10. Sources: `audit400/holdout469/`, D344–D350, M354–M357.

## What was done

469 cells were adjudicated by hand, blind, in ten chunks. Each cell is a `v2_lone_dissent` cell — the
three foreign report-label tables vote unanimously and our v2 differs — intersected with the fold-0
holdout. The rubric was fixed before the first chunk: **does the report AFFIRM the named finding?
1 = affirmed at any severity, 0 = explicitly negated OR not mentioned.** Only the report text was
read; the packs carry no label, no confidence, no column statistics.

Two protocol rules were pre-registered and both held:

1. Each chunk's verdicts were written to disk BEFORE the next chunk was opened.
2. `adjudication_key.csv` (sha `6b7c5bffaf2944a0`) was not opened until the 469th verdict was on disk.

A third rule — never grep the pack's report text against `raw/train.csv` to recover study UIDs — was
never violated and stays in force for any future pack.

## The headline

    v2 correct        51 / 469   0.109
    cluster correct  418 / 469   0.891

The pre-registered expectation was that v2 would be right on ~25% of these cells, from two independent
purity readings. It is right on 10.9%. The dissent set is not a coin flip; it is a near-uniform v2
defect.

Per column, `affirmed` = the flagged finding is actually present in the report:

    column              n   affirmed   v2_acc
    Contusion         120         17    0.167
    Medial Meniscus   111          0    0.000
    Lateral Meniscus   89          0    0.000
    ACL                60          0    0.000
    Medial OA          21          3    0.476
    Lateral OA         19          4    0.632
    MCL                17          0    0.000
    PF OA              11          4    0.364
    Effusion            9          1    0.222
    Baker's             6          3    0.500
    Synovitis           6          0    0.000

**The two rates are different statistics and must not be swapped.** 32 of 469 cells (6.8%) are
affirmed; v2 is right on 10.9%. They differ because on some cells `v2_value` is 0 rather than 1, so
"the finding is absent" and "v2 was right" are the same event there and opposite events elsewhere.

## The rubric did not drift

The split was pre-registered in M355 before chunks 9–10 were read:

    split     n  affirmed   rate
    report   93         5    0.054     (chunks 9-10, never used to derive anything)
    tune    376        27    0.072     (chunks 1-8, the Contusion qualifier was derived from these)

Four hours of adjudication across six context windows, and the clean-read rate agrees with the spent
one. Had the rubric loosened or tightened, these would diverge.

## Three readings

**(a) The tear columns are not a coin flip and never were.** 277 cells — ACL 60, MCL 17, Medial
Meniscus 111, Lateral Meniscus 89. v2 is wrong on **all 277**. Every one of them is intrasubstance
degeneration, mucoid or myxoid degeneration, free-edge fibrillation, or a tear named on the OTHER
meniscus, graded present by v2. This is the criterion-line defect measured by hand rather than by a
reader, and it is total. It overturns the earlier reading that the deletion was "a recall-for-precision
swap": on the contested cells it is not a swap, it is a correction.

**(b) The Contusion qualifier is the smaller half.** Contusion is 120 of the 469 and v2 is right on 20
of them. The clarified criterion sentence reaches the degenerative-marrow-edema class, ~60 cells — one
quarter the size of the tear-column correction.

**(c) The OA columns run the other way and must be left alone.** Lateral OA 0.632, Medial OA 0.476,
Baker's 0.500. v2 is right about half the time there, so a blanket substitution would destroy as much
signal as it repairs. The under-call dip that leans OA is the same boundary seen from the other side.

## Why 277/277 is a population statement, not a sample artefact (D350)

The 469 is `v2_lone_dissent ∩ fold-0 holdout`. Fold assignment is **a hash of the report text**
(`core/data/corpus/folds.py:61`) — deterministic, pseudo-random at study level, designed so duplicate
reports share a fold. No confidence, column, soft margin, or any other label-side variable enters the
draw. The 469 holds 20.1% of the corpus's 1,379 tear-dissent cells against the holdout's 20.6% share of
studies: no enrichment, no depletion. Clustering is mild — mean 1.4 cells per study, max 4 — and the 277
tear cells span **223 distinct studies**, so the 1.000 rests on 223 independent reports.

## What was banked

`labels/report_labels_v6tears.csv` — v5blend with the cluster's soft value written into the **1,379
contested tear cells** (ACL 298, MCL 83, Medial Meniscus 567, Lateral Meniscus 431) across **1,097
studies**, 1,377 of them v2=1 over-calls. Built by `research/scripts/build_v6_tears.py`. 1,371 cells
actually changed; on 8 v5blend already held the cluster's value.

The operation is `column-substitution`, second use — not a blend. `table_blend.py` is a mean/tiebreak
combiner over foreign readers of the same report, a different operation, and building this table
through it would be a misuse.

Two guards, stated before banking rather than after:

1. **Only the contested cells are written.** v2's uncontested tear cells are a different population,
   where the first audit put unanimous-cell error at 1.48%. The script filters the selector to the four
   tear columns before it writes anything, so the agreed mass is untouchable by construction.
2. **The residual-risk budget is ~14 cells.** The cluster's 1.000 rests on 223 studies, so a residual
   error at the ~1% level is invisible at that n; over the corpus's 1,097 tear-dissent studies that is
   about 14 wrong cells. That is the whole budget, and it is the number a later refutation should be
   measured against.

The value written is the cluster's SOFT mean, not its hard vote — v2's targets are confidence-graded,
and writing hard 0/1 into exactly the cells an arm varies would smuggle a target-distribution change
into a one-variable arm. `__conf` and `__verdict` are untouched: a corrected cell stays supervised.

## What is still open

- **The 936 non-tear contested cells.** Refused as a blanket operation on the (c) reading. The open
  question is whether a selector-side statistic — soft margin, cluster tightness, or whether the report
  names a compartment explicitly — separates the cells where the cluster is right from the cells where
  v2 is. 63 of the 469 are OA/Baker's cells with a gold answer: enough to test one rule, not three.
- **The v6 prompt read** — the Contusion qualifier, all twelve columns re-read so the qualifier is the
  only variable (D349). Pre-registered as an INSTRUMENT with near-zero odds of clearing the bar alone;
  it licenses or refuses the clarification axis for the other eleven columns. Gated behind the 58-study
  `LocalReader.annotated` probe, and it cannot share the card with a training arm.
- **`v6tears_span`** (knee-57aec), queued ahead of `labelwise_depth24`. Prediction on record: at or
  below `v5blend_span` on the v2 ruler, at or above on the corrected T1.
