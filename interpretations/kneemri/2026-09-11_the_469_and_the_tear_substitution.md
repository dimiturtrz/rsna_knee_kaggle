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

## The non-tear half: a pooled margin rule is Simpson's paradox (D351 + M358)

D351 measured a soft-margin gradient on the 72 adjudicated OA-family cells: at `|cluster_soft − v2_soft|`
≥ 0.6 the cluster is right 0.83, below it a coin flip, Spearman −0.323 (p=0.006). It proposed
substituting the 365 corpus cells clearing that bar.

Re-cut on **all 192** non-tear adjudicated cells — D351's 72 plus the 120 Contusion cells — the rule
does not survive pooling:

    stratum      n    cluster acc
    hi (>=0.6)  67    0.791
    lo (<0.6)  125    0.704

The "coin flip below the threshold" is an artefact of the 72-cell denominator. Per column:

    column        hi n   hi cl    lo n   lo cl
    Contusion       41   0.780      79   0.861     <-- INVERTED
    Lateral OA       7   0.714      12   0.167
    Medial OA        8   0.750      13   0.385
    PF OA            6   0.833       5   0.400
    Baker's          2   1.000       4   0.250
    Effusion         3   1.000       6   0.667
    Synovitis        -       -        6   1.000

The gradient is real and sharp on the OA family (hi 18/23 = 0.78 against lo 11/34 = 0.32), and it runs
BACKWARD on Contusion, whose whole-column cluster accuracy is 0.833 over 120 cells regardless of margin.
Contusion is 204 of D351's 365 high-margin cells, so the pooled 0.83 is Contusion's high base rate
leaking into an OA statistic through a shared threshold — a textbook Simpson reversal, and the reason
the pooled rule would spend 79 cells the cluster gets right at 0.861 to buy a threshold that hurts them.

The pre-registered report split (chunks 9–10, never used to derive anything) gives a free held-out check
and agrees in direction: hi n=8 cluster 0.875, lo n=27 cluster 0.667. Thin, but not fitted.

**The operation that falls out is per COLUMN, not per margin**: Contusion wholesale (556 corpus cells at
~0.833, ~93 expected breaks — the residual budget, stated before banking), the OA family margin-gated at
≥0.6 (147 cells at ~0.78), Effusion (n=9) and Synovitis (n=6) left alone as too thin to bank either way,
and the other 571 left alone as genuinely contested. Net ~+453 corrected cells against the pooled rule's
~+241.

Not built yet, and deliberately behind `v6tears_span`: folding Contusion into the same table would
confound two corrections in the one arm that can price either. The sequencing hazard D351 raised is
sharper now that Contusion is the wholesale half — the v6 prompt read targets the SAME over-calls by a
different route, so the qualifier has to beat **0.833**, not v2's 0.167, to earn its card-hours.

## The gold ruler has two columns, and only one of them is a ranking (D353–D355 + M362/M363)

`research/scripts/adjudicated_rescore.py` scores an arm's fold-0 holdout npz on these 469 cells against
the blind verdicts. It prints accuracy at cut 0.5 AND `AUC-vs-verdict`; the first three letters reported
only the accuracy, and that column has a trivial ceiling nobody stated.

**32 of the 469 cells are affirmed.** So a constant-0 predictor scores **0.932** on all-469, ~1.000 on
tear-277 and 0.858 on Contusion-120 — above every arm ever scored here. Accuracy at a fixed cut on a
93%-negative split ranks arms by how little they call, and the arms that mask or substitute away v2's
over-calls are exactly the arms that call least. The v2 ruler's artefact with the sign flipped.

The cut-free column settles it. All-469, ten arms:

    arm                acc@0.5    AUC-vs-verdict
    discord_mask        0.832        0.6982
    stack_increments    0.695        0.6834
    v5blend_span        0.693        0.6813
    through_dense       0.542        0.6364
    v2_corrected        0.814        0.6272
    grade_ladder        0.795        0.6070
    band_span (ctrl)    0.625        0.6008
    calib_span          0.729        0.5890
    lr4e-5_ws           0.486        0.5781
    wsfold0_e6          0.601        0.5387

Every arm is above 0.5, so the accuracy ordering was not pure under-calling — but three of the ten move
four or more places. Three consequences:

1. **`discord_mask` holds the top on both columns.** Killed on the v2 ruler at −0.0417 for masking 13.9%
   of cells; on the gold cells it is the best thing the project has. The subtraction route re-opens on a
   cut-free number, which is the strongest form the objection could have been answered in.
2. **`grade_ladder`'s gold lead was calibration.** 0.795 accuracy, 0.6070 AUC — a hair over the control.
   Its ordinal-concordance signal is independent and untouched; the gold ruler simply does not second it.
3. **`lr4e-5_ws` is worst on both columns** among the champion's real seats, and it is the seat the v2
   ruler ranked joint-best of the whole pool. The member most aligned with v2's error distribution wins
   on v2 and loses on gold. That is the same artefact at member level, and it is the mechanism behind
   three differently-seated entries all reading 0.944: the seat search was optimising v2-alignment, and
   every v2-aligned set ties.

Limits, so this column is not the next thing that has to be retracted: **tear-277 AUC is `nan`** — the
verdict has zero variance there — so all-469 AUC is carried by the 192 non-tear cells. 32 positives makes
the interval wide. And accuracy is still the right statistic for how an entry VOTES at a cut; it is only
the wrong statistic for ranking members.

## The stratum no report-text selector can reach, and the first signal that reaches it (D357–D358)

The selector work closed one axis and opened another on the same day, and the arithmetic that did both
is small enough to write in one line.

**v2's errors are not where the selectors look.** v2 is wrong on 1.5% of cells where all four report
extractors agree and 15.6% of cells where they do not; those strata are 69.3% and 30.7% of the corpus.
So the unanimous stratum carries `0.693 x 0.015 = 0.0104` of all v2 error against the discordant
stratum's `0.307 x 0.156 = 0.0479` — **about 18% of every mistake v2 makes is made where all four
extractors agree with each other and are wrong together.** That is a correlated-failure population, and
no selector built on extractor disagreement can flag a single one of its cells at any threshold. It is
not a tuning problem; it is a construction one.

This reframes the whole selector ladder D357 measured. `lone_dissent` at precision 0.727 (~20x base) and
five-way discordance at 0.156 are not two ideas — 99.4% of lone-dissent cells sit inside the discordance
set, so they are one contrast cut at two sharpnesses, and the confidence control below random is the
same contrast cut at zero. The correction axis was never precision-bound. It was coverage-bound, and it
had reached its coverage.

**The image is a second opinion the reports cannot give.** D358 §2 ran the obvious complement: take four
members fitted on four DIFFERENT label families, flag the cells where at least three agree with each
other and disagree with v2. Precision 0.273 against a 0.037 base (~7.4x), recall 1.000 — at n=3, which
must be said — and the number that matters: **90.9% of what it flags is unanimous-stratum.** It fires
almost entirely where extractor consensus already spoke, which is the definition of a different
contrast. The pre-registered negative control — cells where the members disagree with EACH OTHER — flagged
four cells and got **zero** of them right. Model variance is not label error; that was a live alternative
explanation and it is now dead rather than merely unlikely.

The precision is low and that is expected rather than disappointing: hunting a 1.5%-wrong population
yields mostly false positives no matter how good the signal is. What the number licenses is a
second-order design — discordance banks the sharp 2,315 cells, image-consensus becomes the sampler for
the stratum discordance cannot see — and what it does NOT yet license is a table. Three wrong cells in
eighty is where the estimate lives, so the next adjudication batch goes to this selector's flags,
stratified by margin so the answer is a precision curve rather than an inflated point, with unflagged
unanimous cells included as the recall arm (M366).

**And a process point, paid for in the same letter.** v6tears' epoch-2 adjudicated scores looked like a
Contusion breakthrough (acc 0.675) and did not survive to e12 (0.517). No leak, no shared-feature relief
— a checkpoint read too early. An early-epoch adjudicated number is a stand-in exactly as an early-epoch
v2-ruler number is, and the only thing that separated the artefact from the finding was waiting. The
remainder control behaved throughout: agreement with v2-truth on the 10,439 non-adjudicated cells reads
band_span 0.8427, v6tears 0.8154, v5blend 0.8086 — the corrected tables drift from v2 in proportion to
how much they corrected, with no blow-out.

## The image-side axis is closed, and the 18% above is retracted (D360 + D362)

Both numbers in the section above were measured properly and both are now wrong. They are written here
rather than edited out, because the shape of how they failed is the transferable part.

**The image-consensus selector was an n=3 mirage.** D358's precision 0.273 / recall 1.000 licensed a
56-cell blind pack, stratified FOUR ways before any verdict existed — top-third and middle-third of the
unanimous-flag margin, the dissent stratum, and an unflagged recall arm. The result:

| stratum | n | v2-wrong | precision [95% Wilson] |
|---|---|---|---|
| flagU_top (top third of margin) | 14 | 0 | 0.000 [0.000, 0.215] |
| flagU_mid (middle third) | 14 | 0 | 0.000 [0.000, 0.215] |
| flagD (dissent stratum) | 14 | 6 | 0.429 [0.214, 0.674] |
| unflagU (recall arm) | 14 | 1 | 0.071 [0.013, 0.315] |
| **flagU pooled** | **28** | **0** | **0.000 [0.000, 0.117]** |

Zero of 28 in the unanimous stratum at BOTH margin thirds. What the selector actually finds is the
dissent stratum at 0.429 — cells `discord_mask` already masks. The axis is closed on the merits, not on
a threshold, and the pre-registered stratification is the only reason the mirage did not become a
second-order design. **That is the transferable rule: a pre-registered stratification is what separates
a finding from an n=3 mirage**, and it cost 56 hand-adjudicated cells to learn cheaply instead of a
card-week to learn expensively.

**The 18% is retracted as a point estimate.** It was `0.693 x 0.015 / (0.693 x 0.015 + 0.307 x 0.156)`
— arithmetic on a 1.5% that had never been measured. D360's recall arm read 0.071 on n=14 and flagged
it; the denom60 batch measured it on n=60:

- **U-stratum v2-wrong: 2/60 = 0.033 [0.009, 0.114]** (Wilson 95%)
- D-stratum comparator: 10/64 = **0.156 [0.087, 0.264]**
- constant-0 floor on those 60: accuracy 0.717

The share, computed as a RANGE across the interval endpoints rather than as a point: **0.325 at the
point estimates, spanning 0.07 to 0.75.** So the honest line is "about a third of v2's errors sit in
the unanimous stratum, with a range reaching both 7% and 75% until the D interval tightens" — and never
a point. The direction in the section above survives: the dissent stratum still carries the majority,
so `discord_mask` still addresses the bigger half. The size does not.

The measurement also landed strictly BETWEEN the two branches pre-registered for it in M369 — the point
sits where "1.5% stands" would have been restored, but the interval's upper bound (0.114) blows through
that branch's <=0.08 condition. Pre-registration that cannot be satisfied by the number that arrives is
still doing its job: it forbids picking the flattering reading afterwards.

**And the error direction is one-sided across three independent batches.** Both U-stratum errors are v2
firing 1 where the report does not support it — zero under-calls in 60. D360's blind adjudicator sided
with v2 on all 28 flagU cells. D345 found 31 over-corrections in v3. **v2's residual error is over-call;
the corrected chain's residual error is over-deletion.** Same disease, opposite signs. The balance56
batch was commissioned to price the second one, and until it reports, `v2_corrected`'s cell-level
precision is evidence that the chain can FIND cells, not that it can FIX them. Those are different
skills, and the board is still the only ruler that separates them.

## The substitution's two directions are not one operation (D363)

`balance56` was built to ask whether the corrected chain deletes true positives. It answered that, and
then answered a question nobody had asked, which turned out to be the bigger one. 56 cells, blind,
stratified by DIRECTION before any verdict existed:

| direction | n | chain wrong (cluster != verdict) | Wilson 95% |
|---|---|---|---|
| neg - deletions (v2=1 -> cluster 0) | 28 | 2 = 0.071 | [0.020, 0.226] |
| pos - additions (v2=0 -> cluster 1) | 28 | **22 = 0.786** | [0.605, 0.898] |

Constant-0 floor on these 56: **0.857**, which is why neither rate may be quoted as an accuracy.

**Deletions work. Additions fail catastrophically.** And the failure has one mechanism, not twenty-two:
**21 of the 22 pos-direction errors are OA columns** (Lateral OA 11, Medial OA 8, PF OA 2, plus one
Contusion). In every one of them the cluster asserts a compartment the report does not NAME - a report
saying "gonarthrosis" or "femorotibial degenerative change" becomes a positive `Lateral OA` cell. This is
the D319 compartment-assignment problem surfacing in the label table rather than in an image head. The
rule was named in M370, with all nineteen cells listed, BEFORE the key was opened; the key then agreed.
That is the second time in two batches that a pre-registered rule is what separated a mechanism from
scattered noise, and the first time it was the letter rather than the manifest carrying the
pre-registration.

At corpus scale the selector holds 2,315 cells: 2,080 deletions and 235 additions. At the measured
rates, the deletions are right ~93% of the time (about +1,934 net) and the additions are right ~21%,
against a v2 that was right ~87% on those same cells - so the additions cost roughly **-185** relative to
doing nothing at all, about 150 of that on OA columns.

**Every table that writes both directions carries those ~185 known-wrong cells**: `v2_corrected`,
`v6tears`, `v5blend`, `v7nontear`. `v7nontear_span` was on the card while this was being written. The
one-directional correction is the operation the evidence licenses, and
`research/scripts/apply_lone_dissent.py --direction down` now writes it
(`report_labels_v2_delonly.csv`, 2,080 cells changed, 235 selector rows deliberately skipped).

**The consequence for the mask-vs-substitute tie, which nothing offline had been able to move.** Every
prior attempt came back symmetric: the gold ruler picks the mask on the ranking column (AUC 0.6982 vs
0.6272, against a 0.932 constant-0 floor), the selector audit picks the substitution's cells (precision
0.727 / recall 0.533 vs 0.156), and 2,301 of the substitution's 2,315 cells sit inside the mask's 7,369
anyway. D363 is the first asymmetry with a mechanism behind it: **the mask cannot make this error,
because it writes nothing.** Removing supervision from a cell cannot assert an unnamed compartment.
That is a measured argument for the mask on the OA columns specifically, not a philosophical preference
about correction versus filtering.

It does not settle the tie - it prices one side of it. `corrseat` still ships tonight, but it is now
expected to LOSE to `goldseat`, and what it measures is a TRADE (over-call replaced by over-deletion plus
mis-addition) rather than an improvement. Writing that down before the score lands is the whole point of
having written it down.

**The one-sided disease, now across four batches.** v2's residual error is OVER-CALL: denom60's two
wrong cells are both over-calls with zero under-calls in 60; D360's blind adjudicator sided with v2 on
all 28 flagU cells. The corrected chain's residual error is OVER-DELETION and, as of D363,
OVER-ASSERTION on compartments. Same disease, opposite signs, and the direction is consistent across
four independently-built batches. A chain that finds the right cells and then writes the wrong value
into them is demonstrating that FINDING and FIXING are different skills - which is exactly what the
selector audit and the gold ruler have been disagreeing about all along.

## D365 -- a lone dissent is informative in ONE direction only (2026-09-11)

`dstratum120` was adjudicated blind, 120 cells, key sealed until the answers were on disk: 10 affirmed
of 120, constant-0 floor 0.917. The naive batch rate of v2-wrong is 0.433 [0.348, 0.523]. **That number
is not the corpus rate and must never be quoted as one** -- the batch was built DIRECTION-BALANCED, 60
cells each way, while the natural D population is 85.5% pos-direction. Reweighted to the natural mix the
D-stratum v2-wrong rate is **0.138**, which is what audit400's 0.156 [0.087, 0.264] already said with a
looser interval. The pre-registered branch that fires is 2, not 3, and the reading of D362 stands.

The U-share re-derives to **~0.35, range [0.07, 0.55]**. The 18% figure stays retracted. It is quotable
only as "about a third, with a range that still reaches both ends".

### The finding is the split, not the aggregate

| direction | meaning | v2-wrong | n |
|---|---|---|---|
| neg | v2 says PRESENT, cluster says absent | **0.850** [0.703, 0.947] | 60 |
| pos | v2 says absent, cluster says PRESENT | **0.017** [0.000, 0.114] | 60 |

Fifty-fold. OA vs non-OA shows nothing by comparison (0.450 vs 0.417).

So: **a lone dissent against v2 carries information in one direction and noise in the other.**
steven_full's dissents cover 25.7% of the corpus and are ~98% wrong where they ADD a positive, ~85%
right where they REMOVE one. Any future use of a dissent signal -- a weight, a mask, a confidence
feature -- must carry the direction, or it averages a 0.85 signal against a 0.017 one and measures
neither.

This is the third independent arrival at the same table. D362 said v2's residual disease is over-call.
D363 measured the substitution's own two directions at ~93% right (deletions) and 78.6% wrong
(additions), with 21 of 22 addition errors on OA columns. D365 now measures the underlying cells from
the third side and agrees. `report_labels_v2_delonly.csv` -- deletions applied, additions dropped --
was built on the D363 reading alone; it is no longer resting on one letter.

### What it does NOT say

It does not say the substitution is worthless: its deletion direction is the 85%-wrong reservoir and
repairing that is real. It says the additions are the part that must go, and that
`v2_corrected`/`v6tears`/`v5blend`/`v7nontear` each carry ~185 known-wrong cells they cannot shed
without being rebuilt. The mask is exempt by construction -- it writes nothing, so it cannot make an
addition error. That remains a mechanism, not a preference.

### The second column: attribution, not dissent

The same letter carries a count on a different axis. Over all 4,406 studies, v2 positives whose
compartment lexeme appears nowhere in the report:

| column | positives | unnamed | share |
|---|---|---|---|
| Medial OA | 1,619 | 154 | 0.095 |
| Lateral OA | 1,087 | 42 | 0.039 |
| **PF OA** | **1,991** | **515** | **0.259** |

"Named" is presence-anywhere, not attribution, so each share is a FLOOR on the unattributed fraction.
PF OA is the outlier by a factor of three, and 515 cells is a build with a known size before a card
hour is spent on it. Whether it is a NEW axis or a re-cut of `discord_mask`'s cells is the open
question, and it is answered by one crosstab -- {named, unnamed} x {U stratum, D stratum} per column --
requested in M373. If the unnamed cells are mostly unanimous, this reaches cells no dissent-based
operation can see.
