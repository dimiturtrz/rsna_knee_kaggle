# The metric is a plain mean, the refit term is two terms, and three columns hold ~0.02 of board

**Date:** 2026-09-05 · **Beads:** knee-wko44 (adjacent) · **Card spent:** none · **Slots:** two read, none spent

Two instrument slots landed together on 2026-09-04. Both read their pre-registered anchors exactly,
which is rare enough to say out loud, and between them they change what the next arm should be.

## `colnull` read exactly 0.500, so the ruler is finally known

`kneemri-colnull` submitted a constant 0.5 in every cell of every column. Three bands were registered
before the slot: **exactly 0.500** = a plain mean of per-column AUCs with no offset and no non-AUC
term · **0.495-0.505 but not 0.500** = a tie-handling or prevalence term · **anything else** = not a
mean of per-column AUCs at all.

It read **0.500**. `comp/README` has carried this as unmeasured since week one.

The consequence is arithmetic and it is the useful part: **a column improving its AUC by `d` moves the
board by `d/12`.** A per-label lane needs ~0.018 in its own column before the board's third decimal
can show it at all, and ~0.18 to be worth 1.5% of board by itself. Every per-label plan in the ledger
was priced on an assumed formula and is now priced on a measured one.

## `uncalib` read exactly the lower anchor, and the refit term splits in two

`kneemri-uncalib` is `fiveship`'s five seats with four of them swapped from the calibrated `_cf` build
to the uncalibrated `_ff` build, `merged_blend_cf` held fixed. The calibration flag is the only
variable across the pair.

| step | pair | delta |
|---|---|---:|
| pure refit, five seats | `twinfive` 0.935 → `uncalib` 0.937 | +0.002 |
| calibration, five seats | `uncalib` 0.937 → `fiveship` 0.941 | +0.004 |
| pure refit, four seats (independent pair) | `foldfour` 0.936 → `fitfour` 0.937 | +0.001 |

Two independent pairs put pure refit at or under +0.002 — under this project's 0.0062 noise floor on
the 909-study holdout. **Refitting the fleet for the refit is dead.** Roughly thirty hours of card that
the 0.953 chain had budgeted as a +0.006 constant buys +0.002.

What survives: refit *with* `--calibrated-targets` still carries the +0.006, because calibration is a
flag on the run that produces it and not a second run. So the plan does not change; the REASON does. A
seat is refit because calibration needs a refit to attach to, and a refit shipped without the flag
throws 0.004 away.

Caveat on the control: `twinfive` is a fold-0 pool and `uncalib` is fold-full, so the +0.002 also
carries seat-for-seat build differences. The four-seat pair is the cleaner of the two and reads
smaller, not larger.

## What the two together open: the per-label axis, priced for the first time

With the mean confirmed, the per-column map is worth reading. A rank-average of the seven strongest
fold-0 members this project owns, scored with `kneemri.evaluation.scores.LabelScores` over the
909-study holdout:

| finding | AUC | positives | prevalence |
|---|---:|---:|---:|
| MCL | **0.8104** | 171 | 0.188 |
| Fracture | **0.8255** | 68 | 0.075 |
| Lateral OA | **0.8309** | 233 | 0.256 |
| Effusion | 0.8663 | 558 | 0.614 |
| PF OA | 0.8698 | 409 | 0.450 |
| Synovitis | 0.8717 | 124 | 0.136 |
| Lateral Meniscus | 0.8737 | 226 | 0.249 |
| Contusion | 0.8851 | 267 | 0.294 |
| Medial OA | 0.8875 | 333 | 0.366 |
| ACL | 0.9055 | 248 | 0.273 |
| Medial Meniscus | 0.9108 | 482 | 0.530 |
| Baker's | 0.9208 | 217 | 0.239 |
| **mean** | **0.8715** | | |

The spread is 0.110 from worst column to best. Bringing the bottom three to 0.90 — the pool's own
upper-middle, not a frontier number — is +0.09, +0.075 and +0.070, which is 0.235 summed and
**+0.0196 of board**. That clears the 1.5% bar, and it is the first lane in some weeks whose
mechanism could plausibly deliver it. Every ensemble operation the project has run moves all twelve
columns together and has flattened seven times; this axis moves three of them and has never been
tried.

## The free road is closed, on both estimators

The first thing to try on a newly-priced axis is the thing that costs nothing. The metric leaves no
within-column operation alive — `kneemri-monotone` shipped a monotone transform of the champion's table
and read the champion's exact score — so post hoc has to be CROSS-column, and `column_mixing` is built
for exactly that. Its ridge came back a clean null some time ago, and that result bounded DENSE mixing
only: the L1 label-graph rule exists in `mixing_rules` precisely because it is the estimator that can
keep one real pair and zero the other hundred-odd, and it was the open half.

Run nested on the same seven-member pool, the sparse rule reads **mean −0.0085, worst −0.0653, best
+0.0041**. It also fails the specific question that made it worth running: it does not lift the weak
columns. MCL +0.0001, Lateral OA +0.0041, and Fracture **−0.0653** — 68 positives, and the fit destroys
the column.

Post-hoc is now empty as a class. Every remaining mechanism for the per-label axis is training-side and
costs card, which is what ranks it against whatever else holds the GPU.

## Is the weak column a weak label? Mostly not, and the exception is the one being measured

The obvious unification is that our three weak columns are the three our lexicon reads worst. The
2026-09-04 four-reader measurement says `report_labels_v2` makes 108 false positives against 32 misses,
so per-finding precision against the radiologist is available:

| finding | radiologist agrees with a reader YES | model AUC |
|---|---:|---:|
| Contusion | 44.4% (36 calls) | 0.8851 |
| MCL | **50.0%** (18 calls) | **0.8104** |
| Lateral OA | **56.2%** (16 calls) | **0.8309** |
| Medial OA | 63.6% (22) | 0.8875 |
| Lateral Meniscus | 66.7% (30) | 0.8737 |
| PF OA | 66.7% (24) | 0.8698 |
| Effusion | 68.0% (50) | 0.8663 |
| Fracture | 68.4% (19) | **0.8255** |
| Baker's | 68.8% (16) | 0.9208 |
| ACL | 71.9% (32) | 0.9055 |
| Medial Meniscus | 78.1% (32) | 0.9108 |
| Synovitis | 81.2% (16) | 0.8717 |

Spearman between the two orderings is **+0.41 over twelve findings** — positive, and nowhere near
significant at n=12. The four loudest cells contradict each other outright: Contusion is the
worst-labelled column and the eighth-weakest model column, Synovitis is the best-labelled and sixth
weakest, Fracture is eighth-labelled and second weakest, and Baker's is ninth-labelled and the best
column we have. **The weak-column ordering is not the label ordering**, and an arm aimed at the weak
columns cannot be justified as a labelling arm.

Two honest caveats, and they cut in opposite directions. The per-finding precisions rest on 16 to 50
calls each over 57 studies, so every one of those rows has an interval wide enough to swallow its
neighbours — this separates the axes weakly at best, and it is not a refutation of the labelling lane.
And **MCL is the one cell where both rulers agree**: worst-modelled column, second-worst-labelled, and
the one column about to get a board-truth number.

## What this does not say

The per-column map is scored against OUR supervision (`report_labels_v2.csv` binarised at its
midpoint), not against Kaggle truth. It gives the ORDERING of weak columns under our labels, which is
what an arm would be aimed at, and it does not give the board's own per-column AUC.

**That second number arrives on its own.** `kneemri-colmcl` (ref 56020054, still scoring) replaces the
MCL column with a constant after the blend. With the plain mean now confirmed, its drop from the
champion IS `(board AUC of the MCL column − 0.5) / 12`, which is the first per-column read on Kaggle
truth this project will ever have. If it lands near the pre-registered 0.918-0.928 it also confirms
the offline map's level, not just its ordering; if it lands elsewhere the map is ordering-only and any
arm aimed by it must say so.

No arm has been queued on this yet, and no mechanism for lifting a specific column has been chosen.
The lane is priced; it is not started. Nothing here is a score.

## AMENDMENT, 00:25Z — `colmcl` landed outside every band, and it inverts the map above

`kneemri-colmcl` scored **0.904**. The three bands registered before the slot were 0.918-0.928 (a plain
mean at an MCL board AUC of 0.69-0.81, which is where our holdout's 0.8104 sits), below 0.918, and above
0.928. It missed all of them downward.

The harness was checked before the reading was believed, because a 0.040 drop is 96% of the largest drop
a single column can produce and that is the shape of a bug. `run_check` on the kernel log reads `loaded
6 of 6 members`; the scored `submission.csv` carries exactly the twelve `TARGETS` columns with MCL at one
unique value; and the kernel's `chosen` tuple and `FOREIGN_WEIGHT` are byte-identical to the champion's.
Nothing is wrong with the entry.

So the arithmetic is forced. **MCL's board AUC is (0.944 − 0.904) × 12 + 0.5 = 0.980**, and the rounding
interval on two three-decimal scores puts it at 0.968-0.992.

MCL is the **worst** column in the table above and very nearly the **best** column on the board. Two
things follow, and the second is the one that costs.

**The map does not transfer in level.** The board's twelve columns must average 0.944, because that is
the champion's score and the metric is their mean. The pool above averages 0.8715. Every absolute number
in the headroom table — the +0.0196, the +0.0316, the 0.90 target itself — is computed in a coordinate
system the board does not use.

**On the one column with an external read, it does not transfer in ordering either — it inverts.** The
section above closed with "if it lands elsewhere the map is ordering-only and any arm aimed by it must
say so". It landed elsewhere, and worse than that clause allowed for: the map is not ordering-verified,
it is ordering-**refuted** at its single tested point. An arm aimed at MCL by that map would have been
aimed at the strongest column we own.

What it does say, and this is the useful half: the model separates MCL almost perfectly against the
board's labels while scoring 0.8104 against ours. That 0.17 is not the model failing to see the
ligament. It is our supervision, and it lands on exactly the column the four-reader census already calls
second-worst-labelled at 50.0% radiologist agreement. The two rulers that disagreed everywhere else in
this document agree here, and they now agree with a third.

**What this does to the lane.** The per-label *lift* lane is dead as priced — its price came from a map
that has now failed its only external check, and Kimi's literature pass puts the published mechanisms
(per-label heads, class-specific attention, CFT-style per-column calibration) at +0.02-0.04 on a weak
column against the +0.08 the arithmetic needed. The per-label *supervision* lane is the one this
promotes, and it is the same lane the 2026-09-04 four-reader page arrived at from the other direction.

### The mean constraint, which is the most useful line on this page

The twelve board columns must average exactly 0.944, because that is the champion's score and the metric
is their mean. Two consequences fall straight out and neither needs another slot.

**The other eleven columns average 0.9407.** From (0.980 + 11x) / 12 = 0.944. So a typical column sits
near 0.94, and a Baker's ablation should drop the board by about (0.94 − 0.5) / 12 = 0.037 if Baker's is
typical — which is where `colbaker`'s expected branch is drawn.

**The entire remaining headroom on this board is 0.056.** A perfect submission scores 1.000 and we sit at
0.944. The owner's target of 0.953 is +0.009, which is **16% of everything left**. That reframes every
lane on the page: a mechanism does not need to find a weak column, because on the board there may not be
one worth 0.09 — it needs to find 0.009 spread across twelve columns that are already at 0.94. The
per-label lift lane was attractive precisely because it promised a big number in one place, and the mean
constraint says no such place is available to promise it.

It also puts a ceiling on the supervision lane before the card is spent. If fixing our labels lifted
every column to 0.98 — MCL's measured level, the best any column here is known to reach — the board would
read 0.980, and getting to 0.953 needs about a quarter of that. That is the first ceiling on the axis
that comes from the board's own arithmetic rather than from someone else's corpus.

**What would settle it for one slot and no card.** A second ablation, on a column the map calls STRONG.
`colmcl`'s own docstring named this as the refutation it could not make. Baker's is the pool's best
column at 0.9208; if its board drop is also ~0.040 then every column is near-ceiling and our map carries
no information, if it is small the ordering is inverted broadly, and if it is 0.035 the map transfers and
MCL alone is anomalous. Those three answers aim three different next arms, which is what an instrument is
for.

### The relabel lane, priced against the board instead of against a corpus

Kimi returned a literature price for a full-corpus relabel at 00:40Z: point estimate **+0.010 of
board**, 80% interval 0.005-0.015, upside 0.020, built from Wollek's 0.858-vs-0.939 substitution plus
Ghiassi's multi-label noise-response curve and the observation that MCL's 50% radiologist agreement puts
it in the high-sensitivity regime. The mechanism half of that letter is sound and is kept. The price is
not, and `colmcl` is what breaks it.

The price converts "MCL recovers 0.08-0.12 AUC from clean labels" into board points by dividing by
twelve. That step assumes the 0.08-0.12 is board AUC we do not currently have. **We have it.** MCL's
board AUC is 0.980. The remaining headroom on that column is (1.000 − 0.980) / 12 = **0.0017 of board**,
a quarter of this project's 0.0062 noise floor — not 0.007-0.010.

The confusion is between two different injuries that noisy labels inflict, and only one of them is what
Wollek measured:

1. **Noise as TEST truth.** Our holdout AUC understates the model. `colmcl` proves this outright: the
   same weights and the same preprocessing read 0.8104 against our labels and 0.980 against Kaggle's.
2. **Noise as TRAIN supervision.** The model is genuinely worse. This is Wollek's effect, measured
   against a clean test set.

The 0.17 MCL gap is mostly (1), and a relabel is only paid for (2). Whatever (2) costs us on MCL, it is
bounded above by the 0.020 AUC the column has left. On the one column where the axis can be priced
against the board rather than against someone else's corpus, the relabel is worth less than the noise
floor.

What survives is Kimi's instrument, with its threshold moved: a relabel is licensed only if the columns
are noise-limited ON THE BOARD, and our worst-labelled column is not. `colbaker` decides whether MCL is
representative or anomalous, and until it reads, no card goes to the relabel.
