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

### The 0.17 ruler gap, audited — including the half of it I had got wrong myself

Kimi answered Q-RULER1-4 at 01:43Z. Two of the four answers stand, one is corrected here, and the
correction runs OPPOSITE to the direction their reply drew.

**What they found, and it is a real number I had not read.** `interpretations/kneemri/2026-09-03_the-report-reader-is-the-weaker-half.md:50` carries the row

    MCL               0.9762 → 0.9070   -0.0692

and the reply used the 0.9070 as "the same ensemble's MCL column against the radiologist", closing
0.096 of the 0.17 gap and concluding "the model is fine, the lexicon labels are the problem."

**Three things are wrong with that use, and the third is the one that matters.**

*The sample moves with the ruler.* 0.8104 is 909 studies scored against the lexicon. 0.9070 is
**58 studies** scored against the radiologist. Two variables change at once, so none of the 0.096 can be
attributed to the ruler alone. Their own Q-RULER3 says the 58-study panel gives an MCL standard error of
0.08-0.10 — wide enough to swallow the whole difference they attributed to labels.

*It is not the same ensemble.* That row is the fitted model of that arm, not the six-member champion
plus the raptor seat at 1/3 that `colmcl` damaged.

*The cited file says the opposite about MCL specifically.* That table is `supervision_gap`: the report
LABEL as a predictor against the radiologist (0.9762) versus the MODEL against the radiologist (0.9070).
Its headline is that the model beats its own supervision by 0.047 over 696 cells — and that **MCL is the
single inversion**, the one finding where the lexicon label is the BETTER half, with a band excluding
zero. The file's own sentence is: "Where the report is *better* — MCL, and ACL and Medial Meniscus …
no label work can help, because there is nothing left in the label to recover."

So on the one column where the board has been measured, the lexicon is not the weak half. That is the
second time in two days that a relabel price has been argued from a number whose denominator was not
checked, and it lands the same way: **the relabel lane is not licensed by MCL.** It also promotes
candidates (c) case mix and (d) definitional mismatch from residual to leading, and neither is
falsifiable offline.

**What I had got wrong on my own side.** My 0.17 compared the board's champion against the SEVEN-member
fold-0 rank pool — a different model set, which is the same class of confound I had just objected to.
Tested rather than assumed: pooling all **103 rostered members that carry a holdout record** over the
same 909 studies reads **MCL 0.8112** and **Baker's 0.9189** (pool mean 0.8713), against the seven-member
pool's 0.8104 and 0.9208. The column reading is robust to the pool; the gap survives the objection. The
champion's own six cannot be scored honestly here — five are fold -1 and have seen every holdout study —
so the exactly-matched comparison does not exist, and that absence is itself part of the finding.

**Kept without change.** Sampling variance is ruled out as the primary cause: Hanley-McNeil at 171
positives and 738 negatives gives SE ~= 0.0255 at AUC 0.9 (verified: sqrt[0.9*0.1*(1/171 + 1/738)]), so
0.17 is about 7 SE. And the Q-RULER2 framing is right — this is a linear inverse problem, exact recovery
of twelve columns needs twelve independent measurements, and the only cheap path is adaptive group
ablation under a sparsity assumption. That is precisely `colhalf`, which makes it the correct next slot
rather than a guess.

**The clause with teeth is Q-RULER4.** A shared additive label bias cancels in a paired comparison, so
member ORDERING survives — but the assumption fails when the noise interacts differently with the two
members, and the named failure mode is *different supervision*: a member trained on the lexicon can
learn to reproduce its over-calling, scoring higher on our holdout and lower on the board. Every arm on
the card right now is in that class by construction. `errors_rank` and `errors_relabel` differ from the
`band_dense` control in NOTHING BUT supervision. So their offline delta against 0.8578 is exactly the
comparison Q-RULER4 flags as reversible, and whatever they read has to be written down with that
caveat attached rather than as a clean keep/kill.

### The column map is unaffordable, and that is a closure rather than a caveat

Kimi accepted the correction above and withdrew the 0.096. What they returned instead is the more useful
half: **case mix and definitional mismatch cannot be separated offline, at any price.** Both hypotheses
make the identical prediction on every piece of data we hold — the model ranks the board's MCL event
better than it ranks ours — and telling them apart needs either the board's cases or the board's label
definition. Any offline design that claims to separate them is smuggling in a bridge assumption
("if the board used the radiologist definition the AUC would be X") that is itself untestable without
board data. That is a guess wearing a measurement's clothes, and it is exactly the shape of the
0.9070 mistake this same exchange just produced.

**So a reliable per-column board map costs ONE SLOT PER COLUMN.** Twelve columns is twelve slots — 2.4
days in which we spend every slot we get on instruments and ship nothing. That price is not payable, and
naming it is what closes the axis: there is no cheap version, and the expensive version is not worth its
cost against a 0.009 gap.

What survives is the sparsity bet. `colhalf` ablates six columns at once and asks whether the deviation
vector is CONCENTRATED. If it is, adaptive group ablation finds the deviants in a handful of slots
(Fornasier & Rauhut) and a partial map becomes affordable. If the six-column half lands near the six-column
prediction of a flat map, the columns are spread and the map is unbuyable at any slot count we have. Either
way `colhalf` is the last slot this axis gets until it earns another.

The residual reading on MCL stands and is now unexplained rather than explained: MCL is 0.980 on the board
and 0.8112 over our 103-member rostered pool, sampling is ruled out at ~7 SE, label noise is demoted to
possible-but-unmeasured on this column, and the two leading candidates are both unfalsifiable here. An
unexplained 0.17 on the column that carries 0.014 of board is not a comfortable place to leave a finding,
but it is an honest one, and the alternative is to keep buying explanations with slots at a price the gap
does not justify.

### The 58-study panel cannot protect a comparison, and the bound is what proves it

Kimi's Q-RULER6 offered a model-free check: flipping D mislabelled cells can move a finding's AUC by at
most D/min(P,N), so an observed difference LARGER than that bound cannot be reversed by label correction.
It costs no model and no card — only `core.data.labels.reader_agreement`, which already prints the counts.

Computed. N = 57 - P exceeds P for every finding, so the bound is D/P:

    MCL 9/9 = 1.00 · Lateral OA 7/9 = 0.78 · Synovitis 17/27 = 0.63 · Fracture 10/17 = 0.59
    PF OA 11/19 = 0.58 · Medial OA 8/14 = 0.57 · Lateral Meniscus 11/21 = 0.52 · Baker's 6/12 = 0.50
    Effusion 16/34 = 0.47 · ACL 9/23 = 0.39 · Medial Meniscus 8/26 = 0.31

**The smallest bound in the table is 0.31 and the deltas this project measures are ~0.01.** The bound
exceeds the signal by 31x at best and by 100x on MCL. It never binds. By Kimi's own criterion — reversal
is possible whenever the observed difference is below the bound — reversal is possible for all twelve
findings at every effect size we have ever reported or ever will.

This is a stronger negative than the underpowering Kimi already warned about, and it is a different one.
Underpowering says the panel gives a noisy answer. This says the panel gives NO answer: even with the
counts known exactly and no sampling error at all, the arithmetic cannot exclude a reversal. 58 studies
with a third of the reader's positive calls disputed is simply not a ruler that can adjudicate a 0.01
difference, and no cleverness applied to those 58 rows changes that.

**Consequence, and it is uncomfortable.** Every comparison where two arms differ in SUPERVISION — which is
`errors_rank` and `errors_relabel` against their control, by construction — has no offline protection at
all. Not a weak one. None. The only instrument that can adjudicate such a comparison is a board slot, and
the board has already flattened seven member-level entries, so a slot spent adjudicating one is a slot
spent on an axis that has not moved.

**What replaces it.** The panel is the wrong denominator; the full holdout is the right one. This project
holds FOUR reader tables over all 909 studies, not one. A supervision arm that beats its control under
`report_labels_v2` alone is a result about that table. One that beats it under all four is a result about
the studies, and the four disagree with each other far more than sampling noise — which is precisely what
makes agreement across them informative. That is 909 studies rather than 58, costs no card and no slot,
and is the strongest offline ruler this project can actually build. It is not a proof of board transfer.
Nothing offline is. It is the difference between one table's opinion and four.

## 4. Q-LEFT came back a clean negative, and the repo had already priced the one mechanism it named

I asked Kimi the question the closed axes leave: name a PUBLISHED mechanism that moves a mean of twelve
per-column AUCs by 0.01, under our constraints. Their answer, verbatim:

> "**Nothing I can find**, under the constraints you listed… TTA: typically +0.001–0.003 AUC at
> saturation; you already run it. Slice-level attention / 2.5D fusion (MST, DinoAtten3D, CoPAS):
> +0.002–0.005… Multi-task / auxiliary supervision / label-correlation modeling: +0.001–0.004…
> AUC-margin / ranking losses: mixed, usually within noise. Self-supervised pretraining on in-domain
> unlabeled data: the one mechanism that *can* give +0.005–0.010, but only when the pretraining corpus
> is large and task-matched."
>
> "My clean answer: **no published mechanism meets the Q-LEFT bar without more data.**"

I tried to break the closure on its own premise and failed, which is worth recording because the failure
is the useful part. Kimi's supporting claim is "there is no hidden reservoir of unlabeled knee MRIs to
pretrain on", and on this repo's terms that is FALSE: `core/data/corpus/fastmri_corpus.py` and
`rijeka_corpus.py` are both wired up, and 231 fastMRI presigned URLs are held. The reservoir exists.

The closure survives anyway, because the lever was never priced on corpus availability. `knee-fqit7`
priced it on two ledger rows and a compute curve: OrthoFoundation is somebody else's DAPT on our own
modality and our own joint, at 1.2M images, and it is ALREADY A MEMBER HERE as `ortholarge`. It reads
`ortho_stack` 0.8417 against `v3large_stack` 0.8277, so DAPT is worth about **+0.014 at matched ViT-L**
— and the ViT-L lane itself costs about that much against ViT-B, so the net against `base_dense` 0.8512
is a wash. There is no published DINO-objective medical gain at a 12 GPU-hour budget; the positives are
ExPLoRA at 80–100 GPU-h and BrainFound at ~104 GPU-h. P(clears 1.5%) = 0.08.

So the right correction is to my objection, not to Kimi's conclusion. Having the data does not re-open a
lever that was priced on effect size and compute. The bead's own RE-OPEN BY clause is the only live path
and it is cheap: the two ortho rows were both run against ViT-L controls, and the trunk-size handicap was
measured only afterwards (`band_pairmem`, knee-710gi). An ortho arm on the DENSE recipe against a ViT-L
control at the same read separates DAPT from trunk size properly, and that is **2 h of card, not 12**.
That is the next thing to queue when `errors_relabel` clears.
