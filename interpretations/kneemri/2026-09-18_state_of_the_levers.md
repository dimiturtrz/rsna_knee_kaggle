# State of the levers — what could still carry 1.5%, priced — 2026-09-18T01:50Z

## Why this file exists

Tonight's close (`2026-09-18_the_pool_holds_no_per_finding_specialist.md`) says the next member has to be
BUILT rather than selected, so the next question is which mechanism to build. Answering it took seven
greps and one file read, and **every one of them landed on a number this repo had already recorded.**
That re-derivation has now been paid for at least three times. This is the table, so the fourth time is a
grep of one file.

Card cost of this audit: zero. It is arithmetic over things already on disk.

## The levers, each against the 0.011 board gap to first place

| axis | bound measured here | where |
|---|---|---|
| **pool diversity** — some existing member leads a finding | **0 of 12 columns clear** their own 95% paired interval over the shipping solo | `2026-09-18_the_pool_holds_no_per_finding_specialist.md` |
| **capacity ladder** — DINOv2-base/large over small | **−0.0010**, five isolating pairs median 0.0000 | knee-5dx47; `base_dense` 0.8512 vs `base_honest_batch` 0.8502 |
| **planes** — does the pipeline read coronal/axial at all | **already consumed**: six slots span all three planes | `core/data/corpus/slots.py:58` |
| **slot occupancy** — a finding starved of its primary sequence | **no column's weakness tracks slot presence**; mean 4.57 of 6 slots per study | README:10353; `2026-09-05_the-weak-columns-are-not-a-missing-sequence-and-not-a-combination.md` |
| **distillation** — a teacher above the student | **teacher headroom 0.002**, term itself +0.0056 (under the 0.0062 floor) | README:16585 |
| **mutual supervision** — a LIVE peer rather than a converged teacher | **CLOSED −0.0015** on the strongest trunk (`dml_a_s2026` 0.9044 vs `headlr_ship_s2026` 0.9059, ledger75, 909 studies, peak e18 so the verdict is not a last-epoch artefact). The per-epoch deficit is FLAT, not converging: −0.0024, −0.0020, −0.0008, −0.0014, −0.0008, −0.0009, −0.0009. The peer disagreement was real and did not convert | knee-z81wd branch 4; README `### dml_a` 2026-09-18 |
| **slice coverage** — 24 -> 36 slices | **CLOSED.** Behind at every one of 7 matched epochs, mean **-0.0224** = 3.6x the paired floor, at 50% more wall-clock | `2026-09-18_coverage_is_closed_on_a_trajectory_not_on_a_score.md`, knee-jb0ci |
| **in-domain MRI pretraining** — SSL on fastMRI/SKM-TEA | **published wash at our labelled scale** (+0.073 at 80 subjects, −0.005 at 800; we hold ~4,400) | README:7425, Zhang et al. MedIA 2024 |
| **external data as LABELLED augmentation** — the one axis README:7425 left explicitly open | **closed on our own measurement**: `pool_specialise` FLAT, `noisy_student` 0.8548, `--corpus-oversample 2.22` does not lift the mean | README:15216, :12447, :10737 |
| **recipe fragments** — any piece of the champion family | **closed entirely**; sign unanimous across seven leave-one-outs, `backbone_lr` at its optimum both sides | README:16503, knee-rn9py, knee-lm2ol |
| **auxiliary objectives** — the twelve `--*-weight` arms at full budget | **−0.0019** against its own matched converged control, the `<= 0.000` branch | knee-evdfd (read 2026-09-18T02:14Z) |
| **backbone-lr** — the one knob whose corner value was an ENDPOINT, 4/4 upward in the corpus | **−0.0022** at 8e-5 against its own control, converged, both peaking e18 — the sign REVERSES at 24 epochs with dropout | knee-e9f83 (read 2026-09-18T02:30Z) |
| **adaptation depth** — more than six encoder blocks adapting, the segment that RISES +0.0129/block from 4 to 6 | **+0.0007 midpoint / −0.0012 ledger75** at seven blocks against its own control — the gradient does not survive past six | knee-p724l (read 2026-09-18T03:19Z) |
| **readout depth** — which ViT block the head reads, the axis `get_intermediate_layers` exists for | **−0.0227** at block 9 of 12, 3.7x the floor, converged, seat −0.0034; an interior peak at 10/11 would need a sharp non-monotonicity | README:1200. **Spelled `encoder_layer`, not `--layer`** — a grep for the flag name finds nothing and reads as unarmed |
| **through-plane at the dense corner** — each slice's other two channels hold its depth NEIGHBOURS, free by construction | density gradient, not a flat: **12 slices −0.0136\* on Medial Meniscus** (`through224` 0.8505), **20 slices −0.0020** (`through_dense_s2026` 0.8558, seated). The named cause is SPAN LOSS, which shrinks with density | README:4635; knee-4by7q prices 36 slices and BLOCKS on slicecover36's premise |
| **DINOv3 trunk, ours** | best honest member 0.8724 against seats 0.9045 | `2026-09-17_member_candidates.md` |
| labels · cut · head_lr · resolution · overlap TTA | closed previously | prior ledger |

**No unbounded single mechanism remains on available assets.** Every priced axis is bounded below the
gap, most of them below the 0.0062 paired floor.

## The pool's ORACLE is also below the gap

The per-finding close says no single member leads a column. A weaker question is whether the pool leads
*collectively*: take the best member in every one of the twelve columns at once, and grant it every
advantage — the 374-way selection is not deflated, the leaders are read on the studies that chose them,
and combining them is assumed free. That oracle's mean margin over the shipping solo is the mean of the
twelve margins in `2026-09-18_the_pool_holds_no_per_finding_specialist.md`:

```
(0.0016 + 0.0113 + 0.0097 + 0.0057 + 0.0302 + 0.0002 + 0.0125 + 0.0120
 + 0.0124 + 0.0096 + 0.0033 + 0.0062) / 12  =  +0.0096 offline75
```

**+0.0096 is the ceiling of every recombination of all 374 members**, and the board gap is 0.011. So the
gap is not reachable by any reseating, re-weighting or per-column routing of what exists, even with an
oracle doing the routing — and the real figure is smaller, because 0.0302 of that sum is MCL's
selection artefact on the table's widest interval. This is the quantitative form of "the next member has
to be BUILT": not a preference, a bound.

## The composition escape is NOT open by default

Global CLAUDE.md says sub-floor increments still compose. This project has measured it both ways and the
distinction is per pair, not per size:

- **Composes** — the five small fragments of the control recipe sum to +0.0253 and `ctl_stripall` removes
  all five at once for +0.0269, inside the pre-registered band (README:16503).
- **Does not** — `distill_stack_clean` IS `labelwise_drop` plus distillation and reads **−0.0101** at the
  0.75 cut. Distillation subtracts from the pooling stack rather than adding to it (README:15893).

So "compose the surviving sub-floor terms" is a hypothesis owing its own arm each time, never a free sum.
The one composition that was still unspent — **+0.0056 distillation onto the SHIPPING recipe**
(README:16599) — is now **spent and null**, and it cost no card: `dml_a_s2026` IS that recipe plus a
supervision term at w=0.3 against a control that is the same recipe without one, and it reads
**−0.0015** ledger75. Mutual differs from one-directional in who the TEACHER is, not in what reaches the
student, and the strong member's peer sits below it, so `dml_a` is a LOWER bound on the one-directional
form. Three readings, consistent in sign wherever the pooling matches: **+0.0056** at max/mixed pooling
(not the shipping corner), **−0.0101** labelwise at w=0.7, **−0.0015** labelwise at w=0.3 on the corner.
The +0.0056 belongs to a pooling this project does not ship. Re-open by producing a teacher above the
student, never by re-quoting +0.0056.

## And the offline ruler no longer ORDERS members at the top

Three solos have now shipped from the top of the offline ranking. Their offline75 spread is 0.0014 —
far inside the 0.0062 paired floor, i.e. an offline TIE — and their board spread is 0.004:

| member | offline75 | board | entry |
|---|---|---|---|
| `labelwise_ext_drop40_long_s2026` | 0.9045 | 0.937 | `extsolo_v1` |
| `labelwise_depth24_drop40_long_s2026` | 0.9053 | 0.941 | `l75solo_v1` |
| `headlr_ship_s2026` | 0.9059 | 0.937 | `headlrsolo_v1` |

The body line fitted below 0.900 over-predicts all three, 3 of 3 LOW. So above offline75 ~0.904 the
ruler is an upper bound and not a rank: **an arm selected on offline75 up there cannot be priced before
its slot.** The consequence for the schedule is that the board's five slots a day are now the only
instrument with resolution at the top, which is what makes a slot spent on a reseating expensive.

That was an inference from three board points. It is now **measured**, by pointing
`kneemri.evaluation.rank_stability` at the shipping cluster itself — the instrument had only ever been
run on the 0.851-0.863 family (README:12502), where "resolves its top" means the top of *that* band.
Twelve arms, the whole `eb3ba199` population above offline75 0.8998, 909 shared studies, 500 resamples,
ruler ledger75, 2026-09-18T03:47Z:

| | 0.851-0.863 family | the shipping cluster |
|---|---|---|
| score span | 0.012 | **0.0052** — inside the 0.0062 paired floor |
| top arm holds rank 1 | **96.8%** | **29.2%** |
| arms that rank first at least once | 6 of 13 | **11 of 12** |
| Kendall tau, midpoint vs ledger75 | +0.6667 | +0.6061 |

`headlr_ship_s2026` leads by 0.0006 and holds that lead in under a third of resamples; the arm ranked
sixth wins outright 5.6% of the time. **909 studies cannot order the band we ship in.** Two things
follow, and the second is the expensive one:

1. Offline75 cannot choose which solo to ship. Picking the top row is a coin flip with ~29% odds, and
   the 0.004 board spread across the three shipped solos is what that coin flip looks like from the
   board side.
2. **Every keep/kill taken inside this band is not a null — it is no reading.** The three knob closes
   recorded tonight (aux objectives −0.0019, `backbone_lr` −0.0022, adaptation depth ±0.001) all sit
   an order of magnitude inside a span the ruler cannot resolve. They remain correct as *refusals to
   ship* — nothing there earns a slot — but they may not be cited as evidence that those mechanisms do
   nothing. The distinction matters the next time one of those axes is proposed with a stronger
   mechanism: the prior is uninformative, not negative.

The bar this sets for any future corner arm: **a mechanism must be worth enough to clear the band, not
to win inside it.** With the top-12 span at 0.0052 and rank 1 held at 29.2%, an offline-measurable
result at the corner needs to be a step, not an increment — which is the same conclusion the lever
table reaches from the other side, and the reason the remaining two places to look are supervision
quality and slice-to-study aggregation rather than another knob.

## What is actually live

1. `headlr_ship_ff` — the `--fold -1` refit of the shipping solo, FINISHED 02:16Z and unread. Dark arm;
   its reading is a board slot, not a ledger row, and it is priced as an instrument: its honest twin
   `headlr_ship_s2026` shipped at board 0.937, so carrying the refit term off that predicts 0.940-0.945
   against a 0.946 champion. P(beats champion) ~0.10.
3. `slicecover36` (knee-jb0ci) — QUEUED behind a 107 GB corpus decode started 02:32Z, because no
   36-slice cache exists and `run_train` never builds one. Watcher `after_sc36.sh` gates the arm on the
   decode's pid. Honest price ~1.5-2 h decode + 2.4 h train, expected value +0.001 — justified as an
   INSTRUMENT on the aggregation residual below, not as a gain.
2. ~~the auxiliary-objective pair~~ — CANCELLED 2026-09-18T02:14Z. Both had already run on 2026-09-13 and sat
   unpriced; `reading_check --only` refused the re-arm and the pair reads −0.0019, the closing branch
   (knee-evdfd). The card is free behind `headlr_ship_ff`.

## What this says about the schedule

The honest reading is that **offline ranking is exhausted and so is the asset pool**: the ruler ceiling
(offline75 ~0.904) and the per-finding close and this table are three views of one fact. A lever that
could carry 1.5% would have to add information the project does not hold — labelled studies, or a trunk
that has seen MRI at a scale our licences do not reach. The trunk one was closed on premise before the
card was spent; the labelled-data one was closed the harder way, by three arms of our own, and the
caveat README:7425 left open ("not a verdict on external data as LABELLED augmentation") was answered
NO eight days later at README:15216 and sat unjoined to it until now.

What that leaves is not "find the big lever". It is: keep the standing entry protected, spend card only on
arms with a pre-registered branch, and treat any claim that an axis is open as owing a line in this table.

## Where a 1.5% lever can still live, now that the knobs are gone

knee-dry3h priced 22 defaulted leaves and found the corner at its optimum on every one, with a single
exception: `--backbone-lr`, the only leaf whose corner value was an ENDPOINT rather than an interior
point, with 4/4 upward sign agreement across the corpus. That exception is now measured and reads
**−0.0022**. The upward agreement came from 12-epoch 224 arms without slot dropout — knee-dry3h's own
honest limit (2) — and the sign reverses at the corner's 24 epochs with dropout.

So three independent lines now say the same thing: a 4x encoder buys nothing, 22 defaulted knobs are at
their optimum, and the last endpoint reads down. **The residual is downstream of the encoder and
downstream of the optimiser.** Two places remain, and both are named rather than hoped for:

- **supervision quality** — what the labels mean, not how the model is fitted to them;
- **slice-to-study aggregation** — how twelve findings are read out of six slots of slices.

`slicecover36` is the input-side instrument for the second. It is not a candidate ship (expected +0.001);
it decides whether the coverage cell is open at the corner on a harness that cannot be blamed, which is
what the two prior readings — both 12-epoch, both under a max aggregator whose noise floor grows with the
size of the set it maxes over — could not.

## The board record says ensembling has bought this project nothing over its best solo

`rank_stability` printed one more number: the ensemble's margin over the best single member is
**+0.0054 paired, 95% [+0.0020, +0.0058]** on 909 studies. That is a real offline margin — its interval
clears zero — so the obvious expectation is that an ensemble entry outscores a solo entry on the board.
It does not. Over the 147 scored rows of `comp/board.jsonl`, split by member count and by whether the
entry carries an honest offline number:

| population | n | best board | best entry |
|---|---|---|---|
| solo, honest scope | 17 | **0.945** | `extffsolo_v1` |
| 2-4 members, honest scope | 23 | 0.939 | `fasttrio_v1` |
| >= 5 members, honest scope | **62** | **0.945** | `noraptor_v1` |
| >= 5 members, offline unpriceable | 33 | 0.946 | `tripleseat_v2` (the pin) |

**Sixty-two ensemble attempts of five or more members have never once beaten the best of seventeen solo
attempts.** The offline +0.0054 does not transfer: at a top-band slope near +1.1 it should have shown as
roughly +0.006 board, about six ticks, and the measured difference is zero ticks.

Read the fourth row carefully — this is where a first draft of this section was wrong. Those 33 rows are
NOT a greedy-selection family; `scope_reason` says so for only 2 of them. The rest read *"offline is None
by construction"*, and their `selection` field is `designed` 20 times. What actually distinguishes them is
that they contain a full-corpus `_ff`/`_cf` refit or the foreign seat, so no honest offline number exists
for the seating. So the correct statement is about the REFIT, not about selection:

- the best solo we own is itself a refit (`extffsolo_v1`, 0.945);
- the honest-offline ceiling is 0.945 whether the entry has one member or seven;
- the only entries that ever reached 0.946 also contain refits.

That converges with the refit-term question from the other direction. **The lever that has moved the board
in this project is the full-corpus refit; the ensembling is not.** It is also the reason the standing
mandate — one solo checkpoint, never an ensemble — costs nothing to obey: the record says the ensemble
machinery has bought zero board ticks over the best solo across 62 tries, while the refit has bought
every tick we have.

The live consequence for `headlr_ship_ff`'s slot: under the ceiling reading pre-registered on knee-33iwi
(refits converging near 0.9445 rather than adding a term), a solo cannot pass 0.946 by refitting at all,
and the remaining tick has to come from the base recipe rather than from the fit. The separating evidence
is a refit off an honest twin BELOW 0.935, which the additive reading predicts near 0.950 and the ceiling
reading predicts at 0.9445 — and we own honest twins down there, so that experiment costs one slot and no
card.

## The target as a number, and why the ruler is the lever (appended 2026-09-18 05:48Z)

Fit over every board row with both numbers and a scope that is not `excluded` (n=48, all of which sit
at offline75 >= 0.8700, so this IS the top band):

    board = 1.104 * offline75 - 0.0603        residual sd 0.0017

Inverted at the champion:

| quantity | value |
|---|---|
| board needed to beat the pinned champion | 0.947 |
| offline75 that predicts it | **0.9124** |
| our best member today (`headlr_ship_s2026`) | 0.9059 |
| **required gain** | **+0.0065 offline75** |

That single row is the argument for the rotation. The paired floor on fold 0 is **0.0062**, so the gain
we need is *at* the floor: a recipe that genuinely wins and a recipe that ties are the same reading on
today's ruler, and no amount of card time changes that. A rotation covering all 4407 studies takes the
floor to roughly **0.0028**, which makes +0.0065 a **2.3 sigma** effect - visible, and visible without
spending a board slot to find out.

### The winner's-curse test, and why it does not answer the question

`headlrsolo_v1` scored 0.937 where the line predicts 0.940 (-1.8 sd), the third top member in a row to
land low. The proposed mechanism was winner's curse: ~450 members ranked against ONE fixed 909-study
draw, so the top of that ranking carries an in-sample component that evaporates on fresh board data.

The card-free test was to split transfer-line residuals by the `selection` field:

    search       n=  2   mean resid -0.0025
    designed     n= 19   mean resid +0.0004
    unlabelled   n= 27   mean resid -0.0001

The sign is right and the result is **worthless as evidence**, for two independent reasons. First,
**n=2**. Second, and worse, `selection` records how the ENTRY was assembled, not how the MEMBER was
chosen - and the three low readings are all solos. The field cannot express the axis the hypothesis is
about, so this is a wrong-ruler measurement, not a weak one. Winner's curse remains the best available
explanation and remains **unconfirmed**; it is not to be planned around until something can actually
separate it from label mismatch and case-mix shift.

### What the rotation is FOR, after the owner's correction

The owner's point (2026-09-18): since every shipped member is refit with `--fold -1` on all 4407, the
holdout constrains nothing we ship. It is a development instrument, so moving it costs comparability
with past measurements only - and those measurements have just been shown to be biased at the top.

Two consequences. **We are free to change it**, keeping fold 0 only as the legacy ruler for the ~450
existing records, which cannot be re-scored anywhere else. And **the fold members are disposable**: a
rotation prices a RECIPE, not a checkpoint. Train recipe R on k folds, assemble the out-of-fold number,
and if R wins, refit R on everything and ship that. The k fold checkpoints are deleted. That keeps the
one-good-member constraint exactly - no ensemble, no pair - while buying the sharper ruler.

**And therefore k=2.** The floor is the same for every k, because the union of the folds is the corpus
at any k; all a larger k buys is a training share nearer the shipped model's.

| scheme | card | OOF n | floor | each model trains on |
|---|---|---|---|---|
| today | 1x | 909 | 0.0062 | 80% |
| k=2 | **2x (5.2 h)** | 4407 | **~0.0028** | 50% |
| k=3 | 3x (7.8 h) | 4407 | ~0.0028 | 67% |
| k=5 | 5x (13 h) | 4407 | ~0.0028 | 80% |

k=5 costs 2.5x more than k=2 for identical measurement precision. k=3 is the pick: 67% keeps the regime
recognisable against the 100% refit that ships.

### Warm start: which half of it leaks

Fold models may NOT warm-start from any checkpoint we own. Our honest members trained on folds 1-4, so
warm-starting fold 1 from one hands it rows that are now fold 1's holdout - the `raptor_solo` defect at
98.6% in-sample. Fold models start cold from DINOv2, or warm from a foreign trunk (MRNet / Rijeka /
fastMRI never saw our 4407), which is clean and would cut per-fold cost.

The final ship refit is the opposite case: it has no measurement duty, so leakage there is meaningless.
`experiments/README.md:6479` prices warm start as a TIME mechanism with no quality term - one warm epoch
about twelve cold ones - which turns the 2.6 h ship step into roughly 20 minutes.

## CORRECTION: the ruler was not the lever (appended 2026-09-18 06:05Z)

The section above concluded that the offline ruler is the binding constraint, because +0.0065 offline75
is 1.05 sigma against the 0.0062 paired floor. That conclusion used the wrong denominator. It priced the
gain against the ruler we happen to run on the card, and never asked what the BOARD resolves.

THE MEASUREMENT. Take every pair of adjacent board entries whose offline75 differs by less than 0.0008 —
pairs the offline ruler calls tied. There are 33 of them across the whole record. Their board deltas
scatter with sd 1.9 ticks (0.0019), which independently reproduces the 0.0017 residual of the transfer
fit. That is the spread of the board over things offline cannot separate.

    ruler                   scatter    +0.0072 board / +0.0065 offline75 reads as
    offline, today (909)    0.0062     1.05 sigma
    offline, 1/3 (1471)     0.00487    1.33 sigma
    public board            0.0019     3.7 sigma

So the board resolves the target at 3.7 sigma and the offline ruler at 1.05. The board is 3.5x the
sharper instrument and it costs a slot rather than 2.2 h of card, five a day. A public score is also not
a resampled quantity: it is a deterministic read of the model on a fixed test set, so there is no
measurement error in it at all.

WHAT THIS CHANGES. "We cannot tell a winning recipe from a tie" is false as stated. OFFLINE cannot. The
board can, and always could. The binding constraint is therefore CANDIDATE SUPPLY, not measurement: we
have five 3.7-sigma reads a day and nothing mechanism-backed to put in them. The 2.2 h calibration run
that the 1/3 ruler owes is not worth its wall-clock — it moves a SCREEN from 1.05 to 1.33 sigma, and a
screen only has to avoid spending a slot on a dud. It does not change which candidates screen through.

WHAT SURVIVES. `--set run.folds=3` stays the default for new arms: it is free (an existing ConfigLeaf),
better balanced than k=5 (mean prevalence spread 0.0316 against 0.0352), holds 17 gold studies instead of
13, and takes the rarest column from 65 positives to 104. It is a better default, not a project. knee-ycz3f
is re-priced accordingly. The legacy k=5 fold 0 remains the only ruler the ~450 existing holdout records
can be read on.

THE ONE THING THAT WOULD REVERSE THIS. 3.7 sigma is a statement about the PUBLIC test set. Private is a
different draw and repeated selection on public is the shakeup trap. No private score exists while the
competition is live, so the risk cannot be measured here. What bounds it: the move we need is 7 ticks,
well outside the +/-4-tick public wobble the 33 tied pairs show, so a genuine 7-tick winner is unlikely
to be a public artefact. A 1-2 tick "win" IS inside that wobble and must not be shipped as a result.

## The audit closes: no lever remains on owned assets (appended 2026-09-18 06:20Z)

Four independent closures landed within three hours of each other, and together they leave the levers
table with no unbounded mechanism.

**Coverage closed on the card.** `slicecover36` (knee-jb0ci) was the instrument for the input side of
slice-to-study aggregation, one of the two places this file named a 1.5% lever could still live. It
truncated at 03:45Z on CANNOT REACH: e6 0.8255, generous ceiling 0.8604 from the corpus max e6-to-peak
gain over 27 matched runs, against control `headlr_ship_s2026` at 0.8702 midpoint. It loses by 0.0098,
1.6x the paired floor, on the best trajectory the corpus has ever recorded. It never led at any matched
epoch while taking 583 steps to the control's ~350 — more coverage is costly, not flat. knee-4by7q dies
with its premise.

**Supervision closed by exhaustion, three times.** 34 label-themed interpretations, with the lane
recorded closed on 2026-08-26, 2026-08-30 and 2026-09-03, and the axis measured with v2 winning it on
2026-09-17. That was the other of the two named places.

**The trunk was closed on published scaling, not on licence.** In-domain MRI SSL reads +0.073 at 80
labelled subjects and -0.005 at 800 (Zhang et al. MedIA 2024, README:7425); we hold ~4,400. The owner's
2026-09-14 external-data ruling does not reopen it, because licence was never what shut it.

**The composition escape is spent, and it failed on the BOARD.** `headlr_ship_s2026` IS the shipping
recipe plus every directional-positive knob, and at 0.9059 ledger75 it is the highest offline number the
project owns. It read board **0.937**. The three highest offline members read 0.9045 / 0.9053 / 0.9059
against board 0.937 / 0.941 / 0.937 — all three BELOW the fitted transfer line at -2.7 sd. So the knobs
did compose offline and the composition did not transfer. That is a stronger statement than "the knobs
are flat": it says the offline ruler's top end is winner's curse, and stacking more sub-floor terms
against it is fitting the ruler rather than the task.

### What this forbids

Resolving the directional knobs individually was the obvious reply, and it prices badly. An OOF rotation
(knee-700cm) takes the floor 0.0062 -> ~0.0028 but costs 5x training per member, so one knob comparison
is ~9 h and a +0.003 knob still reads at ~1.1 sigma. Nine hours for one-sigma is not a measurement, and
the -2.7 sd transfer failure says the number it produced would not transfer anyway.

### What is actually left

Neither queued refit can win. Under ADDITIVE (+0.005 on the honest twin) `depth24_ff_long36` predicts
0.946 and `headlr_ship_ff` 0.942; under CEILING both predict 0.9445. The champion is 0.946. So no refit
of any base we own clears it, and the base itself has to improve — on an axis the audit has just closed.

That leaves exactly one live question, and it is answered today for one slot and no card: whether the
refit term is ADDITIVE or a CEILING (knee-uh1j5, `calib_span_ff`, ~9 sigma). It does not win by itself.
What it decides is whether honest-base improvement is BANKABLE at +0.005 — the only remaining premise
under which small gains would be worth chasing at all — or whether the fit is exhausted at 0.9445 and
the remaining eleven ticks must come from a member this project has not built and has no mechanism for.

Stated plainly for whoever reads this next: at 0.946 against a 0.957 leader, we are not one knob away.
