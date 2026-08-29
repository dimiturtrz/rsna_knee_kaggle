# Seats per decode: resolution was never a decode, and we have been paying for one anyway

*2026-08-29. Measured through the serve path, not inferred from the ledger.*

## What was believed

Every seat argument in this project has priced a member against a decode budget. `knee-nas5` measured
pixel-group collapse; `merged_span_cf` was justified as buying a cell "nothing else occupies";
`trioseat_v1` was written up as three seats bought with one decode. The shared premise was that a
member fitted at a different resolution is a member in a different pixel group, and therefore costs
another read of the corpus.

That premise is wrong, and the code that serves our submissions has said so all along.

## What the serve path actually does

`Ensemble.reads` (`kneemri/inference/ensemble.py:179-208`) groups members that can share one decode by

    (scheme, slices, tie, reading)

and its own docstring states the exclusion explicitly:

> Resolution is not in that list: `SlotReader.cropped` stops before the resize, so one read serves every
> size exactly — `Study.bags` returns them together and each is byte-identical to a native read at that
> size.

Neither pooling is in the key either. `slice_pooling` and `slot_pooling` are properties of the model, not
of the read. `Ensemble.reads` then builds `sizes=tuple(sorted({member.image_size for member in members}))`
for the group, and `predict_member` indexes `sized[member.image_size]`.

## The measurement

Asserting this from the metrics table would be the failure mode this repo has paid for repeatedly — a
value recorded in a config while the consumer reads something else. So it was checked by the decode
COUNT, which cannot be true by accident.

A two-member probe package (`lr4e-5_ws_cf` at 336, `native224_s2026` at 224) through
`kneemri.kaggle.dry_run --device cpu --studies 4`:

    4 studies, 25 series
      studies 1-4: decode 10s at [224, 336], elapsed 10s

One decode line, both sizes.

Then a 32-member package covering every fold-0 checkpoint at (canonical, 12 slices, band 0.2-0.8),
spread across 224, 336 and 448:

    4 studies, 25 series
      studies 1-4: decode 3s at [224], elapsed 3s
      studies 1-4: decode 4s at [224, 336, 448], elapsed 4s

Two reads for thirty-two members, and the large one carries **three resolutions at once**.

## What splits, and why it is the interesting half

The members that broke away are the tissue-centred ones: `"centre": "tissue"` against `"centre":
"image"`, identical on scheme, slices, band, tie and every pooling. The crop centre lives inside
`reading`, and it changes WHICH PIXELS EXIST rather than how they are sampled — the same reason the
scheme is in the key. So the split is correct, and it is the one axis in this family that genuinely buys
a second read.

That is worth stating as a positive: crop centre is a real decode-diversity axis, and resolution is not.
We have been spending our diversity budget on the one that is free and treating the one that costs as
incidental.

## What it changes

The champion's second read — canonical, 12 slices, band 0.2-0.8, centre image, tie first — is shared by
about thirty fold-0 checkpoints we already own. The champion seats **one** of them. Seven are packaged.
Every other seat on that read costs no decode at all, whatever resolution it was fitted at, and the
resolution differences are then diversity we are not paying for.

Compare the trades:

| entry | decodes spent | seats bought |
|---|---|---|
| `trioseat_v1` | 1 | 3 |
| the champion's second read, as seated today | 1 | 1 |
| the champion's second read, fully seated | 1 | up to ~30 |

`trioseat_v1` is the reason to take this seriously rather than as bookkeeping: it scored 0.938 against a
written pre-registration of 0.925-0.929 by seating three members on one decode, so a read holding several
seats is a measured effect on the board and not only a cost identity.

## The whole map, both reads

The hand grouping that produced the first version of this section was replaced by a tool the same
morning — `python -m kneemri.kaggle.roster_reads`, which groups raw checkpoints by the same
`Member.decode_key` the kernel serves under, so its answer cannot drift from what a submission does.
Pointed at every `.pt` under the processed root (195 files, 7 of which describe no reading and are
teachers or partial saves rather than members):

    108 members at [224, 336, 448]  canonical / 12 slices / band 0.2-0.8 / centre image / crop 130mm
     58 members at [336]            canonical / 20 slices / band 0.1-0.9 / centre image / crop 130mm
      9 members at [336]            canonical / 12 slices / band 0.1-0.9 / centre image / crop 130mm
      3 members at [336]            canonical / 20 slices / band 0.2-0.8 / centre image / crop 130mm
      ... nine more reads of one or two members each
    188 seats on 13 decodes = 14.5 seats per decode

So we own two large reads, and they are not equivalent:

| read | members on disk | packaged | seated today | resolutions present |
|---|---|---|---|---|
| A — canonical, 20 slices, 0.1-0.9 | 58 | 27 | 3 (`trioseat_v1`) | 336 only |
| B — canonical, 12 slices, 0.2-0.8 | 108 | 19 | 1 (the champion's `lr4e-5_ws_cf`) | 224, 336, 448 |

Read A is where `trioseat_v1` proved seats compose, but every one of its members was fitted at 336, so
its only free diversity is treatment. Read B is both the larger pool AND the one carrying three
resolutions, so it offers treatment and resolution diversity for the same single decode — and it is the
read that is seated once. **Read B is the better read, and it is the one we barely use.**

The packaged column is the one that decides what is buildable TODAY. Forty-six members are already in
`dimiturnt/kneemri-members` on those two reads, so an entry seating all of them costs **two decodes, no
card time and no dataset push** — a kernel edit and a slot. The pinned champion spends about four
decodes to seat seven.

The full-disk numbers were wrong in the first draft of this section (49 and ~30, from a hand grouping
over a filtered subset). Writing the tool was what corrected them, which is the argument for the tool.

## What it does not change

This is a COST result. It says the seats are free; it says nothing about whether they are worth taking.
Thirty members fitted in the same cell are plausibly correlated enough that the roster saturates after a
handful, and the honest prior is that most of them price at zero. The gate is
`experiments/arms/trio_seat_value.sh`, which prices these pools offline against both `trioseat_v1` and
the pinned champion before any slot is spent.

What the cost result does is move the bar. There is no decode budget to trade a candidate against any
more, so the decision threshold for a member on an already-paid read is the 0.0062 holdout floor and
nothing else. That is a lower bar than any seat question this project has asked before, and it is lower
for a reason that was measured rather than argued.

## The efficiency reading

Our efficiency-track row is stale (rank 16 at 0.925, against an accuracy rank of 28 at 0.943). That prize
measures score against serve cost, and serve cost is dominated by decodes on a four-CPU host — the
`reads` docstring says so directly: "decode is the I/O-bound half". Seats per decode is therefore not a
proxy for the efficiency metric, it is close to the metric itself. The same lane serves both prizes,
which is rare enough here to be worth naming.

## Provenance

- `kneemri/inference/ensemble.py:179-208` (`reads`), `:433-442` (`sizes`), `:497` (`sized[...]`)
- `kneemri/inference/members.py` (`Member.decode_key`, the key both the serve path and the tool take)
- `kneemri/kaggle/roster_reads.py` — `--roster <file>` or `--candidates <names>`, and `--max-decodes N`
  to refuse a roster that costs more reads than an entry budgeted for
- `logs/dryrun_probe224.log`, `logs/dryrun_group2.log` (the decode counts the serve path actually ran)
- beads `knee-z652` (this lane), `knee-vfnr` (the trio result that opened it), `knee-nas5` (pixel-group
  collapse, which constrains groups and not seats), `knee-s9cr` (224 throughput)

## The seats were free and they are worth nothing — priced, 2026-08-29

The cost result above said the seats cost no decode. It said explicitly that it said nothing about
whether they are worth taking. They were then priced, and the answer is a clean null.

`seat_value --against honest12` over every PACKAGED, UNSEATED member on each of the champion's three
own reads — 24 candidates, no card fit, no dataset push:

| read | priced | best marginal | best swap | bands excluding zero |
|---|---|---|---|---|
| B — 12sl, 0.2-0.8, 224+336+448 | 11 | +0.0003 | +0.0003 | 0 |
| A — 20sl, 0.1-0.9, 336 | 12 | +0.0001 | +0.0003 | 0 |
| span — 12sl, 0.1-0.9, 336 | 1 | -0.0003 | +0.0000 | 0 |

Nothing clears +0.0015, the separation under which this ruler has never been shown to pick a board tick,
let alone the 0.0062 holdout floor. `merged_span_cf`, priced the same way the same morning, is the best
of everything measured today at +0.0003 marginal / +0.0005 as a swap — still inside the ruler.

**The null is stronger than it looks, not weaker.** The control had to be `honest12` (0.937, twelve
members) rather than the pinned champion, because every one of `raptorseat_v1`'s members is a `_cf`
checkpoint — fold -1, fitted on the whole corpus including every holdout study. That is exactly why it is
the champion and exactly why it can never be priced offline; `seat_value` refuses it by name. `honest12`
is 0.006 weaker on the board, and a seat is EASIER to earn in a weaker roster. The pools read zero
against the easier target.

So the pre-registered second branch fired: the reads are saturated at the seats they already hold, and
seats-per-decode is a COST result only. It is still the right lens for the efficiency track — an entry
that serves fewer decodes for the same score is cheaper by the metric that prize measures — but it buys
no score. **The next slot goes to decode DISTANCE — a genuinely new read — not to a wider one.**
(That instruction was carried out within the hour and came back negative as well — see "Decode distance
priced too" below, which also names why the distance pool cannot answer the question it was asked.)

That is not a retraction of the cost finding, which stands and is measured. It is the answer to the
question the cost finding deliberately left open, and the honest reading is that our pools are deep in
members and shallow in mechanisms: 188 checkpoints, 13 reads, one architecture. The champion's own
`architecture-monoculture` principle said this from the other direction, and this pricing is its
independent confirmation from the seat side.

## Decode distance priced too, and it is negative — but the pool is confounded

The section above ended by saying the next slot goes to decode DISTANCE rather than to a wider read.
That was priced the same morning, against the same control, and it is a second null — this time with
every candidate NEGATIVE:

| candidate | nearest neighbour | alone holdout | marginal | best swap |
|---|---|---|---|---|
| `crop80_tissue_s2026` | 0.8375 | 0.8345 | −0.0006 | −0.0004 |
| `adjacent224_s2026` | 0.8617 | 0.8462 | −0.0002 | −0.0002 |
| `extended_ws_s2026` | 0.8775 | 0.8459 | −0.0003 | −0.0001 |
| `centre_tissue_s2026` | 0.8784 | 0.8409 | −0.0009 | −0.0007 |
| `tissue224_s2026` | 0.8979 | 0.8491 | −0.0005 | −0.0002 |
| `band_wide_s2026` | 0.9116 | 0.8553 | −0.0003 | −0.0001 |

Six seats on five decodes — 1.2 seats per decode, the opposite trade to the free seats — and every one
of them costs a read the champion does not already pay for. Best is −0.0001, and `seat_value` charges a
negative at about 3×, so roughly −0.0002 on the board. Not one band excludes zero.

**But read the two middle columns together before concluding anything about distance.** Seat value
tracks ALONE QUALITY almost monotonically here, and distance ANTI-tracks it: the most decorrelated
candidate (`crop80_tissue`, nearest 0.8375) is the weakest and prices second-worst, while the nearest
twin (`band_wide`, nearest 0.9116) is the strongest and prices best. Every distant member we own is
also a weak one. That is a **confound, not a finding** — this pool cannot separate "a new read buys
nothing" from "we have never trained a strong member on a new read".

And the confound has a recorded, named cause in at least the extreme case. The ledger's own
`crop80_tissue` heading says *"the control is not schedule-matched"*: it ran `--epochs 8` against the 12
every canonical member is fitted with. So the single member that could refute "distant members are just
weak" has never been asked the question on equal terms. Its cache
(`images_336px_12sl_80mm_tissue.npy`, 33.4 GB) is still on disk, so the re-ask costs card time and
nothing else — `experiments/arms/crop80_tissue_full.sh`, bead `knee-17kn`, branches pre-registered
there.

There is a third possibility that arm also tests, and it is the one worth the most. **A noisy member is
decorrelated from everything.** `mae_matched_s2026` reads nearest 0.4691 at a 0.6735 holdout — it is not
distant, it is wrong. If `crop80_tissue`'s 0.8375 is the same effect one order milder, then the
nearest-neighbour column is not a twin detector on weak members at all, and it is currently load-bearing
in three write-ups.

So the correct statement today is narrower than the last section's: **the distance pool as it exists
prices negative, and the pool cannot distinguish the read from the member.** The lane is not closed; it
is unmeasured.

### One thing both nulls agree on

In every list priced this morning — 24 free seats and 6 distant ones, 30 candidates, three reads and
five — the only positive move anywhere is a swap for `mae_matched_s2026`. `honest12` carries a dud, and
it is the one seat in that roster any of thirty candidates can improve on.

### Three harness defects, found by running it

None of them was the idea failing, and all three would have produced a confident wrong number.

1. **The pinned champion is unpriceable offline.** All-`_cf` members carry no holdout record. Every
   `seat_value` invocation in `trio_seat_value.sh` that omitted `--against` was asking a question that
   cannot be answered — as was the same arm's `--against trioseat_v1` (three members, all fold -1).
2. **Arms must use `uv run --no-sync --project "$REPO" python`, never bare `python`.** `lib.sh` points
   arms at an isolated `.venv-arm` precisely so a gate run cannot rewrite the interpreter underneath a
   live arm. Both pricing arms called bare `python` and died on `ModuleNotFoundError: jaxtyping` — a
   defect `trio_seat_value.sh` had carried since it was written, which read exactly like a broken venv.
3. **`confident224_s2026` was scored under a different label table.** `RecordMix` refuses it over the
   909 shared studies. Mixing it would have produced a seat value against a different ground truth: a
   number that looks fine and means nothing. Dropped from both candidate lists; a rescore makes it a
   candidate again.

## The null holds on the WHOLE pool, not a sample of it — 41 candidates, 2026-08-29 14:12Z

The pricing above ran over the members that were PACKAGED: 24 candidates across three reads. That is a
sample of a pool, and a null on a sample is a weaker claim than it reads as — the members we happened to
have pushed to a Kaggle dataset are not a random draw from the members we own.

So the pool was enumerated instead of listed. `roster_reads` over all 198 checkpoints on disk gives the
complete decode map: **191 members on 13 reads, 14.7 seats per decode**, and read B alone holds 108 of
them.

| members | resolutions | read |
|---|---|---|
| 108 | 224, 336, 448 | canonical / 12sl / 0.2-0.8 / centre image / 130mm / tie first / spread |
| 59 | 336 | canonical / 20sl / 0.1-0.9 / centre image / 130mm / tie first / spread |
| 9 | 336 | canonical / 12sl / 0.1-0.9 / centre image / 130mm / tie first / spread |
| 4 | 336 | canonical / 20sl / 0.2-0.8 / centre image / 130mm / tie first / spread |
| 11 | mixed | nine further reads, one or two members each |

Of read B's 108, five are smoke runs or other folds (`RecordMix` refuses them by split), one is
`confident224_s2026` (refused by label table), and the rest carry no holdout record — which leaves **41
scoreable candidates**, against the eleven the packaged pricing could reach.

**The number is the same to the fourth decimal.** Best move anywhere: `merged_blend_s2026` at **+0.0003**,
inside the +0.0015 separation floor, and `seat_value` says so itself — *"best is +0.0003, inside +0.0015:
this ruler cannot tell it from the incumbent"*. Second is `blendlabels_s2026` at +0.0001. Everything else
is zero or negative, down to −0.0005.

And the pattern the packaged run noticed in passing is now unmissable at 41 candidates: **every single
one of the 41 has the same best move — swap out `mae_matched_s2026`.** Not one of them is a marginal
ADDITION to the roster. The only value any member of this read carries is that it is not the dud.

So the lane closes on the widest pool available rather than on a sample: **read B is saturated at the
seats it holds**, and the free-seat design of `knee-3yq5` buys no score. The COST finding is untouched —
44 candidate seats on 2 decodes against the champion's 7 on about 4 — and it remains the right lens for
the efficiency track, where an entry that serves fewer decodes at the same score is cheaper by the metric
that prize measures.

What this does NOT close is decode DISTANCE, which the section above showed is confounded: every distant
member we own is also a weak one. `crop80_tissue_full.sh` (bead `knee-17kn`) is queued behind
`raddino_dense` to ask that one on equal terms.

## A second ruler, and it closes the lane the first one left open — full-pool greedy, 2026-08-29 14:30Z

The seat null above was measured with `seat_value`, which prices a candidate's marginal contribution to a
roster's **holdout mean**. That ruler has a known blind spot and it is not a small one: a member that helps
on half the findings and dilutes on the other half reads as nothing. So the null it produced could not, on
its own, close the decode-distance question — the one lane the 41-candidate pricing explicitly left open.

`record_mix --seats greedy --largest-pool` over every checkpoint in the processed root is a different ruler
over the same evidence. It never asks for a mean delta; it asks **at what position does forward selection
reach for this member at all**. Position cannot be diluted by averaging, which is exactly the defect that
made the first ruler unable to settle the question.

### The pool, and what it dropped

`RecordMix` kept 110-odd records and named everything it removed rather than refusing on the first
mismatch: eleven smoke/other-fold checkpoints plus `confident224_s2026` as "not mixable with the majority"
(it is scored under a different label table), and `pool_channels_fold0` / `pool_max_fold0` skipped for
carrying no holdout record. That surviving set is the denominator and it is read off the log line, not off
the number of paths passed.

### The curve saturates at ten

```
greedy  1: 0.8634  +calib_dense_hold_s2026
greedy  5: 0.8757  +finding_calib_dense_s2026
greedy 10: 0.8767  +calib_span_s2026          <- peak
greedy 11: 0.8767  +contrast_dense_s2026      <- plateau
greedy 12: 0.8766  +small_batch6_s2026
greedy 51: 0.8733  +base_dense_s2026
greedy 109: 0.8663
```

**The 0.8767 is not quotable and is not quoted here.** `record_mix`'s own docstring is explicit that
`greedy` chooses and scores on the same studies and "reports its own overfitting as a result"; only the
`honest` rule, refitted on half the studies and scored on the other half, produces a number that may be
cited. What is readable is the **shape** — a peak at ten members and a monotone decline across the
remaining hundred — and the **positions**, which is what everything below rests on.

Only two of the top twelve are `--fold -1` (`merged_span_cf` at 2, `merged_blend_cf` at 8). That matters:
the last time this project read greedy positions, on `raptor_distill`, positions 1, 2, 4, 7, 8 and 18 were
all fold −1 checkpoints whose holdout sits inside their training set, and the peak was "not a number about
anything". Here ten of the top twelve are honestly-trained fold-0 members, so positions between honest
members are comparable.

### Every distant decode enters late, and the ordering tracks weakness rather than distance

```
greedy 29  adjacent224_s2026         alone 0.8462   spacing adjacent
greedy 42  native224_s2026           alone 0.8481
greedy 54  res392_ws_s2026
greedy 60  extended_ws_s2026         alone 0.8462   nine slots, its own cache
greedy 64  tissue224_s2026           alone 0.8491   centre tissue
greedy 76  crop80_tissue_full_s2026  alone 0.8343   80 mm crop
greedy 80  crop80_tissue_s2026       alone 0.8345
greedy 90  centre_tissue_s2026       alone 0.8409
```

Not one is inside the peak set. `seat_value` priced this lane negative on the mean over six members;
greedy prices it negative on position over eight. **Two independent rulers, same sign**, and the second one
is immune to the first one's blind spot. The confound recorded above — every distant member we own is also
a weak one — is still true and is still the honest caveat, but it now has to explain two rulers rather
than one, and within this table distance and position do not co-vary once alone-quality is accounted for.

`extended_ws_s2026` deserves its own line, because it was killed on a mean of **+0.0002** and its position
had never been read. It reads 60. The mean and the position agree, which is the case where the cheap ruler
was right and it is worth saying so.

### What it cost to find out, and what it saved

Nothing on the card. It also caught a queued arm that would have cost about three hours of it:
`crop80_tissue_full.sh` was queued behind `raddino_dense` on the argument that `crop80_tissue`'s weakness
had a schedule cause rather than a mechanism cause. `experiments/README.md:618` had already closed that —
"the schedule defect is CLOSED, and it was never the explanation", 0.8343 at twelve epochs against a 0.8492
control, −0.0006 — and the checkpoint was already on disk. The watcher is killed and the arm dequeued.

### What it does not close

A distant **trunk** is not a distant **decode**. `raddino_dense` reads read A — canonical, 20 slices, band
0.1–0.9, 336 px — the decode fifty-nine existing members already sit on, and differs from its matched
control in the encoder's pretraining and nothing else the decode key can see. Distance in pixels is now
measured twice and does not buy a seat; distance in **features** is untested, and that arm is the only
member this project has ever trained that isolates it. Its gate is pre-registered at `knee-fzpy`: position
inside the peak set on a solo AUC well below control, read by re-running this same greedy over the same
pool with the new checkpoint in it.

## The honest ruler, and the leak I nearly quoted from it — 2026-08-29 14:40Z

`record_mix --seats honest --largest-pool` finished over the same ~110-record pool as the greedy run
above. It is the only one of the three seat rulers whose number may be quoted, because it refits the
greedy choice on half the studies and scores it on the other half, both directions, rather than choosing
and scoring on the same 881.

```
  1 members: held-out 0.8624   in-sample 0.8634   bias +0.0010
  2 members: held-out 0.8689   in-sample 0.8710   bias +0.0021
  4 members: held-out 0.8723   in-sample 0.8747   bias +0.0024
  8 members: held-out 0.8743   in-sample 0.8764   bias +0.0021
 11 members: held-out 0.8744   in-sample 0.8767   bias +0.0023
 16 members: held-out 0.8745   in-sample 0.8763   bias +0.0018
 21 members: held-out 0.8748   in-sample 0.8760   bias +0.0012   <- best honest
 24 members: held-out 0.8747   in-sample 0.8758   bias +0.0010
```

**THE SHAPE IS THE RESULT AND THE PEAK IS NOT.** Held-out runs 0.8743 at eight members and 0.8748 at
twenty-one — a span of +0.0005 over thirteen extra members, an eighth of the 0.0062 holdout floor and a
third of the +0.0015 board-tick separation. "Best honest count 21" is what the argmax of a flat line
prints; it is not a recommendation to seat twenty-one. The honest ruler's finding is that **the roster
saturates at eight**, and that everything the pool has to offer past the eighth seat is unmeasurable.

That is the third independent ruler to say the pool is spent, and the first one entitled to say it. The
41-candidate `seat_value` null priced marginal contribution to the mean; greedy priced entry position;
this one prices the whole roster under a fitted-elsewhere choice. Three rulers, three blind spots, one
sign.

**Selection optimism is small, which is its own finding.** The in-sample column sits +0.0010 to +0.0032
above the held-out one and the gap SHRINKS as seats are added (+0.0032 at ten, +0.0010 at twenty-four).
So the greedy ordering quoted above was not fantasy: its 0.8767 peak is about +0.0023 optimistic, not the
manufactured result that twelve choices over 881 studies could have produced. The reason to distrust
greedy was real and the distrust turned out to be cheap — worth recording, because it means the greedy
ordering can be read for POSITION (which is what `knee-fzpy` gates raddino on) without an honest re-run
each time.

**A LEAK I REPORTED HERE AND THEN REFUTED — RETRACTED 15:09Z, the text kept because the retraction is the
finding.** Three members of this pool end in `_cf`, and I wrote that `_cf` meant fold −1: trained on every
study including the fold-0 holdout these numbers are computed on, so their "holdout" AUC would be a
training-set AUC. Two of them are seated inside the first eight:

```
greedy  2: +merged_span_cf.pt
greedy  8: +merged_blend_cf.pt
greedy 18: +merged_dense_cf.pt
```

The reasoning was right and the premise was invented. A split-half estimator genuinely cannot touch member
leakage — it removes optimism from the CHOICE by fitting the choice on studies it does not score, while
both halves would still lie inside a leaked member's training set. All of that stands. What does not
stand is that any member here is leaked.

**`_cf` DOES NOT MEAN FOLD −1. IT MEANS THE MERGED CORPUS.** The checkpoints say so themselves:

```
merged_blend_cf   fold=0 folds=5 split_seed=2026 scheme=merged train_fraction=1.0
merged_span_cf    fold=0 folds=5 split_seed=2026 scheme=merged train_fraction=1.0
merged_dense_cf   fold=0 folds=5 split_seed=2026 scheme=merged train_fraction=1.0
```

Fold 0 of 5 on split seed 2026 — the same split every other member in the pool holds out. `merged` is a
`FOREIGN_SCHEME`: the training corpus carries foreign studies alongside the RSNA ones, and the fold
assignment still holds out RSNA fold 0. There is no leak, and **0.8743–0.8748 is quotable as it stands.**
The `_cf`-free re-run is cancelled; there is nothing to correct.

**WHAT ACTUALLY WENT WRONG, which is worth more than the false alarm.** `RecordMix` raises "a whole-corpus
member (--fold -1) writes no `.holdout.npz`" — so the existence of `merged_*_cf.holdout.npz` was already
proof, on disk and in the source, that these are not fold −1. I had the rule and the counter-evidence in
the same session and asserted the opposite from a NAME. Two suffix letters carried a claim about training
data that one `torch.load` of the checkpoint's own `options` refutes in nine seconds.

The tell I ignored: I "remembered" that the routed `blendlabels` pricing had dropped these two for this
reason. It had. That earlier drop was the same unverified inference, repeated — which is how a guess
becomes a fact about the repo. A belief that arrives already attached to a justification is the one to
check, because the justification is what makes it feel checked.

**AND IT WAS THE SECOND ONE THIS HOUR.** The transfer-slope section below records nearly filing `+1.82` as
refuted off a range-restricted eyeball. Both were caught, neither by a gate — by re-deriving a premise
that had only ever been asserted. The difference between the two is that the slope check took a minute and
this one takes nine seconds, and the nine-second one had already produced an interpretation section and a
P2 bead before it was run.

**WHAT IT IS WORTH IF IT HOLDS.** Eight seats scoring what twenty-four score is a score-lane null and an
efficiency-lane result: the shipped champion seats twelve. A roster a third smaller for a held-out
difference of +0.0005 is the shape of an efficiency-prize entry, and it costs a submission slot to learn
whether the board agrees. It is not worth a slot on its own today — but if raddino enters the peak set
under `knee-fzpy`'s gate, the eight-seat roster rebuilt around it is one submission that serves both
prizes, and that is the shot worth holding the slot for.

## What the offline ruler can actually resolve on the board — 2026-08-29 15:06Z

The leak above raised a second question worth one minute of arithmetic: two `_cf` members sit in the
SHIPPED champion (`honest12`), so is its offline number inflated relative to the board the way the honest
curve's is? `comp/board.jsonl` answers it directly — 22 scored entries whose offline column is our own
ruler on our own members, `raptor_solo` excluded because its offline is a foreign member's number on a
different ruler.

**The `_cf` confound does not survive contact with the board — and it turned out not to exist at all.**
This check was run BEFORE the retraction above, against a leak that was never there, and it is kept
because it independently reached the right answer and because it is the measurement the transfer finding
came out of. Ranked by board score, the gap
`board - offline` runs +0.058 to +0.064 across seventeen consecutive entries from 0.919 to 0.937.
`honest12`, the only `_cf`-carrying entry among them, sits at +0.0619 against a pack mean of ~+0.0613 —
its offline is about +0.0006 high, which is a third of the scatter. Two leaked members out of twelve are
real and unmeasurable at the roster level. The shipped champion's offline number stays usable.

Read against the retraction, this is a null explained: the gap is ordinary because the members are
ordinary. A +0.0006 residual on the one entry I had singled out is what a roster with NO leaked member
looks like, and I read it as an upper bound on a leak instead of as its absence. **An effect that is
merely small when your hypothesis predicts small is not confirmation — check the premise before you
spend a section bounding the size of something.**

**I nearly filed the transfer constant as wrong, and the reason I was wrong is worth more than the check.**
Reading down the top eighteen entries the gap looked like a flat OFFSET of +0.061, not a slope — which
would have contradicted `board_transfer ≈ +1.82` outright. It does not. Those eighteen span an offline
range of 0.017; fitting over all 22 (offline 0.8518-0.8751) gives:

```
slope 2.099   95% CI [1.197, 3.012]   r = 0.855   residual sd 0.0078
forced slope 1.00 -> residual sd 0.0101
forced slope 1.82 -> residual sd 0.0078
```

Range restriction flattens a slope toward zero, and eyeballing the top of a sorted list is range
restriction. **+1.82 sits comfortably inside the interval and fits as well as the free slope.** The
constant stands.

**BUT THE INTERVAL IS THE FINDING, NOT THE POINT ESTIMATE.** [1.20, 3.01] is not a number to multiply an
offline gain by and believe the product. And the residual sd is the sharper statement: **0.0078**. That
is what predicting a board score from an offline score is worth — and it is comparable to the ENTIRE gap
to first place (0.009 at the 14:00Z read). Two consequences, both about how slots get spent:

1. **An offline difference under ~0.004 predicts nothing about the board.** Below that the two rosters'
   predicted board scores overlap almost completely. This is the quantitative reason sixteen board entries
   sit inside 0.005 of each other despite being chosen as improvements — the ruler that chose them cannot
   resolve them.
2. **It prices the honest curve's plateau.** Eight seats to twenty-four is +0.0005 offline, an eighth of
   the resolution floor. Submitting to find out which is better is spending a slot on a coin flip.

The lane where a slot buys knowledge is a change big enough to clear 0.004 offline, and nothing in the
current pool is. That is the case for the raddino arm and against another reseating, stated in the
instrument's own units.

## One number explains the sign flip, the plateau, and why I had to rewrite my own gate — 15:12Z

`knee-0ceh` has stood since 2026-08-27 as the strongest negative result on this project's instruments:
offline-greedy seat selection **refuted**, because `honest12` turned a +0.0050 offline gain into 0.937 on
a board where six seatings ordered themselves *inverse* to their offline gains. A selector whose gains
arrive with the opposite sign is broken, not weak.

The transfer residual measured above says otherwise, and says it arithmetically. **Residual sd 0.0078.**
`honest12`'s +0.0050 is 0.64 of one residual sd. The six entries span 0.937–0.941 — a board range of
0.004 — against an offline spread of ~0.005 whose *predicted* board spread is ~0.009. Six points inside
half a residual sd, ordered at random. That is not an instrument returning the wrong sign. It is an
instrument returning noise, which has an even chance of looking like the wrong sign and did.

**The two diagnoses differ in what they forbid.** "Broken" forbids offline-greedy seating forever.
"Below resolution" forbids it under ~0.004 and says nothing above. The six-entry set cannot distinguish
them, because nothing in it was ever more than 0.005 apart — and `knee-0ceh` chose the stronger reading.
Its own stated mechanism (a 454-study fit cannot resolve a seat worth less than the 0.0062 paired floor)
*is* the weaker reading, already written down.

The honest curve closes the circle from the other side: +0.0005 across thirteen seats is a sixteenth of
the transfer resolution. **The seat axis has nothing left to resolve, so every selector ranking those
seatings must return noise.** Saturation, the sign flip, and the plateau are one fact seen three ways.

**AND IT COST ME MY OWN PRE-REGISTRATION.** `knee-fzpy` gated raddino on greedy POSITION — control 51,
peak 10–11. But the greedy curve reads 0.8767 at ten and 0.8733 at fifty-one: those two branches are
~0.003 apart, under both the holdout floor and the transfer residual. **A gate that cannot resolve its own
two branches is not a gate.** Worse, I had cited `knee-0ceh`'s surviving half — "offline holdout still
rules a SINGLE member" — to license reading position, when position is not a single-member number at all.
It is a seating rank, which is precisely the half that issue refutes.

Rewritten before the arm landed, which is the only honest time to change a pre-registration: raddino is
now gated on whether the **honest eight-seat held-out number moves**, ≥ +0.004 to be a real member,
≤ +0.001 to confirm the head-monoculture prediction. Stricter than what it replaced — the roster's score
must actually move, on the one ruler that fits its choice on studies it does not score.

**The through-line of this whole session is one habit.** Three claims re-derived from their premises today:
`_cf` meaning fold −1 (false — two suffix letters, refuted by one `torch.load`), `board_transfer` being
refuted (false — range restriction), and greedy position being a usable gate (false — below resolution).
None was caught by a gate. All three were caught by asking what a number was measured *on*, and all three
had already produced downstream writing before the question was asked.

## Per-finding routing, measured at last — and the cheat loses — 2026-08-29 15:20Z

The seat lane closed on the mean. The obvious next question is whether the mean was the wrong ruler:
the competition metric is twelve independent AUCs, so a pool with no marginal member could still hold
members that are *each* strong on a different finding. Nothing in this repository had ever asked. The
per-label module was written on 2026-08-20 and cited twice as prior art without producing a number.

It turns out it had produced one all along, in a reading nobody read: `record_mix` with no flag runs
`MemberCurve.logged`, and `logged` has called `LabelCombine.curve` the whole time. I discovered this by
starting to add a `--records` mode to `label_combine` and getting an import cycle —
`label_combine -> record_mix -> member_curve -> label_combine`. The cycle was not an obstacle. It was
the gate saying the feature already had a home, and it took nine minutes to hear it. Reverted.

Over the eight honest seats, 909 shared studies, no encoder and no card:

```
top- 1 per label:  in-sample 0.8736   split-half 0.8669
top- 2 per label:  in-sample 0.8796   split-half 0.8737
top- 3 per label:  in-sample 0.8808   split-half 0.8767
top- 4 per label:  in-sample 0.8806   split-half 0.8770
top- 5 per label:  in-sample 0.8798   split-half 0.8774   <- best routed
top- 6 per label:  in-sample 0.8789   split-half 0.8772
top- 7 per label:  in-sample 0.8775   split-half 0.8772
top- 8 per label:  in-sample 0.8764   split-half 0.8766   (= the flat rank-average)
oracle 0.8736  vs plain average 0.8764  vs best single 0.8634  (headroom -0.0029)
```

Routing is worth **+0.0008** against its own matched control under one estimator — an eighth of the
holdout floor, a fifth of what the board can resolve. The number to keep is not that one. It is the
**negative oracle**: selecting per label WITH THE ANSWERS scores 0.8736 against the flat average's
0.8764. The cheat loses to the thing it is supposed to bound.

Read carefully, the oracle compares one member per label against an eight-member average, so
correlation alone would favour the average and the headroom is not a clean ceiling. The top-k curve is
the fair test, and it says the same thing more slowly: a small rise to k=5, then a walk back to the
control. Both agree there is no per-label specialism here to route on.

That is head-monoculture, measured on the axis where it had not yet been measured. The seat curve said
the members carry the same information in the mean; this says they carry it in the same PLACES. Two
rulers, one shape. And it sharpens what a new member has to be: not a better score, a different
per-label profile. If raddino lands with one, this curve is a minute of CPU to re-run — the axis has a
re-run condition, not a verdict.

### The label headroom on the same mix, and why it is not a lever

The same records answer the other per-label question for free — where the eight-seat mix's score
actually lives:

```
mix of 8: 0.8764 over 12 labels, each worth 0.0833
  MCL                0.8243  n= 171   to 0.90: 0.8827  (+0.0063)
  Lateral OA         0.8374  n= 233   to 0.90: 0.8816  (+0.0052)
  Fracture           0.8409  n=  68   to 0.90: 0.8814  (+0.0049)
  ...
  Baker's            0.9246  n= 217   to 0.90: 0.8744  (-0.0021)
```

Three findings hold +0.0164 between them, which clears the bar — and it is the trap the ledger already
disarmed on 2026-08-23. That column is a PHYSICAL headroom, not a purchasable one: put our columns and
the published weights on the same 909 studies and we beat their average single member on ten of twelve
labels, losing only MCL (−0.0346) and Fracture (−0.0167). Closing both completely, against weights that
demonstrably exist, is +0.0043 — the ceiling of the entire weak-label programme, a quarter of the bar.
`focus_mcl` was cancelled on that arithmetic and stays cancelled.

Recorded here only because the routing run printed the headroom column beside the routing curve, and a
+0.0164 sitting unlabelled in a log is how a settled question gets re-opened by whoever reads it next.

## The pool has an aggregation ceiling near 0.876, and the method does not matter — 15:27Z

Eight curated members is the wrong pool to ask about specialists. One specialist among a hundred is
exactly what a mean-based seat scan hides, so the routing question gets SHARPER with pool size — and the
109-member pool was the one pool the reading could not reach, because the only door to it also
enumerated 2**109 subsets. Splitting `MemberCurve.routing` out of `logged` is one method and no new
logic, and it is the difference between a question that can be asked and one that cannot.

```
top-  1 per label:  in-sample 0.8760   split-half 0.8652
top-  9 per label:  in-sample 0.8827   split-half 0.8761   <- best
top-109 per label:  in-sample 0.8663   split-half 0.8664   (= the flat rank-average of all 109)
```

**+0.0097 over its own control, and that number is not routing's value.** The control is the flat
average of 109 members, most of them weak; nearly all of that gap is ordinary member selection, which
the greedy scan already performs. A matched control has to hold selection fixed and vary only the
per-label part. Put every honest reading of this same pool on one line:

| strategy, all split-half over the same 909 studies | score |
| --- | --- |
| routed top-9 over 109 members | 0.8761 |
| honest greedy seats over 109 members (peak at 21) | 0.8748 |
| flat rank-average of the curated 8 | 0.8766 |

Routing beats mean-greedy by **+0.0013** and loses to a flat average of eight hand-picked members by
**−0.0005**. Three strategies that share nothing but the pool — pick by label, pick by mean, pick by
hand and then do not pick at all — land inside 0.0018 of one another.

That is the result, and it is larger than any of its rows. **The pool has an aggregation ceiling near
0.876 and the method of combining does not matter.** The seat curve said the members carry the same
information in the mean. The per-label curve said they carry it in the same places. This says no
combining rule finds anything the others missed, which is the strongest form the claim can take: the
ceiling is a property of the members, not of the arithmetic applied to them.

Everything downstream follows. There is no aggregation work left worth a slot, and the next number can
only come from a member of a different KIND. That is exactly what is on the card.

## A fourth strategy, and the first one that was allowed to fit weights (knee-669u)

The three strategies above all *select* members; none of them *weights* them. `vote_ceiling` closes that
hole, and it closes it honestly: weights are fitted on one half of the 909 studies and scored on the
other, both directions, with the fit-on-self number printed beside each honest one so the overfit is
visible rather than assumed. It is the read `QualityWeightedRank` asked for in its own docstring and
never got. No card, no slot — the holdout records were already on disk.

Over all 109 scoreable members:

| rule | honest | vs flat | overfit gap |
| --- | --- | --- | --- |
| `RankAverage` | 0.8661 | — | +0.0000 |
| `QualityWeightedRank` | 0.8670 | +0.0009 | +0.0001 |
| `LabelQualityRank` | 0.8672 | +0.0011 | +0.0007 |
| `MedianLogit` | 0.8677 | +0.0015 | +0.0000 |
| `LogitAverage` | 0.8688 | +0.0027 | +0.0000 |

Over four members — the greedy top of the whole-stack pool, and the shape that actually ships:

| rule | honest | vs flat |
| --- | --- | --- |
| `RankAverage` | 0.8616 | — |
| `QualityWeightedRank` | 0.8616 | −0.0000 |
| `LabelQualityRank` | 0.8617 | +0.0000 |
| `LogitAverage` | 0.8619 | +0.0003 |
| `MedianLogit` | 0.8611 | −0.0005 |

At the pool size that ships, all five rules lie inside 0.0005 against a 0.0053 Hanley-McNeil floor. The
winner at BOTH sizes is the one rule with zero fitted parameters, so nothing here is bought by fitting;
the two fitted rules carry overfit gaps of +0.0001 and +0.0007, which is small in the reassuring sense
and worthless in the useful one — they do not overfit because there is nothing left for them to fit.

Read the +0.0027 at 109 members for what it is. It is `LogitAverage` tolerating a mostly-junk pool
better than `RankAverage` does, and it collapses to +0.0003 the moment the pool is four good members.
Quoting it as a weighting gain would be quoting a statement about the junk.

So the ceiling now holds against four strategies that share nothing but the pool: routed top-k per
label, honest greedy seats, a flat rank-average of a curated eight, and learned weights. The
conclusion above does not change, it only stops being provisional — and the aggregation lane is closed
rather than merely unpromising. The champion already serves `LogitAverage`, which is the honest maximum.
