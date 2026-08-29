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

**WHAT I ALMOST QUOTED, AND WHY IT IS WRONG.** Three members of this pool end in `_cf` — fold −1, trained
on every study including the fold-0 holdout these numbers are computed on. Their "holdout" AUC is a
training-set AUC. Two of them are seated inside the first eight:

```
greedy  2: +merged_span_cf.pt
greedy  8: +merged_blend_cf.pt
greedy 18: +merged_dense_cf.pt
```

A split-half estimator does not touch this. It removes optimism from the CHOICE, by fitting the choice on
studies it does not score; it cannot remove leakage from a MEMBER, because both halves are inside that
member's training set. So the honest column is fitted-elsewhere and still contaminated, and the absolute
0.8743-0.8748 is not a number about held-out performance.

I knew this. The routed `blendlabels` pricing launched an hour earlier dropped exactly these two from
honest12's roster for exactly this reason, and I did not carry the rule across to the pool run. The
denominator discipline is not a thing you apply once per session.

**WHAT SURVIVES THE LEAK.** The SHAPE probably does and the LEVEL does not. Two leaked members inflate
every row from the second onward by roughly the same amount, so a curve that is flat from eight to
twenty-four stays flat when they are removed — leakage shifts a level, it does not manufacture a plateau.
The bias column survives for the same reason. What must be re-measured is the level, and with it the
question of whether eight is still the saturation point once two of the eight are gone (the honest answer
is that it may become ten, and that ten and eight are the same claim about a spent pool).

**THE CORRECTED RUN IS QUEUED, NOT DONE.** Re-run with the `_cf` members excluded from `--records` once
the card frees — it is offline arithmetic over cached predictions, but it is CPU-heavy enough to have
already cost the running arm a measurable epoch, which is the other half of `knee-8hau`.

**WHAT IT IS WORTH IF IT HOLDS.** Eight seats scoring what twenty-four score is a score-lane null and an
efficiency-lane result: the shipped champion seats twelve. A roster a third smaller for a held-out
difference of +0.0005 is the shape of an efficiency-prize entry, and it costs a submission slot to learn
whether the board agrees. It is not worth a slot on its own today — but if raddino enters the peak set
under `knee-fzpy`'s gate, the eight-seat roster rebuilt around it is one submission that serves both
prizes, and that is the shot worth holding the slot for.
