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
