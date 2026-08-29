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

The same grouping applied to the other large family — and `crop_mm` joins `centre` as a reading field,
which the numbers show directly:

    canonical / 20 slices / band 0.1-0.9 / centre image / crop 130mm   49 members
    canonical / 20 slices / band 0.1-0.9 / centre image / crop 140mm    1 member   (band140)
    merged    / 20 slices / band 0.1-0.9 / centre image / crop 130mm    1 member

So we own two large reads, and they are not equivalent:

| read | members owned | seated today | resolutions present |
|---|---|---|---|
| A — canonical, 20 slices, 0.1-0.9 | 49 | 3 (`trioseat_v1`) | 336 only |
| B — canonical, 12 slices, 0.2-0.8 | ~30 | 1 (the champion's `lr4e-5_ws_cf`) | 224, 336, 448 |

Read A is where `trioseat_v1` proved seats compose, and it is the larger pool — but every one of its
members was fitted at 336, so its only free diversity is treatment. Read B is smaller and carries three
resolutions, so it offers treatment AND resolution diversity for the same single decode. **Read B is the
better read, and it is the one seated once.**

Two decodes could therefore seat roughly seventy-nine members we already own. The pinned champion spends
about four decodes to seat seven.

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
- `logs/dryrun_probe224.log`, `logs/dryrun_group2.log`
- beads `knee-z652` (this lane), `knee-vfnr` (the trio result that opened it), `knee-nas5` (pixel-group
  collapse, which constrains groups and not seats), `knee-s9cr` (224 throughput)
