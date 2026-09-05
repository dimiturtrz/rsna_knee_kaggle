# Our 909-study holdout is a harder and wider ruler than either thing we are trying to predict

2026-09-05. Three measurements taken this week for three unrelated reasons turn out to be one
observation. None of them was designed to make this point, which is most of why it is worth writing
down.

## The three readings

**1. Same weights, two rulers, +0.039 apart.** `band_dense_hold_s2026` is `band_dense`'s recipe refit
with `--hold-annotations`, so the 58 radiologist-annotated studies are outside its loss and can score it.
It reads **0.8956 on the gold 58** and **~0.857 on the 909-study report holdout**. One model, one set of
weights, two truth tables, six times the 0.0062 holdout noise floor between them. Our report labels
over-call — 108 confident false positives against 32 misses over those same 58 studies — and the holdout
carries the identical noise, so it charges the model for cells where the model is right and the report is
not.

**2. The board's columns are all near the top, and their spread is below its own resolution.** Two
slots were spent ablating single columns to a constant 0.5 and reading the drop, with the metric first
pinned by `colnull` as a plain unweighted mean of twelve per-column AUCs. MCL — the column our holdout
scores **lowest**, 0.8104 — reads **0.980** on the board. Baker's — the column our holdout scores
**highest**, 0.9208 — reads **0.992**. The two endpoints of a 0.110-wide offline spread land 0.012 apart
on the board, with overlapping intervals at three-decimal rounding. Our column ordering does not survive
the trip, and the arithmetic forces the other ten columns to average 0.9356.

**3. Eight consecutive board flats.** Nine entries at 0.943-0.944 across six, seven and eight seats,
including `relabel7_v1`, the first of them to move supervision rather than seating. Every one of those
entries was assembled by ordering candidates on the holdout in reading 1.

## What they say together

Readings 1 and 2 are the same statement made against two different upper rulers. Against the
radiologists, our holdout is **too harsh** — it scores a model 0.039 below what a physician's reading of
the same images gives it. Against the board, our holdout is **too wide** — a 0.110 spread of per-column
difficulty that is 0.012 there. Both are consistent with one cause: our truth table is report-derived,
and report-derived noise is neither small nor symmetric. It is class-conditional and one-directional
(over-calling), which depresses the level and inflates the apparent difficulty of exactly the columns
where the reader is most verbose.

Reading 3 is then the predicted consequence rather than an independent puzzle. A search that orders
candidates by a ruler whose spread is inflated relative to the target will pick differences that are
partly noise structure, and those differences arrive at the board compressed toward nothing. Eight flats
is what that looks like from inside.

## What this does NOT license

It is not a claim that our holdout is uninformative, and nothing here says a member that reads better
offline reads worse on the board. Reading 2 measured LEVEL and explicitly failed to resolve ORDERING —
the two column intervals overlap, which is why `colbaker_v1`'s row retracts `colmcl`'s earlier "the map
inverts" reading. One of these is a level gap on one model; the other is a level-and-spread observation
on two columns. Neither is a transfer function, and `board_transfer` remains the only fitted statement
about how an offline delta arrives on the board.

It is also not a reason to switch rulers. The gold table is 58 studies and 696 cells. It cannot order a
roster, it cannot price a seat, and its cells were used to fit `Supervision.MEASURED_AGREEMENT` and the
`level_calibration` rates — so most of this project's supervision arms cannot be scored on it at all
without reading their own constants back (knee-vrioa). It is a second opinion, not a replacement.

## What it changes

The label axis is not closed by `relabel7_v1`'s flat, and that flat is now the third piece of evidence
for the ruler account rather than evidence against corrected supervision. The open question is whether
corrected supervision pays when read by something other than the ruler it is meant to fix, and the arms
that can answer it are the ones whose supervision carries no gold-derived constant:

- `relabel_hold` — `--unread-targets` from the confident-learning overlay, running tonight, scored on
  both rulers against `band_dense_hold`. Its four branches are pre-registered in its header and the
  SIGN PATTERN across the two rulers is the evidence, not either level.
- `rank966_hold` — `--label-errors` at the ranking selector and a 966-cell budget, queued behind it.
  It answers knee-jom26: `errors_dense`'s -0.0158 belongs to the confident-joint MASK at 5,360 cells,
  more than half of which the radiologist agreed with, and the 4.05x precision that licensed the axis
  was never actually the selector that ran.

Both are clean of both leakage routes, both are read on the gold ruler first, and between them they
separate SELECTOR from TREATMENT from RULER on an axis that until today had one refuted point on it.
