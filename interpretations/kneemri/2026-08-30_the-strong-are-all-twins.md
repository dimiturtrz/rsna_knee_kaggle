# The strong are all twins: the monoculture, measured over 117 members

2026-08-30. `python -m kneemri.evaluation.pool_census`, card-free, no slots.
Answers the open form of knee-nj4l.

## The question, and why it was still open

knee-nj4l says the champion is six copies of one lens and the way out is a member whose errors are
somewhere else. Its own note, added the same week, says the proposition **has never been measured**: the
diverse-seating curve that appeared to refute it rested on `mae_matched` (DINOv2-small, four epochs
against a 12-epoch family, peak 0.6735) and `focus_mcl` (0.5332 on the twelve-label mean — near chance on
eleven of twelve columns). Those two are decorrelated because they are undertrained and narrow. The pool
contained no strong-and-distant member to seat, so the curve could not test the proposition its conclusion
was stated about.

## The measurement

117 members share a holdout (150 records on disk; `RecordMix.narrow` keeps the largest agreeing pool).
For each: its own mean AUC, its NEAREST rank agreement to any other member, its nearest belief divergence.

```
nearest-agreement range among the 14 strong: 0.9011 to 0.9496
nearest-agreement range across all 117:      0.0688 to 0.9664
```

`strong` = within 0.0062 — this project's measured arm-vs-arm floor — of the pool's best solo, 0.8642
(`stack_increments_s2026`). Fourteen members qualify.

**The distance axis has 0.90 of range in the pool and 0.05 of it among the strong.** Not one of the
fourteen strong members has a nearest neighbour below 0.9011, while the pool reaches down to 0.0688.
Every strong member the project owns is a twin of another strong member. That is the monoculture as a
measurement rather than as a configuration dump, and it is the mechanical reason seven differently
seated entries land inside 0.004.

## The cell is occupied and the occupancy means nothing

Three members clear both cuts — `labelwise_dense` (0.8613, agreement 0.9011), `finding_calib_dense`
(0.8600, 0.9179), `small_batch6` (0.8598, 0.9159). Read that as an artefact of the cut, not a result: the
`distant` cut is this pool's MEDIAN agreement, which by construction puts about half the pool below it, so
the three sit 0.002 to 0.019 under a line drawn through the middle of a tight cluster. The members this
project has actually measured as far — `margin_seat` at 0.8772, `mae_matched` at 0.4691 — are not in the
strong set at all. Occupancy under a median is a weak reading; the RANGE is the one that does not move
when the threshold does, which is why the tool now prints both.

## What it licenses, and what it does not

**A pre-registration bar for the non-DINOv2 lane.** The cnx / swin / vjepa arms are justified by
diversity, and diversity has until now been an argument rather than a number. It can now be one BEFORE a
slot is spent: a new member earns the diversity claim by landing **solo >= 0.8580 AND nearest agreement
< 0.90** — outside the band all fourteen strong members occupy. Anything inside that band is another
copy of the lens however different its trunk is, and the census will say so for free.

**It does not license a slot.** Nothing here is a board reading. `seat_value`'s own doc holds: the
offline ruler ranks correctly and understates what a weak member costs, and every seat value in the
current pool sits under the +0.0015 the board has never been shown to resolve.

## Two caveats that bound the claim

**Ours-only.** Seven foreign records are excluded by default, and correctly: `foreign_overlap` measured
896 of our 909 holdout studies inside the labels `raptor` was distilled from, which is why `raptor_solo`'s
0.8938 was voided on 2026-08-29. So the one genuinely different-family model the project holds is not in
the 117 — and its offline number is not trustworthy enough to put it there. The census describes OUR
pool, which is the pool a seat search actually draws from.

**Nearest, not mean.** A member could sit far from 116 others and near exactly one, and this reading
would still call it a twin. That is deliberate — the question is whether a member contributes an ordering
nothing else already contributes — but it means the census cannot be read as a statement about average
diversity.

## Side reading, free with the same run

`recal_dense_s2026` (0.8610) and `calib_dense_hold_s2026` (0.8634) report identical nearest agreement
0.9496 and identical nearest divergence 0.0065 bits: they are each other's nearest neighbour. Whatever the
gold ruler says about `recal_dense` under knee-9198, it is not new information for the ensemble.
`convergence_check` reads it converged, so its 0.8610 is a real peak rather than a truncation floor.
