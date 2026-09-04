# The board stops amplifying near the top, and that resets what the target costs

**Read:** `python -m kneemri.evaluation.board_transfer`, 2026-09-04 11:16Z, 39 entries on the `ensemble`
scope, 598 orderable pairs.

## The numbers

| Band (offline holdout) | Resolved pairs | Median board points per holdout point |
|---|---|---|
| whole record | 598 orderable | +1.13 |
| at or above 0.8670 | 17 | **+0.98** [+0.94, +1.17] |
| at or above 0.8695 | 2 | +0.80 — direction only |

The tool labels the whole-record figure itself as the wrong number to price an arm with, and the banding
says why: compression is a decreasing function of level. Low in the record a holdout point bought more
than a board point; at the level this project now operates at it buys about one.

## What was being quoted instead

`+1.55`, `+1.72` and `+1.82` appear across `kneemri-allfit`, `kneemri-dark6`, `kneemri-dark7`,
`kneemri-v3seat`, `kneemri-calibsix`, `kneemri-holdfive` and `CLAUDE.md`. Every one of them is a
**whole-record** fit. They were not wrong when written — they described the record as it then stood, and
the +1.82-vs-+1.55 pair in `CLAUDE.md` is making a point about record COMPLETENESS, not about level. But
the record has since filled in at the top, and a whole-record median now averages a regime we have left
with the one we are in. Pricing a 2026-09 arm off any of them overstates its board return by roughly 60%.

## The consequence, which is the reason this is written down

The standing target is 0.953 against a champion at 0.937 — a gap of **+0.016 board**.

At +1.55 that gap reads as +0.010 offline, which is inside the range a seating, a weighting or a
re-lensing operation has occasionally produced. At **+0.98** the same gap reads as **+0.0163 offline**,
which is above this project's own 1.5% result bar. It is not a knob-sized distance.

So the reading is not "the board got harder". It is that the class of operation that could close the gap
changed: the member axis being closed (seven board flats) is no longer a separate observation from the
target being far — they are the same fact seen twice. Only something that moves the offline number by
more than 1.5% absolute can land 0.953, and on the current record that is a new trunk family or a data
change, not a re-arrangement of the pool we have.

## What to do with a slope from here

Quote the band, not the record. `board_transfer --near <the offline score the arm would be priced
against>` is the reading; the bare invocation is a summary of history. An arm priced at the champion's
level takes the 0.8670-band number, and takes it with its interval, because 17 pairs is what the top of
this record actually contains.

Related: `knee-j4jog`.
