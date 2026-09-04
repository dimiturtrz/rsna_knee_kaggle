# The +0.02 that made ortholarge look special was the kit era

**2026-09-04. knee-l67zk closed. Two submission slots, no card time.**

## What was ambiguous

`ortho_solo_v1` sat about +0.02 above the line the board record fits through DINOv2 solos. Two readings
produced that identically and the record could not separate them, because the only current-kit solo in
the record was the ortholarge one:

- **the trunk** — the knee-continued ViT-L transfers better to the hidden test set than its offline
  number says, which would make the in-domain trunk lane the most valuable open axis in the project; or
- **the era** — everything shipped in the current kit sits above a line fitted mostly over older kits,
  and `ortho_solo_v1` inherited that lift by being recent rather than by being ortholarge.

Offline level and submission date were collinear across every row that could have told them apart.

## The instrument

Two solos, current kit, fold 0, DINOv2, differing only in which member sits in the seat — the cell the
record never held. Pre-registered in both kernel docstrings **before either score**: both ≥0.900 = the
era explains it; both ≤0.885 = it does not; 0.886–0.899 = partial, close nothing.

| entry | member | offline | board |
|---|---|---|---|
| `spansolo_v1` (ref 55996812) | `calib_span_s2026` | 0.8545 | **0.921** |
| `holdsolo_v1` (ref 56009579) | `calib_dense_hold_s2026` | 0.8634 | **0.928** |

Both land in the ≥0.900 branch, and not marginally.

## The reading

**The era carries it.** A plain current-kit DINOv2 solo at 0.8545 offline already clears 0.92 on the
board. `ortho_solo_v1`'s excess over the DINOv2-fitted line is a property of when it shipped, not of the
trunk it shipped. The in-domain trunk lane does not get to count that +0.02 as evidence for itself — and
that matters right now, because `ortho_stack`'s whole value rests on whether the knee continuation is
worth anything at the shipped operating point. Offline, `ortho_stack − v3large_stack = +0.0140` still
stands on its own matched contrast; what it no longer gets is a board-side +0.02 corroborating it.

**`board_transfer`'s whole-record slope is era-contaminated.** It is fitted across kits, so it is
absorbing the era step as if it were compression. It should be refitted within era before it is quoted
again, and the two constants already quoted from it (+1.82 from the incomplete record, +1.55 from the
complete one) are both suspect for this separate reason.

## What this does NOT establish

The two points give 0.007 of board per 0.0089 of offline — a within-era slope near 0.79, far below
either fitted constant. That is **two points, at three-decimal board granularity, 0.0089 apart**. It is a
direction and nothing may be planned on it. It is quoted here only because it is the same evidence that
says the whole-record slope mixes eras; the refit is the follow-up, not this number.

Neither anchor is a bid on the champion, and neither was scored against one.
