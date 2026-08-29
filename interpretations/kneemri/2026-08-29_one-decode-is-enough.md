# One decode is enough — the second and third buy nothing measurable

2026-08-29. Task: kneemri. Tool: `python -m kneemri.kaggle.decode_entry` (new this day).

## The question nothing could ask before

The efficiency track pays its own podium and weights runtime hard, and runtime on the four-CPU Kaggle
host is DECODE-dominated — a fleet pass measured the card at 0% while pydicom worked. So the efficiency
question is not *"what is the smallest roster"* but *"what is the best score available at N decodes"*,
and until today no tool spanned it: `roster_reads` prices a set's cost in decodes and never its score,
`seat_value` prices one member's marginal score against a fixed roster and never its cost, and neither
searches over rosters at all.

`decode_entry` is that join — greedy forward selection over a pool, refusing any candidate whose read
would push the entry past a ceiling on decodes.

## The measurement

37 members (the packaged read-A / read-B / span pools plus the six decode-distance candidates), riding
8 distinct decodes, fold-0 holdout, rank average.

| decode budget | best score | seats | the walk's last step |
|---|---|---|---|
| 1 | 0.8736 | 7 | +0.0000 |
| 2 | 0.8744 | 7 | +0.0001 |
| 3 | 0.8746 | 7 | +0.0001 |

The second decode buys **+0.0008**. The third buys **+0.0002**. Both are an order of magnitude inside
the 0.0062 holdout noise floor and well inside the +0.0015 separation under which `seat_value` has never
been shown to pick a board tick.

The control makes the same point from the other side. `honest12` — the only board entry priceable
offline — scores **0.8750** on this holdout with **twelve seats across three decodes**. The greedy
one-decode entry is 0.8736 with **seven seats on one**: −0.0014, inside the floor, at a THIRD of the
decode cost and seven-twelfths of the inference.

## And the board says it independently

This is not an offline-only reading, which matters because every offline number here carries the usual
transfer caveat. `trioseat_v1` — three seats, ONE decode — scored **0.938**. `honest12` — twelve seats,
THREE decodes — scored **0.937**. The cheaper entry is the higher one, on the hidden test set, with no
offline ruler involved.

Two independent lines, one offline and one on the board, agree that decodes two and three are not
earning their runtime.

## WHICH decode matters more than HOW MANY

Same tool, one budget, three pools — each read searched alone:

| the one decode | best score | seats |
|---|---|---|
| read A — canonical / 20 slices / band 0.1-0.9 / 336 | **0.8736** | 7 |
| read B — canonical / 12 slices / band 0.2-0.8 / 224+336+448 | 0.8669 | 4 |
| span — canonical / 12 slices / band 0.1-0.9 / 336 | 0.8654 | 3 |

Read A beats read B by **+0.0067**, which CLEARS the 0.0062 floor — the only separation measured today
that does. So the decode budget is the wrong knob and the decode CHOICE is the right one: adding a
second read buys +0.0008, and picking the better single read buys eight times that.

`trioseat_v1` already serves read A (`final_ws_cf`, `grade_ladder_cf`, `through_dense_cf`), so the
offline ruler and the board's best one-decode entry independently chose the same decode. That is a
harness check passing, not a new fact — but it is the check that would have caught the ruler pointing at
the wrong read.

One thing read A's walk says that is not yet spent: it reaches 0.8705 at three seats and 0.8736 at
seven. +0.0031 for four more seats on a decode already paid for — under the floor, over the +0.0015
ruler, and on the ONE axis `knee-nas5` measured AMPLIFYING to the board at 2.4x.

**But the obvious way to spend it is already refuted, by the ledger rather than by a slot.** Those four
seats exist only as `_s2026` — fold-0 — and `greedy7_v1` is exactly that experiment: seven fold-0
members, 0.8712 offline, **0.933** on the board, against `trioseat_v1`'s three `_cf` seats at **0.938**.
The one-decode seven here is 0.8736 offline, +0.0024 over greedy7, which interpolates to roughly 0.9355
— still under the trio it would be extending. Bolting fold-0 seats onto an all-`_cf` roster dilutes it
faster than the seat count repays.

So the seat term on read A is real and the cheap route to it is closed. The route that stays open is
refitting those four members at `--fold -1`, which costs card time rather than a slot, and turns
`trioseat_v1` into a seven-seat entry on the same single decode. That is the efficiency play worth
running: about three hours of card against a podium with three places on it, where we are rank 16.

## What it does and does not license

It licenses the efficiency entry being a ONE-DECODE entry, and we already have one: `trioseat_v1` at
0.938 is exactly that shape, so the next efficiency move costs no slot — it is waiting for the
efficiency snapshot to catch up (our row is stale as of 2026-08-27 22:11, before `trioseat` and
`onepair` landed).

It does NOT license dropping decodes from the accuracy champion. `raptorseat_v1` seats six members
across four reads including the foreign raptor's own, and none of it is priceable offline — every member
is `_cf`. −0.0014 offline is charged at about 3× when negative (knee-c0u9), so ~−0.004 on the board is
the honest worst case for the trade, and the accuracy track has no runtime term to pay it back.

It also does not say diversity is worthless — it says the diversity we can currently BUY with a second
decode is worth less than the noise floor. That is the third null of the day pointing the same way (the
free-seat pool, the decode-distance pool, and now the budget curve), and the confound named in
[2026-08-29_seats-per-decode.md](2026-08-29_seats-per-decode.md) still stands: every distant member we
own is also a weak one. `crop80_tissue_full` is the arm testing exactly that.

## The one thing the walk itself flags

Every budget stopped at SEVEN seats, not at the ceiling. The walk is score-limited, not budget-limited —
an eighth member did not raise the mean at any budget. Whatever is capping this ensemble, it is not the
decode count and it is not the seat count.
