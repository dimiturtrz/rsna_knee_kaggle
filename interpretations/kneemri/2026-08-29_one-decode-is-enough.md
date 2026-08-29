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
