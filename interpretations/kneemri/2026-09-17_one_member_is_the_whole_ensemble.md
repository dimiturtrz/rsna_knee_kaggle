# One member reads 0.944 against a seven-member 0.946: the ensemble premium is confirmed ON THE BOARD

*2026-09-17. Board row `56296698`, submitted 06:09Z, kernel `kneemri-ffsolo`, labelled INSTRUMENT at
submission time. It cost one slot and it settles a question that had only ever been answered offline.*

## The reading

```
ref        date        board  members  entry
56296698   2026-09-17  0.944      1    labelwise_depth24_drop40_long_ff
56232529   2026-09-15  0.946      7    tripleseat_v2  (champion, comp/champion.json)
```

A single member is **0.002** behind the seven-member ensemble this project has been building toward for
weeks.

The board reports at 0.001 — all 43 distinct scores in `comp/board.jsonl` are exact multiples of it — so
0.002 is two ticks of the competition's own resolution. Detrended on `ledger75` the top band (offline ≥
0.8700) has **residual sd 0.0013**, so the gap is **1.5 sd**. It is not a measurement that six additional
members bought anything.

## Why this is a board reading and not an offline one

Every prior estimate of the ensemble premium was offline, and every offline estimate of an ensemble is
made on the same 909 studies its seats were chosen against. The entry's `offline75` is null and must stay
null: `entry_offline --ruler ledger75 --write` refused it in writing —

```
ffsolo_v1 DECLINED in-sample: fitted at --fold -1, has seen every holdout study
(labelwise_depth24_drop40_long_ff)
```

The `_ff` suffix is a full-corpus fit. It holds nothing out, so no offline number about it is admissible
and the board score is its only honest read. That is the point of the instrument: the one ruler that has
never seen any of our training data says the same thing the offline estimates said.

## What it costs the plan

The ensemble premium was already priced at **≤0.002 offline**, and the gap to first place is **0.011**
board — **8.5 sd** on the same ruler. So:

- Reseating, member swaps, seat-count sweeps and cross-family diversity all live inside a ±0.002 budget
  against an 0.011 deficit. None of them can reach first place, whatever they measure offline.
- knee-bzhpv (cross-fold decorrelation, ~20 h, P ~0.15) buys ENSEMBLE diversity. This page prices what
  ensemble diversity is worth on the board and the answer is two ticks. It stays unarmed and now has a
  measured reason rather than a ranking.
- The corollary is the useful half: **the solo member is the deliverable.** Whatever moves
  `labelwise_depth24_drop40_long` moves the board almost one-for-one, without seven copies of the cost.
  A lever tested on one member no longer needs an ensemble rebuild to be read.

This is also the second reason the slot was worth spending. The first was that the board is the only
ruler `_ff` members have; the second is that it converts a standing offline assumption into a board fact
before more wall-clock is spent on seats.

## What it does not say

It does not say ensembling is worthless — 0.002 is a real if tiny gain, and the champion should keep it.
It says the premium is too small to be a PATH, not that it is zero. And it is one entry: a single board
row at 1.5 sd cannot separate "+0.002" from "0.000", only bound the premium below the scatter.

## Denominators

Board: the competition's own held-out set, reported at 0.001 resolution. Top-band residual sd 0.0013 is
`board_check --live --ruler ledger75`, offline ≥ 0.8700, n=15, detrended against slope +1.09. Gap to
first: leader 0.957 against our 0.946, read live 2026-09-17T12:39:10Z, rank 65 of 3885.
