# The offline ruler saturates, and two levers close on the same day

2026-08-24. Two things were settled: the holdout stopped predicting the board at the top of its own
range, and a claim about the corpus that had sat unrun for a week turned out to be a measurement of our
own tooling.

## 1. Offline greedy selection is closed

`peak8` shipped the best subset the greedy holdout search has ever found — eight members, 0.8731, the
highest offline number this project owns — and scored **0.932**, in its pre-registered kill band.
`allfit`, seven members at 0.8712-equivalent quality, scored **0.937**.

| entry | offline | board |
|---|---|---|
| weakest-seven | 0.8562 | 0.908 |
| greedy7 | 0.8712 | 0.933 |
| **peak8** | **0.8731** | **0.932** |

The first two pairs give a slope near 1.67 and were the basis of the `board_transfer` +1.92 constant.
The third pair moves offline **up** by 0.0019 and the board **down** by 0.001. The transfer is real
where members are weak and **saturated** where they are not.

The operational consequence is narrow and firm: **a greedy holdout peak is evidence about the holdout.**
It no longer licenses a submission, and the transfer constant must not be extrapolated above ~0.933.
That retires the single most-used decision rule this project had.

### What it does not say

`peak8`'s eight seats are all fold-0. Its all-data dose is 0/8, against `allfit`'s 6/7 — the one axis
ever measured worth +0.003. **Member count and all-data dose are fully confounded in this entry**, and
the −0.005 is the dose gap's own size with the sign it already had. Reading `peak8` as "eight members
is worse than seven" is exactly the denominator mistake this repo keeps catching itself in: the rate
that matters is over the variable you moved, not over the one that moved with it.

Count-at-dark-dose has never been on the board and *cannot be built yet*. Only six dark checkpoints
exist, so every entry to date is six-dark-plus-one and the pool has never had a seventh to offer. That
is what `knee-rndp` is for.

## 2. The corpus is not not-English

`2026-08-17_the-corpus-is-not-english.md` claimed 33.1% of the corpus carries severity vocabulary the
extraction path "cannot see at all", and closed **unrun**. Sized as a P1 (`knee-2voi`), it was the
largest untouched lever on the label axis. It was refuted in twelve minutes, before any GPU, by asking
one question of the shipped label table: what is the severity spread of the three YES levels, **by
report language**, over all 4,407 studies?

```
en 6253  .401/.399/.199    tr 1963  .400/.412/.188    bg  687  .456/.338/.207
hr 2209  .435/.342/.223    el 1198  .369/.404/.227    nl  541  .370/.379/.251
es 2201  .331/.498/.170    de 1070  .481/.326/.193
```

Every language grades at essentially the English rate. `report_labels_v2.csv` is **already a
multilingual read**, and there is no language gap in it to close.

The 2026-08-17 measurement was of `report_language.SEVERITY` — *our own English regex* — and it read
that tool's blind spot as a property of the data. The doc's own Limits section names this failure mode.
The lesson is the denominator one again, stated the other way round: a rate computed over the subset a
tool could parse is a fact about the tool.

## What both have in common

Neither cost card time, and both killed a lever that had been priced as large. The audit that reads the
repo's own prior measurements before proposing work closed six further axes the same day — member
selection, per-label weighting, slice depth, slot coverage, pooling operator, encoder capacity — each
already refuted by a run this project had paid for and written down.

The surviving structural axis is count-at-dark-dose, and it needs a checkpoint before it needs a slot.
