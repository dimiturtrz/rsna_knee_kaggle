# Nine axes were closed on a ruler that cannot read them, and seven of them have no other ruler

2026-09-06. Subject: every axis this project recorded as CLOSED on a board FLAT, re-read against the
measured resolution of the board itself. Instrument: `kneemri/evaluation/board_floor.py` (knee-6j8b4).
Verdict rule pre-registered by mastermind at 18:48Z, before any of the numbers below were computed.

## The resolution the closures were read at

The public leaderboard scores roughly 30% of the 1,322 hidden studies — about four hundred. A paired
study bootstrap of a one-seat swap at that size reads a 95% half-width of **0.0011** on our holdout; at
our own 909 it is 0.0007. The board publishes three decimals, so a true 0.0005 effect is unreadable by
rounding whatever the sample size. **The board's resolution for a one-seat swap is therefore ~0.0011**,
and every closure below was written on a board difference of 0.000-0.002.

A board flat inside that width says "below the board's resolution". It does not say "inert". The two
readings license completely different work: the second closes an axis, the first only says that spending
slots on it cannot settle it.

Caveat carried from `board_floor`'s docstring: the width is measured on a ONE-SEAT swap, whose arms are
strongly correlated, so a larger disturbance is wider, not narrower. The rows are our 909, so this is our
case mix, and it says nothing about the LEVEL either ruler reads (knee-l4xx6 untouched).

## Verdict rule (pre-registered)

| offline delta on the 909 | with a board flat | verdict |
|---|---|---|
| >= 0.003 | board could not see it, offline could | **REOPENED** |
| 0.0011 - 0.003 | | directional, composes |
| < 0.0011 | neither ruler can see it | closed on both rulers |
| no offline number exists | the only ruler used cannot resolve it | **UNREAD** — never closed at all |

## The table

| axis | entry pair | board delta | offline delta on the 909 | verdict |
|---|---|---|---|---|
| seventh seat (severity) | `sevengrade_v1` 0.931 / `sixfresh_v1` 0.931 | 0.000 | +0.0002 | closed on both rulers |
| kit refresh | `sixfresh_v2` 0.931 / `sixfresh_v1` 0.931 | 0.000 | 0.0000 | closed on both rulers |
| count at high redundancy | `sixstrong_v1` 0.930 / `sixfresh_v1` 0.931 | −0.001 | +0.0005 | closed on both rulers |
| ordinal separation | `accumseat_v1` 0.933 / `gradeswap_v1` 0.933 | 0.000 | +0.0001 | closed on both rulers |
| count 6→7 (dark curve) | `dark7_v1` 0.937 / `dark6_v1` 0.937 | 0.000 | none — both arms in-sample | UNREAD (fold-0 analogue: +0.0005) |
| encoder family at seat 7 | `v3seat_v1` 0.936 / `dark6_v1` 0.937 | −0.001 | none — both arms in-sample | UNREAD (fold-0 analogue: −0.0004) |
| member type (regularised) | `augseat_v1` 0.936 / `lastseat_v1` 0.937 | −0.001 | none — both arms in-sample | UNREAD (fold-0 analogue: +0.0003) |
| the vote rule | `votefit_v1` 0.938 / `dark6_v1` 0.937 | +0.001 | none — both arms in-sample | UNREAD (fold-0 analogue: +0.0003) |
| serving views (jitter) | `jitterfit_v1` 0.937 / `votefit_v1` 0.938 | −0.001 | none — both arms in-sample, and no offline serving exists | UNREAD, and unreadable off-card |

`farseat_v1` is not a row: it ran six of seven named members and is void as a seven-seat reading.

## The finding the table is actually about

Four of the nine pairs carry offline numbers on the same scope. All four read **under 0.0011**, so those
axes are closed on both rulers — the board flat was right, for the wrong reason.

**The other five carry `offline: null` by construction.** They are `--fold -1` entries: every member has
seen every holdout study, so no honest offline delta exists for them and none can be recovered. Those
five axes — count 6→7, encoder family at seat seven, member type, the vote rule, serving views — were
closed on the ONLY ruler that was ever applied to them, and that ruler's resolution is 0.0011 while the
deltas in question are 0.001. **They were never closed. They were never read.**

Where a fold-0 analogue of the same swap exists on honest members it is given above: the same axis, a
different (six-seat, fold-0) base, so it is evidence about the axis and not about the entry. All four
computable analogues land under 0.0011 as well, which is a real reading and downgrades those axes from
UNREAD to closed-on-the-analogue. In full, against the six of `sixfresh_v1` (909 studies, paired
bootstrap, `logs/board_flat_axes.log`):

| fold-0 analogue | offline delta | 95% interval at n=400 |
|---|---|---|
| count 6→7 — six + `small_batch6_s2026` vs six | +0.0005 | [−0.0005, +0.0016] |
| encoder family — six + `v3base_dense_s2026` vs six | −0.0004 | [−0.0016, +0.0008] |
| severity seat — six + `grade_ladder_s2026` vs six | +0.0002 | [−0.0009, +0.0012] |
| member type — six + `small_batch6` vs six + `small_accum` | +0.0003 | [−0.0009, +0.0015] |
| the vote rule — `LogitAverage` 0.8690 vs `RankAverage` 0.8687 over the six | +0.0003 | — |

The severity row is the control: it reproduces the ledger's own `sevengrade_v1` − `sixfresh_v1` reading
of +0.0002 from the records rather than from the board, so the analogue construction is measuring the
swap it claims to. The vote-rule row does NOT contradict the champion twin's pre-registered +0.0034 at
6× — that is a different pool with a foreign seat in it, and a vote rule prices what the members
disagree about; a base of six same-family members has little for it to arbitrate.

## What this changes about tomorrow's slots

Mastermind's framing was that three reopened +0.003 axes stacked on the foreign seat would be a +0.009
candidate. **Nothing in this table clears +0.003.** Every axis that could be re-read offline reads
0.0000-0.0005 — an order of magnitude under the bar — and the axes that could not be re-read are the
seat-permutation family the ledger already refuses slots to (knee-xuj5). The stacking entry does not
exist, and a slot spent building one would be spent on a sum of four numbers that are each smaller than
the rounding of the ruler that would score it.

What the table does license: **stop spending slots to close seat axes in either direction.** A flat is
uninformative and a 0.001 win is equally uninformative; neither ruler can arbitrate a one-seat swap, so
the only member-side work worth a slot is one whose offline delta is already >= 0.003 before it ships.
That is the same conclusion the closed member axis reached, now with a measured reason rather than seven
flats.
