# The priceable pool has a ceiling, and it is 0.938 of board — 2026-09-05

Card-free, slot-free. Three findings, in the order they were forced.

## 1. `entry_offline` resolved holdout records against the wrong root

`EntryOffline` was constructed with `root.processed()` while every `HoldoutRecord` lives under
`root.models()`. The record check could therefore never succeed, so every entry surviving the in-sample
test was falsely declined `no-record`. Two entries were hidden that way — `distinct14` (14 members) and
`fitfour_v1` (4) — and all eighteen of their members had a `.holdout.npz` on disk the whole time.

Every other tool in `kneemri/kaggle/` resolves checkpoints with `root.models()`; `champion_uncertainty`
even names the local variable `processed` while assigning `models()`. The contrast is the tell.

Fixed. The two entries price at **0.8749** and **0.8711**, both above 0.8695. `board_transfer`'s top
regime went from 2 to **4** resolved pairs and its 0.8678 band from 17 to **24**, median +0.98
[+0.94, +1.08] — tight enough to price an arm with, which it was not this morning.

It also retires a claim `priceable_entries` states in its own docstring: *the board's ENTIRE TOP IS
UNPRICEABLE, and structurally so… the highest FULLY scoreable entry is `gradeswap_v1` at 0.933*.
`distinct14` is board **0.937** and now carries an offline number. The top is priceable. That docstring
needs correcting and the correction is not cosmetic — it is the anchor every prediction below uses.

## 2. A member scored against a different ruler wins every ranking — already known, guard missing

**The repo knew this and I did not read it first.** `comp/members.declines` line 84 states it in full:
*DO NOT QUOTE 0.8964 from its training log: the arm passed `--labels v3` with `--eval-labels` unset, so
its holdout truth was binarised against the RE-MAPPED values and the ruler moved with the arm;
`run_comparison` refuses the pair outright.* The member is declined as a seat and kept as an instrument.
What follows is therefore a re-derivation, and the only NEW part is the last paragraph: the ranking path
had no guard. Filing it as a discovery cost a P1 that was closed the same hour (knee-w4pyq).


`recal_holdannot_s2026` reads **0.8964** on the 909-study holdout, 0.032 clear of the best comparable
member. Its holdout truth differs from the corpus label table in **10,752 of 10,908 cells**. It was
fitted on a supervision table whose values were re-mapped from the 58 annotated studies, and it records
its holdout against THAT table. The number is not a better member. It is a different question.

This is the `raptor_solo` failure mode again — one contaminated cell carrying a whole reading — and it
is more dangerous, because nothing about the name says so and it sorts to the top of any AUC ranking of
the pool. `RecordMix.narrow` catches it only when it has company: it drops the minority ruler as soon as
a second member joins, which is why a forward selection seeded on it picks it first and then cannot add
a second seat. A single-member read never triggers that guard at all.

`priceable_blend.on_the_corpus_ruler` now takes the MAJORITY truth table and drops the rest, chosen by
majority rather than by name so a future re-mapped arm is caught the same way.

## 3. The ceiling, and what it says the schedule must do

Forward selection over the 195 priceable members, on the corpus ruler:

| | offline | board |
|---|---------|-------|
| best single member (`stack_increments_s2026`) | 0.8642 | — |
| greedy blend of 7 | **0.8763** | predicted 0.938 [0.938, 0.938] |
| `distinct14`, 14 members | 0.8749 | **0.937** measured |
| champion `monotone_v1`, `_ff` members | unpriceable | 0.944 |

Seven seats buy **+0.0121** over the best member; fourteen buy less. Ensembling over this pool is
saturated, and 0.8763 is a SELECTED number — optimistic by construction, so the true ceiling is lower.

At the top-regime compression (+0.82 [+0.71, +0.93] per offline point) the 0.953 target needs offline
about **0.891**. That is **+0.015 over the best blend this project can assemble from every member it
owns**. No selection, weighting, or seating operation closes a gap that size — those axes are closed on
the board seven times over, and this is the offline arithmetic saying the same thing independently.
It has to come from better SINGLE members, which means the card.

The gap to the champion is separately explained and is not a quality gap: converting members to
`--fold -1` moved `finalfit_v1` 0.934 to `allfit_v1` 0.937 and then saturated (knee-97ib). The priceable
pool sits ~0.007 under the shipped pool for that reason alone.

## What is pre-registered against the next slot

The compression above 0.8695 rests on four pairs and `board_transfer` calls it DIRECTION ONLY, yet every
arm decision now runs through it. The 7-member blend predicts **0.938** with a band of [0.938, 0.938]
that is tight only because the anchor is one point. Shipping it tests the ruler, not the pool:

- **0.936-0.940** — compression validated in the regime that matters. Arm pricing at ~1:1 is licensed,
  and an arm needing +0.015 of offline is confirmed as the only thing that reaches the target.
- **0.941-0.944** — the compression UNDER-predicts. The priceable pool is better than its offline says,
  the `_ff` dose effect is smaller than 0.007, and held-out members stop costing board position.
- **above 0.944** — the ruler inverts at the top. Every offline-based decision this project holds is
  suspect and the honest next move is to stop pricing arms offline at all.
- **below 0.936** — selection optimism exceeds the gain. An offline-selected blend does not transfer,
  and greedy search over the holdout is not a tool this project may use to choose an entry.

It is an INSTRUMENT. It is predicted to score BELOW the champion and cannot improve board position.
