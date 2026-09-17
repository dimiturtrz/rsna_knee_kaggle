# The full-corpus refit term is a per-member measurement, not a constant

2026-09-17. Beads knee-22nk5, knee-e9f83. No card spent: two board slots and an OLS.

## What a "refit term" is, and why every plan carried one

A member is fitted at `--fold 0`, holding out 909 studies, so it can be priced honestly on a ledger75
row. Shipping it means refitting the same recipe at `--fold -1` on the whole corpus — the same model
with a fifth more data, and no honest row of its own. The **refit term** is the board delta between
those two, and it is the coefficient at the end of every plan that reads "raise offline75, refit, ship".

## The three readings, and how they were reconciled wrongly

| pair | axis | term |
|---|---|---|
| `foldfour_v1` 0.936 / `fitfour_v1` 0.937 | four seats, matched | +0.001 |
| `tripleseat_v2` / `fullseat_v1` / `fullseat2_v1` — 0/1/2 full-fit seats | ensemble dose | 0.946 / 0.946 / 0.946, **flat** |
| `l75solo_v1` 0.941 / `ffsolo_v1` 0.944 (`labelwise_depth24_drop40_long`) | **solo, matched** | +0.003 |

Only the third is on the axis a solo plan multiplies. `extffsolo_v1` (0.945) then appeared to confirm
it at n=2 and licensed carrying **~+0.004** — but that reading compared a shipped full-fit member
against its honest twin's **PREDICTED** board, because the twin had never shipped. A term measured
against a prediction is a term measured against the ruler, not against the board.

## The measurement that settles it

`extsolo_v1` (ref 56315030) is that twin, shipped: `labelwise_ext_drop40_long_s2026`, board **0.937**.

    term(labelwise_ext_drop40_long)      = 0.945 - 0.937 = +0.008
    term(labelwise_depth24_drop40_long)  = 0.944 - 0.941 = +0.003

**2.7x apart on two members**, beside a dose series that reads flat on a seat of the first of them.
The +0.004 carry is retracted. A refit is a measurement to take per member, and no plan may multiply
by it — which removes the coefficient from the whole family of "refit and collect" plans, including
the one that priced the head_lr arms.

## What it does NOT rule out

That refitting helps: both measured terms are positive, and the larger is +0.008, worth more than the
entire ensemble premium (+0.001 to +0.002). Refitting a member before shipping stays right. What is
dead is **predicting how much** it will pay before the slot is spent.

## The confound, stated because it is not resolved

Both facts landed on the same read. `extsolo_v1` is also the solo transfer line's first miss at its
top (predicted 0.9403, read 0.937, −2.1 sd — see
[the transfer-fit interpretation](2026-09-17_the_solo_transfer_fit_predicts_the_board_to_the_thousandth.md)).
The two are the same 0.004, seen from two directions: if that member simply transfers 0.004 worse than
its offline75 says, then its refit term is inflated by exactly that amount and the term may still be
~+0.004 after all. **Nothing here separates them**, and the entry that can is a third point at the top
of the line — `kneemri-headlrsolo` (`headlr_ship_s2026`, offline75 0.9059). Read this file beside that
entry's result, not before it.
