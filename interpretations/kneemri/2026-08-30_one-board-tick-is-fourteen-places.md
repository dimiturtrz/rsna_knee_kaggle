# One board tick is fourteen places, and top-ten is +0.003 rather than +0.008

**2026-08-30 18:58 UTC · read off the live leaderboard, no card time, no slot**

## The measurement

`board_read --live` places us at **rank 29 of 2714, 0.944, tied with five**. The leader is 0.952 and the
gap has been quoted as 0.008 ever since. That framing is what this file corrects: the gap to the *leader*
is not the quantity that governs what to run next, because the field between us and them is dense.

Counted off the leaderboard's own top page:

| score | teams at it | best rank at that score | ticks from us |
|---|---|---|---|
| 0.952 | 1 | 1 | +8 |
| 0.950 | 2 | 2 | +6 |
| 0.948 | 2 | 4 | +4 |
| 0.947 | 5 | **6** | +3 |
| 0.946 | 4 | **11** | +2 |
| 0.945 | 6 | **15** | +1 |
| 0.944 | ≥6 | ~21 (we sit 29) | — |

Roughly **five teams per thousandth** through this band. So:

- **+0.001 → about rank 15.** Fourteen places, for one board tick.
- **+0.002 → about rank 11.**
- **+0.003 → about rank 6.** Top ten, and inside the score prize.

## What that does to the pricing rule

The house bar is ~1.5% absolute, and this project has correctly refused to call anything smaller a result.
That bar is about **what may be claimed**, and it does not move. What moves is the **objective**: at this
density the competition's own scoring is a step function on the third decimal, and the distance to a
top-ten finish is **three ticks**, not the 0.008 the gap line has been reporting, and nowhere near the
0.054 frontier figure that priced earlier plans.

The consequence is the corollary CLAUDE.md already states and this project has not been spending against:
*increments below the bar still compose, and a chain of them that sums past it IS a result.* Two of this
project's honest directional readings — the kind written up as "+0.002, under the floor, not a finding" —
compose to a top-ten position if they add. Whether they add is the open question and it is the one thing
nobody has tested (`knee-orjq`).

So the spend rule for the rest of this competition:

- A change with a **named mechanism** and a directional reading is now worth a slot even at +0.001, because
  the rank payoff is real where the score payoff is not. It must still be written up as directional.
- A change with **no mechanism** is still worth nothing, at any size. The density does not make noise into
  signal; it makes a small *real* gain worth more than it was.
- **Report rank and score separately, and never launder one into the other.** "+0.003 takes us to rank 6"
  is true. "+0.003 is a result" is false. Both sentences belong in the same paragraph.

## What this does NOT say

It is one read, at one instant, of a board that moves. The density will change — the 0.944 shelf we are on
is itself six teams deep, so the same arithmetic works against us: a competitor's single tick costs us
places without our score changing at all. `comp/leaderboard.json` carries the UTC instant of the read and
`champion_check` refuses one over two days old, for exactly this reason. **Do not quote the table above from
this file.** Re-read it: `python -m kneemri.kaggle.board_read --live`. The gap constant in `champion.json`
decayed silently twice — 0.008, then 0.015 — because it was quoted from prose, and this table is a much
more attractive thing to quote than a single number was.

Nor does it name a lever. It says what a lever would have to be worth, and the honest state is that the
cheap serving-side ones are spent: test-time views were tried (`kneemri-views`, 2026-08-17, blank public
score — three forwards did not fit the kernel ceiling), overlap TTA prices at 0.0000 for whole-stack
members (`knee-eeo`), the all-data dose is banked and sub-linear, the foreign seat is flat at two weights
and two counts, and the eighth seat is inert (`knee-uco3`, now with a third support in `vjepaseat_v1`).
Finding the three ticks is still the whole problem. What changed is only that it is **three**.
