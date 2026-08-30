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

The tempting consequence is CLAUDE.md's corollary — *increments below the bar still compose, and a chain of
them that sums past it IS a result* — and two of this project's honest "+0.002, under the floor" readings
would compose to a top-ten position **if they added**. They do not, and that is measured rather than feared.
`knee-orjq` asked exactly this and was closed on 2026-08-29 by seven arms stacked on the same control
(`calib_dense_hold` 0.8634): `--ema` −0.0003, labelwise pooling −0.0021, `--finding-calibration` −0.0039,
warm start −0.0072, `--contrast-weight` −0.0087, through-plane+attend+sided −0.0125, `--auc-margin` −0.0399.
Seven of seven on the same side of zero is p = 0.008 under a null-additive model, and the ordering is
monotone in how much the second term disturbs the recipe — the shape of a local optimum, not of seven
unrelated bad bets. **The best member is at a local optimum for recipe knobs, and stacking is closed.**

So the density does not license a chain of small knobs. It licenses a slot for a small gain that comes from
a mechanism the current optimum does not already contain.

So the spend rule for the rest of this competition:

- A change with a **named mechanism** and a directional reading is now worth a slot even at +0.001, because
  the rank payoff is real where the score payoff is not. It must still be written up as directional.
- A change with **no mechanism** is still worth nothing, at any size. The density does not make noise into
  signal; it makes a small *real* gain worth more than it was.
- **Report rank and score separately, and never launder one into the other.** "+0.003 takes us to rank 6"
  is true. "+0.003 is a result" is false. Both sentences belong in the same paragraph.

## Where the three ticks would have to come from

`knee-urvf` fits the transfer law from three dated board pairs: **board(ensemble) ≈ board(best single) +
0.015**, with the second term saturated across fourteen days while the pool grew from one member to seven
and roughly thirty seating/vote/count/TTA entries were spent on it. Member gains transfer to the ensemble
at roughly one-for-one. (Three points, no intervals — an order of magnitude, enough to rank axes.)

Cross that with the density table and the target becomes a single sentence: **+0.003 on the board means
+0.003 on the best single member.** Not on the seating, not on the vote rule, not on the seat count.

Which closes the exploitation question rather than opening it. Every route that acts on the combination is
spent or saturated — the bonus term has no trend (`knee-urvf`), the ensemble axis fits inside 0.005 across
sixteen entries (`knee-va22`), seat permutations span 0.002 across nine entries (`knee-xuj5`), the eighth
seat is inert with three supports (`knee-uco3`), views blew the kernel ceiling and overlap TTA prices at
0.0000. And the route that acts on the best member's recipe is a local optimum, 7/7 (`knee-orjq`).

So there is no cheap +0.003. The density finding does not produce a lever; it re-prices the levers that
already needed a new mechanism, and it says a **smaller** new mechanism now suffices than the plans written
against the 0.008 gap assumed. Concretely, the three surviving routes and what the density does to each:

| route | what it needs | priced against 0.008 | priced against 0.003 |
|---|---|---|---|
| non-DINO trunk seated (`knee-23wa`, `knee-8009`) | +0.017–0.020 solo to buy a seat | out of reach | **unchanged** — the seat bar is set by the pool, not the board |
| a better DINOv2 member | +0.003 solo, a mechanism outside the current optimum | needed +0.008 | **needed +0.003** |
| external data / new pretraining | a labels-free stage | the named route | the named route |

### The obvious objection, tested the same hour: the pool grew and it changed nothing

Every seat scan this project ever ran was made against a pool that was missing members. `roster_check`'s
2026-08-29 sweep found **76 scoreable checkpoints in neither the roster nor anywhere else**, including
`stackfit_s2026` at 0.8631 — the project's second-best solo — and thirty of them joined the roster, which now
holds 93. So the honest first move on "the seating is saturated" is to check whether it was saturated or
merely under-supplied.

It costs nothing to ask: eleven of the twelve strongest new members already have cached holdout predictions,
so `seat_value --against greedyseat_v1` is a CPU scan — no card, no slot. Result, over `greedyseat_v1`'s
seven seats:

| candidate | solo | seat value | nearest-neighbour |
|---|---|---|---|
| `attend_dense_s2026` | 0.8569 | **−0.0002** | 0.8814 (most distant) |
| `raptor_distill_s2026` | 0.8613 | −0.0003 | 0.9134 |
| `stackfit_s2026` | 0.8631 | −0.0004 | **0.9384** (nearest twin) |
| `band_density_s2026` | 0.8540 | −0.0009 | 0.9107 |

All eleven negative. **Zero clear +0.0015**, the threshold under which this ruler has never been shown to
pick a board tick, and no candidate's interval excludes zero. The best possible swap is −0.0002, which the
board charges at roughly 3x (`knee-c0u9`) — about −0.0005, so not even a free look.

The distance column is what makes it a finding rather than eleven bad draws. `stackfit` is the strongest
solo of the eleven and the **most redundant** member of the lot at 0.9384 nearest-neighbour; the most
distant candidate, `attend_dense` at 0.8814, is still nowhere near the 0.8346 `swin_supervised` reaches and
still prices negative. Thirty new members, and every one of them is the same lens — which is exactly what the
monoculture principle in `champion.json` predicts, now measured on the enlarged pool instead of the old one.

So "the pool was under-supplied" is refused. The saturation is real, and it survives a 5x larger candidate
set than the one it was originally read off.

The middle row is the one the density moves, and it is the only row it moves. It was previously an
+0.008 problem with no candidate; it is now a +0.003 problem, which is inside the range a genuinely new
supervision or decode-path mechanism has produced before (`vote.py` records decode-path diversity at
0.0059 offline / 0.014 on the board). That is where an exploitation slot should go, and it still needs a
mechanism named before it is armed.

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
