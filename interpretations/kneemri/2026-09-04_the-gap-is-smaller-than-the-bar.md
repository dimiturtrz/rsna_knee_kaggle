# The gap to first place is smaller than the bar we have been pricing arms against

**Read taken 2026-09-04T03:22:52Z** (`comp/leaderboard.json`, written by `board_read --live`; re-read it
rather than quoting these numbers, which decay every day slots are spent).

```
ours      0.944   rank 35 of 3020, tied with 9
frontier  0.9502  (top-5 mean)
leader    0.952
gap       0.008
deadline  2026-10-22 — 48 days
```

## The arithmetic that matters

The project's standing significance bar is **~1.5% absolute**, and the loop policy has for days named the
trunk-family arm as *"the only open lever that could plausibly deliver 1.5%"*. That framing asks every arm
to be a breakthrough.

**The gap to the leader is 0.008 — a little over half that bar.** So the thing we need is strictly smaller
than the thing we have been requiring of each experiment. `CLAUDE.md` already legislates the resolution:

> Track the running total against the ORIGINAL baseline — increments below the bar still compose, and a
> chain of them that sums past it IS a result.

At a 0.008 target, **four directional +0.002s win the competition** and no single one of them would ever
have been called a finding. That is not a licence to chase noise: each increment still needs a mechanism
and still has to survive the 0.0062 report floor before it is quoted. It is a change in what counts as
worth *running*. An arm whose honest pre-registration predicts +0.003 was, under the old framing, a
near-zero-expected-score run to be justified only as an instrument. Under the real gap it is a quarter of
the win.

## The rates say we are not behind

Over 64 recorded reads spanning 8.2 days:

| series | slope |
|---|---|
| ours | **+0.00050/day** |
| frontier (top 5) | +0.00020/day |
| closing | +0.00030/day |

We are improving at **2.5x the frontier's rate**. At the closing rate the 0.008 gap shuts in **26.7 days**
— about 2026-10-01, three weeks inside the deadline. At our own rate against a static frontier it is 16
days.

## What I do not trust about that, stated before it is used

1. **Our series is a step function, not a slope.** Board score moves only when a slot is spent and the
   entry happens to beat the incumbent. A per-day rate fit to it is a convenience, and it embeds the
   submission cadence of the last 8.2 days. If that cadence changes, the slope is not transportable.
2. **The frontier's +0.0002/day almost certainly accelerates.** Leaderboards compress toward the deadline
   as people spend their remaining slots and merge teams. Extrapolating a mid-competition frontier slope
   48 days forward is the optimistic branch, not the expected one.
3. **The plateau is crowded and rank is the headline.** We are tied with 9 others at 0.944, and
   `leaderboard.json`'s own preamble says so: *"a gap across a crowded plateau is not one step."* Moving
   0.944 to 0.945 may buy many rank places or none; moving to 0.952 has to pass everyone on that shelf.
4. **8.2 days, 64 reads.** Short baseline for a two-slope comparison.

None of that reverses the conclusion, because the conclusion does not depend on the extrapolation. It
depends on one comparison: **0.008 < 0.015**. The rates only say the situation is not desperate enough to
justify a gamble.

## What changes

- **Stop gating arms on a 1.5% mechanism.** Price them against 0.008 and against the days remaining. An
  arm with a sound mechanism and a +0.003 pre-registration is now fundable on its own terms.
- **Keep the running total against 0.944**, not against each arm's own control. The question an arm has to
  answer is what it adds to the chain, not whether it is individually impressive.
- **The closed axes stay closed.** Seating, calibration, vote rule, resolution and the rest were closed on
  measurements, and a smaller target does not reopen a refuted mechanism — it only lowers the bar for
  mechanisms that are still live. Reopening on the strength of a smaller gap would be the "hacking around
  a failure" this project forbids.
- **Do not spend the remaining 48 days waiting for one big arm.** The trunk-family lane
  (`v3large_stack`, then `radiodino_corpus`) is still the largest single open lever and it stays on the
  card. But it is now one contributor to a chain rather than the thing everything else waits on.

## The one number to re-read

`rank`, not `gap`. Re-run `board_read --live` before planning against any of this; `champion_check`
refuses a read older than two days for exactly this reason.
