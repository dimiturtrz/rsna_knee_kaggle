# Slice coverage is closed, and it is closed on a TRAJECTORY rather than on a score — 2026-09-18T08:05Z

## Why this file exists

The reading lived in `comp/members.declines` and in bead knee-jb0ci and nowhere a reader would look.
`2026-09-18_state_of_the_levers.md` prices every axis that could still carry 1.5%; coverage was on it as
open, and it is not. This is the row that closes it, written where the other closures live.

Card cost of this write-up: zero. The arm it reads was already on disk.

## The claim

Raising slice coverage from 24 to 36 does not pay at the shipping corner. `slicecover36` sits **behind
its controls at every one of its seven matched epochs, mean −0.0224**, which is **3.6× the 0.0062 paired
floor**, and it costs **50% more wall-clock per epoch** (331 s against 222 s) to get there.

## Why the trajectory is the right ruler here, and the final score is not

The obvious reading of a short arm is its best score, and that reading is unavailable: the run was
truncated, so `convergence_check` correctly calls its score a FLOOR and no verdict may be taken from it.
A truncated number is the exact shape of evidence this repo has three times mistaken for a fact about an
architecture when it was a fact about a killed process.

The trajectory is not exposed to that. **Truncation moves a FINAL score; it cannot move a whole matched
trajectory.** Epoch k of the 36-slice arm against epoch k of a 24-slice control is a paired comparison
that exists whether or not either run was allowed to finish, and the sign of that comparison was
constant across all seven pairs. A method that is going to win does not lose every matched epoch first.

The controls are two independent 24-slice members agreeing with each other to within 0.003, so the
−0.0224 is not a fact about one control's scatter.

## What this does NOT say

It does not say depth is worthless — it says that at THIS corner (`--slices 24`, 336 px, whole-stack,
labelwise), buying depth costs more than it returns. The mechanism that would re-open it is a corner
where the extra slices carry information the 24 do not: a different band, a different spacing, or a
finding whose primary sequence is being sampled past. None of those were tested here and none are
refuted by this.

## The ratio that killed the re-run

A converged re-run is ~2.2 h of card for P(clears the 0.0062 floor) ≈ 0.10, against a trajectory that
was negative at every epoch by 3.6× that floor. Bought nothing. Not run.

## Evidence

* `comp/members.declines` — the decline, written as an argument rather than a number.
* bead `knee-jb0ci` — the matched-epoch table.
* `experiments/arms/warm_calib_dense.sh` — the control's line.
* commit `be676b49`.
