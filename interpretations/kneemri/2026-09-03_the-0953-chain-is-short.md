# The 0.953 chain is short, and the refit term is where it broke

**Date:** 2026-09-03 · **Beads:** knee-q18e8 · **Status:** the plan, restated at honest numbers

## What this corrects

The path to the owner's 0.953 target was written with a **fixed +0.006** for the final step —
refitting the shipped seats on the full corpus (`--fold -1`) instead of fold-0. That term came from
one measured board pair:

| entry | seats | fit | board | offline |
|---|---|---|---|---|
| `twinfive_v1` | 5 | fold-0 | 0.935 | 0.8705 |
| `fiveship_v1` | 5 | full-corpus | **0.941** | in-sample, unquotable |

delta **+0.006**.

Treating that as a constant was my error. Kimi's Q-CZ reply supplies the outside view and it is
right: the refit gain is the vertical gap between two points on a **concave** learning curve
(error ∝ N^−α, α ≈ 0.3–0.5 — Hestness et al. 2017, arXiv:1712.00409; Mohr et al. 2024, learning
curves for decision making in supervised ML). A gap measured low on that curve does not transfer to
a point higher on it. It shrinks with:

- **pool quality** — less residual headroom to Bayes error above a better ensemble;
- **ensemble size** — averaging has already absorbed part of the variance that more data removes
  (Breiman 1996, *Bagging predictors*; Caruana et al. 2004, *Ensemble Selection from Libraries of
  Models*);
- **recipe convergence** — and part of the +0.006 is not data at all. The `_ff` checkpoints differ
  from their `_s2026` twins in stopping and annealing, and the seats were selected on the same
  holdout they were then refit against (Cawley & Talbot 2010, JMLR 11, on selection bias in
  performance evaluation).

**Planning number at an honest 0.945: +0.002 to +0.004.** Not +0.006.

## What that does to the chain

The chain as written:

```
fleet moves honest ensemble   0.8751 -> 0.885   offline
x transfer +0.97                              = +0.010 board -> 0.947
full-corpus refit                             = +0.006 board -> 0.953   TARGET
```

The same chain at the corrected term:

```
0.8751 -> 0.885 offline, x0.97                = +0.010 board -> 0.947
full-corpus refit, honest                     = +0.002..+0.004 -> 0.949..0.951   MISSES
```

So the plan no longer reaches the target on its own arithmetic. Two ways to close it, and they are
not alternatives — the second is the real one:

1. **The fleet carries more.** 0.949–0.951 becomes 0.953 only if the honest offline ensemble reaches
   **~0.888–0.890** rather than 0.885. That is +0.003–0.005 offline on top of what the fleet is
   projected to deliver, and nothing currently on disk supplies it.
2. **A lever outside the chain.** Every selection-side axis is closed by measurement (seating, seat
   count, vote rule, calibration, foreign dose, trunk architecture family, diversity losses, slice
   pooling, prediction-space transduction). The chain is entirely a *re-arrangement* argument, and
   re-arrangement has now been priced to its ceiling. The remaining +0.009 has to come from a
   mechanism not yet built.

## The instrument that tests the correction, and how it may be read

`foldfour` (fold-0) and `fitfour` (full-corpus) are the same four seats across the fold line —
`calib_dense_hold`, `calib_span`, `through_dense`, `small_batch6`. Four is not a design compromise
for its own sake: only four members have twins on both sides, and an all-fold-0 side is the only
side that carries a quotable offline number.

**The reading is asymmetric, and this is the part that must not be forgotten.** Four seats and a
weaker pool are exactly where the concave model says the refit term is *largest*. So the pair's
delta is an **upper bound** on the term at the champion's seven seats and better pool.

- **null or small (≤ +0.003)** — decisive. The last step of the 0.953 path is dead for the price of
  two instrument slots, and roughly thirty hours of card that would have gone to refitting the fleet
  is saved.
- **large (≥ +0.004)** — **uninformative**. It does *not* license budgeting +0.006 at 0.945. It is
  consistent with both a constant term and a term that decays to nothing by the time the pool is
  good, because the measurement was taken where both models predict a large gap.

An instrument whose interesting outcome is the null is still worth its slots when the null kills a
thirty-hour commitment. That is the whole case for these two, and it is why they ship as
instruments rather than as bids on the champion.

## What this does not say

It does not say full-corpus refitting is worthless — the measured pair is real and positive, and a
refit still costs nothing at ship time once the checkpoints exist. It says the *size* of the term is
a function of where you stand, and that it cannot be spent twice: once as the reason the ensemble is
good, and again as a bonus on top.
