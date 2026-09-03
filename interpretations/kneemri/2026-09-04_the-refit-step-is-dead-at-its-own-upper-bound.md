# The full-corpus refit step is dead, measured at the seating where it should have been largest

**Date:** 2026-09-04 · **Beads:** knee-q18e8 · **Status:** measured, decisive branch fired

## The measurement

Two submissions, identical seating, one variable — the fold line:

```
foldfour_v1  ref 55994671  4 seats at --fold 0    offline 0.8711   board 0.936
fitfour_v1   ref 55995214  4 seats at --fold -1   offline none      board 0.937
                                                            DELTA  +0.001
```

`calib_dense_hold_s2026`, `calib_span_s2026`, `through_dense_s2026`, `small_batch6_s2026` on both
sides. Four seats because only four members have twins across the fold line. No offline number is
possible for the full-corpus half and none is quoted: every seat there has seen all 909 holdout
studies, which is what the arm is for.

## The branch it fired

Pre-registered in `2026-09-03_the-0953-chain-is-short.md:75` and in both kernel docstrings, before
either slot: **≤ +0.003 is decisive and kills the last step of the 0.953 path.** It read +0.001,
inside board noise.

**The asymmetry is what makes one measurement enough.** Four seats and a weaker pool are exactly
where a concave learning curve (error ∝ N^−α) says the refit term is LARGEST. So +0.001 is an UPPER
BOUND on the term at the champion's seven seats and stronger pool, not an estimate of it. A term dead
at its own upper bound is deader than a term that merely read null where it was expected to.

The step had been carrying **+0.006**, budgeted from a single measured pair (`twinfive` 0.935 →
`fiveship` 0.941) and then treated as a constant. It is worth +0.001 at most.

## What that does to the chain

```
best fleet offline, re-arranged to its ceiling   0.885
x transfer +0.97                                 = +0.010 board -> 0.947
full-corpus refit, MEASURED at its upper bound   = +0.001 board -> 0.948    MISSES BY 0.005
```

The target needs board 0.953. At the +0.97 operating-point slope that is offline **≈ 0.891** —
+0.006 offline above where the whole re-arranged fleet already sits. Nothing in the seating,
selection, weighting, calibration or fold axes can supply it; every one of them is closed in writing.

**It must come from a better MEMBER.** That is now the only open account, and the only lever with a
mechanism sized to it is trunk pretraining domain/recipe — `ortho_stack` on the card, and the
knee-continued SSL lane behind it.

## What it buys

Roughly **thirty hours of card** that would have gone to refitting the fleet at `--fold -1`, spent
on an axis now known to be worth +0.001. It also removes the last reason to keep fold-0 and
full-corpus twins of every member, which is a standing tax on every future arm.

## What this does not say

It does not say full-corpus training is worthless — it says the BOARD does not pay for it at this
pool quality, which is a statement about the remaining headroom, not about the data. It does not
touch the +0.97 transfer constant (this row is `scope: excluded`; no offline number exists for it, so
it never enters `board_transfer`'s fit). And it says nothing about seating: the seating is held fixed
across both halves by construction, which is the whole design.
