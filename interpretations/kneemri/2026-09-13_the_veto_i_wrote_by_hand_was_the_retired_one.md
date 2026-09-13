# The last-epoch veto I wrote by hand was the version the gate retired

2026-09-13. Subject: `labelwise_ext_drop40_long_s2026` (bead knee-cgt9m), and the reading rule that
nearly threw its number away. Bead knee-a21h9, closed the hour it was opened.

## What happened

`labelwise_ext_drop40_long` is `labelwise_extended`'s line with two flags moved — `--slot-dropout 0.40`
and `--epochs 12 -> 24`. It landed at **0.9045 ledger75 / 0.8672 midpoint on the 909 honest rows**, the
best honest solo the project owns. Against its control `labelwise_extended_s2026` at 0.8986 that is
**+0.0059**, which is its pre-registered CLEAR band.

Its argmax is epoch 23. Epochs run 0..23, 24 distinct — so epoch 23 is the **last** epoch. The arm's own
header says:

> `peak` — if this arm peaks ON ITS LAST EPOCH the budget is still binding at 24 and NO verdict may be
> read. Record it and stop; do not quote the delta.

So I recorded the member and refused the verdict. That was the right call at the time: a pre-registration
is not waived because its clause is inconvenient. Then I filed the suspicion it raised, which was that
`$RECIPE` carries `--one-cycle`, the LR anneals to ~0 on the final epoch, and a healthy annealed run
*should* put its best weights there — making the veto mis-specified for the whole family.

## The suspicion was wrong, and the repo had already measured it

`convergence_check.py:104-109` records the base rate, taken over the corpus on 2026-08-24:

- under `one_cycle` a **late** peak is ordinary — 32% of finished twelve-epoch runs peak at e9 or e10,
  deep in the anneal, so "peaked late" says nothing;
- peaking on the **very last** epoch is rare — **3 of 53** finished twelve-epoch runs, against the ~14%
  a flat prior over the observed e5–e11 range would give. It is rarer than chance, and all three carry a
  recipe defect confirmed from somewhere else.

So the anneal does not drag the argmax to the end. I filed a bead against a mechanism the project had
already priced, which is CLAUDE.md's *"before filing a finding, READ THE LEDGER AND THE
INTERPRETATIONS"* for the third time in this project's life.

## The conclusion changes anyway, for a different reason

The project's veto is **not argmax-alone, and has not been since 2026-08-30.**
`Convergence.UNDERTRAINED` requires two things: the argmax is last **and** the gain over the final
`TAIL = 3` epochs clears `RISING_MARGIN = 0.0062`, the 909-study paired holdout floor. The docstring
records why the margin was added — the argmax alone reported eleven runs, **seven of them plateaus** that
had gained 0.0004–0.0020 over their last three epochs.

The pre-registrations in `labelwise_ext_drop40_long.sh` and `labelwise_depth24_drop40_long.sh` carry
only the first half. They are hand-written restatements that dropped the margin, and in dropping it they
re-introduced by prose exactly the version the gate had retired with an argument.

Measured, not argued:

```
labelwise_ext_drop40_long_s2026   e20 0.8666  e21 0.8667  e22 0.8669  e23 0.8672
                                  tail gain over TAIL=3: +0.0006, a tenth of the 0.0062 floor

convergence_check --run labelwise_ext_drop40_long_s2026.metrics.jsonl   ->  1 runs: 1 converged
convergence_check --run labelwise_extended_s2026.metrics.jsonl          ->  1 runs: 1 converged
```

A rise of +0.0006 over three epochs is a plateau that happens to end on its high point. The budget was
not binding. **`convergence_check` says CONVERGED, and that is the verdict that counts.**

## The rule, stated so it binds both ways

**A budget verdict is read from `convergence_check`, never from an argmax by hand.** Arm headers cite the
check instead of restating it; a restatement is a fork of a legislated rule, maintained by whoever wanted
the run.

This is not a waiver bought to rescue a number, and the test of that is that it binds where it hurts:
it applies unchanged to `labelwise_depth24_drop40_long` (knee-9bnkc, still on the card) whichever
direction that arm lands, and it applies to every future arm whose delta is bad. Neither running arm's
header was edited — a pre-registration is not rewritten while its run is in flight; the rule is recorded
here and applied at readout with the reasoning in the open.

## What the member is worth

**`labelwise_ext_drop40_long` clears its band**: +0.0059 ledger75 on a matched, converged control. The
operator pair — slot dropout at 0.40 plus a doubled budget — **transfers off the `max` pooling corner**,
where it was measured (`band_dense` e8 0.8877 -> e16 0.8918, +0.0041), onto labelwise pooling. That
re-prices the other three labelwise arms against the same operators and it is why the second base
(`labelwise_depth24`) was queued blind.

**And the member is still not a submission.** Against the core six on the full 909, LogitAverage, no
selection:

```
  solo 0.9045  labelwise_ext_drop40_long_s2026
  core six            0.9050
  swap ext->new (6)   0.9058   +0.0008
  add new (7)         0.9067   +0.0017
  rank-corr(new, core consensus) = 0.938
```

Both gains sit under the +0.0040 floor, and +0.0017 is exactly the size of optimal reseating
out-of-fold. At `board_transfer` +1.135 the seventh seat is +0.0019 board — one tick at the board's
0.001 resolution. A better solo that correlates at 0.938 with the consensus it joins does not move an
ensemble, which is the same thing `knee-0v8yz` has been saying: only member quality **at count 1** has
ever moved this board 0.005+.

The number that matters is the solo one, and it is now the project's best honest member.
