# Calibrated targets, re-read against the paired floor: the falsifier fires

**2026-08-16. No card. The arm ran on 2026-08-13; this is the same arm re-read on cached predictions.**

## What was blocked

knee-bot's arm was recorded as UNREAD, not refuted: "not one of the twelve deltas clears its own label's
95% sampling interval, so the per-label pattern is UNREAD". That was against the **unpaired** bound —
each label's sampling error from its positive count, which is far too conservative for two arms scored
on the same studies. The bead named the fix itself: "it costs a bootstrap over cached predictions."

Two things had to land before the bootstrap could run, and both did:

1. `LabelScores.paired_interval` / `RunComparison.paired` — this pair's own half-widths.
2. `HoldoutMetrics` (today, 83af24c) — because the control, `wsfold0_e6`, predates the metrics recorder
   and the comparison refused it as "does not exist" while its predictions sat beside the checkpoint.

## The reading

Control `wsfold0_e6`, mean **0.8457** — reconstructed from the kept predictions, and it reproduces the
ledger's recorded 0.8457 exactly, which is what says the right array was found.

```
PF OA            +0.0198*   Fracture         +0.0170    Medial Meniscus  -0.0136*
Contusion        +0.0116*   Lateral Meniscus +0.0111    Effusion         -0.0084
Medial OA        +0.0054    Synovitis        -0.0050    MCL              +0.0045
MEAN             +0.0038    ACL              +0.0023    Baker's          +0.0006
Lateral OA       +0.0005
```

`*` = clears this pair's own paired half-width (up to ±0.0241). **Three clear where zero cleared before.**

## What it says, and what it does not

**The mechanism prediction is contradicted, and by its own pre-registered falsifier.** knee-bot predicted
the gain would concentrate where the miscalibration is largest — MCL and Effusion, the labels most skewed
toward the badly-valued 0.68 — and that PF OA, already nearly calibrated, would barely move. Measured:
PF OA is the **largest** delta and one of the three that clears; MCL (+0.0045) and Effusion (−0.0084)
neither clear nor point the predicted way. The acceptance criterion said "a uniform shift would refute
the story even if the mean moves", and the structure-scale reading is exactly that: focal +0.0043,
regional +0.0032 — **UNIFORM LIFT**.

**The mean is unchanged at +0.0038, still inside the 0.0053 floor. Still directional, still not a finding.**

**Two limits, and they are real.**

- **Both runs' predictions are `re-derived`, not `trained`.** The record's own caveat applies: a
  re-derivation lands within a cross-precision tolerance that is *wider* than the half-widths being
  quoted. So the three stars are **INDICATIVE**, not confirmed. The uniform-lift verdict does not depend
  on the tight floor and survives this; the individual stars do not.
- **The pair cannot be certified matched.** `wsfold0_e6` recorded no recipe, so `unmatched()` can only
  say "recipe unrecorded, so nothing here can say the two runs are matched". knee-bot asserts "same
  recipe otherwise" in prose, and prose is what this project keeps getting burned by.

## The honest state of knee-bot

Unchanged where it matters and worse for the story than "unread": the calibration is free,
mechanism-sound in its **derivation** (four of five assigned values sit outside their measured
agreement intervals), worth +0.0038 directionally — and its predicted per-label signature is now
contradicted rather than merely unmeasured. Keep `--calibrated-targets` available. Do not plan around it.
Do not quote the per-label pattern in either direction: the three that clear are not the three the
mechanism named, on an indicative floor, over an uncertified pair.

## Side finding

`wsfold0_e6.holdout.npz` stamps `epoch 11` while its weights are the epoch-6 best (the mean proves it).
The back-fill wrote the run's length where the kept epoch belongs. Filed separately — it is cosmetic
today and a mis-join tomorrow.
