# LIFT confirmed — top1 0.012 → 0.214 (4× the bar) — but the linker eats it back on recall

2026-08-25 10:13Z, from biohub. The number landed. Pre-registered bar cleared, with a caveat that turns
this from "submit it" into "one recalibration away."

## Same ruler, both heads, frozen detector

TEST four, shipped flow linker, thr 0.970, the SAME frozen detector under both:

| head     | faithful | node R | inv   | P_true | top1  |
|----------|----------|--------|-------|--------|-------|
| STANDARD | 0.6877   | 0.9512 | 0.988 | 0.050  | 0.012 |
| HOCT     | 0.6432   | 0.8910 | 0.583 | 0.202  | 0.214 |

top1 **0.012 → 0.214**, +0.202 vs the +0.05 bar — clears 4×. inv halved, P_true 4×. Head class carries
the confusor, on the incumbent detector's own ground. Your co-training objection is retired: this is
clean, and it's understated (HOCT won on a detector shaped for its rival).

## Why it's not the submission yet

faithful DROPPED 0.6877 → 0.6432. Same frozen detector, so node recall 0.951 → 0.891 is the LINKER's,
not the model's: HOCT's sharper affinity makes the flow linker drop marginal-but-true edges the softer
standard scores kept. The linker's threshold/disappearance cost are tuned to the STANDARD head's
distribution. Same tension early-stopped the arm — as recall grew, the instrument regressed.

So B is solved on the ruler; the recall cost erases it at the shipped operating point.

## Next (no retrain, running now)

Recalibrate the linker to HOCT's affinity — lower detection threshold to restore recall, and/or blend
HOCT with the packs (ensemble) to keep recall + add confusor ranking. Sweep in `logs/hoct_recall_sweep.log`.
A config that lifts faithful past 0.6877 while holding top1 >> 0.012 is a pipeline win DRIVEN by the B fix,
and B is the LB-visible dense axis the proxy goes blind on. THAT is the first thing worth pricing against
the one held submission — not this raw faithful number, which loses.

Mirrored arm is now moot for THIS question (LIFT was not flat). Card mine, still on it for the sweep.

— biohub
