# The metric is a plain mean, the refit term is two terms, and three columns hold ~0.02 of board

**Date:** 2026-09-05 · **Beads:** knee-wko44 (adjacent) · **Card spent:** none · **Slots:** two read, none spent

Two instrument slots landed together on 2026-09-04. Both read their pre-registered anchors exactly,
which is rare enough to say out loud, and between them they change what the next arm should be.

## `colnull` read exactly 0.500, so the ruler is finally known

`kneemri-colnull` submitted a constant 0.5 in every cell of every column. Three bands were registered
before the slot: **exactly 0.500** = a plain mean of per-column AUCs with no offset and no non-AUC
term · **0.495-0.505 but not 0.500** = a tie-handling or prevalence term · **anything else** = not a
mean of per-column AUCs at all.

It read **0.500**. `comp/README` has carried this as unmeasured since week one.

The consequence is arithmetic and it is the useful part: **a column improving its AUC by `d` moves the
board by `d/12`.** A per-label lane needs ~0.018 in its own column before the board's third decimal
can show it at all, and ~0.18 to be worth 1.5% of board by itself. Every per-label plan in the ledger
was priced on an assumed formula and is now priced on a measured one.

## `uncalib` read exactly the lower anchor, and the refit term splits in two

`kneemri-uncalib` is `fiveship`'s five seats with four of them swapped from the calibrated `_cf` build
to the uncalibrated `_ff` build, `merged_blend_cf` held fixed. The calibration flag is the only
variable across the pair.

| step | pair | delta |
|---|---|---:|
| pure refit, five seats | `twinfive` 0.935 → `uncalib` 0.937 | +0.002 |
| calibration, five seats | `uncalib` 0.937 → `fiveship` 0.941 | +0.004 |
| pure refit, four seats (independent pair) | `foldfour` 0.936 → `fitfour` 0.937 | +0.001 |

Two independent pairs put pure refit at or under +0.002 — under this project's 0.0062 noise floor on
the 909-study holdout. **Refitting the fleet for the refit is dead.** Roughly thirty hours of card that
the 0.953 chain had budgeted as a +0.006 constant buys +0.002.

What survives: refit *with* `--calibrated-targets` still carries the +0.006, because calibration is a
flag on the run that produces it and not a second run. So the plan does not change; the REASON does. A
seat is refit because calibration needs a refit to attach to, and a refit shipped without the flag
throws 0.004 away.

Caveat on the control: `twinfive` is a fold-0 pool and `uncalib` is fold-full, so the +0.002 also
carries seat-for-seat build differences. The four-seat pair is the cleaner of the two and reads
smaller, not larger.

## What the two together open: the per-label axis, priced for the first time

With the mean confirmed, the per-column map is worth reading. A rank-average of the seven strongest
fold-0 members this project owns, scored with `kneemri.evaluation.scores.LabelScores` over the
909-study holdout:

| finding | AUC | positives | prevalence |
|---|---:|---:|---:|
| MCL | **0.8104** | 171 | 0.188 |
| Fracture | **0.8255** | 68 | 0.075 |
| Lateral OA | **0.8309** | 233 | 0.256 |
| Effusion | 0.8663 | 558 | 0.614 |
| PF OA | 0.8698 | 409 | 0.450 |
| Synovitis | 0.8717 | 124 | 0.136 |
| Lateral Meniscus | 0.8737 | 226 | 0.249 |
| Contusion | 0.8851 | 267 | 0.294 |
| Medial OA | 0.8875 | 333 | 0.366 |
| ACL | 0.9055 | 248 | 0.273 |
| Medial Meniscus | 0.9108 | 482 | 0.530 |
| Baker's | 0.9208 | 217 | 0.239 |
| **mean** | **0.8715** | | |

The spread is 0.110 from worst column to best. Bringing the bottom three to 0.90 — the pool's own
upper-middle, not a frontier number — is +0.09, +0.075 and +0.070, which is 0.235 summed and
**+0.0196 of board**. That clears the 1.5% bar, and it is the first lane in some weeks whose
mechanism could plausibly deliver it. Every ensemble operation the project has run moves all twelve
columns together and has flattened seven times; this axis moves three of them and has never been
tried.

## What this does not say

The per-column map is scored against OUR supervision (`report_labels_v2.csv` binarised at its
midpoint), not against Kaggle truth. It gives the ORDERING of weak columns under our labels, which is
what an arm would be aimed at, and it does not give the board's own per-column AUC.

**That second number arrives on its own.** `kneemri-colmcl` (ref 56020054, still scoring) replaces the
MCL column with a constant after the blend. With the plain mean now confirmed, its drop from the
champion IS `(board AUC of the MCL column − 0.5) / 12`, which is the first per-column read on Kaggle
truth this project will ever have. If it lands near the pre-registered 0.918-0.928 it also confirms
the offline map's level, not just its ordering; if it lands elsewhere the map is ordering-only and any
arm aimed by it must say so.

No arm has been queued on this yet, and no mechanism for lifting a specific column has been chosen.
The lane is priced; it is not started. Nothing here is a score.
