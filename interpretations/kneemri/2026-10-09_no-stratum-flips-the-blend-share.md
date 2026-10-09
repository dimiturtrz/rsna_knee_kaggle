# No acquisition stratum flips the blend share; the per-finding profile is the only structure

**Date:** 2026-10-09 · **Bead:** knee-dbj03 · **Card spent:** none (CPU) · **Slots:** none

The question: is there a per-study gate — scanner, field strength, protocol, series count, model
disagreement — under which the public raptor family should take a larger share of the blend than ours?
If yes, a reliability-gated share beats `triseat`'s flat 1/3.

## Setup

- **Ours:** three honest fold-0 members (`labelwise_{depth24,ext,dense}_drop40_long_s2026`), rank-averaged.
  The four `_cf`/`_ff` seats of the shipped seven are trained on all studies and have no honest holdout.
- **Raptor proxy:** our fold-0 replicas of the raptor recipe (`raptor_ft_repl2x2_{a,c}_swa`,
  `raptor_ft_effv2l_f0_swa`), rank-averaged. They are weaker than the real nartaa CoAtNet, which
  trained on every study and so cannot be read here.
- **Ruler:** `ledger75`, 896 studies, the same truth arrays in all six files (max |diff| 0.0).
- Stratum metadata: `metadata/canonical_metadata.csv` (scanner, field strength, sequence) and
  `train_series.csv` (series count, fat-sat and fluid-sensitive planes). 200-draw study bootstrap.

## Result

Whole holdout: ours 0.9098 · replicas 0.8840 · 1/3 blend 0.8990 · best share 0.9 → 0.9104.

| stratum | n | ours − replicas | 95% |
|---|---:|---:|---|
| Philips | 276 | +0.039 | [+0.027, +0.051] |
| Siemens | 403 | +0.022 | [+0.012, +0.033] |
| GE | 157 | +0.021 | [+0.009, +0.034] |
| Canon | 54 | +0.006 | [−0.019, +0.031] |
| field ≠ 1.5/3 T | 61 | +0.005 | [−0.013, +0.025] |
| 4 / 5 / 6 series | 148 / 473 / 274 | +0.028 / +0.029 / +0.022 | all exclude 0 |
| disagreement terciles | 300 each | ours ahead in all three | best share 0.8–1.0 |

**No stratum holds a reversal.** Ours leads in every stratum large enough to read, and the two
strata where the lead shrinks to near zero (Canon, odd field strengths) total 115 studies with
intervals spanning ±0.02. A per-study gate has nothing to switch on.

**The structure is per FINDING, and it repeats the 2026-09 instrument.** Replicas lead only on Fracture
(0.855 vs 0.837, 39 positives). Effusion, Baker's, Contusion and Lateral OA are ties, and blending helps
there (best share 0.5–0.6). ACL, MCL, Lateral Meniscus, PF OA and Synovitis go to ours outright.

## What this does NOT license

- **The absolute share.** The holdout puts the optimum at 0.9 for ours. The board reads 1/3 and 1/2
  both at 0.952. The holdout truth comes from our own label pipeline, so it is biased toward the
  members trained on it (`2026-09-05_the-holdout-is-the-distorted-ruler.md`). The replicas are also
  weaker than real nartaa. Only the board sets the level.
- **A per-finding tilt as a finding.** Per-finding swaps have transferred their SIGN and nothing else
  (memory: member-swap transfers). A tilted share (ours up on ACL/MCL/LMen/PFOA/Synovitis, down on
  Fracture) is a slot-sized instrument. It is not a predicted gain. Pricing: ~0 h of card, one
  slot, plausible board move 0.000–0.002, P(≥0.953) ~0.2.

The reliability-gating lever is closed for the acquisition axes measured here. It is re-opened only
by a stratum the board can see and the holdout cannot, such as a test-only vendor mix.
