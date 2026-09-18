# Four null distillation rows all held one variable fixed, and it is the one the mechanism needs — 2026-09-18T08:05Z

## Why this file exists

`2026-09-18_state_of_the_levers.md` prices distillation as closed: **teacher headroom 0.002, the term
itself +0.0056, under the 0.0062 floor.** That pricing is correct about what was run and wrong about what
was concluded from it. This file says what the four rows share, why that shared thing is not incidental,
and what was measured before spending a card on it.

Card cost of the measurement below: zero. It reads `HoldoutRecord` predictions already on disk.

## What the four null rows share

| row | teacher | result |
|---|---|---|
| `distill_pairs` | our own converged fold-0 twin | seat **+0.0016**, under the floor |
| `raptor_distill` | converged foreign member | right sign, under the floor |
| `foreign_distill` | converged foreign member | distance 0.0249 → 0.0190, **destroys it** |
| feature-distillation pair | converged trunk | flat — returns the trunk's solo score and nothing more |

Every one of them teaches from a **CONVERGED teacher emitting FIXED targets.** That is not a detail of
the setup, it is the whole of the mechanism under test: a fixed target is a function the student is
asked to imitate, and `distill_pairs` already diagnosed why imitation pays nothing here — *a rank
average cannot pay for an ordering it already holds.* Two DINOv2s share a trunk, so they share the
ordering, so there is nothing in the teacher's fixed reading that the student's own labels do not
already say.

**Deep mutual learning does not hold that variable.** Both models train simultaneously and each is taught
by the other's reading *as it currently stands*. Zhang et al. 2018's own claimed mechanism is the high
entropy of peer posteriors EARLY in training — a regulariser, not a knowledge transfer. Nothing in the
ledger tests a regulariser; all four rows test a transfer.

## The measurement that licensed the run

The condition for re-opening, written in the bead before any of this: **does the early-training peer
signal survive a shared trunk?** Measured against the supervised residual it would be added to — five top
members, 909 studies, 12 labels:

| pair | \|pa−pb\| | \|pa−truth\| | ratio | corr |
|---|---|---|---|---|
| `headlr_ship` vs `depth24_drop40_long` | 0.0499 | 0.1758 | **0.284** | 0.957 |
| `headlr_ship` vs `calib_span` | 0.0681 | 0.1758 | **0.387** | 0.921 |
| `depth24_drop40_long` vs `calib_span` | 0.0719 | 0.1730 | **0.415** | 0.916 |
| `clean_calib` vs `stackfit` | 0.0536 | 0.1889 | **0.284** | 0.961 |

**Peer disagreement is 28–42% of the supervised residual. The shared trunk does not collapse it.** The
mutual gradient is about a third the magnitude of the label gradient — not a rounding error.

And this is the **hardest** place to find disagreement: converged members, held-out rows, one family. The
early-training, training-split disagreement the mechanism actually feeds on is larger, not smaller.

## What this does NOT say

**Disagreement is NECESSARY, not SUFFICIENT.** `distill_pairs` (+0.0016) was our own fold-0 twins and
would show this same ratio; mutual training still has to extract it. This licenses a run. It does not
predict a gain, and quoting the table above as if it did would be the error this project has made with
`board_transfer` four times.

## The design consequence: one variable, not two

The bead's first design said *"attach at the output … standard DML is a KL between output posteriors."*
Audit contradicts it. `teacher_targets.py` states `distill_weight` is a pairwise **ordering** term and
argues explicitly against KL — *"Not the usual KL on softened probabilities, which targets calibration the
metric discards."* Since all four null rows used the ORDERING term at `--distill-weight 0.3`:

* **ordering term, same weight** → moves exactly ONE variable against `distill_pairs`: teacher liveness.
* **KL** → moves two, and could not be read against that row at all.

The arm takes the ordering form. It also needs no new plumbing.

## The cadence, and why `--epochs 8` could not buy it

Rounds need the cohort to stop at a known epoch and resume onto the *same* anneal. Lowering `--epochs`
cannot do that: `one_cycle` is built over `--epochs` (`run_setup.py:571`), so a shorter count
**compresses** the schedule into a different recipe — measured head to head, the compressed cycle leads at
epoch 0 (0.7510 against 0.7393) and trails at the end (0.8429 against 0.8457). `patience.py` already
legislates the rule; `--stop-after-epoch` (commit `ae71f68b`) is that mechanism with a deterministic
trigger, bounding the LOOP while the schedule stays built for the full count.

Verified end to end in a 60-study smoke before the card was spent: the resumed round logged
`distill_weight "0.3"` and the peer npz with `epochs "4"` unchanged, and `RunMetrics.begin()` rotated the
finished round aside as `_run1.pt` + `.metrics.1.jsonl` while `.partial`/`.resume` survived.

## The price, and the pre-registration

**~3.5 h of card. P(clears the 0.0062 paired floor) ≈ 0.30. P(clears 1.5%) ≈ 0.08.** Read as the delta of
`dml_a_s2026` against `headlr_ship_s2026` 0.9059 ledger75. A result under the floor is directional only
and opens nothing; a result at or below zero closes mutual supervision for this family at this weight and
does **not** re-open by re-quoting the disagreement table.

The deliverable is ONE checkpoint. The peer is scaffolding and is discarded like a teacher.

## Evidence

* bead `knee-z81wd` — the measurement, the price, the amendment.
* `experiments/arms/dml_cohort.sh` — commit `e3d008d8`.
* `kneemri/training/fitting.py`, `config_tree.py` — commit `ae71f68b`.
* `kneemri/inference/teacher_readings.py` — training-split orderings, refuses `--fold -1`.
