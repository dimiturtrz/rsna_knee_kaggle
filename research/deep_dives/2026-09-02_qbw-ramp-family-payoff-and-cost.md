# Q-BW — Ramp / schedule-shaping family: payoff and cost audit

Claude's question (2026-09-02): audit the ramp / schedule-shaping family with numbers from
`experiments/README.md` and holdout records on disk.

## Family definition

The ramp family here is the set of arms that apply a curriculum / sample-weight RAMP over the
first half of a 12-epoch one-cycle schedule:

- `certainty_ramp` (certainty_curriculum.sh) — label-certainty ramp, control `disagreed_uniform`
- `disagreed_ramp` (disagreed_ramp.sh) — member-disagreement ramp, controls `disagreed_full` and
  `disagreed_uniform`
- `ramp_dense_gate` (rampgate.sh) — transfer test to the champion's dense corner, control
  `calib_dense_hold`
- `rampsix` sub-members — refit five champion seats with the ramp:
  - `final_ws_ramp`
  - `through_dense_ramp`
  - `grade_ladder_ramp` (running)
  - `lr4e-5_ws_ramp` (not started / no metrics)
  - `calib_span_ramp` (not started / no metrics)

## Holdout results

| arm | best AUC | epoch | control | control AUC | delta | clears 0.0062? | clears 0.015? |
|---|---:|---:|---|---:|---:|:---:|:---:|
| certainty_ramp | 0.8584 | 8 | disagreed_uniform | 0.8507 | **+0.0077** | yes | no |
| disagreed_ramp | 0.8574 | 7 | disagreed_uniform | 0.8507 | **+0.0068** | yes | no |
| disagreed_ramp | 0.8574 | 7 | disagreed_full | 0.8471 | +0.0103 | yes | no |
| ramp_dense_gate | 0.8579 | 7 | calib_dense_hold | 0.8634 | **−0.0055** | no (negative) | no |
| final_ws_ramp | nan | — | — | — | unreadable (fold -1) | — | — |
| through_dense_ramp | nan | — | — | — | unreadable (fold -1) | — | — |
| grade_ladder_ramp | running | — | — | — | — | — | — |
| lr4e-5_ws_ramp | not started | — | — | — | — | — | — |
| calib_span_ramp | not started | — | — | — | — | — | — |

Sources: `D:/data/volumetric/mri/models/rsna_knee_abnormality/*_s2026.metrics.jsonl` and `ramp_dense_gate.metrics.jsonl`.

## Counts

- Arms with completed holdouts and a readable delta: **4**
- Deltas clearing the 0.0062 holdout floor: **2** (certainty_ramp, disagreed_ramp)
- Deltas clearing the 0.015 significance bar: **0**
- Board entries containing any ramp-family member: **0** (searched `comp/board.jsonl` on
  `members_named`; no match)

## Card-hours

| arm | wall-clock hours |
|---|---:|
| certainty_ramp | 1.24 |
| disagreed_ramp | 1.34 |
| disagreed_full | 1.22 |
| disagreed_uniform | 1.22 |
| ramp_dense_gate | 1.36 |
| final_ws_ramp | 1.41 |
| through_dense_ramp | 1.73 |
| grade_ladder_ramp (so far, 2 epochs) | 1.73 |
| **total spent to date** | **11.25** |
| rampsix remaining (2 unreadable + 2 not started, ~1.5 h each) | ~6.0 |
| **total if rampsix finishes** | **~17.3** |

Computed from epoch `started` + `seconds` in each metrics file.

## Verdict

The ramp **pays in the corner where it was measured** (+0.007 on the disagreed/certainty
recipe), but it **does not transfer to the champion's dense corner**: `ramp_dense_gate`
0.8579 is −0.0055 against `calib_dense_hold` 0.8634. That is the transfer test `rampgate`
was built to answer, and its own pre-registered kill branch (< 0) has fired.

The remaining `rampsix` calls cannot improve this reading:
- `final_ws_ramp` and `through_dense_ramp` are `--fold -1` and produce no holdout score.
- `grade_ladder_ramp` is running but belongs to the same dense/grade-ladder corner that
  `ramp_dense_gate` just refuted.
- `lr4e-5_ws_ramp` and `calib_span_ramp` have not started.

No ramp-family member has ever taken a board seat, so there is no historical entry-vs-entry
evidence to set against the holdout numbers.

**Honest answer:** 4 completed ramp arms, 2 deltas above the 0.0062 floor, 0 above 0.015, 0
board seats, ~11.25 card-hours spent. The transfer test to the champion geometry is negative.
The ramp family should be killed and the card redirected to the pretrain pair.
