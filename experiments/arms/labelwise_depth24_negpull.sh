#!/usr/bin/env bash
# labelwise_depth24_negpull.sh — knee-bjpq4, the strongest base refitted on the negation-pulled table.
#
# ONE CHANGE. Every flag below is `labelwise_depth24_drop40_long.sh`'s, byte for byte, except `--labels`
# and `--out`. The control is that arm's checkpoint, already on disk at 0.9053 ledger75, so this arm owes
# no control run and the comparison is same-recipe.
#
# WHAT THE TABLE DIFFERS BY. `NegationPull(0.15)` moves 414 cells of 52,872 (0.78%): a positive cell whose
# report DENIES that finding everywhere it names it is pulled toward its column's negative floor, with the
# extractor's measured 15% clean-failure rate left standing as the residual. Never a hard flip. ACL 98,
# Medial Meniscus 78, Lateral Meniscus 68, PF OA 57, MCL 42, the other six 71 between them.
#
# WHY THE GOLD SET IS NOT THE RULER HERE. `supervision_gap --labels` reads the corrected table at 0.8675
# against 0.8658 — but the rule fires exactly THREE times in the 696 gold cells (right all three times).
# +0.0017 on n=3 is not a measurement of a 414-cell change, which is why this costs a card rather than
# being settled on CPU.
#
# THE RULER IS THE OLD TABLE, DELIBERATELY. `--eval-labels "$LABELS"` holds the holdout scoring on
# `report_labels_v2.csv` while training reads the corrected one. Without it the corrected table would be
# both the supervision AND the ruler, and 400 holdout cells score differently under it — a gain would then
# be the ruler moving rather than the model (knee-df7t, caught by `arm_check` before this was armed).
#
# PRE-REGISTERED, against `labelwise_depth24_drop40_long_s2026` at 0.9053 ledger75:
#   >= +0.0040   negation-corrected supervision pays at full weight. Roster it, price it as a SWAP for its
#                own base in the n=7 seating, and relabel the rest of the pool.
#   +0.0015..+0.0039  partial. Keep ONLY if it prices positive as a marginal over the seating — a solo is
#                not a seat, and stack_increments read +0.0020 solo and -0.0001 seated.
#   <= +0.0014   414 cells is too few to move a member, and the negation lane closes at the TABLE rather
#                than at the detector: the rule is right (3/3 on gold, zero recall cost) and too small.
#                That reading licenses the wider question — a pool relabelled throughout — and refuses
#                any further single-member spend on this rule.
#   solo floor   >= 0.9000, or something other than the labels changed and no branch above may be read.
#
# PRICE: ~1.5 h. P(clears +0.0040) ~ 0.20.
#
# deliberate: --labels report_labels_negpull.csv (the whole arm; built by
#             `python -m core.data.labels.table_correction --rule missed_negation`)
# reading: labelwise_depth24_negpull_s2026.holdout.npz
# bead: knee-bjpq4
#
# control  labelwise_depth24_drop40_long_s2026 = 0.9053 ledger75 (on disk, same recipe line)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

NEGPULL="$PROCESSED/labels/report_labels_negpull.csv"
require "$NEGPULL"

run_train labelwise_depth24_negpull \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --labels "$NEGPULL" --eval-labels "$LABELS" --out "$MODELS/labelwise_depth24_negpull_s2026.pt"
