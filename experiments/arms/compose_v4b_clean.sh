#!/usr/bin/env bash
# compose_v4b_clean — does the distillation term SURVIVE a better label table, or was it agreement with v2?
#
# THE QUESTION, and it is not "one more member". Two treatments each cleared their own bar at the same
# 336/20 corner, one variable off `band_dense` (0.8578) apiece:
#   labels_v4hybrid      --labels blend_hybrid_fx_v4b.csv   v2 holdout 0.8524, gold 58 0.9063 (+0.0131)
#   seat_clean_distill   --teacher-targets teacher_clean    read 2026-09-07, leak removed
# `labels_v4hybrid`'s ledger entry closed on MIDDLE, and the MIDDLE rule it wrote down is exactly this:
# "the label axis composes with distillation or it is not worth a seat". This is that composition, with
# the leaky teacher replaced by the clean one so the reading is not the artefact that voided the family.
#
# THE MECHANISM THAT COULD KILL IT, stated before the card is spent, because it is the whole reason the
# arm is informative either way. The five teacher blocks were fitted on **v2** labels. Their targets
# therefore carry v2's construction errors, and a v4b student pulled toward them is pulled BACK toward
# the table the swap was meant to leave. So:
#   * if the distillation term is a general regularizer (dark knowledge about IMAGES), it adds on top
#     of v4b and the composition is roughly additive;
#   * if the term was mostly "agree with v2 more", it CANCELS against a v4b table and the composition
#     reads at or below `labels_v4hybrid` alone.
# Those two predictions differ in sign, which is what makes 1.2 h worth spending on a knob-sized delta:
# the arm prices the distillation lever's NATURE, not one more seat.
#
# ONE VARIABLE against `labels_v4hybrid.sh`: `--distill-weight 0.7 --teacher-targets teacher_clean.npz`.
# Everything else verbatim, `--hold-annotations` and `--eval-labels` included, so the matched control is
# `labels_v4hybrid_s2026` at 0.8524 on the v2 holdout and 0.9063 on the gold 58 — both on disk, no
# control arm is run.
#
# PRE-REGISTERED, against `labels_v4hybrid`'s 0.8524 on the v2 holdout. Let D be the clean distillation
# term measured at this corner on the v2 table, D = seat_clean_distill_s2026 peak - 0.8578. Additive
# composition predicts 0.8524 + D.
#   >= 0.8524 + D - 0.004   ADDITIVE. The term is about images, not about agreeing with v2, and the two
#                           axes compose. This is the member that goes to the board, and the successor
#                           named in `labels_v4hybrid.sh` (table + level map on the best base) is licensed.
#   0.8524 +- 0.004         CANCELS. Distillation buys nothing once the table is better; the term is
#                           re-read as a partial repair of v2's construction errors, which a better table
#                           already made. Neither axis is killed; their COMPOSITION is, and no arm may
#                           stack them again.
#   < 0.8520                ANTAGONISTIC. The v2-fitted teacher actively drags a v4b student back. That
#                           is a finding about teacher provenance and it forbids every distillation arm
#                           on a non-v2 table until a v4b-fitted teacher exists.
# The gold 58 is read beside it against `labels_v4hybrid`'s 0.9063 and is REPORTED, never the verdict:
# the per-label floor at n=58 is ~0.03 (measured, the Synovitis control's own excursion) and a knob-sized
# composition term sits under it. The twelve-mean is the only gold statistic with enough support.
#
# WHY NOT THE oof5 TEACHER. `teacher_oof5.npz` is full-strength but its student has no holdout by
# construction (`--fold -1`), so a composition read through it costs a board slot to price. This arm
# answers the same question on a ruler we own for zero slots, and the starvation handicap (-0.0067 on the
# teachers, measured over four folds) applies EQUALLY to the control's absence of a teacher and to this
# arm's presence of one - it can only push this number DOWN, so a clear ADDITIVE reading survives it.
#
# COST one twelve-epoch fit at 336/20, ~364 s an epoch measured on `seat_clean_distill` -> ~1.2 h.
#
# deliberate: --distill-weight 0.7 (the settled point on the weight curve; not re-searched)
# deliberate: --teacher-targets teacher_clean.npz (the CLEAN teacher; the corpus file is the leaky one)
# deliberate: --labels blend_hybrid_fx_v4b.csv (the table `labels_v4hybrid` was read on, unchanged)
# deliberate: --eval-labels report_labels_v2.csv (the ruler every fold-0 row is scored on)
# deliberate: --hold-annotations (the gold 58 is read here too; an unheld fit scores its own cells)
# deliberate: every other flag verbatim from labels_v4hybrid.sh
# reading: compose_v4b_clean_s2026.holdout.npz
# bead: knee-wko44
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

CANDIDATE="$PROCESSED/labels/blend_hybrid_fx_v4b.csv"
TEACHER="$PROCESSED/teacher_clean.npz"
require "$CANDIDATE"
require "$TEACHER"
require "$LABELS"

run_train compose_v4b_clean \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$CANDIDATE" --eval-labels "$LABELS" --out "$MODELS/compose_v4b_clean_s2026.pt"
