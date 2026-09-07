#!/usr/bin/env bash
# compose_v4b_oof.sh — the two label-side levers that each moved, run TOGETHER on one recipe.
#
# THE QUESTION. `labels_v4hybrid` (better table) read +0.0131 on the gold 58 and `oof_teacher` (softer
# target) read +0.0107 on the v2 holdout. Mastermind's 23:43Z correction is that these are NOT two
# columns: v4hybrid's gold lift is Lateral Meniscus +0.0820, Lateral OA +0.0349, Fracture +0.0250,
# MCL +0.0227 — the same ligament/meniscus/bone columns distillation moved. Two treatments pulling the
# model-bottlenecked findings from opposite sides, a cleaner target and a softer one. So the honest
# prior is OVERLAP, not a sum, and this arm is the only thing that separates the three cases.
#
# ONE ARM, TWO TREATMENTS, EVERYTHING ELSE HELD at band_dense's geometry (20 slices, 0.1-0.9, whole
# stack, max pooling, 4e-5, 12 epochs). Neither treatment is new here; both are lifted verbatim from the
# arm that measured them.
#
# PRE-REGISTERED, from mastermind 23:43Z, on the v2-holdout-equivalent number:
#   * >= 0.8741 (0.8681 + 0.006) — ADDITIVE. Both treatments go on every champion seat.
#   * 0.8641-0.8721 (0.8681 +/- 0.004) — REDUNDANT. Distill alone on the seats; the label table is
#     dropped from the seat recipe, since it costs a second table to ship and buys nothing on top.
#   * < 0.8640 — INTERFERENCE, and the named mechanism is a MIXED TARGET: the teacher's posterior was
#     fitted under v2 supervision while the student is asked to fit v4b, so the two targets disagree
#     wherever the tables do. In that branch the next action is to rebuild the teacher on v4b, and
#     NOTHING ELSE is read off this arm.
#
# TWO RULERS, AND WHICH BRANCH IS CALLED ON WHICH. Report both, always.
#   1. v2 HOLDOUT — the branch above is called on this one, but it is NOT the raw number. A student of
#      v4b scored against v2 measures disagreement with v2 as if it were error: measured directly, the
#      same recipe reads 0.8524 with the v4b table against `band_dense_hold`'s 0.8573, so the
#      construction offset is -0.0049 and it is ADDED BACK before the bands are applied. That offset is
#      a measurement (two runs, one flag apart), not an allowance.
#      The other matched-pair correction is -0.0005: `oof_teacher` (0.8681) did NOT hold annotations and
#      this arm must, so its control moves by band_dense 0.8578 -> band_dense_hold 0.8573.
#   2. GOLD 58, `annotated_readings --per-label`, against `labels_v4hybrid_s2026`'s 0.9063 and
#      `holdannot_s2026`'s 0.8932. This is the ruler the label axis was decided on, and it is the one
#      that can say WHERE a gain landed. It cannot call the branch: the gold instrument's control column
#      moved -0.0298 on a byte-identical table, so its per-label noise floor at these n is ~0.03.
#
# WHAT IS NOT IN THIS ARM, AND WHY — the v3 LEVEL MAP.
# The 23:43Z order names four ingredients: v4b table, verdict mask, v3 map, OOF distill 0.7. Three are
# here. The v3 map is not, because it cannot be built on v4b with the tool that built it on v2, and the
# attempt is on disk: `logs/level_calib_v4b.log`.
# `LevelCalibration` fits one rate per UNIQUE VALUE in the table and its docstring states the
# precondition outright — "over all 4,406 rows the reader's vocabulary is five levels". v2 satisfies it
# (0.08/0.28/0.68/0.82/0.94 -> 0.041/0.136/0.378/0.688/0.958, five parameters on 684 cells). v4b is a
# CONTINUOUS blend: 130 distinct values, most of them appearing in one or two gold cells, so the fit
# reads `n=1 raw 1.000` sixty-odd times and writes the gold panel's own answers back into the training
# table. That is not a recalibration, it is the shared-origin defect at full strength on the exact ruler
# this arm is then read with. The written table was deleted.
# The composition mastermind asked for is still available two ways, and both are ARMS, not this one:
#   (a) v4b's VERDICTS on v2's five-level values, then `level_calibration` — but v4b and v2 agree on
#       99.04% of verdict cells, so that table is v3 with ~1% of cells moved and cannot carry v4b's
#       +0.0337 of gold agreement, which lives in the grading and not in the verdicts;
#   (b) quantise v4b onto the reader's five levels first — which invents a quantiser and is a third
#       treatment, not a composition of two measured ones.
# Neither is licensed without a reading, so the arm that runs is the two-treatment one, which is also
# the only version the pre-registered bands can be applied to: a third and fourth ingredient would make
# a null unattributable.
#
# COST ~80 min of card, zero submission slots. Both inputs already exist.
#
# deliberate: --labels blend_hybrid_fx_v4b.csv (treatment 1, verbatim from labels_v4hybrid)
# deliberate: --distill-weight 0.7 --teacher-targets teacher_oof_corpus.npz (treatment 2, verbatim from
# deliberate: oof_teacher; 0.7 held from two arms that each found it the best point on a monotone line)
# deliberate: --hold-annotations (the gold ruler refuses an unheld fit — it would score its own cells)
# deliberate: --eval-labels report_labels_v2.csv (the ruler must not move with the arm; knee-9198)
# bead: knee-ghjm4
# reading: compose_v4b_oof_s2026.holdout.npz
# control  oof_teacher = 0.8681, labels_v4hybrid = 0.8524, band_dense_hold = 0.8573
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

CANDIDATE="$PROCESSED/labels/blend_hybrid_fx_v4b.csv"
TEACHER="$PROCESSED/teacher_oof_corpus.npz"
require "$CANDIDATE"
require "$TEACHER"
require "$LABELS"

run_train compose_v4b_oof \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --distill-weight 0.7 --teacher-targets "$TEACHER" \
  --labels "$CANDIDATE" --eval-labels "$LABELS" --out "$MODELS/compose_v4b_oof_s2026.pt"
