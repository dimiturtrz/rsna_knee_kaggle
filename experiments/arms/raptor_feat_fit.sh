#!/usr/bin/env bash
# raptor_feat_fit.sh — stage two: the labelled fit, warm from the feature-distilled trunk. THIS is the reading.
#
# ONE VARIABLE against a control already on disk. `swin_supervised` is this invocation with
# `--encoder-weights` removed: 12 epochs, batch 1 / accumulate 6, max pooling, whole-stack, swinbase,
# 384 px, `--backbone-lr 1e-4`, 20 slices, band 0.1-0.9, vram-gb 8, the same $RECIPE and $THROUGHPUT. It
# scored 0.8331 solo. The same control (a) used, deliberately, so the two cells of knee-8009 are read on
# one ruler and the difference between them is the LEVEL of the distillation target and nothing else.
#
# THE EXPORT IS PART OF THE ARM, not a step to remember — `swin_distill_fit`'s reason exactly: nothing
# refuses an encoder file that is simply the PREVIOUS export sitting at the same path, so the conversion
# runs here. It costs seconds and no card.
#
# WHAT NO BRANCH LICENSES. Nothing here says anything about the feature WEIGHT (20 is the stage-one knob,
# held, never swept before the axis is shown to transfer), about warm starting in general (the control is
# cold BY DESIGN, so this arm cannot separate "distilled trunk" from "warm start" — a positive reading
# owes a follow-up warm-from-swin_supervised control before it is called a distillation result), or about
# any trunk that is not swinbase.
#
# AND THE SEAT IS READ WITH THE SCORE OR NOT AT ALL. `swin_supervised` is the most decorrelated member
# this project owns (nearest rank correlation 0.8346 against a seated band of 0.8772-0.9225). A trunk
# shaped toward a FOREIGN model's representation could gain solo score while collapsing toward that
# model's ordering, and a report quoting only the gain would hide it. `run_train` prints the seat value;
# read `member_axes` beside it and quote both.
#
# control  swin_supervised_s2026 = 0.8331 (cold, the same control the logit arm used).
# PRICE: ~30 min stage one + ~3.3 h stage two, 0 slots.
# P(clears +0.0062) ~ 0.25.
#
# PRE-REGISTERED, before the run, against the cold control:
#   * >= +0.0062 — feature distillation IS pretraining where logit distillation was not. knee-8009
#     closes with (b) as the surviving cell, the -0.0077 of (a) is explained by the TARGET LEVEL rather
#     than by distillation, and the next arm is the champion-ensemble teacher's features on this student.
#   * +0.0015 to +0.0061 — directional. Keep, never a finding, no slot. The target level matters and it
#     is not free.
#   * -0.0061 to +0.0014 — FLAT, and this is a real result rather than a null: a representation target
#     ties the cold trunk while a logit target LOST 0.0077, so the level of the target is what the -0.0077
#     measured. knee-8009 closes on both cells, and distillation closes for this project on the trunk
#     cause, leaving only the corpus cause open.
#   * <= -0.0062 — SUSPECT THE HARNESS FIRST. Verify stage one recorded a non-null `teacher_features` and
#     its weight, verify the exported encoder is the file stage two loads via `encoder_weights`, verify
#     the teacher npz JOINS on the studies this fold trains on (a teacher file that joins on nothing
#     trains the control and calls it an arm), and verify the projection actually received gradient.
#     Only after all four come back clean does a negative read as a fact about representation targets.
#
# THE FOURTH SUSPECT IS ANSWERABLE WITHOUT A RERUN. The projection is owned by the LOSS, not by the model,
# and `parameter_groups` collects the head BY EXCLUSION from the backbone — so an unattached projection
# would sit in no parameter group and stay at its initialisation for the whole run (knee-7gyw's defect,
# silently). `Objective.attached` is what hands it to the head group; the unit gate asserts the optimiser
# holds those tensors, and stage one's own loss falling is the run-time witness.
#
# bead: knee-8009
# deliberate: --encoder-weights  (the feature-distilled trunk — the ONE varied term)
# deliberate: --variant
# deliberate: --train-size       (384, pinned by the window tables — see swin_supervised.sh)
# deliberate: --backbone-lr
# deliberate: --batch
# deliberate: --accumulate
# deliberate: --vram-gb
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$PROCESSED/raptor_feat_pre_s2026.pt"

uv run --no-sync --project "$REPO" python -m kneemri.training.fitted_encoder \
  "$PROCESSED/raptor_feat_pre_s2026.pt" "$PROCESSED/encoders/raptor_feat_pre.enc.pt" || exit 1

run_train raptor_feat_fit \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant swinbase --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --encoder-weights "$PROCESSED/encoders/raptor_feat_pre.enc.pt" \
  --vram-gb 8 \
  --labels "$LABELS" --out "$PROCESSED/raptor_feat_fit_s2026.pt"
