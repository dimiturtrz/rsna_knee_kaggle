#!/usr/bin/env bash
# distill_pairs.sh — one student taught the ENSEMBLE'S ORDER, because the labels cannot express it.
#
# WHY A TEACHER AND NOT ANOTHER KNOB. The labels this project trains on come off a report reader that
# emits FIVE DISCRETE LEVELS (0.08 / 0.28 / 0.68 / 0.82 / 0.94, `MEASURED_AGREEMENT`). The objective's
# pairwise term weights a pair by `relu(s_i - s_j)`, so two studies read at the same level contribute
# NOTHING — and the metric is twelve ROC AUCs, which read nothing but order. That is a supervision hole
# with a size: Fracture sits at level 0 in ~93% of studies, so ~87% of its pairs are unordered by the
# labels; Effusion, positive in ~60%, leaves ~25%. An ensemble scores CONTINUOUSLY and orders all of
# them. This arm hands the student that ordering as a second loss term, not as a second training.
#
# THE MECHANISM, STATED SO IT CAN FAIL. If the term works by supplying order the labels withhold, the
# lift must be ORDERED BY RARITY — roughly 7.7x the teachable pair mass on Fracture against 1.3x on
# Effusion, a six-fold spread. A UNIFORM lift across the twelve labels REFUTES this mechanism even if
# the mean moves: uniform is what a regulariser looks like, and a regulariser is not worth a teacher.
# Read the per-label table before the mean.
#
# WHY PAIRWISE AND NOT KL. Standard distillation is KL on temperature-softened probabilities, which
# targets CALIBRATION. AUC is rank-based, so every monotone rewrite of a prediction scores identically
# and the temperature knob is provably inert here. `DistilledObjective.imitation` reuses `Objective.pairs`
# wholesale against the teacher's scores instead, so the term teaches the one thing the metric reads.
#
# WHY THE TEACHER IS THE WEAK ENSEMBLE. The champion's seven members were fitted with `--fold -1` and
# have seen every study including fold 0, so teaching from them would carry this arm's own evaluation
# labels into the student and the run would read as a win with nothing behind it. The teacher here is
# the six fold-0 `_s2026` twins (~0.869 joint), which is a weaker ensemble bought for a holdout that
# stays honest. That the teacher ALSO trained on these training studies is ordinary distillation: its
# readings there are partly memorised, which caps what the student can learn but cannot manufacture a
# score. `kneemri.inference.teacher_readings` refuses `--fold -1` outright.
#
# ONE VARIABLE against the control: `--distill-weight`. The control is `band_dense` byte-for-byte, and
# `--distill-weight 0` is the matched arm to run beside it if the ledger's entry is ever doubted.
#
# THE FALSIFIABLE PART. Bands separated before the run, against the teacher's 0.869 and the best solo
# member ever measured (0.8598, `small_batch6_s2026`):
#   * >= 0.869 — the student MATCHES the ensemble that taught it. Ship it solo: seven members' inference
#     collapses to one, which is the Efficiency Prize and a large serving win either way.
#   * 0.860-0.869 — beats every member this project has ever fitted, so a seat is near-certain and the
#     next arm is the DINOv3 student (a teacher's order is architecture-agnostic, which is the whole
#     reason this is worth more than a knob).
#   * 0.850-0.860 — the per-label clause decides. Rarity-ordered gains keep the mechanism alive at a
#     wrong operating point (weight, teacher strength); a flat profile does not.
#   * <= its own control WITH A FLAT PER-LABEL PROFILE — the term did not transfer at this operating
#     point and the DINOv3 student is NOT licensed. File the profile, not a verdict on distillation.
#
# COST. Teacher cache ~36 min of card (six members x 3,526 training studies), then ~45 min for the
# student. ZERO submission slots. Build the teacher first:
#   python -m kneemri.inference.teacher_readings --fold 0 \
#     --checkpoints "$PROCESSED"/{band_dense,calib_span,grade_ladder,base_dense,lr4e-5_ws,through_dense}_s2026.pt \
#     --out "$PROCESSED/teacher_fold0.npz"
# SMOKE IT before queueing: `python -m kneemri.experiments.arm_check --smoke`, then --studies 60 --epochs 1.
#
# THE FIRST RUN OF THIS ARM WAS VOID, AND ITS NUMBER MUST NOT BE READ. On 2026-08-24 it ran without
# `--backbone-lr`, so it took the trainer's default 8e-6 (trainer.py) while the control holds 4e-5 — a
# fifth of the control's backbone rate, against a docstring two lines up claiming byte-for-byte. It was
# killed at epoch 8 (0.8423, mean -0.0138) once `run_comparison` printed the config diff. The per-label
# profile was every label down and worst on the FOCAL structures (focal -0.0178, regional -0.0082): the
# signature of a starved backbone, not of a distillation null, and not the rarity ordering the pair-mass
# mechanism predicts either. The distill term itself DID land (the diff carried distill_weight=0.3 and
# the teacher path), so the mechanism fired into a crippled recipe. The bands below are unread.
#
# bead: knee-1xsu
# deliberate: --distill-weight
# deliberate: --teacher-targets
# control  band_dense = 0.8578 (best epoch 8), byte-identical recipe with no teacher.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# 0.3 is the report term's weight, and it is borrowed DELIBERATELY rather than derived: it is the only
# auxiliary-term strength this project has ever measured as neither inert nor dominant, and a weight
# swept before the mechanism is shown to fire spends the card on a curve through a null.
run_train distill_pairs \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/teacher_fold0.npz" \
  --labels "$LABELS" --out "$MODELS/distill_pairs_s2026.pt"
