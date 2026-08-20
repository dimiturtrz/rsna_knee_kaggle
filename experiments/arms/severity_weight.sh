#!/usr/bin/env bash
# severity_weight.sh — the reader graded every positive and the loss threw the grade away.
#
# THE DEFECT, and it is a WEIGHT defect rather than a value one. The reader emits three YES levels —
# 0.68 mild, 0.82 moderate, 0.94 severe — and a `__conf` column that is CONSTANT at 0.95 across all
# three, measured over all 52,872 populated cells (min == max == 0.95 at every YES level). So the loss
# pulls exactly as hard on a mild degeneration as on a complete tear. The grade is per-study
# information that reaches the table and dies there.
#
# WHAT THE GRADE IS ACTUALLY WORTH, against the 57 fully-annotated studies — how often the radiologist
# confirmed the finding the reader graded:
#
#     level     measured   95% CI            n
#      0.68      0.378     [0.282, 0.474]    98
#      0.82      0.688     [0.611, 0.764]   141
#      0.94      0.958     [0.912, 1.004]    72
#
# THREE INTERVALS THAT DO NOT OVERLAP. The ordering is not soft: a severe call is right two and a half
# times as often as a mild one, and the loss currently cannot tell them apart. This arm makes the
# weight that rate. A NO (0.08) is graded from the other side — it asserts ABSENCE and was right 95.9%
# of the time — so it pulls at 0.959 rather than at 0.041.
#
# 0.28 IS LEFT ALONE, deliberately. It asserts neither side, so its 0.136 is a base rate rather than a
# hit rate and there is nothing for the radiologist to have agreed WITH. Those cells keep the near-zero
# confidence the reader already gives them.
#
# NOT calib_span, WHICH LOST 0.0020, and the distinction is the whole reason this is worth a slot.
# `--calibrated-targets` spends the same table on the VALUE: it rewrote 6,461 mild cells from 0.68 to
# 0.378, moving them across 0.5 and INVERTING their sign — the model was told those findings were
# absent. This leaves every target exactly where it is and changes only how loudly each speaks. Same
# measurement, opposite half of the pair, and the failure of one is not evidence about the other.
#
# COMPOSES WITH THE SHIPPED FLOOR RATHER THAN FIGHTING IT. `--weight-floor` is in $RECIPE and is an
# AFFINE rescale (`floor + (1-floor)*conf`), not a clamp, so it preserves order: the three grades land
# at 0.534 / 0.766 / 0.969 where today they all sit at 0.9625. Spread across the YES grades goes from
# 0.000 to 0.435. The mechanism survives the flag it runs under, which was the one plumbing trap here.
#
# TWO THINGS MOVE, AND THE SECOND IS THE BIGGER BUCKET — said here because the pre-registered clause
# below is what separates them, not because it is small. Measured on the real table: 6,363 mild cells
# drop 0.9625 -> 0.5335 and 6,286 moderate drop to 0.7660, but 22,102 NO cells RISE 0.8875 -> 0.9693.
# So this arm both grades the positives and strengthens the denials, and a bare mean cannot say which
# paid. The denial uplift is UNIFORM across all twelve labels (every label's NOs move by the same
# +0.082); the grade effect can only appear on labels a report actually grades. That is exactly the
# discriminator the clause below asks for, which is why a uniform lift REFUTES rather than confirms.
#
# AND THE PILOT'S FABRICATED DENIALS DO NOT UNDERMINE THE UPLIFT, though they look like they should.
# The 2026-08-20 label-contest pilot found the reader's dominant error is inventing NOs the report never
# made (23 of 48). But 0.959 is measured against the IMAGES on 196 cells, not against the rulebook — a
# denial the report never wrote is still CORRECT about the image when the finding is genuinely absent,
# and the image ruler has already counted those cells. The fabrication is priced in, not ignored.
#
# THE RULER DOES NOT MOVE, for the same reason it does not under a calibration: severity is applied to
# the TRAINING table only. Grading the holdout too would change which cells it counts and how much,
# and the AUC would move by construction. `Supervision.moves_the_ruler` is what forces the second read.
#
# NOR DOES IT LEAK INTO THE NUMBER IT IS JUDGED BY. The rates are fitted on the 58 IMAGE-READ
# annotations; this arm is scored on the fold-0 holdout of the REPORT labels. Different rulers. What it
# spends is some independence of the annotation-based check (`annot`), stated rather than hidden.
#
# THE FALSIFIABLE PART, written before the run so it cannot be decided afterwards: the gain must
# CONCENTRATE in the labels whose positives are graded, and be near zero where they are not. The three
# YES levels only separate on findings a report grades — meniscal degeneration, cartilage/OA, ligament
# sprain-to-tear — so Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA and PF OA must carry it.
# Effusion and Baker's are graded far more rarely (mostly present/absent) and should barely move.
# A UNIFORM LIFT ACROSS ALL TWELVE REFUTES THE MECHANISM — it would mean the arm bought a general
# down-weighting of noisy cells, which `--weight-floor` already prices, and not the grade at all.
# A LIFT CONCENTRATED IN MCL REFUTES IT TOO: the label-contest pilot found 0 rulebook errors on MCL in
# 100 studies and `resolution.sh` argues it is bounded by what we can SEE. Supervision should not reach
# it, and if it does, the mechanism is not what moved the number.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical recipe with the flag off.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train severity_weight \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --severity-weight \
  --labels "$LABELS" --out "$PROCESSED/severity_weight_s2026.pt"
