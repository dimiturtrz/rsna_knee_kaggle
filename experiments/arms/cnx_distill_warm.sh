#!/usr/bin/env bash
# cnx_distill_warm.sh — the same teacher, a trunk from a different pixel group.
#
# WHAT knee-for6 CLOSED ON, AND WHY THAT CLOSE IS AN ARGUMENT RATHER THAN A MEASUREMENT. Its close
# reason: "a student trained on our DECODE, our CORPUS and our TRUNK lands back in our pixel group
# whatever ordering it was taught — distillation moves ORDERING, the seat ruler prices PIXEL GROUP."
# Three named causes, and both distillation arms this project has run held all three fixed at DINOv2
# dense 224 px. `distill_pairs` read +0.0016 with a seat value of -0.0003; `raptor_distill` read +0.0035
# and its nearest neighbour still sat inside the pool. Neither varied the term the close blamed.
#
# THIS ARM VARIES EXACTLY THAT TERM AND NOTHING ELSE. The teacher is `raptor_teacher.npz`, already on
# disk, already the stronger of the two teachers tried (a foreign 0.923 solo, +0.0035 against +0.0016 for
# our own pool — an ordering from outside the family transfers more than one from inside it, which is the
# mechanism's own prediction holding). The weight is 0.3, HELD not re-derived, for the reason
# `raptor_distill` and `distill_v3base` both held it: sweeping an operating point before the axis is
# shown to transfer confounds two questions. The student is a 448 px convolutional trunk warm-started
# from `cnx_own_recipe`'s peak encoder — a different pixel group BY CONSTRUCTION, not by argument.
#
# THE CONTROL IS FREE AND IT IS MATCHED. `cnx_warm_anneal` is this invocation with the two teacher flags
# removed and every other flag identical: cnxbase, 448 px, 20 slices, band 0.1-0.9, whole-stack, max
# pooling, batch 1 / accumulate 6, vram-gb 16, `--backbone-lr 1e-4`, 6 epochs, the same encoder file. So
# this arm costs no control card, and it is single-variable in the strict sense `mismatched` can check.
# It therefore runs AFTER that arm, never instead of it.
#
# WHAT IT IS BOUGHT FOR, said plainly because the quality case is weak. Priced honestly: two distillation
# arms read +0.0016 and +0.0035, both inside the 0.0062 paired floor, and nothing licenses expecting more
# from a third. `stack_increments` predicted 0.8703 additively and measured 0.8642 — distillation competes
# for headroom rather than adding to it. The case here is SEAT DISTANCE, the quantity knee-for6 actually
# declined on and the axis knee-nas5 measures transferring at ~2.4x the board slope. A distant member that
# is merely level can seat; a near member that is better cannot.
#
# THE HONEST COUNTER, which cuts against the arm. cnx solo is 0.8186 against a 0.8460 roster merit level.
# A weak member seats rarely whatever its distance. If `cnx_warm_anneal` lands in its CONVERGED branch
# then this arm inherits a weak student, and a distance claim priced against a member nobody would seat is
# a measurement about the ruler rather than about the trunk. That is still worth twenty minutes — it is
# the only cell that answers knee-for6's close on its own terms — but it is not sold as a score.
#
# cost   ~20 min at 6 warm epochs. The teacher pass is already written and costs nothing. Zero slots.
#
# PRE-REGISTERED, against `cnx_warm_anneal`'s peak — WHATEVER IT READS, because it has not run yet and a
# branch written against a number I already know is not a pre-registration. Floor 0.0062, deltas signed
# against that control:
#   THE TRUNK TERM WAS REAL — >= +0.0062 AND `record_mix` gives the student a positive seat value. Then
#     knee-for6's third cause is the one that mattered, the distillation axis re-opens on TRUNK rather
#     than on teacher strength, and the next arm is the champion-ensemble teacher (knee-7ybf's second
#     cell) on this same student — teacher strength varied with the trunk term now known to move.
#   DISTANT BUT FLAT — inside +/-0.0062 with a nearest-neighbour distance BELOW the pool's own median.
#     The student is where the mechanism said it would be and carries no quality. Keep it if free, read
#     it as a seat candidate through `record_mix` and nothing more; do NOT spend a second cnx distill arm.
#   NEITHER — inside +/-0.0062 and near the pool anyway. Then the pixel-group story is wrong: a 448 px
#     conv trunk taught by a foreign teacher still lands in our group, and what places a member is the
#     CORPUS, not the trunk. That closes distillation for this project on all three named causes bar
#     corpus, and points the remaining effort at knee-8009 (feature distillation as PRETRAINING), which
#     is the only cell that moves what the trunk was fitted ON.
#   SUSPECT THE HARNESS FIRST — <= -0.0062. Before writing any of the above: check the run recorded
#     `distill_weight: 0.3` and a non-null `teacher_targets`, and that the npz joins on the train studies
#     THIS fold uses. A teacher file that joins on nothing trains the control and calls it an arm. Only
#     after that comes clean does a negative here mean the foreign ordering fights the conv trunk.
#
# WHAT NO BRANCH LICENSES. Nothing here says anything about the ConvNeXt rate axis (bracketed by the
# parent's own trajectory), about warm starting in general (that is `cnx_warm_anneal`'s question, and this
# arm inherits its answer rather than testing it), or about the distillation WEIGHT, which is held.
#
# bead: knee-7ybf
# bead: knee-for6
# control  cnx_warm_anneal = the matched no-teacher run at this identical warm recipe.
# deliberate: --distill-weight
# deliberate: --teacher-targets
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$PROCESSED/encoders/cnx_own_recipe.enc.pt"
require "$PROCESSED/raptor_teacher.npz"

run_train cnx_distill_warm \
  $RECIPE --epochs 6 --batch 1 --accumulate 6 --slice-pooling max --whole-stack \
  --variant cnxbase --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 448 \
  --encoder-weights "$PROCESSED/encoders/cnx_own_recipe.enc.pt" \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --vram-gb 16 \
  --labels "$LABELS" --out "$PROCESSED/cnx_distill_warm_s2026.pt"
