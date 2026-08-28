#!/usr/bin/env bash
# raptor_distill.sh — the first teacher whose ordering our pool does not already hold.
#
# WHAT IS HELD AND WHAT MOVES. This is `band_dense`'s invocation with the teacher term added and nothing
# else touched — same variant, rate, batch, slices, band, pooling, whole-stack, epochs and schedule. So
# the control is `band_dense` at 0.8578 (peak e8, 2026-08-19), it is already in the ledger, and this arm
# costs ZERO extra card to have a matched control. The single variable is `--distill-weight`.
#
# WHY THIS IS NOT A RERUN OF distill_pairs. That arm read +0.0016 with a seat value of -0.0003, and the
# ledger names the cause rather than guessing: "at 0.9159 nearest, this member sits closer to a seated
# member than any seated member sits to its own nearest neighbour. A rank average cannot pay for an
# ordering it already holds." Its teacher was our own pool distilled into a student of the same family.
# That is a verdict on THAT TEACHER, and the module docstring of `kneemri.kaggle.raptor_teacher`
# carries the whole argument.
#
# WHAT CHANGED, AND IT IS A BOARD MEASUREMENT AND NOT AN ARGUMENT. On 2026-08-28 `raptor_v4_swa` served
# alone scored 0.923 (submission 55829913), EXACTLY level with our own best single (calibsolo 0.923) on
# the same day, the same kit and the same one-seat serving. So this teacher is equal in quality by the
# ruler that decides the competition, and different in architecture, decode and training corpus. Equal
# and distant is the pairing distillation needs and the one we have never had.
#
# AND IT IS THE ROUTE THE CHAMPION PIN NAMES. comp/champion.json's last principle: at the measured seat
# pass-through the member-quality lane ceilings near 0.943, "so 0.950 is NOT reachable by making members
# better, and the remaining routes are external data and stacking". A student taught by a foreign model
# is the external-data route expressed as a member we can SEAT — which the foreign member itself cannot
# cheaply be, since its decode costs a second forward path inside the kernel's nine-hour budget.
#
# THE WEIGHT IS distill_pairs' 0.3, HELD RATHER THAN RE-DERIVED. Sweeping the weight on a new teacher
# before the teacher is shown to transfer at all confounds the teacher question with the operating-point
# question, which is the same reason distill_v3base held it.
#
# PRICE: stated before the run. ~50 min of card (12 x ~250 s) plus the ~30 min teacher pass, which is
# written once and reused free by every later arm on this axis. ZERO submission slots.
# P(clears the 0.0062 holdout floor) ~ 0.35. P(clears the house 1.5% bar) ~ 0.05 — no single arm in 110
# has, and this one is not sold as the one that will. The case is the RATIO: it is the only unrun arm
# whose mechanism is the one the champion pin says the remaining routes run through, and the card-time
# it takes was otherwise going to a head/backbone rate knob.
#
# THE FALSIFIABLE PART, against band_dense's 0.8578 and the 0.0062 paired floor:
#   * >= 0.8640 — the foreign ordering transfers, and it does what a self-teacher could not. Seat it and
#     read its member_distance: the claim this arm actually makes is that the student is BOTH strong and
#     far, and a strong-but-near student is a weaker result than the number alone would say.
#   * 0.8578-0.8640 — directional and inside the floor, the distill_pairs shape at a bigger separation.
#     It does NOT license a weight sweep; it licenses reading the DISTANCE column, because a flat mean
#     with a far student is still a seat candidate and that is the quantity the board rewards.
#   * <= 0.8578 — the teacher's ordering does not survive the family gap. That is a real negative and it
#     closes the distillation axis for this project, both teachers having now been tried.
#   * <= 0.8500 — suspect the harness before the idea: check the recipe recorded `distill_weight: 0.3`
#     and `teacher_targets` non-null, and that the npz covers the train studies this fold actually uses.
#     A teacher file that joins on nothing trains a control and calls it an arm.
#
# WHAT IT CANNOT SAY. The teacher was fitted on all 4,406 studies, so its scores on our train studies are
# partly memorised. Sound for copying an ORDERING, unsound for validating one — no number here is
# evidence about the teacher's accuracy, and the student's fold-0 holdout is clean only because no fold-0
# teacher target is ever shown to it. It also cannot separate "the teacher is foreign" from "the teacher
# is good": both are true of this checkpoint and telling them apart needs a weak foreign teacher.
#
# bead: knee-for6
# control  band_dense = 0.8578 (peak e8), byte-identical recipe with no teacher.
# deliberate: --distill-weight
# deliberate: --teacher-targets
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$PROCESSED/raptor_teacher.npz"

run_train raptor_distill \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --labels "$LABELS" --out "$PROCESSED/raptor_distill_s2026.pt"
