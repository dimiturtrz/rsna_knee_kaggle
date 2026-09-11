#!/usr/bin/env bash
# distill_stack_max.sh -- every positive attribute marginal in ONE member, because 296 one-variable
# arms have already been run and the pool they produced tops out 0.0011 short of the target.
#
# bead: knee-dstackmax
# reading: distill_stack_max_s2026.holdout.npz
#
# WHY A STACK AND NOT ONE VARIABLE. This arm deliberately breaks the one-variable rule and says so
# before the number. The justification is the pool census, not impatience: 296 ruler-consistent
# honestly-held solos @holdout909 midpoint have median 0.8487 and MAX 0.8685 (oof_teacher_s2026),
# total spread 0.0198, top twenty inside 0.0072. The researcher's independent cut (319 scored, 302
# non-void) replicates the same ceiling and the same member to 0.0002. The needle needs ~0.8696: at
# the fitted board_transfer slope +1.55, the +0.010 board gap to the leader is +0.0065 offline, and
# stackfit_s2026 anchors both ends at 0.8631 offline -> 0.933 board. So the target sits PAST the
# maximum of everything this project has ever trained. One more sample of the same recipe
# distribution cannot reach it by construction; only a cell nobody has occupied can.
#
# THE CELL, from D371's per-attribute marginals @honest302 (median with vs median without):
#   distill                +0.0151 (n=22)
#   blend-three llm labels +0.0133 (n=8)
#   labelwise head         +0.0132 (n=11)
#   calibrated targets     +0.0116 (n=40)
#   hold-annotations       +0.0113 (n=39)
# and everything architectural null at this operating point: res336 +0.0019, slices>=20 +0.0030,
# span +0.0014, epochs +0.0005, dinov2-base +0.0001, every non-dinov2 trunk negative. So the arm
# moves NOTHING architectural and stacks only the five attributes that price positive.
#
# WHAT MAKES IT EMPTY. foreign_distill_mix07_s2026 is 2nd of 296 at 0.8657 and carries distill
# ALONE: its arm is `--slice-pooling max`, `--labels $LABELS` (v2), and neither --calibrated-targets
# nor --hold-annotations. Every blend-three member lacks distill; all 11 labelwise members are e12
# no-distill. The four strongest positive marginals have never shared a member with distillation.
#
# THE CONFOUND, STATED AS THE RESEARCHER STATED IT. These marginals co-occur in the same ~30 members
# and realised combinations OVER-ADD about 2x against an additive sketch (sketch 0.8785-0.8896 for
# the full cross vs realised best 0.8685). The additive sum here is meaningless and is NOT the
# prediction. The honest prediction is the researcher's post-over-add estimate: +0.003 to +0.008
# over the mix07 anchor 0.8657, i.e. 0.869-0.874, which BRACKETS the 0.8696 target without
# decisively clearing it. This arm is therefore not sold as a win; it is the only cell with a
# measured basis for even reaching the bar.
#
# NO ATTRIBUTION IS CLAIMED. Five variables move at once and the result cannot say which carried it.
# That is accepted deliberately: the deliverable is a member above 0.8685, not a coefficient. If it
# clears, the one-variable peel-back is the NEXT card, not this one.
#
# HARNESS CHECK BEFORE READING THE NUMBER. The printed recipe must name blend_three_v1.csv on
# "labels", report_labels_v2.csv on "eval_labels", "slice_pooling": "labelwise", a non-zero
# distill weight with teacher_foreign_corpus.npz, calibrated_targets true and hold_annotations true.
# Three runs have entered this ledger as facts about an architecture when they were facts about a
# flag that never landed.
#
# VRAM, DERIVED BEFORE THE CARD. labelwise_dense reserves 28.26 GB at six slots on the canonical
# six-channel scheme, which is the scheme here (extended's nine slots are what forced
# labelwise_extended down to --batch 4 --accumulate 3). --batch 6 --accumulate 2 holds the effective
# batch at 12 with the launch line required to print a cap BELOW 31.8 GB; if it does not, kill it
# rather than let Windows page over PCIe at 11x.
#
# PRE-REGISTERED against the pool ceiling 0.8685 and the mix07 anchor 0.8657, @holdout909 midpoint:
#   >= 0.8696   the target is REACHED offline. The stack is the lever, the cell is real, and the
#               next card goes to the one-variable peel-back plus a board solo of this member.
#   0.8686-0.8695  above the 296-member ceiling but short of the needle. The stack is real and
#               under-delivering; peel-back is still licensed, a board slot is not.
#   0.8658-0.8685  inside the pool's existing top band. The over-add warning is confirmed at full
#               strength and stacking positive marginals is closed as a route to the needle.
#   <= 0.8657   the stack is at or below its own distill-only parent. Then the marginals are not
#               additive in any degree and the attribute-marginal framing itself is retired.
#
# PRICE: ~1.5 h of card, ZERO submission slots. Teacher and label table both on disk. Expected gain
# +0.003-0.008 with the bar at +0.0065 -- the only arm currently priceable that can touch it.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND_THREE="$PROCESSED/labels/blend_three_v1.csv"
require "$LABELS"
require "$BLEND_THREE"
require "$PROCESSED/teacher_foreign_corpus.npz"

run_train distill_stack_max \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations \
  --batch 6 --accumulate 2 \
  --distill-weight 0.7 --teacher-targets "$PROCESSED/teacher_foreign_corpus.npz" \
  --labels "$BLEND_THREE" --eval-labels "$LABELS" --out "$MODELS/distill_stack_max_s2026.pt"
