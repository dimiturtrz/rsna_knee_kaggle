#!/usr/bin/env bash
# merged_blend.sh — the two mechanisms that each moved a DIFFERENT half of the split, composed.
#
# bead: knee-twtr
#
# WHY THESE TWO AND NOT ANY OTHER PAIR. `supervision_gap` divides the twelve findings by which side is
# the weak one, and this week produced exactly one mechanism that moved each side:
#
#   merged_corpus (+0.0020 mean)   deficit six +0.0061   surplus six -0.0022
#   blendlabels   (+0.0034 mean)   its own six label-bottlenecked findings +0.0070
#
# merged_corpus adds a second SITE and acts on the encoder — its largest mover was Lateral Meniscus
# +0.0196*, a finding Rijeka never labels, so the gain cannot be supervision. blendlabels changes the
# TARGET RESOLUTION our rows carry and can only act where the reader is the weak part; its own
# pre-registration said so before it ran, and it landed +0.0070 there and MCL +0.0239* — our single
# weakest finding. The two act on disjoint halves of the same split through disjoint mechanisms. That
# is the whole argument for composing them, and it is why this is not "stack two knobs and hope".
#
# WHAT MOVES, against lr4e-5_ws. Three flags: `--scheme merged` (the 5,143-study cache, already built),
# `--labels merged_blend_v1.csv` (the union table whose OUR rows carry the three-reader blend and whose
# 736 Rijeka rows carry ACL alone), and `--eval-labels report_labels_v2.csv`.
#
# THE THIRD FLAG IS NOT OPTIONAL, and merged_corpus did not need it. That arm's table carried v2 values
# for our rows, so an unpinned holdout was a v2 holdout by accident. This arm's do not. Scored unpinned
# it would measure DISAGREEMENT WITH V2 and report it as quality — a member fitted on the blend predicts
# the blend, so it scores worse against v2 the better it fits. Without the pin this arm answers nothing.
#
# CONTROL. lr4e-5_ws = 0.8525: same twelve slices, same 0.20-0.80 band, same whole-stack, same 4e-5, same
# twelve epochs, same 336px read, same 909-study fold-0 holdout. NOT native224 — the merged cache exists
# only at 336, and the pixel axis measured flat (-0.0023) so the geometry is not the variable here.
#
# COST. ~301 s/epoch measured on merged_corpus itself, so twelve epochs is ~1.0 h. No cache is built:
# the images are merged_corpus's own arrays and the label table was written by
# `build_merged --labels-only` in seconds.
#
# THE FALSIFIABLE PART, written before the run. The mean is the weak reading here and the SPLIT is the
# strong one, because each parent moved a different half and neither moved the mean past the floor.
#   * >= +0.0062 mean over the control. The two gains composed. This is the first single model to clear
#     the floor over lr4e-5_ws all week and it earns a board slot on its own.
#   * +0.003 to +0.0062, WITH the deficit six up and the surplus six up. Both mechanisms fired and the
#     composition is additive; directional under the house bar, kept as a member, and the case for
#     MRNet-on-KIND strengthens because the encoder half held up under a changed target.
#   * The mean moves but ONE side carries all of it. Only one mechanism fired. The other is the suspect,
#     not the composition: check the printed recipe named both tables before concluding anything.
#   * Flat, or the deficit six DOWN. The two interact — the likeliest mechanism is that a finer target on
#     our rows sharpens the gradient our 4,406 studies contribute and drowns the 736 foreign ones, which
#     would be an argument for upweighting the foreign block, not against the second site.
#   * The mean falls past -0.0062. Suspect the harness first: `Targets.pair` must still weight the
#     eleven columns Rijeka is silent on at 0.0, and the blend must not have filled them.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The printed recipe must name merged on "scheme" AND
# merged_blend_v1.csv on "labels" AND report_labels_v2.csv on "eval_labels" AND 336 on "image_size", and
# the log must load 5,143 studies and hold out 909. Three runs have entered the ledger as facts about an
# architecture when they were facts about a flag that never landed, and this arm carries three flags.
#
# deliberate: --eval-labels report_labels_v2.csv (see THE THIRD FLAG above — without it the holdout
# deliberate: measures disagreement with the ruler every other fold-0 row was scored against)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# control  lr4e-5_ws = 0.8525 (fold 0, scored against report_labels_v2)
#
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"
require "$LABELS"

run_train merged_blend \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged \
  --labels "$MERGED_BLEND" --eval-labels "$LABELS" --out "$MODELS/merged_blend_s2026.pt"
