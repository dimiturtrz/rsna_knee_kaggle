#!/usr/bin/env bash
# merged3_corpus.sh — a THIRD site added to the corpus, and the first foreign block that fills more
# than one slot.
#
# bead: knee-m179
#
# WHAT MOVES, against merged_corpus. Two flags, and they are one variable: `--scheme merged3` (the
# 6,393-study cache built 2026-09-01 by merging the 5,143-study canonical+Rijeka cache with MRNet) and
# `--labels merged3_labels.csv` (that cache's union table, our rows carrying v2 exactly as
# merged_labels.csv does, and the 1,248 MRNet rows carrying ACL alone at 0.08/0.82). Everything else is
# merged_corpus's invocation character for character.
#
# WHY MRNET IS NOT JUST MORE RIJEKA. Rijeka fills ONE of the six canonical slots, so every foreign study
# in the merged cache reaches the encoder as a knee with five slots empty — and the merged_corpus result
# (+0.0020 mean) cannot separate "a second site transfers weakly" from "a degenerate input teaches the
# encoder to tolerate holes". MRNet ships sagittal, axial AND coronal, so its exams fill three. If the
# gain scales with slots filled rather than with studies added, this arm reads well above the 1,250/5,143
# study-count ratio would predict, and the corpus axis is a slot-coverage axis rather than a volume one.
# That is the reading this arm exists for; the mean is the weaker half of it.
#
# THE LABEL ASYMMETRY IS DELIBERATE AND IS NOT A DEFECT. MRNet supervises ACL alone — its `meniscus`
# column does not distinguish medial from lateral and its `abnormal` column has no counterpart of ours,
# so both are refused rather than mapped onto columns they do not mean. `Targets.pair` weights the
# eleven silent columns at 0.0, exactly as it does for Rijeka. A foreign block that taught eleven
# findings it does not know would be the defect; one that teaches a twelfth of them is the design.
#
# CONTROL. merged_corpus = the 5,143-study arm on the same recipe, same twelve slices, same 0.20-0.80
# band, same whole-stack, same 4e-5, same twelve epochs, same 336px read, same 909-study fold-0 holdout.
# The comparison is corpus-for-corpus with the ruler held.
#
# COST. ~301 s/epoch on merged_corpus over 5,143 studies; 6,393 studies is ~24% more, so ~370 s/epoch
# and ~1.25 h for twelve. No cache is built — `build_merged --merged-scheme merged3` wrote both the
# array and the table already.
#
# THE FALSIFIABLE PART, written before the run. merged_corpus read +0.0020 mean over lr4e-5_ws with
# deficit six +0.0061 and surplus six -0.0022.
#   * >= +0.0062 over merged_corpus. The third site clears the floor on its own and the corpus axis is
#     the live lever it was argued to be; the next arm on it is a fourth corpus, not a knob.
#   * +0.002 to +0.0062, with the ACL column among the movers. Directional and kept as a member. The
#     slot-coverage reading is supported only if the gain exceeds what 24% more studies buys — compare
#     against merged_corpus's own gain over lr4e-5_ws, which 736 one-slot studies bought.
#   * Flat. The corpus axis saturates at two sites under this recipe, and the honest next question is
#     whether the foreign block is being drowned (upweight it) rather than whether more data helps.
#   * Down past -0.0062, and ACL specifically down. Suspect MRNET_LEVEL first: 0.82 for a positive is a
#     deliberate discount for a binary label that lumps partial with complete tears, and if it is wrong
#     it is wrong in the direction that would show here and nowhere else.
#
# HARNESS CHECK BEFORE READING ANY OF THAT. The printed recipe must name merged3 on "scheme" AND
# merged3_labels.csv on "labels" AND report_labels_v2.csv on "eval_labels" AND 336 on "image_size", and
# the log must load 6,393 studies and hold out 909. Three runs have entered the ledger as facts about an
# architecture when they were facts about a flag that never landed.
#
# deliberate: --eval-labels report_labels_v2.csv — merged_corpus did without it because its table
# deliberate: carried v2 values for our rows and an unpinned holdout was a v2 holdout by accident. The
# deliberate: pin makes that an assertion instead of a coincidence; the values are the same either way.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED3_LABELS="$PROCESSED/labels/merged3_labels.csv"
require "$MERGED3_LABELS"
require "$LABELS"

run_train merged3_corpus \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged3 \
  --labels "$MERGED3_LABELS" --eval-labels "$LABELS" --out "$PROCESSED/merged3_corpus_s2026.pt"
