#!/usr/bin/env bash
# merged_dense_cf.sh — the DENSE stack geometry applied to the best member this project holds.
#
# bead: knee-4hxv
#
# WHY THIS ARM, AND WHY IT IS THE ONLY SHAPE LEFT. Every decode knob this repo owns is closed on its own
# measurement: resolution (336->392 is -0.0002 once the rate is held), crop extent and centre (-0.0099 at
# 80 mm, -0.0011 at 130 mm), depth regions (a substitute for span, not a complement), slice pooling
# per-diagnosis (+0.0015, inside the floor), window geometry (-0.0040), the trunk family (ConvNeXt at its
# own fitted rate lands 0.8186, seat 53 of 67), SSL-as-init (five collapses), the vote, the count and the
# seventh seat. The ledger has said "the limit is upstream" from four directions.
#
# UPSTREAM IS EXACTLY WHERE OUR TWO BEST NUMBERS COME FROM. `merged_blend_s2026` (0.8589) and its
# calibrated twin `merged_blend_cf` (0.8597) are the top two holdout scores in the whole project, above
# `res392_ws` 0.8523 and above `native224` 0.8502. Both are the SAME member: the RSNA corpus merged with
# Rijeka's external expert IMAGE reads, supervised by the three-reader blended table. That axis is the one
# still paying, and it has never been run at anything but the 12-slice 0.2-0.8 geometry it was first
# written with.
#
# ONE VARIABLE against `merged_blend_cf`: the stack geometry, 12 slices at 0.2-0.8 -> 20 slices at 0.1-0.9.
# Corpus, label table, eval ruler, pooling, rate, epochs, precision and `--calibrated-targets` all held.
# That geometry is not a guess: it is the `band_dense` operating point five seated members were fitted at,
# and the span half of it measured +0.0039 alone (`band_span`, at slice_regions=1, which is what this arm
# runs). The claim under test is that the two levers COMPOSE — a lever measured on the report-labelled
# corpus should still pay on the merged one, because they act on different parts of the chain (what the
# network SEES vs what it is TOLD).
#
# THE COMPOSITION IS THE POINT, NOT THE SIZE. +0.0039 does not clear the 0.0062 floor and this arm is not
# sold as clearing it. It is the third link in a chain that is already +0.0095 over `native224` on the same
# ruler (merged corpus + blended labels +0.0087, calibrated targets +0.0010) and the project's own rule is
# that increments below the bar still compose against the ORIGINAL baseline. If the chain reaches ~0.864 it
# is +0.014 over the 224 baseline and about +0.007 over the best member any seat has ever held, which at
# board_transfer +1.72 is the size of the frontier gap.
#
# THE FAILURE MODE THIS ARM IS WRITTEN AGAINST. `depth_regions_span` and `crop80_span` both showed a lever
# that paid alone and REVERSED when stacked on a second lever acting on the same axis. Density and span
# both answer "where along the stack", so they are the same axis and are taken TOGETHER as one settled
# operating point rather than stacked on each other. The merged corpus answers a different question
# entirely, which is why the composition is plausible here and was not there.
#
# THE FALSIFIABLE PART, written before the run. Control `merged_blend_cf` = 0.8597.
#   * >= 0.8659 (+0.0062, clears the floor). The dense geometry composes with the corpus axis and the
#     member is the new best by a real margin. Refit the champion's seats under merged+dense+cf and take
#     the roster there; the data axis is the schedule.
#   * 0.8626 to 0.8658 (+0.0029 to +0.0061). The composition holds at roughly the size `band_span` measured
#     alone. Directional, under the bar, and KEPT: it is a chain link, and it seats as the strongest member
#     we own. The follow-up is a fourth link, not a confirmation seed.
#   * within +/- 0.0029. The geometry does not transfer to the merged corpus. Read it as the merged corpus
#     ALREADY supplying whatever the extra slices supply — more studies covering the same anatomy — and
#     stop stacking decode levers onto it. The follow-up is then a fourth CORPUS, not a fifth knob.
#   * <= -0.0029. The 21% of studies Rijeka contributes are thinner stacks, and reading 20 slices from them
#     resamples further than they hold. Check the merged corpus's own slice counts before anything else;
#     this would be a harness finding, not a verdict on density.
#
# HARNESS CHECK BEFORE READING ANY OF THAT, and this arm carries FOUR flags. The printed recipe must name
# `merged` on "scheme", `merged_blend_v1.csv` on "labels", `report_labels_v2.csv` on "eval_labels",
# `"slices": 20`, `band [0.1, 0.9]` and `calibrated_targets true`. The log must load 5,143 studies and hold
# out 909. Three runs have entered this ledger as facts about an architecture when they were facts about a
# flag that never landed.
#
# VRAM. 20 slices at 336 px is 20 * 1.335 + 0.22 = 26.9 GB by the rule five arms have now confirmed, which
# fits the card with about 5 GB of headroom and does NOT fit alongside a second job. Check `card_check`.
#
# control  merged_blend_cf = 0.8597 (fold 0, scored against report_labels_v2)
#
# deliberate: --slices 20 (the arm's thesis: the settled dense stack geometry)
# deliberate: --band-low 0.1 (the span half of that geometry, +0.0039 alone in `band_span`)
# deliberate: --band-high 0.9 (the span half of that geometry, +0.0039 alone in `band_span`)
# deliberate: --eval-labels report_labels_v2.csv (without it the holdout measures disagreement with the
# deliberate: ruler every other fold-0 row was scored against, not the model)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"
require "$LABELS"

run_train merged_dense_cf \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --scheme merged --calibrated-targets \
  --labels "$MERGED_BLEND" --eval-labels "$LABELS" --out "$MODELS/merged_dense_cf.pt"
