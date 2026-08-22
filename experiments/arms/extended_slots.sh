#!/usr/bin/env bash
# Route the three sequences the canonical six slots cannot express. Nine slots instead of six.
#
# THE LOSS, measured over all 4,407 studies by core.data.corpus.slot_ties --lost, one DICOM header per series
# and no pixels: 2,421 of 24,371 series (9.9%) match NO slot predicate, so nothing can ever route them.
# Three sequences are 96% of that:
#
#   Coronal | fluid | no fat-sat   1,139
#   Axial   | T1    | no fat-sat     621
#   Axial   | fluid | no fat-sat     575
#
# WHY THIS IS THE SAME FAMILY AS THE ONE THING THAT WORKED. Every large effect in this project has
# come from giving the model more of the data it was already handed - whole-stack reading was +0.0147
# and its mechanism is exactly that. Every knob INSIDE the design returned ~0.005. This is a data
# recovery, not a knob.
#
# IT IS NOT AIMED AT MCL, AND AN EARLIER VERSION OF THIS HEADER CLAIMED IT WAS. A coverage audit over
# mask_336px_12sl.npy killed that framing: 95.5% of studies already carry a coronal (COR_FLUID_FS) and
# 64.1% already carry an ANATOMICAL coronal (COR_T1), which is the contrast a collateral ligament is
# actually read on. So the 1,139 Coronal|fluid|nofs series mostly add a THIRD coronal view to studies
# that already have two - not a first coronal to studies that have none.
#
# WHAT REMAINS TRUE: 83,297 slices are never seen by any step, and this is the same 'more data' family
# as whole-stack. What is gone is any reason to expect the gain to land on a NAMED label, so read the
# mean - which is also the only number this holdout can resolve, since per-label AUC carries a
# +/-0.03 to +/-0.06 error bar at these positive counts (knee-2rx).
#
# control  wsfold0_e6 = 0.8457, the same whole-stack recipe over six slots.
# cost     a 9-slot cache is 1.5x the 6-slot one (~54 GB) and a step encodes 1.5x the images.
#
# NOT INTEROPERABLE, on purpose: a nine-slot member has a wider slot embedding than the published
# six-slot ones, so it cannot join an ensemble with them. That is a cost this arm pays knowingly, and
# it is cheaper now than it was yesterday - fold and seed ensembling are out, and the entry we are
# moving toward is a single model.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# The scheme is part of the cache's NAME and therefore part of this gate. The canonical mask already
# exists, so gating without the scheme would pass immediately and train on a cache that is not there.
wait_for_cache 336 12 extended

run_train extended_slots \
  $RECIPE --epochs 12 \
  --slice-pooling max --whole-stack --scheme extended \
  $THROUGHPUT \
  --labels "$LABELS" --out "$PROCESSED/extended_ws_s2026.pt"
