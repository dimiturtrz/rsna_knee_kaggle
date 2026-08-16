#!/usr/bin/env bash
# Read every sequence TWICE — 130 mm for context and 80 mm for detail — and let slot attention pool
# over scale. The build knee-1nn specifies, gated by crop80_tissue.sh.
#
# WHY TWO EXTENTS RATHER THAN A BETTER ONE. Both directions have already paid, separately and against
# each other: res392 (finer pixels) is our best single member at 0.8523, and whole_stack (more context)
# is the biggest arm we have run at +0.0147. A single crop has to trade them, because the twelve
# findings do not want the same field of view — measured against radiologist annotations (knee-dtk),
# the model BEATS its own labels on the diffuse findings and loses badly on the small ones, a split by
# structure SIZE. Two extents need not trade: the head already pools over slots, so scale enters as six
# more slots and NOT as a new mechanism, a second head, or a fusion rule to tune.
#
# THE GATE THIS IS DOWNSTREAM OF. crop80_tissue.sh reads 80 mm ALONE and predicts a two-sided per-label
# split (small structures up, diffuse findings down). That arm decides whether resolution is the
# mechanism at all; this one is only worth its cache if the split appeared. A uniform fall there
# refutes both.
#
# WHY --centre tissue IS NOT OPTIONAL HERE. `SlotReader.window` takes the reader's centre for BOTH
# extents, so the tight crop lands wherever the wide one is centred. At 80 mm the half-extent is 40 mm
# against a joint some 70-80 mm wide, and the measured p90 positioning error is 18 mm — so an
# image-centred tight slot would sometimes miss the joint, and the detail half would fail for a reason
# that has nothing to do with scale (knee-yrf).
#
# control  the canonical six-slot whole-stack members on the same fold and labels — extended_slots.sh
#          is the OTHER way to spend twelve slots (more sequences, one extent) and is the comparison
#          that says whether the gain is SCALE or merely slot count.
#
# NEEDS ITS OWN CACHE, ~71.6 GB, which no other arm builds. Twelve slots against the canonical six, so
# 2x that cache. RUN IT ALONE — build_cache.sh has no resume and once killed a trainer and itself:
#   experiments/arms/build_cache.sh 336 12 multiscale 6 130 tissue
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

# BATCH 6, NOT THE STANDING 12, and this is arithmetic rather than caution. whole_stack.sh measured its
# 20 GB budget at 12 studies x 6 slots x 12 slices = 864 images a step; twelve slots at batch 12 is
# 1728, and extended_slots.sh already needed 26 GB for NINE. Scaling that linearly puts twelve slots
# near 35 GB on a 32.6 GB card. Batch 6 restores the images-per-step this budget was measured at, so
# the memory point is the known one and only the step count changes.
run_train multiscale \
  $RECIPE --epochs 8 --batch 6 \
  --slice-pooling max --whole-stack --recompute-activations --vram-gb 20 \
  --scheme multiscale --centre tissue \
  --labels "$LABELS" --out "$PROCESSED/multiscale_s2026.pt"
